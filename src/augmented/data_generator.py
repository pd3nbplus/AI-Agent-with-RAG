# augmented/data_generator.py
# 主编排器（策略模式）：
# Milvus 取 chunk -> 按策略构建任务 -> LLM 生成评估样本 -> PostgreSQL 持久化。
import ast
import json
import logging
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.augmented.config import GeneratorConfig, build_default_config
from src.augmented.llm_router import LLMRouter
from src.augmented.models import EvalSample
from src.augmented.prompts import PromptRegistry
from src.augmented.sinks import PostgresSink
from src.augmented.sources import MilvusSource
from src.augmented.strategies import StrategyTask, build_strategies

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, config: Optional[GeneratorConfig] = None):
        # [初始化-1] 读取运行配置（环境变量 + 默认值）
        self.config = config or build_default_config()
        # [初始化-2] 准备 Prompt 注册器与缓存（不同策略使用不同模板）
        self.prompt_registry = PromptRegistry()
        self.prompt_cache: Dict[str, ChatPromptTemplate] = {}
        # [初始化-3] 解析策略参数（JSON 字符串 -> dict）
        self.strategy_params = self._safe_parse_strategy_params(self.config.strategy_params_json)
        # [初始化-4] 为 standard 策略补默认题目数（向后兼容旧配置）
        self.strategy_params.setdefault("standard", {})
        self.strategy_params["standard"].setdefault("num_questions", self.config.num_questions_per_chunk)

        # [初始化-5] 准备核心组件：LLM 路由、数据源、存储端、策略集合
        self.router = LLMRouter(config=self.config)
        self.source = MilvusSource()
        self.sink = PostgresSink()
        self.strategies = build_strategies(self.config.enabled_strategies, self.strategy_params)

    def _get_prompt(self, profile: str) -> ChatPromptTemplate:
        if profile not in self.prompt_cache:
            self.prompt_cache[profile] = ChatPromptTemplate.from_template(self.prompt_registry.get(profile))
        return self.prompt_cache[profile]

    @staticmethod
    def _safe_parse_strategy_params(raw: str) -> Dict[str, Dict[str, Any]]:
        try:
            parsed = json.loads(raw) if raw else {}
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _extract_json_candidate(text: str) -> str:
        l_arr = text.find("[")
        r_arr = text.rfind("]")
        if l_arr != -1 and r_arr != -1 and r_arr > l_arr:
            return text[l_arr : r_arr + 1]

        l_obj = text.find("{")
        r_obj = text.rfind("}")
        if l_obj != -1 and r_obj != -1 and r_obj > l_obj:
            return text[l_obj : r_obj + 1]

        return text

    @staticmethod
    def _safe_parse_json(content: str) -> List[Dict[str, Any]]:
        text = content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            candidate = DatasetGenerator._extract_json_candidate(text)
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                data = ast.literal_eval(candidate)

        if isinstance(data, dict):
            data = [data]
        return data if isinstance(data, list) else []

    def generate_from_task(self, task: StrategyTask) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        # [任务执行-1] 按策略名选择 Prompt 模板
        last_error_msg = ""
        prompt = self._get_prompt(task.prompt_profile)
        # [任务执行-2] 单任务重试循环（网络抖动/模型输出异常时重试）
        for attempt in range(self.config.max_retries_per_chunk + 1):
            try:
                # [任务执行-3] 调用 LLM（返回文本 + 实际使用模型名）
                content, model_used = self.router.invoke(prompt=prompt, payload=task.payload)
                # [任务执行-4] 解析 JSON（带容错）
                raw_samples = self._safe_parse_json(content)
                valid_samples = []
                validation_failed = 0
                for raw in raw_samples:
                    try:
                        # [任务执行-5] 字段清洗与结构校验（Pydantic）
                        if isinstance(raw.get("ground_truth_context"), str):
                            raw["ground_truth_context"] = [raw["ground_truth_context"]]
                        valid_samples.append(EvalSample(**raw).model_dump())
                    except ValidationError:
                        validation_failed += 1
                        continue
                if valid_samples:
                    return valid_samples, model_used
                last_error_msg = (
                    f"JSON解析成功但无有效样本(raw={len(raw_samples)}, validation_failed={validation_failed})"
                )
            except Exception as e:
                last_error_msg = str(e)
                logger.error(
                    "任务生成失败 (attempt=%s): %s | strategy=%s | content_preview=%s",
                    attempt + 1,
                    e,
                    task.strategy_name,
                    content[:200] if "content" in locals() else "",
                )
            time.sleep(0.2)

        logger.warning(
            "任务生成失败，已放弃。strategy=%s reason=%s | task_preview=%s",
            task.strategy_name,
            last_error_msg,
            str(task.payload)[:160],
        )
        return [], None

    def generate(self) -> List[Dict[str, Any]]:
        # [主流程-1] 从 Milvus 拉取候选 chunk
        chunks = self.source.load_chunks(limit=self.config.chunks_limit)
        logger.info(
            "开始生成评估集: chunks_limit=%s, min_chunk_length=%s, standard_num_questions=%s",
            self.config.chunks_limit,
            self.config.min_chunk_length,
            self.config.num_questions_per_chunk,
        )
        logger.info("启用策略=%s", [s.name for s in self.strategies])
        logger.info("策略参数=%s", self.strategy_params)
        logger.info("Milvus加载到chunk数量=%s", len(chunks))
        if not chunks:
            logger.warning("⚠️ 未加载到任何 chunk，请检查 source 配置。")
            return []

        # [主流程-2] 预过滤过短 chunk，避免低质量样本
        filtered_chunks: List[Dict[str, Any]] = []
        filtered_short = 0
        for item in chunks:
            text = item.get("text", "")
            if len(text.strip()) < self.config.min_chunk_length:
                filtered_short += 1
                continue
            filtered_chunks.append(item)
        logger.info("过滤过短chunk=%s, 可用chunk=%s", filtered_short, len(filtered_chunks))

        # [主流程-3] 各策略构建任务（standard / adversarial / mixed_pair）
        strategy_tasks: List[Tuple[StrategyTask, str]] = []
        strategy_map = {}
        for strategy in self.strategies:
            strategy_map[strategy.name] = strategy
            tasks = strategy.build_tasks(filtered_chunks)
            for t in tasks:
                strategy_tasks.append((t, strategy.name))
        logger.info("策略任务总数=%s", len(strategy_tasks))

        # [主流程-4] 逐任务执行 LLM + 策略后处理（如对抗题答案强制注入）
        raw_results: List[Tuple[StrategyTask, str, List[Dict[str, Any]], Optional[str]]] = []
        for task, strategy_name in strategy_tasks:
            samples, model_used = self.generate_from_task(task)
            samples = strategy_map[strategy_name].postprocess_samples(samples, task)
            raw_results.append((task, strategy_name, samples, model_used))

        success_tasks = sum(1 for _, _, samples, _ in raw_results if samples)
        failed_tasks = len(raw_results) - success_tasks
        total_generated = sum(len(samples) for _, _, samples, _ in raw_results)
        logger.info(
            "任务生成统计: success_tasks=%s, failed_tasks=%s, generated_samples=%s",
            success_tasks,
            failed_tasks,
            total_generated,
        )

        # [主流程-5] 统一组装落库数据（补 model_name、strategy、source 索引等追踪字段）
        all_samples: List[Dict[str, Any]] = []
        now_ts = int(time.time())
        today = date.today().isoformat()

        for task, strategy_name, samples, model_used in raw_results:
            first_idx = task.source_chunk_indices[0] if task.source_chunk_indices else -1
            if strategy_name == "mixed_pair":
                a_src = task.source_metadata.get("chunk_a_metadata", {}).get("source")
                b_src = task.source_metadata.get("chunk_b_metadata", {}).get("source")
                source_document = f"{a_src}|{b_src}"
            else:
                source_document = task.source_metadata.get("source")

            for sample_idx, sample in enumerate(samples):
                sample_meta = sample.get("metadata", {}) or {}
                sample_meta.setdefault("generated_by", model_used or self.config.default_model_name)
                sample_meta.setdefault("generation_date", today)
                sample_meta.setdefault("strategy", strategy_name)
                sample_meta.setdefault("source_chunk_indices", task.source_chunk_indices)

                all_samples.append(
                    {
                        "id": f"gen_{now_ts}_{strategy_name}_{first_idx}_{sample_idx}",
                        "category": sample.get("category", task.source_metadata.get("category", "general")),
                        "difficulty": sample["difficulty"],
                        "query": sample["query"],
                        "ground_truth_context": sample["ground_truth_context"],
                        "ground_truth_answer": sample["ground_truth_answer"],
                        "source_document": sample.get("source_document") or source_document,
                        "model_name": model_used or self.config.default_model_name,
                        "metadata": sample_meta,
                        "source_chunk_index": first_idx,
                        "source_backend": "milvus",
                        "created_at": now_ts,
                        # 记录该题目属于哪个评测批次，后续可按批次回放评估。
                        "batch_id": int(self.config.sample_batch_id),
                    }
                )

        # [主流程-6] 批量写入 PostgreSQL
        self.sink.save(all_samples)
        logger.info("✅ 数据集生成完成，样本数=%s，已写入 PostgreSQL", len(all_samples))
        return all_samples


if __name__ == "__main__":
    # python -m src.augmented.data_generator
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    generator = DatasetGenerator()
    generator.generate()
