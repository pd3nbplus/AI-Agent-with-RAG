"""RAG 评估结果分析器。通过 LLMRouter 进行诊断生成。"""
# src/augmented/analyst.py
from __future__ import annotations

import asyncio
import ast
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from src.augmented.config import GeneratorConfig, build_default_config
from src.core.models import RagEvalDiagnosis, RagEvalResult
from src.core.postgres_client import get_postgres_client
from src.augmented.llm_router import LLMRouter
from sqlalchemy import desc

logger = logging.getLogger(__name__)


class RAGAnalyst:
    """分析低分样本并给出诊断建议。"""

    def __init__(self, generator_config: Optional[GeneratorConfig] = None):
        # 使用 llm_router 管理多模型降级调用。
        self.generator_config = generator_config or build_default_config()
        self.router = LLMRouter(self.generator_config, llm_group="analyst_llms")
        # 保存最近一次分析生成的完整 markdown 报告。
        self.report: str = ""

        self._prompt = ChatPromptTemplate.from_template(
            """
你是一个资深的 RAG 系统调试专家。
以下是一条低分样本，请输出结构化诊断：

【用户问题】{question}
【检索上下文】{contexts}
【模型回答】{answer}
【标准答案】{ground_truth}
【标准上下文】{ground_truth_contexts}

【指标】
- Faithfulness: {faithfulness}
- Answer Relevancy: {answer_relevancy}
- Context Precision: {context_precision}
- Context Recall: {context_recall}

请输出：
1) 根因判断（检索失败 / 排序失败 / 生成失败 / 混合问题）
2) 证据（引用上下文或回答中的具体片段）
3) 1条可执行优化建议（具体到参数或策略）

直接输出分析结论，不要客套。
"""
        )

    @staticmethod
    def _run_async(coro):
        """在同步代码里安全运行协程。"""
        try:
            asyncio.get_running_loop()
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        except RuntimeError:
            return asyncio.run(coro)

    @staticmethod
    def _pick(row: pd.Series, *keys: str, default=None):
        for k in keys:
            if k not in row:
                continue

            v = row[k]
            # 复杂结构（list/dict 等）直接视为有效值，避免 pd.notna 返回数组导致歧义。
            if isinstance(v, (list, tuple, dict, set)):
                return v
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                # 对不支持 isna 判断的对象，按有效值处理。
                pass
            return v
        return default

    @staticmethod
    def _to_text_list(value) -> List[str]:
        """将上下文字段统一为字符串列表，兼容 CSV 中的字符串化 list。"""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x) for x in value]
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            # 兼容 "['a', 'b']" 这类字符串化列表。
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            return [s]
        return [str(value)]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """归一化文本以便做规则匹配：去掉 <think>、空白和中英文常见标点。"""
        s = str(text or "")
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
        s = s.strip().lower()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[，。！？；：、,.!?;:\"'`~\-_\(\)\[\]{}<>]", "", s)
        return s

    def _is_unanswerable_text(self, text: str) -> bool:
        """判断文本是否表达“依据上下文无法回答”。"""
        s = self._normalize_text(text)
        cues = [
            "根据提供的上下文无法回答",
            "根据上下文无法回答",
            "无法根据提供的上下文回答",
            "上下文无法支持答案",
            "未明确提及",
            "信息不足",
            "无法依据现有信息判断",
            "无法确定",
            "无法回答",
        ]
        return any(cue in s for cue in cues)

    def _is_expected_unanswerable_and_answered_correctly(self, row: pd.Series) -> bool:
        """
        识别“正确拒答”样本：
        - 标准答案本身是不可回答
        - 模型回答也明确不可回答
        这类样本不应作为坏案例。
        """
        gt = self._pick(row, "ground_truth", "reference", "ground_truth_answer", default="")
        ans = self._pick(row, "answer", "response", default="")
        return self._is_unanswerable_text(gt) and self._is_unanswerable_text(ans)

    def _normalize_case(self, row: pd.Series) -> Dict:
        """兼容 evaluator 输出的多种字段名。"""
        contexts = self._to_text_list(self._pick(row, "contexts", "retrieved_contexts", default=[]))
        gt_contexts = self._to_text_list(
            self._pick(row, "ground_truth_contexts", "reference_contexts", default=[])
        )

        return {
            "sample_id": self._pick(row, "sample_id", "id", default=""),
            "sample_batch_id": int(self._pick(row, "sample_batch_id", "batch_id", default=1) or 1),
            "question": self._pick(row, "question", "user_input", default=""),
            "answer": self._pick(row, "answer", "response", default=""),
            "contexts": contexts,
            "ground_truth": self._pick(row, "ground_truth", "reference", default=""),
            "ground_truth_contexts": gt_contexts,
            "faithfulness": float(self._pick(row, "faithfulness", default=0.0) or 0.0),
            "answer_relevancy": float(self._pick(row, "answer_relevancy", default=0.0) or 0.0),
            "context_precision": float(self._pick(row, "context_precision", default=0.0) or 0.0),
            "context_recall": float(self._pick(row, "context_recall", default=0.0) or 0.0),
        }

    def _score_and_select(self, df_results: pd.DataFrame, top_k: int) -> pd.DataFrame:
        required = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        missing = [c for c in required if c not in df_results.columns]
        if missing:
            raise ValueError(f"分析失败：缺少指标列 {missing}")

        work = df_results.copy()
        for c in required:
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

        # 先按评估分数构建候选集。
        work["avg_score"] = work[required].mean(axis=1)

        # 关键改进：排除“正确拒答”样本，避免将其误报为坏案例。
        # 典型场景：标准答案就是“无法回答”，模型也确实返回“无法回答”。
        work["is_correct_abstention"] = work.apply(
            self._is_expected_unanswerable_and_answered_correctly, axis=1
        )
        candidates = work[~work["is_correct_abstention"]].copy()

        if candidates.empty:
            logger.info("坏案例筛选后为空：所有低分样本均为‘正确拒答’或无需诊断。")
            return candidates

        return candidates.nsmallest(top_k, "avg_score")

    async def _diagnose_one(self, payload: Dict) -> Dict[str, str]:
        """在线程池中执行同步 router.invoke，避免阻塞事件循环。"""
        text, model = await asyncio.to_thread(self.router.invoke, self._prompt, payload)
        return {"diagnosis": text, "model": model or self.generator_config.default_model_name}

    async def analyze_bad_cases(self, df_results: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        """异步分析低分 Top-K 样本。"""
        if df_results is None or df_results.empty:
            return []

        worst_cases = self._score_and_select(df_results, top_k)
        logger.info("开始分析低分样本：top_k=%s", top_k)

        reports: List[Dict] = []
        for idx, row in worst_cases.iterrows():
            item = self._normalize_case(row)
            payload = {
                "question": item["question"],
                "contexts": "\n".join(item["contexts"]),
                "answer": item["answer"],
                "ground_truth": item["ground_truth"],
                "ground_truth_contexts": "\n".join(item["ground_truth_contexts"]),
                "faithfulness": item["faithfulness"],
                "answer_relevancy": item["answer_relevancy"],
                "context_precision": item["context_precision"],
                "context_recall": item["context_recall"],
            }

            try:
                diagnose_out = await self._diagnose_one(payload)
                diagnosis = diagnose_out["diagnosis"]
                diagnosis_model = diagnose_out["model"]
            except Exception as e:
                logger.exception("低分样本诊断失败 idx=%s", idx)
                diagnosis = f"诊断失败：{e}"
                diagnosis_model = self.generator_config.default_model_name

            reports.append(
                {
                    "id": idx,
                    "sample_id": item["sample_id"],
                    "sample_batch_id": item["sample_batch_id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "contexts": item["contexts"],
                    "ground_truth": item["ground_truth"],
                    "ground_truth_contexts": item["ground_truth_contexts"],
                    "scores": {
                        "faithfulness": item["faithfulness"],
                        "answer_relevancy": item["answer_relevancy"],
                        "context_precision": item["context_precision"],
                        "context_recall": item["context_recall"],
                        "avg_score": float(worst_cases.loc[idx, "avg_score"]),
                    },
                    "predicted_category": self._auto_categorize_error(item),
                    "diagnosis": diagnosis,
                    "diagnosis_model": diagnosis_model,
                }
            )

        return reports

    def analyze_bad_cases_sync(self, df_results: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        """同步入口，便于在脚本中直接调用。"""
        return self._run_async(self.analyze_bad_cases(df_results=df_results, top_k=top_k))

    @staticmethod
    def _clip_text(text: str, max_len: int = 500) -> str:
        if not isinstance(text, str):
            return ""
        s = text.strip()
        return s if len(s) <= max_len else s[:max_len] + "..."

    @staticmethod
    def _strip_think(text: str) -> str:
        """移除模型返回中的 <think> 思考片段，保留最终可展示答案。"""
        if not isinstance(text, str):
            return ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()

    @staticmethod
    def _format_diagnosis_text(text: str) -> str:
        """将诊断文本整理为更易读的段落/列表结构。"""
        if not isinstance(text, str):
            return ""

        formatted = text.strip().replace("\r\n", "\n").replace("\r", "\n")
        if not formatted:
            return ""
        formatted = re.sub(r"\*\*([^\n*]+)\n\*\*", r"**\1**", formatted)
        formatted = re.sub(r"\s*(#{2,6}\s*)", r"\n\1", formatted)
        formatted = re.sub(r"\s*(\d+\)\s*)", r"\n\1", formatted)
        formatted = re.sub(r"\s*(-\s+)", r"\n- ", formatted)
        formatted = re.sub(r"\s*(\*\*(?:根因判断|证据|可执行优化建议|优化建议)\*\*)", r"\n\1", formatted)
        formatted = re.sub(r"[ \t]+\n", "\n", formatted)
        formatted = re.sub(r"\n{3,}", "\n\n", formatted)
        return formatted.strip()

    @staticmethod
    def _to_blockquote(text: str) -> str:
        lines = (text or "").splitlines() or [""]
        return "\n".join([f"> {line}" if line.strip() else ">" for line in lines])

    def _format_context_blocks(self, contexts, max_items: int = 3, each_max_len: int = 300) -> str:
        items = self._to_text_list(contexts)
        if not items:
            return "> （无）"

        blocks = []
        for i, raw_ctx in enumerate(items[:max_items], start=1):
            ctx = self._clip_text(self._strip_think(str(raw_ctx)), max_len=each_max_len)
            blocks.append(f"#### 引用 {i}\n{self._to_blockquote(ctx)}")
        return "\n\n".join(blocks)

    def _build_case_markdown(self, report: dict, rank: int) -> str:
        """构建单个坏案例 markdown，同时用于数据库 diagnosis_markdown 字段。"""
        q = report.get("question", "")
        scores = report.get("scores", {})
        diagnosis = self._format_diagnosis_text(report.get("diagnosis", ""))
        category = report.get("predicted_category", "")
        model = report.get("diagnosis_model", "")
        answer = self._strip_think(report.get("answer", ""))
        contexts = report.get("contexts", [])
        ground_truth = report.get("ground_truth", "")
        gt_contexts = report.get("ground_truth_contexts", [])

        return (
            f"## 案例 {rank}: {category}\n"
            f"**问题**: {q}\n"
            f"**得分**: Faithfulness={float(scores.get('faithfulness', 0.0)):.2f}, "
            f"Recall={float(scores.get('context_recall', 0.0)):.2f}\n"
            f"**诊断模型**: {model}\n\n"
            f"### 🤖 模型生成答案\n{self._clip_text(answer, max_len=1000)}\n\n"
            f"### 📚 检索引用信息（Top-3）\n{self._format_context_blocks(contexts, max_items=3, each_max_len=500)}\n\n"
            f"### ✅ 标准答案（Ground Truth）\n{self._clip_text(ground_truth, max_len=1000)}\n\n"
            f"### 📖 标准参考上下文（Top-2）\n{self._format_context_blocks(gt_contexts, max_items=2, each_max_len=400)}\n\n"
            f"### 🩺 LLM 专家诊断\n{diagnosis}\n\n"
        )

    def _build_report_markdown(self, eval_run_id: str, reports: List[Dict]) -> str:
        head = f"# 🚨 RAG 坏案例深度诊断报告\n\n**评估运行ID**: `{eval_run_id}`\n\n"
        if not reports:
            return head + "（无坏案例）\n"
        body = []
        for i, report in enumerate(reports, start=1):
            body.append(self._build_case_markdown(report, i) + "---\n")
        return head + "\n".join(body)

    def _save_diagnosis_results(self, eval_run_id: str, reports: List[Dict]) -> int:
        """将坏案例诊断结果写入 rag_eval_diagnoses。"""
        if not reports:
            return 0

        # 关键点：诊断结果与评估结果共享 eval_run_id，支持后续追踪同一轮运行。
        rows = []
        now = datetime.utcnow()
        for rank, report in enumerate(reports, start=1):
            scores = report.get("scores", {}) or {}
            rows.append(
                {
                    "eval_run_id": eval_run_id,
                    "sample_id": str(report.get("sample_id", report.get("id", ""))),
                    "sample_batch_id": int(report.get("sample_batch_id", 1) or 1),
                    "bad_case_rank": rank,
                    "predicted_category": str(report.get("predicted_category", "")),
                    "diagnosis_model": str(report.get("diagnosis_model", "")),
                    "diagnosis": self._format_diagnosis_text(report.get("diagnosis", "")),
                    "diagnosis_markdown": self._build_case_markdown(report, rank),
                    "question": str(report.get("question", "")),
                    "answer": self._strip_think(str(report.get("answer", ""))),
                    "contexts": self._to_text_list(report.get("contexts", [])),
                    "ground_truth": str(report.get("ground_truth", "")),
                    "ground_truth_contexts": self._to_text_list(report.get("ground_truth_contexts", [])),
                    "faithfulness": float(scores.get("faithfulness", 0.0) or 0.0),
                    "answer_relevancy": float(scores.get("answer_relevancy", 0.0) or 0.0),
                    "context_precision": float(scores.get("context_precision", 0.0) or 0.0),
                    "context_recall": float(scores.get("context_recall", 0.0) or 0.0),
                    "avg_score": float(scores.get("avg_score", 0.0) or 0.0),
                    "created_at": now,
                }
            )

        pg = get_postgres_client()
        RagEvalDiagnosis.__table__.create(bind=pg.engine, checkfirst=True)
        with pg.get_session() as session:
            session.bulk_insert_mappings(RagEvalDiagnosis, rows)
            session.commit()
        return len(rows)

    def to_markdown(self, path: str) -> str:
        """将最近一次分析报告写入指定 markdown 文件。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.report or "# 🚨 RAG 坏案例深度诊断报告\n\n（空报告）\n")
        return str(target)

    def analyze(
        self,
        limit: int = 200,
        batch_ids: Optional[List[int]] = None,
        eval_run_id: Optional[str] = None,
        bad_case_top_k: int = 5,
    ) -> Dict:
        """
        一体化分析入口：
        1) 从评估结果表 rag_eval_results 按 batch_ids / eval_run_id 读取数据
        2) 分析坏案例并写入诊断表 rag_eval_diagnoses
        3) 生成完整 markdown 报告到 self.report
        """
        pg = get_postgres_client()
        with pg.get_session() as session:
            q = session.query(RagEvalResult)
            if batch_ids:
                q = q.filter(RagEvalResult.sample_batch_id.in_(batch_ids))
            if eval_run_id:
                q = q.filter(RagEvalResult.eval_run_id == eval_run_id)
            q = q.order_by(desc(RagEvalResult.created_at))
            rows = q.limit(limit).all()

        if not rows:
            run_id = eval_run_id or f"diag_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.report = self._build_report_markdown(eval_run_id=run_id, reports=[])
            return {
                "eval_run_id": run_id,
                "eval_count": 0,
                "diagnosis_count": 0,
                "df_results": pd.DataFrame(),
                "reports": [],
            }

        # 若未指定 eval_run_id，默认取最近一条记录所属的运行批次，避免混入多次运行数据。
        selected_run_id = eval_run_id or rows[0].eval_run_id
        selected_rows = [r for r in rows if r.eval_run_id == selected_run_id]
        df = pd.DataFrame(
            [
                {
                    "sample_id": r.sample_id,
                    "sample_batch_id": r.sample_batch_id,
                    "question": r.question,
                    "answer": r.answer,
                    "contexts": r.contexts or [],
                    "ground_truth": r.ground_truth,
                    "ground_truth_contexts": r.ground_truth_contexts or [],
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                }
                for r in selected_rows
            ]
        )

        if df is None or df.empty:
            self.report = self._build_report_markdown(eval_run_id=selected_run_id, reports=[])
            return {
                "eval_run_id": selected_run_id,
                "eval_count": 0,
                "diagnosis_count": 0,
                "df_results": df,
                "reports": [],
            }

        reports = self.analyze_bad_cases_sync(df_results=df, top_k=bad_case_top_k)
        saved_diag = self._save_diagnosis_results(eval_run_id=selected_run_id, reports=reports)
        self.report = self._build_report_markdown(eval_run_id=selected_run_id, reports=reports)

        logger.info(
            "分析完成 eval_run_id=%s source_eval_rows=%s diagnosis_saved=%s",
            selected_run_id,
            len(df),
            saved_diag,
        )
        return {
            "eval_run_id": selected_run_id,
            "eval_count": int(len(df)),
            "diagnosis_count": int(saved_diag),
            "df_results": df,
            "reports": reports,
        }

    def _auto_categorize_error(self, row: Dict) -> str:
        """基于指标 heuristic 自动归类。"""
        if row["context_recall"] < 0.5:
            return "检索失败 (Retrieval Failure)"
        if row["context_precision"] < 0.5:
            return "排序失败 (Ranking Failure)"
        if row["faithfulness"] < 0.6:
            return "幻觉问题 (Hallucination)"
        if row["answer_relevancy"] < 0.6:
            return "答非所问 (Irrelevance)"
        return "混合问题 / 需人工复核"
