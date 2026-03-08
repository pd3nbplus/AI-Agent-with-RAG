# augmented/strategies.py
# 策略模式：将“如何从 chunk 组织出题任务”从主流程剥离，便于扩展新策略。
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StrategyTask:
    # 该任务由哪个策略生成（standard/adversarial/mixed_pair）
    strategy_name: str
    # 对应 PromptRegistry 的模板名
    prompt_profile: str
    # 传给 LLM 的输入参数（上下文、问题数等）
    payload: Dict[str, Any]
    # 任务来源的 chunk 索引（单块策略 1 个，拼接策略 2 个）
    source_chunk_indices: List[int]
    # 来源元数据，供落库和追踪使用
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    # 该任务期望生成的问题数
    num_questions: int = 1


class BaseGenerationStrategy(ABC):
    name: str

    @abstractmethod
    def build_tasks(self, chunks: List[Dict[str, Any]]) -> List[StrategyTask]:
        # 将输入 chunks 转成“可执行任务列表”
        raise NotImplementedError

    def postprocess_samples(self, samples: List[Dict[str, Any]], _task: StrategyTask) -> List[Dict[str, Any]]:
        # 默认不处理；子类可覆盖（如对抗策略强制无答案）
        return samples


class StandardChunkStrategy(BaseGenerationStrategy):
    name = "standard"
    default_num_questions = 2

    def __init__(self, num_questions: int | None = None):
        self.num_questions = num_questions if num_questions is not None else self.default_num_questions

    def build_tasks(self, chunks: List[Dict[str, Any]]) -> List[StrategyTask]:
        # 一块文档对应一个任务，生成 num_questions 个常规题。
        tasks: List[StrategyTask] = []
        for idx, item in enumerate(chunks):
            tasks.append(
                StrategyTask(
                    strategy_name=self.name,
                    prompt_profile="standard",
                    payload={
                        "context_chunk": item.get("text", ""),
                        "num_questions": self.num_questions,
                    },
                    source_chunk_indices=[idx],
                    source_metadata=item.get("metadata", {}) or {},
                    num_questions=self.num_questions,
                )
            )
        return tasks


class AdversarialStrategy(BaseGenerationStrategy):
    name = "adversarial"
    fixed_answer = "根据提供的上下文无法回答"
    default_num_questions = 1

    def __init__(self, num_questions: int | None = None):
        self.num_questions = num_questions if num_questions is not None else self.default_num_questions

    def build_tasks(self, chunks: List[Dict[str, Any]]) -> List[StrategyTask]:
        # 一块文档对应一个“对抗任务”：问题看似相关，但答案应不可从上下文得到。
        tasks: List[StrategyTask] = []
        for idx, item in enumerate(chunks):
            tasks.append(
                StrategyTask(
                    strategy_name=self.name,
                    prompt_profile="adversarial",
                    payload={
                        "context_chunk": item.get("text", ""),
                        "num_questions": self.num_questions,
                    },
                    source_chunk_indices=[idx],
                    source_metadata=item.get("metadata", {}) or {},
                    num_questions=self.num_questions,
                )
            )
        return tasks

    def postprocess_samples(self, samples: List[Dict[str, Any]], _task: StrategyTask) -> List[Dict[str, Any]]:
        # 对抗题目的标准答案由程序强制注入，避免模型偏离指令。
        for s in samples:
            s["ground_truth_answer"] = self.fixed_answer
            s["difficulty"] = "hard"
            # 对抗题不应存在可支持答案的证据片段。
            s["ground_truth_context"] = []
        return samples


class MixedPairStrategy(BaseGenerationStrategy):
    name = "mixed_pair"
    default_pair_count = 2
    default_num_questions = 3

    def __init__(self, pair_count: int | None = None, num_questions: int | None = None, seed: int | None = None):
        self.pair_count = pair_count if pair_count is not None else self.default_pair_count
        # 业务约束：该策略必须生成 3 题（E/M/H），若传入非法值则强制回到 3。
        n = num_questions if num_questions is not None else self.default_num_questions
        self.num_questions = 3 if n != 3 else n
        self.seed = seed

    def build_tasks(self, chunks: List[Dict[str, Any]]) -> List[StrategyTask]:
        # 该策略要求“跨块推理”，因此至少需要两个 chunk。
        if len(chunks) < 2:
            return []

        indices = list(range(len(chunks)))
        if self.seed is not None:
            # 固定随机种子可复现“随机配对”结果，便于实验对比。
            random.seed(self.seed)
        random.shuffle(indices)
        # 每两个 chunk 组成一对；pair_count 控制最多取几对。
        max_pairs = max(0, min(self.pair_count, len(indices) // 2))
        
        tasks: List[StrategyTask] = []
        for i in range(max_pairs):
            a_idx = indices[2 * i]
            b_idx = indices[2 * i + 1]
            a = chunks[a_idx]
            b = chunks[b_idx]
            tasks.append(
                StrategyTask(
                    strategy_name=self.name,
                    prompt_profile="mixed_pair",
                    payload={
                        "chunk_a": a.get("text", ""),
                        "chunk_b": b.get("text", ""),
                        "num_questions": self.num_questions,
                    },
                    source_chunk_indices=[a_idx, b_idx],
                    source_metadata={
                        "chunk_a_metadata": a.get("metadata", {}) or {},
                        "chunk_b_metadata": b.get("metadata", {}) or {},
                    },
                    num_questions=self.num_questions,
                )
            )
        return tasks


def build_strategies(enabled_strategies: str, strategy_params: Dict[str, Dict[str, Any]] | None = None) -> List[BaseGenerationStrategy]:
    # 根据开关和参数构建策略实例，顺序决定执行顺序。
    enabled = {x.strip().lower() for x in enabled_strategies.split(",") if x.strip()}
    params = strategy_params or {}

    strategy_map = {
        "standard": StandardChunkStrategy(**params.get("standard", {})),
        "adversarial": AdversarialStrategy(**params.get("adversarial", {})),
        "mixed_pair": MixedPairStrategy(**params.get("mixed_pair", {})),
    }
    return [strategy_map[name] for name in ("standard", "adversarial", "mixed_pair") if name in enabled]
