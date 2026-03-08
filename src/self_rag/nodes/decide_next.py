from __future__ import annotations

from src.self_rag.schemas.judge import JudgeResult


class DecideNextNode:
    """根据评判分数与轮次决定下一步动作。"""

    def run(
        self,
        hop: int,
        max_hops: int,
        relevance: JudgeResult,
        grounding: JudgeResult,
        utility: JudgeResult,
    ) -> str:
        if relevance.passed and grounding.passed and utility.passed:
            return "finish"
        if hop >= max_hops:
            return "fallback"
        return "rewrite"
