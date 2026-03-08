from __future__ import annotations

from typing import Any

from src.self_rag.schemas.judge import JudgeResult


class JudgeUtilityNode:
    """评判回答对用户问题是否有用。"""

    def __init__(self, llm: Any, prompt: str, threshold: float):
        self.llm = llm
        self.prompt = prompt
        self.threshold = threshold

    async def run(self, query: str, answer: str) -> JudgeResult:
        result = await self.llm.generate_structured(
            template=self.prompt,
            payload={"query": query, "answer": answer},
            output_model=JudgeResult,
        )
        result.passed = result.score >= self.threshold
        return result
