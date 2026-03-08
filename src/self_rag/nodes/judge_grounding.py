from __future__ import annotations

from typing import Any, List
from src.self_rag.schemas.judge import JudgeResult


class JudgeGroundingNode:
    """评判回答是否被证据支撑。"""

    def __init__(self, llm: Any, prompt: str, threshold: float):
        self.llm = llm
        self.prompt = prompt
        self.threshold = threshold

    async def run(self, query: str, answer: str, contexts: List[str]) -> JudgeResult:
        result = await self.llm.generate_structured(
            template=self.prompt,
            payload={
                "query": query,
                "answer": answer,
                "contexts": "\n\n".join(contexts) if contexts else "（无可用上下文）",
            },
            output_model=JudgeResult,
        )
        result.passed = result.score >= self.threshold
        return result
