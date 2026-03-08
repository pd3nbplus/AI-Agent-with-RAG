from __future__ import annotations

from typing import Any, List


class RewriteQueryNode:
    """查询改写节点：在评判失败时生成新的检索 query。"""

    def __init__(self, llm: Any, prompt: str):
        self.llm = llm
        self.prompt = prompt

    async def run(
        self,
        original_query: str,
        current_query: str,
        answer: str,
        failure_reasons: List[str],
    ) -> str:
        rewritten = await self.llm.generate_text(
            template=self.prompt,
            payload={
                "original_query": original_query,
                "current_query": current_query,
                "answer": answer,
                "failure_reasons": "; ".join(failure_reasons) if failure_reasons else "评分未达阈值",
            },
        )
        return rewritten.strip()
