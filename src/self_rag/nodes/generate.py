from __future__ import annotations

from typing import Any, List


class GenerateNode:
    """生成节点：基于问题与上下文生成回答。"""

    def __init__(self, llm: Any, prompt: str):
        self.llm = llm
        self.prompt = prompt

    async def run(self, query: str, contexts: List[str]) -> str:
        return await self.llm.generate_text(
            template=self.prompt,
            payload={
                "query": query,
                "contexts": "\n\n".join(contexts) if contexts else "（无可用上下文）",
            },
        )
