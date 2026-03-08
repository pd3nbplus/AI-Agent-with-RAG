from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.rag.strategies.base import SearchResult


class RetrieveNode:
    """检索节点：调用现有 RAG pipeline。"""

    def __init__(self, adapter: Any):
        self.adapter = adapter

    async def run(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        return await self.adapter.retrieve(query=query, config=config, category=category)
