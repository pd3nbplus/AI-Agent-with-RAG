from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.rag.pipeline import RetrievalPipeline
from src.rag.strategies.base import SearchResult


class RAGPipelineAdapter:
    """对现有 RetrievalPipeline 做轻量适配。"""

    def __init__(self, pipeline: Optional[RetrievalPipeline] = None):
        self.pipeline = pipeline or RetrievalPipeline()

    async def retrieve(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        run_config = dict(config or {})
        if category:
            run_config.setdefault("filter", {})
            run_config["filter"]["category"] = category
        return await self.pipeline.run(query=query, config=run_config)
