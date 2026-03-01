# src/rag/strategies/retrievers/vector_text.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.milvus_client import get_milvus_client
from src.core.config import settings
import logging
from typing import List
import os

logger = logging.getLogger(__name__)

class VectorTextRetriever(BaseRetrievalStrategy):
    """
    主路: 标准向量检索
    目标字段：text
    输入查询：原始查询
    """
    def __init__(self):
        self.milvus = get_milvus_client()
        self.use_parent_context = settings.rag_offline.chunk_strategy == "parent_child"
        logger.info(f"🔌 [Plugin] 加载原文向量插件 VectorTextRetriever 检索模式: ({'查子反父' if self.use_parent_context else 'normal'})")

    def search(self, query: str, top_k: int, filter_expr: str = None, **kwargs) -> List[SearchResult]:
        logger.debug(f"🔍 [VectorText] 检索：{query}")
        hits = self.milvus.search(
            query=query,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=["text", "metadata"]
        )
        
        results = []
        for hit in hits:
            meta = hit['metadata'] or {}
            original_text = hit['text']
            
            final_text = original_text
            # source_tag = "vector_text"
            source_tag = os.path.splitext(os.path.basename(__file__))[0]
            # 👇 核心逻辑：查子返父
            if self.use_parent_context and meta.get("parent_text"):
                final_text = meta["parent_text"]
                source_tag = source_tag + "_parent"
                # 可选：在 metadata 中记录原始子块文本，方便调试
                meta['_matched_child_text'] = original_text

        results.append(SearchResult(
                text=final_text, # 返回大块文本给 LLM
                score=hit['score'], # 分数基于小子块匹配 (精准)
                metadata=meta,
                source_field=source_tag
            ))

        return results