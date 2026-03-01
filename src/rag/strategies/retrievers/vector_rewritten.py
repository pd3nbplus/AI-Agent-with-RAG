# src/rag/strategies/retrievers/vector_rewritten.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.milvus_client import get_milvus_client
from src.rag.rewriter import rewriter_instance # 复用之前的重写器
from src.core.config import settings
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)

class VectorRewrittenRetriever(BaseRetrievalStrategy):
    """
    插件 2: 变体向量检索
    策略：先让 LLM 重写查询 (Query Rewriting)，再用重写后的句子进行向量搜索。
    """
    def __init__(self):
        self.milvus = get_milvus_client()
        self.rewriter = rewriter_instance
        # 从配置读取是否启用父子上下文
        self.use_parent_context = settings.rag_offline.chunk_strategy == "parent_child"
        logger.info(f"🔌 [Plugin] 加载变体向量检索插件 (Rewritten) 检索模式: ({'查子反父' if self.use_parent_context else 'normal'})")

    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        # 1. 生成变体查询
        try:
            rewritten_query = self.rewriter.rewrite(query)
            if rewritten_query == query:
                logger.debug("⚠️ [Vector-Rewritten] 重写后无变化，跳过此路以避免重复")
                return []
            logger.info(f"🔄 [Vector-Rewritten] 变体查询：{rewritten_query}")
        except Exception as e:
            logger.error(f"❌ [Vector-Rewritten] 重写失败：{e}")
            return []
        
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