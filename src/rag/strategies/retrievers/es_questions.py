# src/rag/strategies/retrievers/es_questions.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.es_client import es_client_instance
from src.core.config import settings
from typing import List, Optional
import logging
import os
logger = logging.getLogger(__name__)

class ESQuestionsRetriever(BaseRetrievalStrategy):
    """
    插件 3: ES 关键词检索 (Questions 字段)
    """
    def __init__(self):
        self.es = es_client_instance
        self.index_name = settings.db.es_index_questions
        if self.es.is_available():
            logger.info(f"🔌 [Plugin] ES 检索插件{self.index_name}已就绪")
        else:
            logger.warning("🔌 [Plugin] ES 不可用，此插件将自动跳过")

    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        if not self.es.is_available():
            return []
            
        # 调用封装好的搜索方法
        # 注意：ES 原生不支持复杂的 JSON 过滤表达式 (如 Milvus 语法)，这里暂不实现 filter_expr
        # 如果需要，可以在 ES 查询中添加 term 过滤
        hits = self.es.search_questions(query, top_k=top_k)
        
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
                text=final_text,
                score=hit['score'],
                metadata=meta,
                source_field=source_tag
            ))
        return results