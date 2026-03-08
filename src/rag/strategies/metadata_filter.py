# src/rag/strategies/metadata_filter.py
from .base import BaseFilterStrategy
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MetadataFilterBuilder(BaseFilterStrategy):
    def build_expr(self, **kwargs) -> Optional[str]:
        """
        构建 Milvus 标量过滤表达式
        支持：category, source, min_page
        """
        filters = []
        
        # 1. 分类过滤
        category = kwargs.get("category")
        if category:
            # Milvus JSON 字段过滤语法：metadata["key"] == "value"
            filters.append(f'(metadata["category"] == "{category}")')
            
        # 2. 来源文件过滤
        source = kwargs.get("source")
        if source:
            filters.append(f'(metadata["source"] == "{source}")')
            
        # 3. 页码过滤
        min_page = kwargs.get("min_page")
        if min_page is not None:
            filters.append(f'(metadata["page"] >= {int(min_page)})')
            
        if not filters:
            return None
            
        expr = " and ".join(filters)
        logger.info(f"🔒 [Filter] 应用过滤策略：{expr}")
        return expr