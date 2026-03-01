# src/rag/factories.py
from src.rag.chunkers import RecursiveChunker, FixedChunker, BaseChunker, ParentChildChunker
from src.rag.reranker import Reranker # 引入之前的重排类
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class ChunkerFactory:
    @staticmethod
    def get_chunker() -> BaseChunker:
        strategy = settings.rag_offline.chunk_strategy.lower()
        
        # 解析分隔符字符串为列表
        separators = settings.rag_offline.chunk_separators.split(',') if settings.rag_offline.chunk_separators else ["\n\n"]
        
        if strategy == "recursive":
            return RecursiveChunker(
                chunk_size=settings.rag_offline.chunk_size,
                chunk_overlap=settings.rag_offline.chunk_overlap,
                separators=separators
            )
        elif strategy == "fixed":
            return FixedChunker(
                chunk_size=settings.rag_offline.chunk_size,
                chunk_overlap=settings.rag_offline.chunk_overlap
            )
        elif strategy == "parent_child":
            return ParentChildChunker(
                parent_size=settings.rag_offline.chunk_size,
                child_size=settings.rag_offline.child_chunk_size, 
                overlap=settings.rag_offline.chunk_overlap,
                child_overlap=settings.rag_offline.child_chunk_overlap,
                separators=separators
            )
        else:
            logger.warning(f"⚠️ 未知分块策略 '{strategy}'，降级使用 recursive")
            return RecursiveChunker(
                chunk_size=settings.rag_offline.chunk_size,
                chunk_overlap=settings.rag_offline.chunk_overlap,
                separators=separators
            )

class RerankerFactory:
    @staticmethod
    def get_reranker() -> Reranker | None:
        if not settings.rag_online.enable_rerank:
            logger.info("🚫 重排序已禁用，返回 None")
            return None
        
        logger.info(f"🏭 工厂正在构建重排器：{settings.rag_online.rerank_model_name}")
        # 直接实例化之前的 Reranker 类，它内部会读取 config
        return Reranker()