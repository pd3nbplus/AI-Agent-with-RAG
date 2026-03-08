# src/rag/factories.py
from src.rag.chunkers import (
    RecursiveChunker, 
    FixedChunker, 
    BaseChunker, 
    ParentChildChunker,
    ChildSplitterFactory  # 👈 导入子分块器工厂
)
from src.rag.reranker import Reranker # 引入之前的重排类
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class ChunkerFactory:
    @staticmethod
    def get_chunker() -> BaseChunker:
        strategy = settings.rag_offline.chunk_strategy.lower()
        logger.info(f"└─ 分块策略：{strategy}")
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
            # ==========================================
            # 1. 解析子分块器相关配置
            # ==========================================
            child_strategy = settings.rag_offline.child_splitter_strategy
            child_chunk_size = settings.rag_offline.child_chunk_size
            child_chunk_overlap = settings.rag_offline.child_chunk_overlap
            min_sentence_len = settings.rag_offline.min_sentence_length
            parent_size = settings.rag_offline.chunk_size
            parent_overlap = settings.rag_offline.chunk_overlap
            
            logger.info(f"   └─ 子分块策略：{child_strategy}")
            logger.info(f"   └─ 父块大小：{parent_size}, 重叠：{parent_overlap}")

            # ==========================================
            # 2. 使用工厂创建【子分块器实例】 (依赖注入的核心)
            # ==========================================
            child_splitter = ChildSplitterFactory.create(
                strategy=child_strategy,
                chunk_size=child_chunk_size,
                chunk_overlap=child_chunk_overlap,
                min_sentence_len=min_sentence_len,
                separators=separators
            )
            
            # ==========================================
            # 3. 创建【父分块器】并注入子分块器
            # ==========================================
            return ParentChildChunker(
                parent_size=parent_size,
                parent_overlap=parent_overlap,
                child_splitter=child_splitter,  # 👈 关键：注入已经初始化好的子分块器
                parent_separators=["\n\n", "\n"] # 父块通常只按段落切分
            )
        else:
            logger.warning(f"⚠️ 未知分块策略 '{strategy}'，降级使用 recursive")
            return RecursiveChunker(
                chunk_size=settings.rag_offline.chunk_size,
                chunk_overlap=settings.rag_offline.chunk_overlap,
                separators=separators
            )

class RerankerFactory:
    _instance: Reranker | None = None

    @staticmethod
    def get_reranker() -> Reranker | None:
        if not settings.rag_online.enable_rerank:
            logger.info("🚫 重排序已禁用，返回 None")
            return None

        # 单进程单例：避免同一进程内多次加载 bge-reranker-base
        if RerankerFactory._instance is None:
            logger.info(f"🏭 工厂正在构建重排器：{settings.rag_online.rerank_model_name}")
            # 直接实例化之前的 Reranker 类，它内部会读取 config
            RerankerFactory._instance = Reranker()
        else:
            logger.info("♻️ 复用已加载的重排器实例")
        return RerankerFactory._instance
