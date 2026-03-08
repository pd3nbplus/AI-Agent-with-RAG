# src/rag/strategies/composer.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.rag.strategies.retrievers.vector_text import VectorTextRetriever
from src.rag.strategies.retrievers.vector_rewritten import VectorRewrittenRetriever
from src.rag.strategies.retrievers.es_questions import ESQuestionsRetriever
from src.rag.strategies.retrievers.es_summaries import ESSummariesRetriever # 导入新插件
from src.rag.fusion.rrf import RRFFusionEngine
from src.core.milvus_client import get_milvus_client
from src.core.es_client import get_es_client
from src.core.config import settings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ComposerConfig:
    """RetrieverComposer 运行时配置，可由实验代码动态注入。"""
    enable_hybrid_search: bool = settings.search.enable_hybrid_search
    plugin_rewritten_query: bool = settings.search.plugin_rewritten_query
    plugin_rewritten_hyde: bool = settings.search.plugin_rewritten_hyde
    plugin_es_questions: bool = settings.search.plugin_es_questions
    plugin_es_summaries: bool = settings.search.plugin_es_summaries
    rrf_k: int = settings.search.rrf_k
    es_host: Optional[str] = settings.db.es_host
    milvus_config: Optional[Dict[str, Any]] = None
    es_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_any(cls, value: Optional["ComposerConfig | Dict[str, Any]"]) -> "ComposerConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            payload = {k: v for k, v in value.items() if k in allowed}
            return cls(**payload)
        raise TypeError("composer config must be None, ComposerConfig or dict")


class RetrieverComposer:
    """
    检索器组装器
    职责：根据配置动态加载多个检索插件，并行执行，并使用 RRF 融合结果。
    """
    def __init__(self, config: Optional[ComposerConfig | Dict[str, Any]] = None):
        self.config = ComposerConfig.from_any(config)
        self.retrievers: List[BaseRetrievalStrategy] = []
        self.rrf_engine = RRFFusionEngine(k=self.config.rrf_k)
        self.milvus_client = get_milvus_client(self.config.milvus_config)
        self.es_client = get_es_client(self.config.es_config)
        
        self._load_plugins()

    def _load_plugins(self):
        """根据配置动态加载插件"""
        # 1. 主路：永远加载
        self.retrievers.append(VectorTextRetriever(milvus_client=self.milvus_client))
        logger.info("✅ [Composer] 已加载主路：VectorText")
        
        # 2. 变体路：如果开启混合检索
        if self.config.enable_hybrid_search:
            # 2. 改写路：如果开启改写
            if self.config.plugin_rewritten_query:
                self.retrievers.append(VectorRewrittenRetriever('standard', milvus_client=self.milvus_client))
                logger.info("✅ [Composer] 已加载变体路：VectorRewritten-standard")
            
            if self.config.plugin_rewritten_hyde:
                self.retrievers.append(VectorRewrittenRetriever('hyde', milvus_client=self.milvus_client))
                logger.info("✅ [Composer] 已加载变体路：VectorRewritten-hyde")

            # 3. ES 路：如果配置了 ES
            if self.config.es_host:
                # 3. ES - Questions 路
                if self.config.plugin_es_questions:
                    es_retriever = ESQuestionsRetriever(es_client=self.es_client)
                    if es_retriever.es.is_available(): # 只有连接成功才加入
                        self.retrievers.append(es_retriever)
                        logger.info("✅ [Composer] 已加载 ES - Questions 路：ESQuestions")
                # 4. ES - Summaries 路
                if self.config.plugin_es_summaries:
                    es_retriever = ESSummariesRetriever(es_client=self.es_client)
                    if es_retriever.es.is_available(): # 只有连接成功才加入
                        self.retrievers.append(es_retriever)
                        logger.info("✅ [Composer] 已加载 ES - Summaries 路：ESSummaries")

    async def search(self, query: str, rough_top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        """
        异步并行执行多路检索
        """
        logger.info(f"🚀 [Composer] 开始异步多路检索 ({len(self.retrievers)} 路)...")
        
        # 定义一个内部异步包装器，用于在线程池中运行同步的 retriever.search
        async def run_retriever(retriever: BaseRetrievalStrategy) -> List[SearchResult]:
            try:
                # asyncio.to_thread 将阻塞的同步代码放入线程池，避免阻塞主事件循环
                results = await asyncio.to_thread(
                    retriever.search, 
                    query, 
                    rough_top_k, 
                    filter_expr=filter_expr, 
                    **kwargs
                )
                if results:
                    logger.debug(f"✅ [{retriever.__class__.__name__}] 完成，召回 {len(results)} 条")
                    return results
                return []
            except Exception as e:
                logger.error(f"❌ [Composer] 插件 {retriever.__class__.__name__} 执行失败：{e}")
                return []

        # 1. 创建所有检索任务
        tasks = [run_retriever(r) for r in self.retrievers]
        
        # 2. 并行执行 (gather)
        # return_exceptions=True 确保某个插件崩溃不会导致整个 gather 失败
        all_results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 3. 过滤掉异常对象和空列表，只保留有效的结果列表
        valid_results_lists = [
            res for res in all_results_lists 
            if isinstance(res, list) and len(res) > 0
        ]
        
        if not valid_results_lists:
            logger.warning("⚠️ [Composer] 所有检索路均无结果或失败")
            return []
            
        # 4. 如果只有一路有效，直接返回并排序 (跳过 RRF 以节省时间)
        if len(valid_results_lists) == 1:
            final_results = sorted(valid_results_lists[0], key=lambda x: x.score, reverse=True)
            logger.info(f"✨ [Composer] 单路有效检索完成，共 {len(final_results)} 条")
            return final_results[:rough_top_k]
        
        # 5. 多路结果，执行 RRF 融合
        logger.info(f"🔄 [Composer] 执行 RRF 融合 ({len(valid_results_lists)} 路输入)")
        fused_results = self.rrf_engine.fuse(valid_results_lists, rough_top_k)
        return fused_results

# 单例
composer_instance = RetrieverComposer()
