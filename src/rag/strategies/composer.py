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
        self.retriever_map: Dict[str, BaseRetrievalStrategy] = {}
        self.milvus_client = get_milvus_client(self.config.milvus_config)
        self.es_client = get_es_client(self.config.es_config)
        
        self._load_plugins()

    def _load_plugins(self):
        """加载可用插件池，实际启停由 search 的 runtime_config 决定。"""
        self.retriever_map["vector_text"] = VectorTextRetriever(milvus_client=self.milvus_client)
        logger.info("✅ [Composer] 已加载主路：VectorText")

        self.retriever_map["vector_rewritten_query"] = VectorRewrittenRetriever(
            "standard", milvus_client=self.milvus_client
        )
        logger.info("✅ [Composer] 已加载变体路：VectorRewritten-standard")

        self.retriever_map["vector_rewritten_hyde"] = VectorRewrittenRetriever(
            "hyde", milvus_client=self.milvus_client
        )
        logger.info("✅ [Composer] 已加载变体路：VectorRewritten-hyde")

        if self.config.es_host:
            es_questions = ESQuestionsRetriever(es_client=self.es_client)
            if es_questions.es.is_available():
                self.retriever_map["es_questions"] = es_questions
                logger.info("✅ [Composer] 已加载 ES - Questions 路：ESQuestions")

            es_summaries = ESSummariesRetriever(es_client=self.es_client)
            if es_summaries.es.is_available():
                self.retriever_map["es_summaries"] = es_summaries
                logger.info("✅ [Composer] 已加载 ES - Summaries 路：ESSummaries")

    def _resolve_runtime_config(self, runtime_config: Optional[Dict[str, Any]] = None) -> ComposerConfig:
        payload: Dict[str, Any] = {
            "enable_hybrid_search": self.config.enable_hybrid_search,
            "plugin_rewritten_query": self.config.plugin_rewritten_query,
            "plugin_rewritten_hyde": self.config.plugin_rewritten_hyde,
            "plugin_es_questions": self.config.plugin_es_questions,
            "plugin_es_summaries": self.config.plugin_es_summaries,
            "rrf_k": self.config.rrf_k,
            "es_host": self.config.es_host,
        }
        if runtime_config:
            payload.update(runtime_config)
        return ComposerConfig.from_any(payload)

    def _select_retrievers(self, cfg: ComposerConfig) -> List[BaseRetrievalStrategy]:
        selected: List[BaseRetrievalStrategy] = []
        vector_text = self.retriever_map.get("vector_text")
        if vector_text:
            selected.append(vector_text)

        if not cfg.enable_hybrid_search:
            return selected

        if cfg.plugin_rewritten_query:
            retriever = self.retriever_map.get("vector_rewritten_query")
            if retriever:
                selected.append(retriever)
        if cfg.plugin_rewritten_hyde:
            retriever = self.retriever_map.get("vector_rewritten_hyde")
            if retriever:
                selected.append(retriever)
        if cfg.plugin_es_questions:
            retriever = self.retriever_map.get("es_questions")
            if retriever:
                selected.append(retriever)
        if cfg.plugin_es_summaries:
            retriever = self.retriever_map.get("es_summaries")
            if retriever:
                selected.append(retriever)

        return selected

    async def search(
        self,
        query: str,
        rough_top_k: int,
        filter_expr: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        异步并行执行多路检索
        """
        effective_cfg = self._resolve_runtime_config(runtime_config)
        selected_retrievers = self._select_retrievers(effective_cfg)
        logger.info(f"🚀 [Composer] 开始异步多路检索 ({len(selected_retrievers)} 路)...")
        if not selected_retrievers:
            logger.warning("⚠️ [Composer] 当前配置下无可用检索路")
            return []
        
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
        tasks = [run_retriever(r) for r in selected_retrievers]
        
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
        rrf_engine = RRFFusionEngine(k=effective_cfg.rrf_k)
        fused_results = rrf_engine.fuse(valid_results_lists, rough_top_k)
        return fused_results

# 单例
composer_instance = RetrieverComposer()
