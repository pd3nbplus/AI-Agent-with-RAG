# src/rag/pipeline.py
from src.rag.factories import RerankerFactory # 👈 新增导入
from src.rag.strategies.metadata_filter import MetadataFilterBuilder
from src.rag.strategies.composer import RetrieverComposer, ComposerConfig # 导入多路召回组件
from src.rag.strategies.base import SearchResult # 👈 导入 SearchResult 类
from src.core.config import settings
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    def __init__(self, composer_config: Optional[ComposerConfig | Dict[str, Any]] = None):
        # 导入元数据过滤组件
        self.filter_builder = MetadataFilterBuilder()
        self.default_filter_category = settings.search.default_filter_category
        # 导入多路召回组件（支持运行时注入配置，便于实验）
        self.composer = RetrieverComposer(config=composer_config)
        # 👇 修改点：使用工厂获取重排器
        self.reranker = RerankerFactory.get_reranker()
        logger.info(f"⚙️ Pipeline 初始化完成 (重排器：{'已加载' if self.reranker else '未加载'})")
        
        # 读取配置
        self.rough_top_k = settings.rag_online.rough_top_k
        self.final_top_k = settings.rag_online.final_top_k
        self.dynamic_threshold = settings.rag_online.score_threshold
        
        logger.info(f"⚙️ Pipeline 初始化：粗排Top{self.rough_top_k}, 动态阈值={self.dynamic_threshold}")

    def _should_trigger_rerank(self, candidates: List[SearchResult]) -> bool:
        """
        动态判断是否需要重排 (支持 SearchResult 对象)
        策略：如果第 1 名和第 2 名的分数差距很小，说明难以抉择，需要重排。
        """
        if not self.reranker:
            return False
        if len(candidates) < 2:
            return False # 只有一个结果，没必要重排
        
        score_1 = candidates[0].score
        score_2 = candidates[1].score
        gap = score_1 - score_2
        
        logger.debug(f"📊 粗排分数分析：Top1={score_1:.4f}, Top2={score_2:.4f}, 差距={gap:.4f}")
        
        # 如果差距小于阈值，触发重排
        if gap <= self.dynamic_threshold:
            logger.info("⚡ 分数差距较小，触发重排序...")
            return True
        else:
            logger.info("✅ 分数差距明显，跳过重排序 (节省资源)")
            return False
    async def run(self, query: str, top_k: int = 3, category: Optional[str] = None) -> List[SearchResult]:
        """
        执行完整的检索流程 (Advanced RAG Online Flow)
        Flow: Query -> filter -> 多路Search -> Result
        """
        final_query = query

        # step 1: 构建过滤表达式
        filter_category = category or self.default_filter_category
        filter_expr = self.filter_builder.build_expr(category=filter_category)
        

        # Step 2: 执行多路检索与融合
        rough_results = await self.composer.search(
            query=query, 
            rough_top_k=self.rough_top_k, 
            filter_expr=filter_expr
        )
        logger.info(f"🔍 [Pipeline] 多路检索与融合：{final_query} (召回 {len(rough_results)} 条)")
        
        if not rough_results:
            return []
        
        # Step 4: ⚡ 动态重排决策 (Re-ranking)
        if self._should_trigger_rerank(rough_results):
            # 触发重排
            rough_results = self.reranker.rerank(final_query, rough_results, top_k=top_k)
        
        # Step 5: 取消阈值过滤，直接返回 top_k 个结果 (保留所有分数信息，由调用方决定如何使用)
        return rough_results[:top_k]

# 单例
pipeline_instance = RetrievalPipeline()
