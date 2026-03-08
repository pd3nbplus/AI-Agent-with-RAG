# src/rag/pipeline.py
from src.rag.factories import RerankerFactory # 👈 新增导入
from src.rag.strategies.metadata_filter import MetadataFilterBuilder
from src.rag.strategies.composer import RetrieverComposer, ComposerConfig # 导入多路召回组件
from src.rag.strategies.base import SearchResult # 👈 导入 SearchResult 类
from src.core.config import settings
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineOnlineConfig:
    """Pipeline 在线配置：支持在策略层按场景注入参数。"""

    enable_rerank: bool = settings.rag_online.enable_rerank
    rough_top_k: int = settings.rag_online.rough_top_k
    final_top_k: int = settings.rag_online.final_top_k
    dynamic_threshold: float = settings.rag_online.score_threshold

    @classmethod
    def from_any(cls, value: Optional["PipelineOnlineConfig | Dict[str, Any]"]) -> "PipelineOnlineConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            payload = {k: v for k, v in value.items() if k in allowed}
            return cls(**payload)
        raise TypeError("online config must be None, PipelineOnlineConfig or dict")


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class RetrievalPipeline:
    def __init__(
        self,
        composer_config: Optional[ComposerConfig | Dict[str, Any]] = None,
        online_config: Optional[PipelineOnlineConfig | Dict[str, Any]] = None,
    ):
        # 导入元数据过滤组件
        self.filter_builder = MetadataFilterBuilder()
        self.default_filter_category = settings.search.default_filter_category
        # 导入多路召回组件（支持运行时注入配置，便于实验）
        self.composer = RetrieverComposer(config=composer_config)
        self.online_config = PipelineOnlineConfig.from_any(online_config)

        # 重排器是否启用由在线配置决定
        self.reranker = RerankerFactory.get_reranker() if self.online_config.enable_rerank else None
        logger.info(f"⚙️ Pipeline 初始化完成 (重排器：{'已加载' if self.reranker else '未加载'})")

        self.rough_top_k = self.online_config.rough_top_k
        self.final_top_k = self.online_config.final_top_k
        self.dynamic_threshold = self.online_config.dynamic_threshold

        logger.info(
            "⚙️ Pipeline 初始化：粗排Top%s, 最终Top%s, 动态阈值=%s",
            self.rough_top_k,
            self.final_top_k,
            self.dynamic_threshold,
        )

    def _should_trigger_rerank(self, candidates: List[SearchResult], dynamic_threshold: Optional[float] = None) -> bool:
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
        threshold = self.dynamic_threshold if dynamic_threshold is None else dynamic_threshold
        
        logger.debug(f"📊 粗排分数分析：Top1={score_1:.4f}, Top2={score_2:.4f}, 差距={gap:.4f}")
        
        # 如果差距小于阈值，触发重排
        if gap <= threshold:
            logger.info("⚡ 分数差距较小，触发重排序...")
            return True
        else:
            logger.info("✅ 分数差距明显，跳过重排序 (节省资源)")
            return False

    def _build_default_run_config(self) -> Dict[str, Any]:
        return {
            "retrieval": {
                "rough_top_k": self.rough_top_k,
                "top_k": self.final_top_k,
            },
            "online": {
                "enable_rerank": self.online_config.enable_rerank,
                "dynamic_threshold": self.dynamic_threshold,
            },
            "filter": {
                "category": self.default_filter_category,
                "source": None,
                "min_page": None,
            },
            "composer": {},
        }

    async def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        执行完整的检索流程 (Advanced RAG Online Flow)
        Flow: Query -> filter -> 多路Search -> Result
        """
        final_query = query

        # 兼容旧参数，并支持新的嵌套 config 入参
        run_cfg = self._build_default_run_config()
        if config:
            run_cfg = _deep_merge(run_cfg, config)
        if top_k is not None:
            run_cfg["retrieval"]["top_k"] = top_k
        if category is not None:
            run_cfg["filter"]["category"] = category

        rough_top_k = int(run_cfg["retrieval"].get("rough_top_k", self.rough_top_k))
        final_top_k = int(run_cfg["retrieval"].get("top_k", self.final_top_k))
        enable_rerank = bool(run_cfg["online"].get("enable_rerank", self.online_config.enable_rerank))
        dynamic_threshold = float(run_cfg["online"].get("dynamic_threshold", self.dynamic_threshold))
        composer_runtime_config = run_cfg.get("composer", {}) or {}

        # step 1: 构建过滤表达式
        filter_expr = self.filter_builder.build_expr(
            category=run_cfg["filter"].get("category"),
            source=run_cfg["filter"].get("source"),
            min_page=run_cfg["filter"].get("min_page"),
        )
        
        # Step 2: 执行多路检索与融合
        rough_results = await self.composer.search(
            query=query,
            rough_top_k=rough_top_k,
            filter_expr=filter_expr,
            runtime_config=composer_runtime_config,
        )
        logger.info(f"🔍 [Pipeline] 多路检索与融合：{final_query} (召回 {len(rough_results)} 条)")
        
        if not rough_results:
            return []
        
        # Step 4: ⚡ 动态重排决策 (Re-ranking)
        if enable_rerank and self._should_trigger_rerank(rough_results, dynamic_threshold=dynamic_threshold):
            # 触发重排
            rough_results = self.reranker.rerank(final_query, rough_results, top_k=final_top_k)
        
        # Step 5: 取消阈值过滤，直接返回 top_k 个结果 (保留所有分数信息，由调用方决定如何使用)
        return rough_results[:final_top_k]

# 单例
pipeline_instance = RetrievalPipeline()
