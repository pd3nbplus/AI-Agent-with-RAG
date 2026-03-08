from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from src.rag.pipeline import RetrievalPipeline
from src.core.config import settings

from .base import BaseAgentStrategy, StrategyContext, StrategyResult


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class DirectReplyStrategy(BaseAgentStrategy):
    """闲聊类策略：不触发检索，交给上游生成模型直接作答。"""

    name = "direct_reply"

    async def execute(self, context: StrategyContext) -> StrategyResult:
        return StrategyResult(
            strategy=self.name,
            message="无需检索，直接调用生成模型回复。",
            metadata={"retrieval_enabled": False},
        )


class FastRetrievalStrategy(BaseAgentStrategy):
    """事实查询策略：主路 + ES 双路增强，返回 Top-3。"""

    name = "fast_retrieval"

    def __init__(
        self,
        pipeline_config: Optional[Dict[str, Any]] = None,
        pipeline: Optional[RetrievalPipeline] = None,
    ):
        self.pipeline_config = {
            "retrieval": {
                "top_k": 3,
                "rough_top_k": 3,
            },
            "online": {
                "enable_rerank": True,
                "dynamic_threshold": settings.rag_online.score_threshold,
            },
            "composer": {
                "enable_hybrid_search": True,
                "plugin_rewritten_query": False,
                "plugin_rewritten_hyde": False,
                "plugin_es_questions": True,
                "plugin_es_summaries": True,
            },
            "filter": {},
        }
        if pipeline_config:
            # 约定：策略只关心 nested config，pipeline 负责解析与透传
            self.pipeline_config = _deep_merge(self.pipeline_config, pipeline_config)
        self.pipeline = pipeline or RetrievalPipeline()

    async def execute(self, context: StrategyContext) -> StrategyResult:
        run_config = deepcopy(self.pipeline_config)
        if context.category:
            run_config.setdefault("filter", {})
            run_config["filter"]["category"] = context.category

        results = await self.pipeline.run(
            query=context.query,
            config=run_config,
        )

        return StrategyResult(
            strategy=self.name,
            message="完成单次向量检索（通过预置 pipeline 配置执行）。",
            results=results,
            metadata={
                "retrieval_enabled": True,
                "top_k": run_config.get("retrieval", {}).get("top_k"),
                "pipeline_config": run_config,
            },
        )


class StandardRetrievalStrategy(BaseAgentStrategy):
    """标准检索策略：五路召回 + 可选重排。"""

    name = "standard_retrieval"

    def __init__(
        self,
        pipeline_config: Optional[Dict[str, Any]] = None,
        pipeline: Optional[RetrievalPipeline] = None,
    ):
        self.pipeline_config = {
            "retrieval": {
                "top_k": 5,
                "rough_top_k": 8,
            },
            "online": {
                "enable_rerank": True,
                "dynamic_threshold": settings.rag_online.score_threshold,
            },
            "composer": {
                "enable_hybrid_search": True,
                "plugin_rewritten_query": True,
                "plugin_rewritten_hyde": True,
                "plugin_es_questions": True,
                "plugin_es_summaries": True,
            },
            "filter": {},
        }
        if pipeline_config:
            self.pipeline_config = _deep_merge(self.pipeline_config, pipeline_config)
        self.pipeline = pipeline or RetrievalPipeline()

    async def execute(self, context: StrategyContext) -> StrategyResult:
        run_config = deepcopy(self.pipeline_config)
        if context.category:
            run_config.setdefault("filter", {})
            run_config["filter"]["category"] = context.category

        results = await self.pipeline.run(
            query=context.query,
            config=run_config,
        )
        return StrategyResult(
            strategy=self.name,
            message="完成混合检索与重排。",
            results=results,
            metadata={
                "retrieval_enabled": True,
                "top_k": run_config.get("retrieval", {}).get("top_k"),
                "pipeline_config": run_config,
            },
        )


class DeepSearchStrategy(StandardRetrievalStrategy):
    """深度检索策略：暂时使用标准检索同款配置，保留独立扩展入口。"""

    name = "deep_search"

    async def execute(self, context: StrategyContext) -> StrategyResult:
        # 当前阶段暂时复用 standard_retrieval
        standard = await super().execute(context)
        return StrategyResult(
            strategy=self.name,
            message="deep_search 暂由 standard_retrieval 执行。",
            results=standard.results,
            metadata={**standard.metadata, "delegated_to": "standard_retrieval"},
        )
