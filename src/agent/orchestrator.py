from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.agent.router import IntentRouter, RouteDecision
from src.agent.strategies import (
    BaseAgentStrategy,
    ClarifyNeededStrategy,
    CodeSearchStrategy,
    DeepSearchStrategy,
    DirectReplyStrategy,
    FastRetrievalStrategy,
    FallbackStrategy,
    StandardRetrievalStrategy,
    StrategyContext,
    StrategyName,
    StrategyResult,
)

# intent 到执行策略的固定映射（用于稳定线上行为）
INTENT_TO_STRATEGY: Dict[str, StrategyName] = {
    "CHIT_CHAT": "direct_reply",
    "FACT_LOOKUP": "fast_retrieval",
    "HOW_TO": "standard_retrieval",
    "COMPARISON": "deep_search",
    "CODE_SEARCH": "code_search",
    "UNKNOWN": "fallback",
}


@dataclass(slots=True)
class RoutedExecution:
    """一次路由执行的完整结果：包含路由决策与策略执行产物。"""

    decision: RouteDecision
    result: StrategyResult


class AgentStrategyOrchestrator:
    """策略编排器：负责将 RouteDecision 分发到具体策略实现。"""

    def __init__(self, registry: Optional[Dict[StrategyName, BaseAgentStrategy]] = None):
        self.registry = registry or self._build_default_registry()

    @staticmethod
    def _build_default_registry() -> Dict[StrategyName, BaseAgentStrategy]:
        return {
            "direct_reply": DirectReplyStrategy(),
            "fast_retrieval": FastRetrievalStrategy(),
            "standard_retrieval": StandardRetrievalStrategy(),
            "deep_search": DeepSearchStrategy(),
            "code_search": CodeSearchStrategy(),
            "fallback": FallbackStrategy(),
            "clarify_needed": ClarifyNeededStrategy(),
        }

    def resolve_strategy_name(self, decision: RouteDecision) -> StrategyName:
        """解析最终执行策略。

        规则：
        1) 明确的 clarify_needed 优先；
        2) 低置信度自动转 clarify_needed；
        3) 否则按 intent 强映射；
        4) 最后才回退到 decision.strategy/fallback。
        """
        if decision.strategy == "clarify_needed" and "clarify_needed" in self.registry:
            return "clarify_needed"

        if decision.confidence < 0.6 and "clarify_needed" in self.registry:
            return "clarify_needed"

        # 优先按 intent 做强映射，避免 LLM 输出 strategy 偏移导致执行不稳定。
        by_intent = INTENT_TO_STRATEGY.get(decision.intent)
        if by_intent and by_intent in self.registry:
            return by_intent

        by_strategy = decision.strategy.lower().strip()
        if by_strategy in self.registry:
            return by_strategy  # type: ignore[return-value]
        return "fallback"

    async def execute(
        self,
        query: str,
        decision: RouteDecision,
        category: Optional[str] = None,
    ) -> StrategyResult:
        strategy_name = self.resolve_strategy_name(decision)
        strategy = self.registry[strategy_name]
        return await strategy.execute(
            StrategyContext(
                query=query,
                category=category,
                intent=decision.intent,
                route_strategy=decision.strategy,
                clarification_questions=decision.clarification_questions,
            )
        )


class RoutedAgentExecutor:
    """高层执行入口：先路由，再按策略编排执行。"""

    def __init__(
        self,
        router: Optional[IntentRouter] = None,
        orchestrator: Optional[AgentStrategyOrchestrator] = None,
    ):
        self.router = router or IntentRouter()
        self.orchestrator = orchestrator or AgentStrategyOrchestrator()

    async def run(self, query: str, category: Optional[str] = None) -> RoutedExecution:
        """对外统一调用方法。"""
        decision = await self.router.route(query)
        result = await self.orchestrator.execute(query=query, decision=decision, category=category)
        return RoutedExecution(decision=decision, result=result)
