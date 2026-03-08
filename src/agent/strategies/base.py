from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from src.rag.strategies.base import SearchResult

# 策略名称白名单：和 router 产出的 strategy 字段保持一致
StrategyName = Literal[
    "direct_reply",
    "fast_retrieval",
    "standard_retrieval",
    "deep_search",
    "code_search",
    "fallback",
    "clarify_needed",
]


@dataclass(slots=True)
class StrategyContext:
    """策略执行上下文：由 orchestrator 统一组装后传入具体策略。"""

    query: str
    category: Optional[str] = None
    intent: Optional[str] = None
    route_strategy: Optional[str] = None
    clarification_questions: Optional[List[str]] = None


@dataclass(slots=True)
class StrategyResult:
    """策略统一输出：兼容“纯回复”和“检索结果”两类场景。"""

    strategy: StrategyName
    message: str
    results: List[SearchResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgentStrategy(ABC):
    """所有 agent 策略的抽象基类。"""

    name: StrategyName

    @abstractmethod
    async def execute(self, context: StrategyContext) -> StrategyResult:
        """执行策略并返回统一结果结构。"""
        raise NotImplementedError
