from __future__ import annotations

from .base import BaseAgentStrategy, StrategyContext, StrategyResult


class CodeSearchStrategy(BaseAgentStrategy):
    """代码检索占位策略：当前版本明确返回不支持。"""

    name = "code_search"

    async def execute(self, context: StrategyContext) -> StrategyResult:
        return StrategyResult(
            strategy=self.name,
            message="暂不支持代码分析，请改为通用问题或知识库问题。",
            metadata={"retrieval_enabled": False},
        )


class FallbackStrategy(BaseAgentStrategy):
    """兜底策略：用于无法理解或潜在有害请求。"""

    name = "fallback"

    async def execute(self, context: StrategyContext) -> StrategyResult:
        return StrategyResult(
            strategy=self.name,
            message="无法理解该请求，或请求可能有害，已拒绝处理。",
            metadata={"retrieval_enabled": False},
        )


class ClarifyNeededStrategy(BaseAgentStrategy):
    """澄清策略：在低置信度时先向用户追问关键信息。"""

    name = "clarify_needed"

    async def execute(self, context: StrategyContext) -> StrategyResult:
        questions = context.clarification_questions or []
        if questions:
            # 将路由器给出的澄清问题整理成可直接发送给用户的文本
            formatted = "\n".join([f"{idx}. {q}" for idx, q in enumerate(questions, 1)])
            message = f"在继续检索前，请先澄清以下问题：\n{formatted}"
        else:
            message = "在继续检索前，请补充更具体的场景、对象或目标。"

        return StrategyResult(
            strategy=self.name,
            message=message,
            metadata={"retrieval_enabled": False, "clarification_questions": questions},
        )
