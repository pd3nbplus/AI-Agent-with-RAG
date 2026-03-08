from __future__ import annotations

import asyncio
from typing import Any, List, Literal, Optional

import logging
from langgraph.graph import END, START, StateGraph

from src.core.prompt_registry import PROMPT_KEYS, core_prompt_registry
from src.self_rag.adapters.trace_adapter import TraceAdapter
from src.self_rag.config import SelfRAGConfig
from src.self_rag.nodes import (
    DecideNextNode,
    GenerateNode,
    JudgeGroundingNode,
    JudgeRelevanceNode,
    JudgeUtilityNode,
    RetrieveNode,
    RewriteQueryNode,
    RouteNode,
)
from src.self_rag.schemas.judge import JudgeResult
from src.self_rag.schemas.output import SelfRAGOutput
from src.self_rag.state import HopTrace, SelfRAGGraphState

logger = logging.getLogger(__name__)


class SelfRAGEngine:
    """Self-RAG 主编排引擎（LangGraph 版本）。"""

    def __init__(
        self,
        config: Optional[SelfRAGConfig] = None,
        rag_adapter: Optional[Any] = None,
        llm_adapter: Optional[Any] = None,
        judge_llm_adapter: Optional[Any] = None,
        trace_adapter: Optional[TraceAdapter] = None,
    ):
        self.config = config or SelfRAGConfig()
        self.trace = trace_adapter or TraceAdapter()

        if llm_adapter is None:
            from src.common.llm_adapter import LLMAdapter

            llm = LLMAdapter()
        else:
            llm = llm_adapter

        if judge_llm_adapter is not None:
            judge_llm = judge_llm_adapter
        elif llm_adapter is not None:
            # 测试/注入场景：若外部已显式传入 llm_adapter，则默认复用，避免强依赖外部 endpoint 文件。
            judge_llm = llm_adapter
        else:
            from src.self_rag.adapters.judge_llm_adapter import JudgeLLMAdapter

            # 生产默认：judge_* 三节点走外部大模型路由，缓解本地单模型并发与稳定性问题。
            judge_llm = JudgeLLMAdapter(
                llm_json_path=self.config.judge_llm_json_path,
                llm_group=self.config.judge_llm_group,
            )

        if rag_adapter is None:
            from src.self_rag.adapters.rag_pipeline_adapter import RAGPipelineAdapter

            rag = RAGPipelineAdapter()
        else:
            rag = rag_adapter

        self.route_node = RouteNode()
        self.retrieve_node = RetrieveNode(rag)
        self.generate_node = GenerateNode(llm, core_prompt_registry.get(PROMPT_KEYS.SELF_RAG_GENERATE))
        self.judge_relevance_node = JudgeRelevanceNode(
            judge_llm,
            core_prompt_registry.get(PROMPT_KEYS.SELF_RAG_JUDGE_RELEVANCE),
            threshold=self.config.relevance_threshold,
        )
        self.judge_grounding_node = JudgeGroundingNode(
            judge_llm,
            core_prompt_registry.get(PROMPT_KEYS.SELF_RAG_JUDGE_GROUNDING),
            threshold=self.config.grounding_threshold,
        )
        self.judge_utility_node = JudgeUtilityNode(
            judge_llm,
            core_prompt_registry.get(PROMPT_KEYS.SELF_RAG_JUDGE_UTILITY),
            threshold=self.config.utility_threshold,
        )
        self.rewrite_query_node = RewriteQueryNode(
            llm,
            core_prompt_registry.get(PROMPT_KEYS.SELF_RAG_REWRITE_QUERY),
        )
        self.decide_node = DecideNextNode()
        self.app = self._build_graph().compile()

    def _build_graph(self):
        """构建 Self-RAG 执行图。

        关键设计：
        1. 将“多轮循环”交给 LangGraph 条件边处理，而不是手写 for-loop；
        2. route 节点只负责“是否进入 Self-RAG”；
        3. run_hop 节点完成一整轮：检索->生成->评判->决策->可选改写。
        """

        workflow = StateGraph(SelfRAGGraphState)
        workflow.add_node("route", self._route)
        workflow.add_node("run_hop", self._run_hop)

        workflow.add_edge(START, "route")
        workflow.add_conditional_edges(
            "route",
            self._next_after_route,
            {
                "run_hop": "run_hop",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "run_hop",
            self._next_after_hop,
            {
                "run_hop": "run_hop",
                "end": END,
            },
        )
        return workflow

    @staticmethod
    def _failure_reasons(
        relevance: JudgeResult,
        grounding: JudgeResult,
        utility: JudgeResult,
    ) -> List[str]:
        reasons: List[str] = []
        if not relevance.passed:
            reasons.append(f"相关性不足(score={relevance.score:.2f})")
        if not grounding.passed:
            reasons.append(f"证据支撑不足(score={grounding.score:.2f})")
        if not utility.passed:
            reasons.append(f"回答效用不足(score={utility.score:.2f})")
        return reasons

    @staticmethod
    def _trace_to_dict(trace: HopTrace) -> dict:
        return {
            "hop": trace.hop,
            "query": trace.query,
            "answer": trace.answer,
            "contexts": trace.contexts,
            "relevance": trace.relevance.model_dump(),
            "grounding": trace.grounding.model_dump(),
            "utility": trace.utility.model_dump(),
            "decision": trace.decision,
            "rewritten_query": trace.rewritten_query,
        }

    @staticmethod
    def _fallback_judge(reason: str) -> JudgeResult:
        return JudgeResult(score=0.0, passed=False, reasoning=reason)

    async def _route(self, state: SelfRAGGraphState) -> SelfRAGGraphState:
        current_query = state["current_query"]
        if self.route_node.run(current_query):
            return {"route_passed": True}
        return {
            "route_passed": False,
            "final_decision": "fallback",
            "last_answer": self.config.fallback_answer,
            "last_contexts": [],
            "last_rewritten_query": None,
        }

    @staticmethod
    def _next_after_route(state: SelfRAGGraphState) -> Literal["run_hop", "end"]:
        return "run_hop" if state.get("route_passed", False) else "end"

    async def _run_hop(self, state: SelfRAGGraphState) -> SelfRAGGraphState:
        """执行一轮 Self-RAG。

        注意：
        - 检索/生成失败：直接 fallback（保持旧行为，且不记录本轮 trace）。
        - 评判失败：降级为低分判定，流程继续可改写。
        """

        hop = state.get("hop", 0) + 1
        max_hops = state.get("max_hops", self.config.max_hops)
        current_query = state["current_query"]
        category = state.get("category")
        traces = list(state.get("traces", []))

        try:
            results = await self.retrieve_node.run(
                query=current_query,
                config=self.config.retrieval_config,
                category=category,
            )
        except Exception:
            logger.exception("self-rag retrieve failed at hop=%s", hop)
            return {
                "hop": hop,
                "final_decision": "fallback",
                "last_answer": self.config.fallback_answer,
            }
        contexts = [item.text for item in results]

        try:
            answer = await self.generate_node.run(current_query, contexts)
        except Exception:
            logger.exception("self-rag generate failed at hop=%s", hop)
            return {
                "hop": hop,
                "final_decision": "fallback",
                "last_answer": self.config.fallback_answer,
            }

        try:
            relevance = await self.judge_relevance_node.run(current_query, contexts)
        except Exception:
            relevance = self._fallback_judge("相关性评判失败")
        try:
            grounding = await self.judge_grounding_node.run(current_query, answer, contexts)
        except Exception:
            grounding = self._fallback_judge("证据支撑评判失败")
        try:
            utility = await self.judge_utility_node.run(current_query, answer)
        except Exception:
            utility = self._fallback_judge("效用评判失败")

        decision = self.decide_node.run(
            hop=hop,
            max_hops=max_hops,
            relevance=relevance,
            grounding=grounding,
            utility=utility,
        )

        rewritten_query: Optional[str] = None
        if decision == "rewrite":
            try:
                rewritten_query = await self.rewrite_query_node.run(
                    original_query=state["original_query"],
                    current_query=current_query,
                    answer=answer,
                    failure_reasons=self._failure_reasons(relevance, grounding, utility),
                )
            except Exception:
                rewritten_query = None

            # 关键兜底：改写失败或改写后未变化，直接终止并 fallback，防止无效循环。
            if not rewritten_query or rewritten_query == current_query:
                decision = "fallback"

        trace = HopTrace(
            hop=hop,
            query=current_query,
            answer=answer,
            contexts=contexts,
            relevance=relevance,
            grounding=grounding,
            utility=utility,
            decision=decision,
            rewritten_query=rewritten_query,
        )
        traces.append(trace)
        self.trace.log(self._trace_to_dict(trace))

        next_query = rewritten_query if decision == "rewrite" and rewritten_query else current_query
        return {
            "hop": hop,
            "traces": traces,
            "current_query": next_query,
            "final_decision": decision,
            "last_answer": answer or self.config.fallback_answer,
            "last_contexts": contexts,
            "last_rewritten_query": rewritten_query,
        }

    @staticmethod
    def _next_after_hop(state: SelfRAGGraphState) -> Literal["run_hop", "end"]:
        # 只有 rewrite 才继续下一轮；finish/fallback 都结束图执行。
        return "run_hop" if state.get("final_decision") == "rewrite" else "end"

    async def run(self, query: str, category: Optional[str] = None) -> SelfRAGOutput:
        initial_state: SelfRAGGraphState = {
            "original_query": query,
            "current_query": query,
            "category": category,
            "max_hops": self.config.max_hops,
            "hop": 0,
            "traces": [],
            "route_passed": True,
            "final_decision": "fallback",
            "last_answer": self.config.fallback_answer,
            "last_contexts": [],
            "last_rewritten_query": None,
        }

        final_state = await self.app.ainvoke(initial_state)
        traces = list(final_state.get("traces", []))
        trace_payload = [self._trace_to_dict(item) for item in traces]

        return SelfRAGOutput(
            query=query,
            final_answer=final_state.get("last_answer", self.config.fallback_answer),
            final_decision=final_state.get("final_decision", "fallback"),
            hops_used=len(traces),
            contexts=final_state.get("last_contexts", []),
            rewritten_query=final_state.get("last_rewritten_query"),
            trace=trace_payload,
        )

    def run_sync(self, query: str, category: Optional[str] = None) -> SelfRAGOutput:
        try:
            asyncio.get_running_loop()
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.run(query=query, category=category))
            finally:
                loop.close()
        except RuntimeError:
            return asyncio.run(self.run(query=query, category=category))
