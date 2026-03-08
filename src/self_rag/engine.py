from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

import logging

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
from src.self_rag.state import HopTrace, SelfRAGState

logger = logging.getLogger(__name__)


class SelfRAGEngine:
    """Self-RAG 主编排引擎。"""

    def __init__(
        self,
        config: Optional[SelfRAGConfig] = None,
        rag_adapter: Optional[Any] = None,
        llm_adapter: Optional[Any] = None,
        trace_adapter: Optional[TraceAdapter] = None,
    ):
        self.config = config or SelfRAGConfig()
        self.trace = trace_adapter or TraceAdapter()

        if llm_adapter is None:
            from src.self_rag.adapters.llm_adapter import LLMAdapter

            llm = LLMAdapter()
            prompt_loader = LLMAdapter.load_prompt
        else:
            llm = llm_adapter

            def prompt_loader(path: Path) -> str:
                with path.open("r", encoding="utf-8") as f:
                    return f.read().strip()

        if rag_adapter is None:
            from src.self_rag.adapters.rag_pipeline_adapter import RAGPipelineAdapter

            rag = RAGPipelineAdapter()
        else:
            rag = rag_adapter

        prompts_dir = Path(__file__).resolve().parent / "prompts"

        self.route_node = RouteNode()
        self.retrieve_node = RetrieveNode(rag)
        self.generate_node = GenerateNode(llm, prompt_loader(prompts_dir / "generate.txt"))
        self.judge_relevance_node = JudgeRelevanceNode(
            llm,
            prompt_loader(prompts_dir / "judge_relevance.txt"),
            threshold=self.config.relevance_threshold,
        )
        self.judge_grounding_node = JudgeGroundingNode(
            llm,
            prompt_loader(prompts_dir / "judge_grounding.txt"),
            threshold=self.config.grounding_threshold,
        )
        self.judge_utility_node = JudgeUtilityNode(
            llm,
            prompt_loader(prompts_dir / "judge_utility.txt"),
            threshold=self.config.utility_threshold,
        )
        self.rewrite_query_node = RewriteQueryNode(
            llm,
            prompt_loader(prompts_dir / "rewrite_query.txt"),
        )
        self.decide_node = DecideNextNode()

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

    async def run(self, query: str, category: Optional[str] = None) -> SelfRAGOutput:
        if not self.route_node.run(query):
            return SelfRAGOutput(
                query=query,
                final_answer=self.config.fallback_answer,
                final_decision="fallback",
                hops_used=0,
                contexts=[],
                trace=[],
            )

        state = SelfRAGState(
            original_query=query,
            current_query=query,
            max_hops=self.config.max_hops,
        )
        last_answer = self.config.fallback_answer
        last_contexts: List[str] = []
        last_rewritten_query: Optional[str] = None
        final_decision = "fallback"

        for hop in range(1, self.config.max_hops + 1):
            try:
                results = await self.retrieve_node.run(
                    query=state.current_query,
                    config=self.config.retrieval_config,
                    category=category,
                )
            except Exception as exc:
                logger.exception("self-rag retrieve failed at hop=%s", hop)
                final_decision = "fallback"
                last_answer = self.config.fallback_answer
                break
            contexts = [item.text for item in results]
            try:
                answer = await self.generate_node.run(state.current_query, contexts)
            except Exception:
                logger.exception("self-rag generate failed at hop=%s", hop)
                final_decision = "fallback"
                last_answer = self.config.fallback_answer
                break
            try:
                relevance = await self.judge_relevance_node.run(state.current_query, contexts)
            except Exception:
                relevance = self._fallback_judge("相关性评判失败")
            try:
                grounding = await self.judge_grounding_node.run(state.current_query, answer, contexts)
            except Exception:
                grounding = self._fallback_judge("证据支撑评判失败")
            try:
                utility = await self.judge_utility_node.run(state.current_query, answer)
            except Exception:
                utility = self._fallback_judge("效用评判失败")
            decision = self.decide_node.run(
                hop=hop,
                max_hops=self.config.max_hops,
                relevance=relevance,
                grounding=grounding,
                utility=utility,
            )

            rewritten_query: Optional[str] = None
            if decision == "rewrite":
                try:
                    rewritten_query = await self.rewrite_query_node.run(
                        original_query=state.original_query,
                        current_query=state.current_query,
                        answer=answer,
                        failure_reasons=self._failure_reasons(relevance, grounding, utility),
                    )
                except Exception:
                    rewritten_query = None
                if not rewritten_query or rewritten_query == state.current_query:
                    decision = "fallback"

            trace = HopTrace(
                hop=hop,
                query=state.current_query,
                answer=answer,
                contexts=contexts,
                relevance=relevance,
                grounding=grounding,
                utility=utility,
                decision=decision,
                rewritten_query=rewritten_query,
            )
            state.traces.append(trace)
            self.trace.log(self._trace_to_dict(trace))

            last_answer = answer or self.config.fallback_answer
            last_contexts = contexts
            last_rewritten_query = rewritten_query
            final_decision = decision

            if decision == "finish":
                break
            if decision == "fallback":
                break

            state.current_query = rewritten_query or state.current_query

        return SelfRAGOutput(
            query=query,
            final_answer=last_answer,
            final_decision=final_decision,  # type: ignore[arg-type]
            hops_used=len(state.traces),
            contexts=last_contexts,
            rewritten_query=last_rewritten_query,
            trace=[self._trace_to_dict(t) for t in state.traces],
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
