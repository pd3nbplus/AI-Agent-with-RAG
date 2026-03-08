"""Self-RAG 流程测试脚本（无外部依赖版本）。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.rag.strategies.base import SearchResult
from src.self_rag import SelfRAGConfig, SelfRAGEngine


class FakeRAGPipelineAdapter:
    async def retrieve(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ):
        _ = config, category
        if "报销流程" in query:
            return [
                SearchResult(
                    text="报销流程：提交申请->主管审批->财务打款。",
                    score=0.92,
                    metadata={"source": "policy.md"},
                )
            ]
        if "重置密码" in query:
            return [
                SearchResult(
                    text="重置密码流程：进入设置页面，点击重置密码，验证手机号后设置新密码。",
                    score=0.95,
                    metadata={"source": "help.md"},
                )
            ]
        return [
            SearchResult(
                text="这是一段无关信息。",
                score=0.31,
                metadata={"source": "unknown.md"},
            )
        ]


class FakeLLMAdapter:
    async def generate_text(self, template: str, payload: Dict[str, Any]) -> str:
        _ = template
        # 改写节点
        if "failure_reasons" in payload:
            return "如何重置密码"

        # 生成节点
        query = payload.get("query", "")
        contexts = payload.get("contexts", "")
        if "报销流程" in query and "报销流程" in contexts:
            return "报销步骤是：提交申请、主管审批、财务打款。"
        if "重置密码" in query and "重置密码流程" in contexts:
            return "重置密码步骤是：进入设置，点击重置密码，验证手机号后设置新密码。"
        return "我不确定，当前上下文不足。"

    async def generate_structured(self, template: str, payload: Dict[str, Any], output_model):
        _ = template
        query = payload.get("query", "")
        contexts = payload.get("contexts", "")
        answer = payload.get("answer", "")

        # relevance
        if "answer" not in payload and "contexts" in payload:
            if ("报销流程" in query and "报销流程" in contexts) or ("重置密码" in query and "重置密码流程" in contexts):
                return output_model(score=0.95, reasoning="上下文与问题高度相关")
            return output_model(score=0.25, reasoning="上下文与问题相关性低")

        # grounding
        if "answer" in payload and "contexts" in payload:
            if ("报销步骤" in answer and "报销流程" in contexts) or ("重置密码步骤" in answer and "重置密码流程" in contexts):
                return output_model(score=0.90, reasoning="回答可由上下文直接支持")
            return output_model(score=0.20, reasoning="回答无法由上下文支撑")

        # utility
        if "步骤" in answer:
            return output_model(score=0.88, reasoning="回答具备可执行性")
        return output_model(score=0.30, reasoning="回答不够有用")


def test_finish_in_first_hop():
    config = SelfRAGConfig(max_hops=3)
    engine = SelfRAGEngine(
        config=config,
        rag_adapter=FakeRAGPipelineAdapter(),
        llm_adapter=FakeLLMAdapter(),
    )
    out = engine.run_sync("报销流程怎么走")
    assert out.final_decision == "finish"
    assert out.hops_used == 1
    assert "报销步骤" in out.final_answer
    print("✅ 场景1通过：首轮完成。")


def test_rewrite_then_finish():
    config = SelfRAGConfig(max_hops=3)
    engine = SelfRAGEngine(
        config=config,
        rag_adapter=FakeRAGPipelineAdapter(),
        llm_adapter=FakeLLMAdapter(),
    )
    out = engine.run_sync("那个怎么弄")
    assert out.final_decision == "finish"
    assert out.hops_used == 2
    assert out.trace[0]["rewritten_query"] == "如何重置密码"
    assert "重置密码步骤" in out.final_answer
    print("✅ 场景2通过：先改写再完成。")


if __name__ == "__main__":
    test_finish_in_first_hop()
    test_rewrite_then_finish()
    print("🎯 Self-RAG 流程已跑通。")
