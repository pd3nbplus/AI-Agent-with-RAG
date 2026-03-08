"""Self-RAG LangGraph 最小自测脚本。

运行方式（项目根目录）：
PYTHONPATH=. python3 src/self_rag/self_test.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class _FakeSearchResult:
    """最小检索结果对象。

    这里只保留 `text` 即可，因为 SelfRAGEngine 当前只消费该字段。
    """

    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


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
                _FakeSearchResult(
                    text="报销流程：提交申请->主管审批->财务打款。",
                    score=0.92,
                    metadata={"source": "policy.md"},
                )
            ]
        if "重置密码" in query:
            return [
                _FakeSearchResult(
                    text="重置密码流程：进入设置页面，点击重置密码，验证手机号后设置新密码。",
                    score=0.95,
                    metadata={"source": "help.md"},
                )
            ]
        return [
            _FakeSearchResult(
                text="这是一段无关信息。",
                score=0.31,
                metadata={"source": "unknown.md"},
            )
        ]


class FakeLLMAdapter:
    async def generate_text(self, template: str, payload: Dict[str, Any]) -> str:
        _ = template

        # 改写节点：当评分不达标时，稳定产出一个可检索的新 query。
        if "failure_reasons" in payload:
            return "如何重置密码"

        # 生成节点：
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

        # relevance：只有 query + contexts
        if "answer" not in payload and "contexts" in payload:
            if ("报销流程" in query and "报销流程" in contexts) or ("重置密码" in query and "重置密码流程" in contexts):
                return output_model(score=0.95, reasoning="上下文与问题高度相关")
            return output_model(score=0.25, reasoning="上下文与问题相关性低")

        # grounding：包含 query + answer + contexts
        if "answer" in payload and "contexts" in payload:
            if ("报销步骤" in answer and "报销流程" in contexts) or ("重置密码步骤" in answer and "重置密码流程" in contexts):
                return output_model(score=0.90, reasoning="回答可由上下文直接支持")
            return output_model(score=0.20, reasoning="回答无法由上下文支撑")

        # utility：只有 query + answer
        if "步骤" in answer:
            return output_model(score=0.88, reasoning="回答具备可执行性")
        return output_model(score=0.30, reasoning="回答不够有用")


def run_self_test() -> None:
    """执行两条核心路径的最小回归。"""

    from src.self_rag import SelfRAGConfig, SelfRAGEngine

    engine = SelfRAGEngine(
        config=SelfRAGConfig(max_hops=3),
        rag_adapter=FakeRAGPipelineAdapter(),
        llm_adapter=FakeLLMAdapter(),
    )

    # 路径1：首轮即满足三个评判维度，直接 finish。
    out1 = engine.run_sync("报销流程怎么走")
    assert out1.final_decision == "finish"
    assert out1.hops_used == 1
    assert "报销步骤" in out1.final_answer
    print("✅ 场景1通过：首轮完成。")

    # 路径2：首轮失败 -> rewrite -> 第二轮命中 -> finish。
    out2 = engine.run_sync("那个怎么弄")
    assert out2.final_decision == "finish"
    assert out2.hops_used == 2
    assert out2.trace[0]["rewritten_query"] == "如何重置密码"
    assert "重置密码步骤" in out2.final_answer
    print("✅ 场景2通过：先改写再完成。")

    print("🎯 Self-RAG LangGraph 最小自测通过。")


if __name__ == "__main__":
    try:
        run_self_test()
    except ModuleNotFoundError as exc:
        if exc.name == "langgraph":
            print("❌ 缺少依赖：langgraph。请先安装后再运行自测。")
            print("建议命令：pip install langgraph")
            raise SystemExit(1)
        raise
