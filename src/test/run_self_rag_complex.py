"""真实链路 Self-RAG 复杂问题测试脚本（不使用任何 Fake 数据）。

运行示例：
python -m src.test.run_self_rag_complex
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


# 运行配置：按你的要求直接用字典写死，不走命令行参数。
RUN_CONFIG: Dict[str, Any] = {
    "query": (
        "如果一名员工想通过步行出差获得最高荣誉，他每走一公里能拿到多少补贴？" + "请分点说明，并给出对应规则/章节依据。"
    ),
    "category": None,
    "max_hops": 3,
    "judge_llm_json_path": "src/augmented/llm_endpoints.json",
    "judge_llm_group": "analyst_llms",
    "require_finish": False,
}


def _print_trace(trace: List[Dict[str, Any]]) -> None:
    print("trace:")
    if not trace:
        print("  (empty)")
        return

    for item in trace:
        relevance = item.get("relevance", {})
        grounding = item.get("grounding", {})
        utility = item.get("utility", {})
        print(
            "  - hop={hop} decision={decision} rewritten={rewritten}".format(
                hop=item.get("hop"),
                decision=item.get("decision"),
                rewritten=item.get("rewritten_query"),
            )
        )
        print(
            "    relevance={:.2f} grounding={:.2f} utility={:.2f}".format(
                float(relevance.get("score", 0.0)),
                float(grounding.get("score", 0.0)),
                float(utility.get("score", 0.0)),
            )
        )


def main() -> int:
    # 延迟导入，便于在缺依赖时给出更明确提示。
    from src.self_rag import SelfRAGConfig, SelfRAGEngine

    # 使用真实链路：不注入 rag_adapter / llm_adapter，直接走项目默认配置与本地服务。
    config = SelfRAGConfig(
        max_hops=int(RUN_CONFIG["max_hops"]),
        judge_llm_json_path=str(RUN_CONFIG["judge_llm_json_path"]),
        judge_llm_group=str(RUN_CONFIG["judge_llm_group"]),
    )
    engine = SelfRAGEngine(config=config)

    out = engine.run_sync(
        query=str(RUN_CONFIG["query"]),
        category=RUN_CONFIG.get("category"),
    )

    print("=== Self-RAG Real Complex Run ===")
    print(f"query: {RUN_CONFIG['query']}")
    print(f"category: {RUN_CONFIG.get('category')}")
    print(f"final_decision: {out.final_decision}")
    print(f"hops_used: {out.hops_used}")
    print(f"rewritten_query: {out.rewritten_query}")
    print(f"final_answer: {out.final_answer}")
    _print_trace(out.trace)

    # 额外打印结构化 JSON，方便你做日志采集或后续 diff。
    print("output_json:")
    print(json.dumps(out.model_dump(), ensure_ascii=False, indent=2))

    if bool(RUN_CONFIG.get("require_finish")) and out.final_decision != "finish":
        print("❌ require-finish 已开启，但最终决策不是 finish。")
        return 2

    print("✅ self-rag 真实链路已执行完成。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        if exc.name == "langgraph":
            print("❌ 缺少依赖：langgraph。请先安装后再运行该脚本。")
            print("建议命令：pip install langgraph")
            raise SystemExit(1)
        raise
    except Exception as exc:
        # 真实链路常见失败点：本地 LLM 服务未启动、向量库/ES/数据库未就绪等。
        print(f"❌ 运行失败：{exc}")
        print("请检查 .env 配置与本地依赖服务状态（LLM API / Milvus / Elasticsearch / 数据库）。")
        raise SystemExit(1)
