from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TypedDict

from src.self_rag.schemas.judge import JudgeResult


@dataclass
class HopTrace:
    """Self-RAG 单轮（1 hop）执行轨迹。

    含义：
    1. Self-RAG 每一轮都会产出一条 HopTrace，记录“检索 -> 生成 -> 评判 -> 决策”的关键结果；
    2. 多轮执行时，外层会把这些记录按顺序放入 traces 列表，形成完整链路；
    3. 这些轨迹最终会进入输出的 `trace` 字段，便于排查“为什么 finish / rewrite / fallback”。

    关键字段：
    - query: 本轮实际使用的检索问题（可能是改写后的问题）
    - answer: 本轮生成回答
    - contexts: 本轮命中的证据上下文
    - relevance/grounding/utility: 三个评判维度的结构化结果
    - decision: 本轮决策（finish / rewrite / fallback）
    - rewritten_query: 当 decision=rewrite 时生成的新 query（否则为 None）
    """

    hop: int
    query: str
    answer: str
    contexts: List[str]
    relevance: JudgeResult
    grounding: JudgeResult
    utility: JudgeResult
    decision: str
    rewritten_query: Optional[str] = None


@dataclass
class SelfRAGState:
    original_query: str
    current_query: str
    max_hops: int
    traces: List[HopTrace] = field(default_factory=list)


class SelfRAGGraphState(TypedDict, total=False):
    """LangGraph 运行状态定义。

    说明：
    1. `original_query/current_query` 区分“原始问题”和“当前轮检索问题”；
    2. `hop` 表示已执行到第几轮；
    3. `final_decision` 由 `run_hop` 节点写入，用于 conditional edge 决定是否继续循环；
    4. `last_*` 字段用于在图结束时直接组装最终输出，避免再做二次推导。
    """

    original_query: str
    current_query: str
    category: Optional[str]
    max_hops: int
    hop: int
    traces: List[HopTrace]
    route_passed: bool
    final_decision: str
    last_answer: str
    last_contexts: List[str]
    last_rewritten_query: Optional[str]
