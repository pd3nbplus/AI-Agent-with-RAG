from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.self_rag.schemas.judge import JudgeResult


@dataclass
class HopTrace:
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
