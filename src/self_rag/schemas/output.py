from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class SelfRAGOutput:
    """Self-RAG 最终输出。"""

    query: str
    final_answer: str
    final_decision: str
    hops_used: int
    contexts: List[str] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    trace: List[dict] = field(default_factory=list)

    def model_dump(self) -> dict:
        return asdict(self)
