from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class JudgeResult:
    """单维度评判结果。"""

    score: float
    reasoning: str
    passed: bool = False

    def model_dump(self) -> dict:
        return asdict(self)
