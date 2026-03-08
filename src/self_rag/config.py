from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

try:
    from src.core.config import settings  # type: ignore
    _ENABLE_RERANK = settings.rag_online.enable_rerank
    _DYNAMIC_THRESHOLD = settings.rag_online.score_threshold
except Exception:
    # 测试场景下若未安装完整依赖，使用保守默认值。
    _ENABLE_RERANK = True
    _DYNAMIC_THRESHOLD = 0.5


@dataclass
class SelfRAGConfig:
    """Self-RAG 全局配置。"""

    max_hops: int = 3
    relevance_threshold: float = 0.65
    grounding_threshold: float = 0.70
    utility_threshold: float = 0.65
    fallback_answer: str = "根据当前可用信息，暂时无法给出可靠答案。"
    retrieval_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "retrieval": {"top_k": 5, "rough_top_k": 8},
            "online": {
                "enable_rerank": _ENABLE_RERANK,
                "dynamic_threshold": _DYNAMIC_THRESHOLD,
            },
            "filter": {},
            "composer": {
                "enable_hybrid_search": True,
                "plugin_rewritten_query": True,
                "plugin_rewritten_hyde": False,
                "plugin_es_questions": False,
                "plugin_es_summaries": False,
            },
        }
    )
