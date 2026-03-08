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
    # judge 三节点专用外部 LLM 路由配置（默认复用 augmented 的 endpoint 文件）。
    judge_llm_json_path: str = "src/augmented/llm_endpoints.json"
    judge_llm_group: str = "analyst_llms"
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
                # 因为self.rag自己会改写
                "plugin_rewritten_query": False,
                "plugin_rewritten_hyde": False,
                "plugin_es_questions": True,
                "plugin_es_summaries": True,
            },
        }
    )
