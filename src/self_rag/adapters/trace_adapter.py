from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TraceAdapter:
    """Self-RAG 运行轨迹记录器。"""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        logger.debug("self-rag-trace: %s", event)

    def dump(self) -> List[Dict[str, Any]]:
        return list(self.events)
