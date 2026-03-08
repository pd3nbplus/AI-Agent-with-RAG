from __future__ import annotations


class RouteNode:
    """决定是否进入 Self-RAG（当前默认启用，可在 M2 扩展）。"""

    def run(self, _query: str) -> bool:
        return True
