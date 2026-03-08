# src/rag/strategies/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class SearchResult:
    """统一检索结果对象"""
    text: str
    score: float
    metadata: Dict[str, Any]
    source_field: str = "text"
    rank: int = 0 # 用于 RRF 计算
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "source_field": self.source_field
        }

class BaseRetrievalStrategy(ABC):
    """检索策略基类"""
    @abstractmethod
    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        pass

class BaseFilterStrategy(ABC):
    """过滤策略基类"""
    @abstractmethod
    def build_expr(self, **kwargs) -> Optional[str]:
        pass