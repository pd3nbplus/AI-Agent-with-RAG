# src/rag/fusion/rrf.py
from src.rag.strategies.base import SearchResult
from typing import List, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RRFFusionEngine:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, result_lists: List[List[SearchResult]], top_k: int) -> List[SearchResult]:
        """
        执行倒数排名融合 (RRF)
        """
        if not result_lists:
            return []
            
        score_map: Dict[str, SearchResult] = {} # Key: text (简化去重)
        
        for list_idx, results in enumerate(result_lists):
            for rank, res in enumerate(results):
                # RRF 分数 = 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                
                # 使用 text 前 60 字符作为唯一键 (生产环境建议用 doc_id)
                key = res.text[:60]
                
                if key in score_map:
                    # 累加分数
                    score_map[key].score += rrf_score
                    # 保留最高原始分
                    if res.score > score_map[key].score: # 注意：这里 score 被复用为 RRF 总分了，需小心
                         # 实际上我们应该有个 separate field for original score, 但为了简单，我们只更新 RRF 分
                         pass
                else:
                    # 克隆对象，分数初始化为 RRF 分
                    new_res = SearchResult(
                        text=res.text,
                        score=rrf_score, # 初始为 RRF 分
                        metadata=res.metadata.copy(),
                        source_field="fused"
                    )
                    # 保存原始分数供参考
                    new_res.metadata['_original_score'] = res.score
                    score_map[key] = new_res
        
        # 按 RRF 总分降序排列
        final_results = sorted(score_map.values(), key=lambda x: x.score, reverse=True)
        
        logger.info(f"✨ [RRF] 融合完成：{len(result_lists)} 路 -> {len(final_results)} 条 (Top-{top_k})")
        return final_results[:top_k]
