# src/rag/reranker.py
from sentence_transformers import CrossEncoder
from src.core.config import settings
import logging
import torch
from typing import List, Dict
from src.rag.strategies.base import SearchResult # 👈 导入 SearchResult 类


logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        self.model_name = settings.rag_online.rerank_model_name
        
        self.device = settings.rag_online.rerank_device
        # 1. 应用 CPU 线程限制 (必须在加载模型前设置)
        self.num_threads = settings.rag_online.torch_num_threads
        torch.set_num_threads(self.num_threads)
        
        logger.info(f"🔄 加载 CrossEncoder 重排模型：{self.model_name} (运行在 {self.device})")
        logger.info(f"🧵 CPU 线程数已限制为：{self.num_threads} (防止资源独占)")
        
        # 2. 处理最大长度：如果配置了则使用，否则让模型自己决定 (通常是 512)
        max_length_arg = {}
        self.rerank_max_length = settings.rag_online.rerank_max_length
        if self.rerank_max_length:
            max_length_arg['max_length'] = self.rerank_max_length
            logger.info(f"⚙️ 限制最大输入长度：{self.rerank_max_length}")
        else:
            logger.info("⚙️ 使用模型默认最大长度 (不截断，保证语义完整)")

        try:
            self.reranker = CrossEncoder(
                model_name_or_path=self.model_name,
                device=self.device,
                trust_remote_code=True,
                **max_length_arg
            )
            logger.info("✅ CrossEncoder 模型加载成功")
        except Exception as e:
            logger.error(f"❌ 模型加载失败：{e}")
            raise e

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
        if not candidates:
            return []
        
        # 准备输入对
        pairs = [[query, cand.text] for cand in candidates]
        
        logger.debug(f"⚖️ 开始对 {len(pairs)} 个候选项进行重排序...")
        
        try:
            # 批量计算分数
            scores = self.reranker.predict(
                pairs, 
                convert_to_numpy=True, 
                show_progress_bar=False
            )
            
            # 确保 scores 是列表 (某些版本单元素可能返回 float)
            if isinstance(scores, (int, float)):
                scores = [scores] * len(candidates)
            else:
                scores = scores.tolist()

        except Exception as e:
            logger.error(f"❌ 重排序计算失败：{e}")
            return candidates[:top_k]
        
        # 回填分数
        for i, cand in enumerate(candidates):
            cand.rerank_score = scores[i]
        
        ranked_candidates = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
        final_results = ranked_candidates[:top_k]
        
        if final_results:
            logger.info(f"✨ 重排序完成：最高分 {final_results[0].rerank_score:.4f}")
        
        # 转换为字典格式返回
        return [
            SearchResult(
                text=cand.text,
                score=cand.rerank_score,
                source_field=cand.source_field,
                metadata=cand.metadata,
                rank = i,
            ) for i, cand in enumerate(final_results)
        ]

# 注意：不要在模块导入时创建全局实例。
# 统一通过 RerankerFactory.get_reranker() 懒加载并复用。
