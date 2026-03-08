from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class GeneratedSample(BaseModel):
    category: str = Field(default="general", description="问题所属类别")
    difficulty: str = Field(description="难度等级：easy, medium, hard")
    question: str = Field(description="用户可能提出的具体问题")
    ground_truth_contexts: List[str] = Field(description="可支持答案的文档片段列表")
    ground_truth: str = Field(description="基于上下文的标准答案")
    source_document: str | None = Field(default=None, description="来源文档名")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")


class EvalInputSample(BaseModel):
    sample_id: str
    sample_batch_id: int = 1
    question: str
    ground_truth: str
    ground_truth_contexts: List[str] = Field(default_factory=list)


class EvalResultSample(BaseModel):
    sample_id: str
    sample_batch_id: int = 1
    question: str
    answer: str
    contexts: List[str] = Field(default_factory=list)
    ground_truth: str
    ground_truth_contexts: List[str] = Field(default_factory=list)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

