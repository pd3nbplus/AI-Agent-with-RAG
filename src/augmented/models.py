# augmented/models.py
# Pydantic 输出模型：约束 LLM 返回字段，避免脏数据写入数据库。
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    category: str = Field(default="general", description="问题所属类别")
    difficulty: str = Field(description="难度等级：easy, medium, hard")
    query: str = Field(description="用户可能提出的具体问题")
    ground_truth_context: List[str] = Field(description="能够回答该问题的文档原文片段列表（关键句）")
    ground_truth_answer: str = Field(description="基于上下文的标准答案，简洁准确")
    source_document: Optional[str] = Field(default=None, description="来源文档名")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
