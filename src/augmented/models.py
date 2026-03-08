# augmented/models.py
# Pydantic 输出模型：约束 LLM 返回字段，避免脏数据写入数据库。

from src.schema.augmented_schema import EvalInputSample, EvalResultSample, GeneratedSample

__all__ = ["GeneratedSample", "EvalInputSample", "EvalResultSample"]
