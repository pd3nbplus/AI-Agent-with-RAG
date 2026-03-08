# 该模块负责评估数据生成器的最小运行配置：
# 只保留任务规模、提示词配置和 LLM JSON 配置文件路径。
# augmented/config.py
import os
from dataclasses import dataclass

from src.core.config import settings


@dataclass
class GeneratorConfig:
    # Prompt 模板档位
    prompt_profile: str = "default"
    # 本次最多处理多少个 chunk
    chunks_limit: int = 10
    # 过滤过短 chunk，避免无效出题
    min_chunk_length: int = 50
    # 每个 chunk 生成的问题数
    num_questions_per_chunk: int = 2
    # LLM 端点配置 JSON 路径
    llm_json_path: str = "src/Augmented/llm_endpoints.json"
    # 当生成失败或无返回模型信息时的兜底名
    default_model_name: str = "unknown"
    # 单个 chunk 的最大重试次数
    max_retries_per_chunk: int = 2
    # 启用策略列表（逗号分隔）：standard,adversarial,mixed_pair
    enabled_strategies: str = "standard,adversarial,mixed_pair"
    # 策略参数透传（JSON 字符串），示例：
    # {"standard":{"num_questions":2},"adversarial":{"num_questions":1},"mixed_pair":{"pair_count":2,"num_questions":3}}
    strategy_params_json: str = "{}"
    # 评测样本批次号：写入 rag_eval_samples.batch_id，便于按批次抽样评估。
    sample_batch_id: int = 1


def build_default_config() -> GeneratorConfig:
    # 运行参数主要通过环境变量注入，便于本地和线上统一调参。
    return GeneratorConfig(
        prompt_profile=os.getenv("EVAL_PROMPT_PROFILE", "default"),
        chunks_limit=int(os.getenv("EVAL_CHUNKS_LIMIT", "10")),
        min_chunk_length=int(os.getenv("EVAL_MIN_CHUNK_LENGTH", "50")),
        num_questions_per_chunk=int(os.getenv("EVAL_NUM_QUESTIONS", "2")),
        llm_json_path=os.getenv("EVAL_LLM_JSON_PATH", "src/Augmented/llm_endpoints.json"),
        default_model_name=settings.llm.model_name,
        enabled_strategies=os.getenv("EVAL_ENABLED_STRATEGIES", "adversarial,mixed_pair"),
        strategy_params_json=os.getenv("EVAL_STRATEGY_PARAMS_JSON", "{}"),
        sample_batch_id=int(os.getenv("EVAL_SAMPLE_BATCH_ID", "1")),
    )
