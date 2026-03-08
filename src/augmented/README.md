# Augmented 模块说明

`src/augmented` 负责评测数据生成、RAG 评估与坏案例诊断。

## 目录职责

- `config.py`: 生成与分析任务的基础配置。
- `data_generator.py`: 从 Milvus 构建评测题并写入 PostgreSQL。
- `evaluator.py`: 执行检索+回答并用 RAGAS 计算评估指标。
- `analyst.py`: 对低分样本做根因诊断并生成报告。
- `llm_router.py`: LLM endpoint 加载与降级调用。
- `prompts.py`: 生成任务使用的 Prompt 模板。
- `sources.py` / `sinks.py`: 数据源与落库适配层。
- `strategies.py`: 评测题任务策略（standard/adversarial/mixed_pair）。

## Evaluator 的 Pipeline 配置（新）

`RAGEvaluator` 通过嵌套 `pipeline_config` 调用 `pipeline.run(query, config=...)`。

配置结构：

```python
pipeline_config = {
    "retrieval": {
        "top_k": 3,
        "rough_top_k": 8,
    },
    "online": {
        "enable_rerank": True,
        "dynamic_threshold": 0.5,
    },
    "filter": {
        "category": None,
        "source": None,
        "min_page": None,
    },
    "composer": {
        "enable_hybrid_search": True,
        "plugin_rewritten_query": True,
        "plugin_rewritten_hyde": False,
        "plugin_es_questions": False,
        "plugin_es_summaries": False,
        "rrf_k": 60,
    },
}
```

说明：

- `retrieval/online/filter` 由 `RetrievalPipeline` 解析。
- `composer` 原样透传到 `RetrieverComposer.search(runtime_config=...)`。

## 使用示例

```python
from src.augmented.evaluator import RAGEvaluator

evaluator = RAGEvaluator(
    pipeline_config={
        "retrieval": {"top_k": 5, "rough_top_k": 10},
        "composer": {
            "plugin_es_questions": True,
            "plugin_es_summaries": True,
        },
        "online": {"dynamic_threshold": 0.3},
    },
)
df, eval_run_id = evaluator.evaluate_from_postgres(limit=50, batch_ids=[1])
```
