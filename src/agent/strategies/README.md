# Agent 策略配置说明

本目录的检索策略（`fast_retrieval` / `standard_retrieval` / `deep_search`）使用统一的嵌套配置驱动 `RetrievalPipeline`。

## 目标

- 在策略层维护一份 `pipeline_config`。
- 调用 `pipeline.run(query, config=...)` 时由 Pipeline 解析自身参数。
- `config["composer"]` 原样透传给 `RetrieverComposer.search(runtime_config=...)`。

## 嵌套配置结构

```python
pipeline_config = {
    "retrieval": {
        "top_k": 5,          # 最终返回条数
        "rough_top_k": 8,    # 多路召回粗排条数
    },
    "online": {
        "enable_rerank": True,
        "dynamic_threshold": 0.5,  # Top1-Top2 分差触发阈值
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

## 透传规则

- `retrieval` / `online` / `filter`：由 `RetrievalPipeline` 解析并执行。
- `composer`：Pipeline 不展开处理，直接透传给 `RetrieverComposer`。
- 运行时如果策略补充了 `context.category`，会覆盖 `filter.category`。

## 默认策略差异

- `FastRetrievalStrategy`
  - 单路向量检索
  - `enable_hybrid_search=False`
  - `enable_rerank=False`
  - `top_k=3`
- `StandardRetrievalStrategy`
  - 混合检索 + 可选重排
  - 默认 `plugin_rewritten_query=True`
  - `top_k=5`
- `DeepSearchStrategy`
  - 当前复用 Standard 的配置和流程
  - 后续可单独扩展配置

## 自定义示例

```python
custom_cfg = {
    "retrieval": {"top_k": 6, "rough_top_k": 12},
    "online": {"enable_rerank": True, "dynamic_threshold": 0.3},
    "composer": {
        "enable_hybrid_search": True,
        "plugin_rewritten_query": True,
        "plugin_es_questions": True,
        "plugin_es_summaries": True,
        "rrf_k": 50,
    },
}

strategy = StandardRetrievalStrategy(pipeline_config=custom_cfg)
```
