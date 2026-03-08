# 🚀 AI Agent 开发实战笔记：从 Docker 容器化环境到智能体架构

> **学习时间**：2026-02-26  
> **环境背景**：WSL2 + Ubuntu 24.04 + Docker Desktop (MySQL, Redis, Qwen 容器化)  
> **核心目标**：构建可移植、可复现的“环境即代码”AI Agent 开发流

---

## 📌 前言：为什么你的环境是“天胡”开局？

兄弟，你现在的这套 **WSL2 + Docker Compose** 架构简直是 AI Agent 开发的“梦中情机”。

很多初学者还在纠结“我本地 Python 版本不对”、“Redis 连不上”、“大模型 API 密钥配置乱了”，而你直接通过 `docker-compose up` 就拥有了：
- **隔离的计算单元**：Qwen 大模型在独立容器跑，不占宿主机资源。
- **持久化记忆层**：MySQL 存业务数据，Redis 做高速缓存/消息队列。
- **一致性**：今天写的代码，明天换个电脑，`git clone` + `docker compose up` 原地复活。

**AI Agent 的本质** = **LLM (大脑)** + **Memory (记忆)** + **Tools (手脚)** + **Planning (规划)**。
既然基础设施都好了，我们直接跳过“配环境”的坑，进入核心的**架构设计与实战路线**。

---

## 🗺️ 学习路线大纲 (Roadmap)

我们将学习过程分为五个阶段，每个阶段都紧扣你的 Docker 环境。

### 第一阶段：基石构建 —— 容器化交互与 Prompt 工程
**目标**：在 Python 代码中优雅地调用容器内的 Qwen，并掌握结构化输出。

1.  **Docker 网络打通**
    *   理解 Docker Compose 的内部 DNS（如何在 Python 容器里访问 `http://qwen:8000` 而不是 `localhost`）。
    *   *避坑指南*：WSL2 端口转发陷阱与 Docker Network 桥接模式。
2.  **LLM 抽象层封装**
    *   使用 `LangChain` 或 `LlamaIndex` (或者轻量级的 `LiteLLM`) 对接本地 Qwen。
    *   实现统一的 `ChatModel` 接口，方便未来切换模型。
3.  **结构化输出 (Structured Output)**
    *   Agent 需要的是 JSON，不是散文。
    *   利用 Pydantic 强制 Qwen 输出符合 Schema 的数据（这是 Agent 执行工具的前提）。

### 第二阶段：记忆系统 (Memory) —— 让 Agent 拥有“海马体”
**目标**：利用 MySQL 和 Redis 构建短期与长期记忆。

1.  **短期记忆 (Context Window)**
    *   对话历史的管理策略：滑动窗口 vs 摘要总结。
    *   实战：将对话列表存入 Redis List，设置 TTL 自动过期。
2.  **长期记忆 (Vector Store & RAG)**
    *   *拓展*：虽然你目前只有 MySQL/Redis，但 Agent 通常需要向量库。
    *   **方案 A (轻量)**：利用 Redis Stack 的向量搜索功能 (RedisVL)。
    *   **方案 B (传统)**：在 MySQL 中存储文本块，配合简单的关键词检索 (BM25) 作为起步，后续再引入 Milvus/Qdrant 容器。
3.  **状态持久化**
    *   将 Agent 的运行状态（State）序列化存入 MySQL，支持断点续跑。

### 第三阶段：工具链 (Tools) —— 给 Agent 装上“机械臂”
**目标**：让 LLM 能够执行代码、查询数据库、调用外部 API。

1.  **Function Calling 原理**
    *   如何定义工具描述（Description），让 Qwen 知道何时调用。
    *   参数解析与异常处理（LLM 传参错了怎么办？）。
2.  **自定义工具开发**
    *   **DB Tool**：编写一个工具，让 Agent 能用自然语言查询你的 MySQL 业务数据（Text-to-SQL 雏形）。
    *   **System Tool**：让 Agent 能读取日志、重启服务（注意权限隔离！）。
3.  **沙箱执行环境**
    *   *安全最佳实践*：不要直接在主容器运行 Agent 生成的代码。
    *   动态启动临时 Docker 容器来执行不安全代码（Sandboxing）。

### 第四阶段：规划与编排 (Planning & Orchestration)
**目标**：从“一问一答”进化为“自主完成任务”。

1.  **ReAct 模式 (Reason + Act)**
    *   经典的“思考 - 行动 - 观察”循环。
    *   实战：手写一个 ReAct 循环，不调用重型框架，理解底层逻辑。
2.  **多 Agent 协作 (Multi-Agent)**
    *   角色分工： Planner (规划者) + Executor (执行者) + Reviewer (审查者)。
    *   利用 Redis Pub/Sub 或 RabbitMQ (如果需要可以加一个容器) 实现 Agent 间的消息传递。
3.  **工作流引擎**
    *   对比 LangGraph (基于图的状态机) 与 简单线性链。
    *   场景：复杂任务拆解与并行执行。

### 第五阶段：部署与观测 (Ops for Agents)
**目标**：让你的 Agent 从 Demo 变成生产级服务。

1.  **可观测性 (Observability)**
    *   记录 Token 消耗、延迟、LLM 决策路径。
    *   集成 LangSmith 或自建日志面板（ELK/Loki）。
2.  **评估体系 (Evaluation)**
    *   如何测试 Agent 的好坏？(准确性、幻觉率、工具调用成功率)。
    *   构建自动化测试集。
3.  **高可用架构**
    *   Docker Compose 的资源限制 (CPU/Memory limits) 防止 OOM。
    *   异步并发处理 (FastAPI + asyncio) 应对多用户请求。

---

## 💡 核心概念深度解析：记忆系统设计

在你的环境中，**Redis** 和 **MySQL** 的组合非常经典。我们来聊聊怎么把它们用在 Agent 上。

### 1. Redis：高速缓冲与会话状态
Agent 的“短期记忆”必须快。想象你在和人聊天，如果每说一句话都要去查硬盘（MySQL），那对话节奏就崩了。

*   **使用场景**：
    *   存储最近 N 轮对话历史 (`LPUSH` + `LTRIM`)。
    *   存储 Agent 当前的思维链状态（比如 ReAct 循环中的 `thought`, `action` 中间态）。
    *   分布式锁：防止同一个用户触发两个并发的 Agent 任务。
*   **代码类比**：
    ```python
    # 伪代码：获取会话上下文
    async def get_context(session_id):
        # Redis 速度极快，适合高频读写
        messages = await redis.lrange(f"session:{session_id}", 0, -1)
        return deserialize(messages)
    ```

### 2. MySQL：长期知识与业务事实
Agent 需要记住用户的偏好、订单信息，或者知识库文档。这些数据结构化强，且需要持久保存。

*   **使用场景**：
    *   用户画像（User Profile）。
    *   RAG 的元数据索引（如果不用专用向量库）。
    *   任务执行的历史审计日志。
*   **最佳实践**：
    *   不要直接把整个向量塞进 MySQL 做相似度搜索（除非用 MySQL 8.0+ 的向量插件且数据量小）。
    *   **模式**：Redis 做热点缓存 -> MySQL 做持久化兜底。

---

## ⚠️ 常见误区与避坑指南

1.  **过度依赖框架**：
    *   *误区*：一开始就上 LangChain 全套，结果被复杂的 Callback 和抽象层搞晕，调试困难。
    *   *建议*：前两周，尝试用纯 Python + `requests` 调用 Qwen API 手写一个最简单的 ReAct 循环。理解了 `Prompt -> LLM -> Parse -> Action -> Observation` 的闭环后，再引入框架。

2.  **忽视 Token 成本与上下文限制**：
    *   *误区*：把所有历史记录都塞进 Prompt。
    *   *后果*：超出 Qwen 上下文窗口，或者响应极慢，费用爆炸（如果是商用 API）。
    *   *对策*：必须实现**记忆压缩**（Summary）或**滑动窗口**。

3.  **本地模型的性能陷阱**：
    *   *现状*：你在 Docker 里跑 Qwen。如果是量化版（如 Qwen-7B-Int4），推理速度尚可；如果是全量版，显存可能爆掉。
    *   *建议*：监控 Docker 容器的资源使用 (`docker stats`)。如果推理太慢，考虑使用 vLLM 或 Ollama 作为后端容器替代原生 HuggingFace 加载，它们对并发和显存优化更好。

4.  **网络隔离问题**：
    *   *痛点*：Python 代码里写 `localhost` 连不上数据库。
    *   *解决*：在 Docker Compose 中，服务间通信请使用**服务名**（如 `mysql`, `redis`, `qwen`），而不是 `127.0.0.1`。

---

## 🛠️ 下一步行动建议 (Next Steps)

既然大纲已经理清，我们不要贪多。建议你按照以下顺序开始**第一讲**的实战：

1.  **检查环境**：确保 `docker-compose ps` 所有服务都是 `Up` 状态。
2.  **Hello World 连通性**：
    *   创建一个简单的 Python 脚本（在宿主机或新建一个 dev 容器）。
    *   尝试通过 HTTP 请求调用容器内的 Qwen 接口。
    *   尝试连接 Redis 写入一个 Key，再从 MySQL 读取一行数据。
3.  **第一个微型 Agent**：
    *   不依赖 LangChain。
    *   实现一个功能：用户问“现在几点了？”，Agent 识别意图，调用本地 `datetime` 工具，返回时间。

**接下来你可以选择深入的方向：**
*   👉 **方向 A**：深入研究 **Prompt Engineering 与结构化输出**（如何用 Pydantic 约束 Qwen 的输出）。
*   👉 **方向 B**：直接上手 **LangGraph** 学习状态机编排（适合想快速构建复杂流程）。
*   👉 **方向 C**：先搞定 **RAG 基础**，把 MySQL 里的业务数据变成 Agent 可查询的知识库。

你想先从哪个方向开始？或者我们先写一段代码验证一下你的 Docker 网络连通性？


------

# 📘 第三讲：RAG 进阶实战 —— 从“能搜到”到“搜得准”
### 第一部分：检索前优化 (Offline Optimization) —— 让知识“颗粒度”更精细


**痛点**：直接存入长文导致语义稀释，关键信息被淹没。
**核心任务**：构建智能数据摄入管道 (Data Ingestion Pipeline)。

1.  **文本分块策略 (Chunking Strategies)**
    *   **固定长度分块 (Fixed-size Chunking)**：基础方案，缺点是会切断句子。
    *   **递归字符分块 (Recursive Character Text Splitter)**：⭐ **重点**。按“段落 -> 句子 -> 单词”层级切割，保持语义完整。
    *   **滑动窗口 (Sliding Window)**：引入重叠区域 (Overlap)，防止上下文断裂。
2.  **元数据增强 (Metadata Enrichment)**
    *   自动提取文档标题、章节、页码。
    *   利用 LLM 为每个 Chunk 生成**“摘要”**和**“假设性问题”** (Hypothetical Questions)，存入 Metadata，提升检索命中率。
3.  **实战演练**
    *   编写 `DocumentLoader` 类，支持 PDF/Markdown 读取。
    *   集成 `LangChain Text Splitters`。
    *   实现“批量入库”工具，一键处理本地文件夹。

### 第二部分：检索后优化 (Online Optimization) —— 让结果“更相关”
**痛点**：向量检索召回了 10 条，但前 3 条可能不相关，或者相关度排序不准。
**核心任务**：引入重排序 (Re-ranking) 机制。

1.  **重排序原理 (Re-ranking Mechanism)**
    *   **向量检索 (Bi-Encoder)**：快，但精度一般（粗排）。
    *   **交叉编码 (Cross-Encoder)**：慢，但精度极高（精排）。将 Query 和 Document 拼接输入模型，直接打分。
2.  **实战演练：集成 BGE-Reranker**
    *   引入 `BAAI/bge-reranker-v2-m3` 模型。
    *   修改 `RetrievalPipeline`：先由 Milvus 召回 Top-20，再由 Reranker 精排选出 Top-3。
    *   对比优化前后的检索结果差异。

### 第三部分：架构模块化 (Modular Architecture) —— 打造可插拔组件
**痛点**：代码耦合，想换分块策略或重排模型需要大改代码。
**核心任务**：重构 `rag/` 模块，实现策略模式。

1.  **工厂模式设计**
    *   `ChunkerFactory`：动态切换分块策略（配置驱动）。
    *   `RerankerFactory`：动态切换重排模型（或开关）。
2.  **配置驱动开发**
    *   在 `config.py` 中新增 `CHUNK_SIZE`, `CHUNK_OVERLAP`, `ENABLE_RERANK` 等开关。
    *   实现“零代码”调整检索策略。

### 第四部分：端到端验收 (End-to-End Evaluation)
**任务**：构建一个完整的测试集，量化评估优化效果。

1.  **构建测试集**
    *   准备 5 个典型问题（包含语义模糊、多跳推理等难点）。
2.  **A/B 测试**
    *   **Baseline**：无分块 + 无重排。
    *   **Optimized**：递归分块 + 滑动窗口 + 重排序。
3.  **指标观察**
    *   命中率 (Hit Rate)。
    *   响应质量 (LLM 回答的准确性)。

---

## 🛠️ 技术栈预告
*   **LangChain Text Splitters**: 递归分块核心库。
*   **PyMuPDF / Unstructured**: 文档解析工具。
*   **FlagEmbedding / BGE-Reranker**: 高性能重排序模型。
*   **Config Pattern**: 策略模式在 Python 中的落地。

---

## 💡 预期成果
课程结束后，你将拥有：
1.  一个**支持上传本地文件**（PDF/MD）并自动分块入库的工具。
2.  一个**具备“粗排 + 精排”双重过滤**的高精度检索管道。
3.  一套**可配置、可插拔**的 RAG 模块化代码架构。

---

**👉 请评估：**
1.  这个大纲是否符合你当前的预期？
2.  是否有特定想优先深入的技术点（比如特别关注 PDF 解析，或者特别关注重排序模型的选择）？
3.  是否需要调整节奏（例如将“分块”和“重排”拆分为两讲详细讲解）？

请给出你的反馈，确认后我们立即开始 **第一部分：智能分块与数据摄入** 的实战！


# 📘 第四讲：RAG 深度优化实战 —— 从“可用”到“卓越”

> **核心目标**：填补现有架构中的“未竟之地”，挖掘 Metadata 与 Summary 的深层价值，引入多路召回机制，解决复杂场景下的检索失效问题。
> **演进路线**：**Advanced RAG** $\rightarrow$ **High-Performance / Modular RAG**。
> **关键指标**：召回率 (Recall) 提升 30%+，复杂问答准确率显著提升。

---

## 📅 课程大纲预览

### 第一部分：元数据与摘要的深度利用 (Metadata & Summary Deep Dive)
**痛点**：目前的 Metadata（摘要、假设性问题）仅存储在数据库中，检索时未被充分利用，导致“存而不用”。
**核心任务**：实现**混合检索 (Hybrid Search)** 与 **预过滤/后过滤** 策略。

1.  **摘要增强检索 (Summary-Augmented Retrieval)**
    *   **策略 A**：将 `summary` 字段也进行向量化，与 `text` 字段共同构建索引（双字段检索）。
    *   **策略 B**：在 Prompt 中动态注入 Summary，让 LLM 基于摘要快速判断相关性。
2.  **假设性问题匹配 (Hypothetical Question Matching)**
    *   利用存储的 `questions` 字段进行**关键词匹配 (BM25)** 或 **专用向量检索**。
    *   解决“用户提问”直接匹配“文档陈述句”的语义鸿沟。
3.  **元数据过滤 (Metadata Filtering)**
    *   实现基于 `category`, `source`, `page` 的**预过滤 (Pre-filtering)**。
    *   场景：用户问“HR 政策里的年假规定”，自动过滤掉“产品手册”中的内容。

### 第二部分：多路召回与融合 (Multi-Retrieval & Fusion)
**痛点**：单一向量检索无法覆盖所有场景（如精确匹配专有名词、代码片段、特定术语）。
**核心任务**：构建**多路召回 (Multi-Query / Hybrid Retrieval)** 管道。

1.  **关键词检索 (Sparse Retrieval / BM25)**
    *   引入 **BM25 算法**（Milvus 支持或集成 Elasticsearch/Whoosh）。
    *   优势：精确匹配专有名词、型号、代码错误码。
2.  **多查询生成 (Multi-Query Retrieval)**
    *   让 LLM 从不同角度生成 3-5 个变体查询（如：正面、反面、侧面描述）。
    *   并行执行多次向量检索，扩大召回覆盖面。
3.  **结果融合策略 (Fusion Strategies)**
    *   **倒数排名融合 (RRF, Reciprocal Rank Fusion)**：工业界标准融合算法，无需调参，自动平衡各路结果。
    *   **加权评分融合**：根据业务需求调整向量分与关键词分的权重。

### 第三部分：父子索引与上下文窗口优化 (Parent-Child Indexing)
**痛点**：小 Chunk 检索准但上下文缺失，大 Chunk 上下文全但检索噪声大。
**核心任务**：实现 **父子文档索引 (Parent-Child Retrieval)**。

1.  **大小块双索引策略**
    *   **子块 (Child)**：小粒度分块（200字），用于高精度向量检索。
    *   **父块 (Parent)**：大粒度分块（1000字）或原文档，用于提供给 LLM 完整上下文。
2.  **检索与替换逻辑**
    *   检索命中“子块”后，自动通过 ID 映射找回对应的“父块”。
    *   将“父块”送入 LLM，既保证了检索精度，又提供了充足上下文。
3.  **实战演练**
    *   修改 `ingestion.py` 建立父子映射关系。
    *   修改 `milvus_client.py` 支持 ID 反向查找。

### 第四部分：端到端评估与自动化测试 (Evaluation & Testing)
**痛点**：优化效果靠“体感”，缺乏量化数据支撑。
**核心任务**：构建 **RAG 评估框架**。

1.  **构建金标准数据集 (Golden Dataset)**
    *   准备 20-50 个典型问答对 (Query, Ground Truth Answer, Relevant Chunks)。
2.  **核心指标计算**
    *   **Hit Rate @ K**：正确答案是否在 Top-K 中。
    *   **MRR (Mean Reciprocal Rank)**：正确答案的平均排名倒数。
    *   **Faithfulness**：LLM 回答是否忠实于检索内容（防幻觉）。
3.  **A/B 测试框架**
    *   对比配置：`Base RAG` vs `Hybrid Search` vs `Parent-Child`。
    *   输出可视化报告，量化每一步优化的收益。

---

## 🛠️ 技术栈预告
*   **Milvus Scalar Filtering**: 元数据预过滤。
*   **Rank-BM25 / Milvus Text Embedding**: 稀疏向量/关键词检索。
*   **LangChain RRF**: 倒数排名融合算法。
*   **Ragas / Arize Phoenix** (可选): 自动化评估框架（或自建简易版）。
*   **ID Mapping Logic**: 父子索引核心逻辑。

---

## 💡 预期成果
课程结束后，你将拥有：
1.  **混合检索引擎**：同时支持向量语义匹配 + 关键词精确匹配 + 元数据过滤。
2.  **上下文增强机制**：通过父子索引，彻底解决“检索准但回答缺上下文”的矛盾。
3.  **量化评估体系**：一套可复用的测试脚本，用数据证明优化的价值。
4.  **真正的企业级 RAG**：能够处理复杂、模糊、专业术语多的真实业务场景。

---

## 🗓️ 建议学习节奏
*   **第 1 步**：元数据深度利用 (最快见效，改动最小)。
*   **第 2 步**：多路召回与融合 (核心难点，提升最大)。
*   **第 3 步**：父子索引 (架构升级，解决长上下文问题)。
*   **第 4 步**：自动化评估 (收尾验证，数据驱动)。

---

**👉 请评估：**
1.  这个大纲是否覆盖了你关心的“坑”（Summary, Metadata, 多路召回）？
2.  **父子索引 (Parent-Child)** 是否是你当前迫切需要的？（如果不是，我们可以替换为其他优化点，如“查询路由 Query Routing”）。
3.  是否需要调整顺序？例如先做“多路召回”还是先做“元数据过滤”？

请给出你的反馈，确认后我们将立即开始 **第一部分：元数据与摘要的深度利用**！




除了 **Sentence-as-Child (Sentence Window Retrieval)**，目前 RAG 领域还有几个公认的“版本答案”（SOTA, State-of-the-Art）。它们分别解决了不同维度的痛点。

根据你的架构（模块化、混合检索），以下是三个最值得集成的进阶方案，按**推荐优先级**排序：

---

### 1. 🏆 方案一：GraphRAG (基于知识图谱的 RAG)
**解决痛点**：**全局理解**与**多跳推理**。
*   **场景**：用户问“这两家公司有什么共同的投资人？”或“总结整个文档集中关于‘气候变化’的所有观点”。
*   **原理**：
    1.  利用 LLM 从文本中提取 **实体 (Entities)** 和 **关系 (Relationships)**。
    2.  构建 **知识图谱 (Knowledge Graph)**。
    3.  对图谱进行 **社区检测 (Community Detection)**，生成每个社区的摘要。
    4.  **检索时**：不仅检索向量块，还检索相关的实体、关系和社区摘要。
*   **优势**：能回答跨文档、跨段落的复杂关联问题，这是纯向量检索做不到的。
*   **微软开源实现**：[Microsoft GraphRAG](https://github.com/microsoft/graphrag)。
*   **如何集成到你的架构**：
    *   新增一个 `GraphRetriever` 插件。
    *   在 `composer.py` 中将其作为第 5 路召回。
    *   适合处理你的测试集中的 **“多跳推理 (Hard)”** 类问题。

### 2. 🎯 方案二：HyDE (Hypothetical Document Embeddings)
**解决痛点**：**查询与文档的语义鸿沟**。
*   **场景**：用户问“怎么修电脑？”，但文档里写的是“故障排除指南：重启步骤...”。两者的词汇完全不匹配，向量相似度低。
*   **原理**：
    1.  用户提问后，先让 LLM **虚构**一篇“完美的答案文档”（HyDE）。
    2.  将这篇**虚构的文档**进行 Embedding。
    3.  用这个虚构文档的向量去检索真实的文档库。
*   **优势**：虚构的答案在语义空间上离真实答案更近，能显著提高召回率，尤其是针对**开放式问题**。
*   **如何集成到你的架构**：
    *   修改 `Rewriter` 模块，增加 `HyDEMode`。
    *   在 `VectorRewrittenRetriever` 中，不再只是改写 Query，而是生成 HyDE 文档并嵌入。
    *   **成本**：每次检索多一次 LLM 调用（延迟增加）。

### 3. 🔄 方案三：Self-RAG / Adaptive Retrieval (自适应检索)
**解决痛点**：**过度检索**与**幻觉**。
*   **场景**：有些问题不需要检索（如“你好”、“1+1等于几”），强行检索反而引入噪声；或者检索到的内容质量很差，LLM 应该拒绝回答。
*   **原理**：
    1.  **检索判断**：LLM 先判断“我需要检索吗？”（Retrieve vs. No Retrieve）。
    2.  **段落评估**：检索后，LLM 给每个 retrieved chunk 打分（支持/反驳/无关）。
    3.  **生成评估**：生成答案后，LLM 自我反思“我的答案有依据吗？”（Is Supported?）。
*   **优势**：极大减少幻觉，动态决定检索时机，节省 Token 和延迟。
*   **如何集成到你的架构**：
    *   在 `Pipeline` 的最前端增加一个 `RouterNode` (LangGraph 擅长这个)。
    *   在 `Reranker` 之后增加一个 `CritiqueNode`，让 LLM 对上下文进行打分过滤。

---

### 📊 横向对比：哪个适合你？

| 方案 | 核心能力 | 延迟 | 成本 | 适合你的测试集类别 | 集成难度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentence-as-Child** | **精准定位 + 完整上下文** | 低 | 低 | 精确匹配、语义理解 | ⭐ (已完成) |
| **GraphRAG** | **全局关联、多跳推理** | 高 (离线建图) / 中 (在线) | 高 (建图贵) | **多跳推理 (Hard)** | ⭐⭐⭐ (需引入图数据库) |
| **HyDE** | **语义对齐、模糊查询** | 中 (多一次 LLM) | 中 | 语义理解、开放问答 | ⭐⭐ (修改 Rewriter) |
| **Self-RAG** | **防幻觉、动态决策** | 波动 (取决于是否检索) | 中 | 所有类别 (提升可靠性) | ⭐⭐ (修改 Pipeline 流程) |

---

### 💡 我的建议：下一步做什么？

鉴于你现在的架构已经非常稳健（混合检索 + 父子索引），我建议按以下顺序演进：

#### 第一阶段：低成本高收益 —— **HyDE**
*   **理由**：你的测试集中有“语义理解与推理”类问题，HyDE 能显著提升这类问题的召回率，且代码改动最小（只需修改 `Rewriter`）。
*   **实施**：在 `src/rag/rewriter.py` 中增加 `generate_hypothetical_document()` 方法。

#### 第二阶段：架构升级 —— **Agentic RAG (Self-RAG 简化版)**
*   **理由**：利用你已有的 **LangGraph** 优势。让 Agent 自己决定“要不要调用检索工具”、“要不要调用搜索引擎”、“要不要直接回答”。
*   **实施**：在 `Mini_Agent/graph.py` 中增加条件边（Conditional Edges），根据意图分类路由。

#### 第三阶段：终极杀器 —— **GraphRAG (轻量版)**
*   **理由**：解决最难的“多跳推理”。不需要全量建图，可以只对**高频实体**或**核心文档**建图。
*   **实施**：引入 `NetworkX` 或 `Neo4j`，提取实体关系，作为额外的检索源。

### 🚀 想先尝试哪一个？
如果你想立刻看到效果提升，我们可以先写 **HyDE** 的实现代码，它和你的 `Sentence-as-Child` 是完美互补的（一个优化索引粒度，一个优化查询向量）。

或者，你想深入聊聊如何利用 **LangGraph** 实现 **自适应检索 (Self-RAG)**？


---


你的 RAG 架构现在已经非常成熟：**混合检索 + 父子索引 + 策略化重写 (HyDE) + 动态重排 + 异步并发**。这已经超过了市面上 90% 的开源 RAG 项目。

如果要继续优化，我们不能再做“堆砌功能”的事，而应该转向 **“智能化”、“精细化”和“可观测性”**。

以下是按 **投入产出比 (ROI)** 排序的下一步优化方向：

---

### 🥇 第一阶段：智能化与自适应 (Agentic & Adaptive)
**目标**：让系统学会“思考”，而不是机械地执行固定流程。

#### 1. 路由与自适应检索 (Router & Adaptive Retrieval)
*   **痛点**：简单问题（如“你好”）也要走全套 HyDE+ 重排，浪费钱且慢；有些问题需要查内部库，有些需要查外部网。
*   **方案**：在 `Pipeline` 最前端加一个 **LLM Router**。
    *   **分类**：闲聊 / 知识库检索 / 联网搜索 / 代码执行。
    *   **决策**：是否需要检索？（如果 LLM 自信能回答，直接跳过检索）。
    *   **选择**：用哪种重写策略？（简单问题用 Standard，复杂推理用 HyDE）。
*   **实现难度**：⭐⭐ (利用 LangGraph 的状态机特性极易实现)。
*   **收益**：降低延迟 50%，节省 Token 成本 30%。

#### 2. 查询分解 (Query Decomposition / Multi-Step Retrieval)
*   **痛点**：用户问“A 产品和 B 产品的价格差是多少？”单路检索很难同时找到 A 和 B 的准确价格并计算。
*   **方案**：让 LLM 将复杂问题拆解为多个子问题：
    1.  “A 产品的价格是多少？” $\rightarrow$ 检索
    2.  “B 产品的价格是多少？” $\rightarrow$ 检索
    3.  计算差值 $\rightarrow$ 生成答案。
*   **实现难度**：⭐⭐⭐ (需要维护子任务状态)。
*   **收益**：大幅提升 **多跳推理 (Multi-hop)** 问题的准确率。

---

### 🥈 第二阶段：数据飞轮与评估 (Evaluation & Feedback)
**目标**：从“凭感觉调优”转变为“数据驱动调优”。

#### 3. 自动化评估体系 (RAG Evaluation Framework)
*   **痛点**：每次改代码（如换 HyDE 提示词），不知道效果是变好了还是坏了，只能靠人工看几条。
*   **方案**：引入 **Ragas** 或 **TruLens**。
    *   **指标**：Context Precision (上下文精度), Faithfulness (忠实度), Answer Relevance (答案相关性)。
    *   **流程**：每次提交代码前，自动跑一遍测试集，生成雷达图对比。
*   **实现难度**：⭐⭐ (主要是集成现有库)。
*   **收益**：建立信心，敢于重构，量化每一次优化的收益。

#### 4. 用户反馈闭环 (Human-in-the-Loop)
*   **痛点**：系统答错了，用户点了“踩”，但系统不知道错在哪，下次还错。
*   **方案**：
    *   收集用户的点赞/点踩 + 修改后的答案。
    *   **Bad Case 分析**：自动聚类点踩的问题，发现是检索没召回？还是重排排错了？还是 LLM 生成幻觉？
    *   **微调/增强**：将 Bad Case 加入测试集，或者将优质问答对加入向量库 (Few-Shot)。
*   **实现难度**：⭐⭐⭐ (需要前端配合 + 后端存储分析)。
*   **收益**：系统越用越聪明，形成护城河。

---

### 🥉 第三阶段：架构深水区 (Advanced Architecture)
**目标**：解决极端复杂场景。

#### 5. GraphRAG (知识图谱增强)
*   **痛点**：跨文档的全局性问题（如“总结所有文档中关于‘人工智能’的演进历程”），向量检索只能找到片段，无法构建全局视图。
*   **方案**：
    *   离线提取实体关系构建图谱。
    *   检索时结合图谱邻居节点和社区摘要。
*   **实现难度**：⭐⭐⭐⭐⭐ (需要引入图数据库 Neo4j/NetworkX，离线建图成本高)。
*   **收益**：解决 **全局理解** 和 **复杂关联** 问题的终极方案。

#### 6. 长上下文优化 (Long-Context Optimization)
*   **痛点**：随着文档增多，Context 超长，LLM 注意力分散（Lost in the Middle）。
*   **方案**：
    *   **Context Compression**：检索后，不只截断，而是用 LLM 压缩每个 Chunk，只保留与问题相关的句子。
    *   **LLM 原生长窗口**：如果预算允许，直接切换到支持 128k+ 的模型，减少分块依赖。
*   **实现难度**：⭐⭐。
*   **收益**：提升长文档问答的精准度。

---

### 🚀 我的建议：下一步做什么？

考虑到你现在的架构已经很强，我建议按以下顺序推进：

1.  **立刻做：自动化评估 (Evaluation)**
    *   没有评估，优化就是盲人摸象。先集成 **Ragas**，把你之前的 `recall_test.py` 升级为包含 **Faithfulness** 和 **Answer Correctness** 的综合评估报告。
    *   *理由*：这是后续所有优化的基石。

2.  **接着做：智能路由 (Router)**
    *   利用 LangGraph 实现一个简单的路由节点：判断是否需要检索，以及选择 Standard 还是 HyDE 策略。
    *   *理由*：立竿见影地提升速度和降低成本，让系统显得更“智能”。

3.  **最后做：查询分解 (Decomposition)**
    *   针对你那套测试集中的 "Hard (多跳推理)" 类别进行专项优化。
    *   *理由*：这是攻克最难问题的关键。

**你想先开启哪一个方向？**
*   A. **构建自动化评估看板** (集成 Ragas，量化当前 HyDE/Sentence-Child 的真实收益)
*   B. **实现智能路由 Agent** (让系统自己决定什么时候用 HyDE，什么时候直接回答)
*   C. **挑战 GraphRAG** (引入知识图谱，彻底解决多跳推理)

请告诉我你的选择，我们马上开始！


# 📘 第六部分：RAG 自动化评估体系构建 (RAG Evaluation Framework)
## —— 从“凭感觉调优”到“数据驱动决策”

> **核心目标**：建立一套可量化、可自动化、可视化的评估体系，精准定位 RAG 系统的瓶颈（是检索不准？还是生成幻觉？），并量化每一次代码变更（如引入 HyDE、Sentence-Child）的真实收益。
> **核心理念**：**No Evaluation, No Optimization**.

---

## 📅 课程大纲

### 第 1 讲：评估维度与指标体系设计
**目标**：理解 RAG 评估的“不可能三角”，建立分层评估模型。
*   **1.1 传统 NLP 指标的局限性**：为什么 BLEU/ROUGE 不适用于 RAG？
*   **1.2 RAG 评估的黄金三角**：
    *   **检索层 (Retrieval)**：Context Precision (上下文精度), Context Recall (上下文召回率), MRR。
    *   **生成层 (Generation)**：Faithfulness (忠实度/防幻觉), Answer Relevance (答案相关性)。
    *   **端到端 (End-to-End)**：Answer Correctness (答案正确性 - 需 Ground Truth)。
*   **1.3 无参考评估 (Reference-Free)**：在没有标准答案的情况下，如何评估 Faithfulness 和 Relevance？
*   **1.4 实战演练**：绘制你当前系统的评估指标雷达图草案。

### 第 2 讲：构建黄金测试集 (Golden Dataset)
**目标**：数据是评估的燃料，学习如何构建高质量的评测数据集。
*   **2.1 数据来源**：从历史日志、人工构造、LLM 合成 (Synthetic Data) 获取测试对。
*   **2.2 数据格式规范**：定义 `Query`, `Ground_Truth_Answer`, `Ground_Truth_Context` 的标准 JSON 结构。
*   **2.3 使用 LLM 自动生成测试集**：
    *   利用 `LangChain` + `LLM` 从现有文档自动提取 `(Question, Answer, Context)` 三元组。
    *   数据清洗与去重策略。
*   **2.4 实战演练**：编写脚本，从你的“奇葩星球”文档中自动生成 50 条高质量测试题。

### 第 3 讲：集成 Ragas 框架进行自动化评估
**目标**：掌握业界最流行的开源评估库 `Ragas`，实现代码级集成。
*   **3.1 Ragas 核心原理**：基于 LLM 的“裁判”机制 (LLM-as-a-Judge)。
*   **3.2 环境搭建与配置**：安装 Ragas，配置评估用 LLM (建议使用比主模型更强的模型作为裁判)。
*   **3.3 代码集成**：
    *   将你的 `pipeline_instance` 封装为 Ragas 兼容的接口。
    *   编写 `evaluate.py` 脚本，一键运行全套指标。
*   **3.4 解读评估报告**：如何看懂 Ragas 输出的 DataFrame 和可视化图表？
*   **3.5 实战演练**：运行第一次全量评估，输出基准报告 (Baseline Report)。

### 第 4 讲：深度诊断与瓶颈分析 (Debugging RAG)
**目标**：不仅知道“好不好”，还要知道“哪里不好”。
*   **4.1 检索 vs 生成 归因分析**：
    *   高 Recall + 低 Faithfulness = 生成模型问题 (幻觉)。
    *   低 Recall + 高 Faithfulness = 检索系统问题 (没找到资料)。
    *   低 Recall + 低 Faithfulness = 双重崩溃。
*   **4.2 坏案例 (Bad Case) 挖掘**：
    *   自动筛选得分最低的 Top 10 案例。
    *   可视化分析：展示 Query、检索到的 Chunk、生成的答案、扣分原因。
*   **4.3 针对性优化策略**：
    *   如果是 Context Precision 低 $\rightarrow$ 调整分块策略 (Sentence-Child) 或 重排模型。
    *   如果是 Faithfulness 低 $\rightarrow$ 调整生成 Prompt 或 开启 Self-RAG。
*   **4.4 实战演练**：分析上一轮的 Bad Cases，提出具体的优化假设。

### 第 5 讲：A/B 测试与持续集成 (CI/CD for RAG)
**目标**：将评估融入开发流程，防止“优化变退化”。
*   **5.1 A/B 测试设计**：
    *   对比实验：`Recursive` vs `Sentence-Child`，`Standard` vs `HyDE`。
    *   统计显著性检验：提升是真实的还是随机波动？
*   **5.2 构建回归测试流水线**：
    *   在 Git CI (GitHub Actions/GitLab CI) 中集成评估脚本。
    *   设置阈值门禁 (Quality Gate)：如果 Faithfulness 下降超过 5%，禁止合并代码。
*   **5.3 长期监控看板**：
    *   使用 Streamlit 或 Grafana 搭建实时评估看板。
    *   跟踪线上用户反馈与离线评估分数的相关性。
*   **5.4 实战演练**：配置一个 GitHub Action，每次 Push 自动跑测试集并评论分数变化。

---

## 🎯 课程产出物
完成本部分学习后，你将拥有：
1.  **一套黄金测试集** (`golden_dataset.json`)：覆盖不同难度和场景的标准化测试题。
2.  **一个自动化评估脚本** (`src/eval/evaluate.py`)：一键运行，输出详细报表。
3.  **一份基线诊断报告**：明确指出当前系统（HyDE + Sentence-Child）的强项和弱点。
4.  **一个 CI 门禁配置**：确保未来代码质量不回退。

---

## ❓ 确认环节
请确认以上大纲是否符合你的预期？
*   如果有想**增加**的内容（比如想重点看 Streamlit 看板搭建），请告诉我。
*   如果有想**删减**的内容（比如暂时不想做 CI/CD，只想本地跑），也请告诉我。
*   如果**确认无误**，回复“**确认开始**”，我们将立即进入 **第 1 讲：评估维度与指标体系设计**！


| 架构模式 | 特点 | 是否符合你的图？ |
| :--- | :--- | :--- |
| ReAct | 思考-行动-观察循环，动态调用工具。 | ❌ 不完全符合。你的图是预先定义好策略分支，由路由器一次性选定，而不是在循环中动态决定下一步动作。 |
| Plan-and-Solve | 先制定完整计划，再一步步执行。 | ❌ 你的图没有显式的“制定计划”步骤，而是直接“分类路由”。 |
| Sequential (顺序) | 固定流程 A->B->C。 | ❌ 你的图有分支决策 (`selected_strategy`)，不是固定线性流。 |
| Parallel (并行) | 同时执行多个任务然后汇总。 | ❌ 虽然策略层画在一起，但逻辑上是互斥选择（选其中一个策略），而非同时执行所有策略。 |
| Router (路由) | 根据意图分类，分发到不同子流程。 | ✅ 完全符合。核心就是 `IntentRouter`。 |
| Network (网络/自主协作) | 多个 Agent 自由对话、互相指派任务。 | ❌ 你的图是中心化的星型拓扑，不是去中心化的网状协作。 |
| Evaluator-Optimizer | 生成结果 -> 评估 -> 修正循环。 | ❌ 图中虽有 `compose_final_answer`，但没有显式的“评估-重试”闭环反馈。 |
| Human-in-the-loop | 关键步骤需人工确认。 | ❌ 图中未体现人工介入节点。 |

