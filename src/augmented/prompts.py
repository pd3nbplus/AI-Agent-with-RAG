# augmented/prompts.py
# Prompt 注册中心：集中管理不同出题风格模板，按 profile 选择。

STANDARD_PROMPT = """
你是一个专业的 RAG 评估数据集构建专家。请阅读以下【文档片段】，并生成 {num_questions} 个高质量评估测试题。
【要求】：
1. 题目需要覆盖 easy / medium / hard。
2. `ground_truth_contexts` 必须是字符串数组，数组元素直接摘录原文，不允许杜撰。
3. 每个问题必须可独立回答。
4. 严格输出 JSON 数组，字段固定为：
   category, difficulty, question, ground_truth_contexts, ground_truth, source_document, metadata
5. `category` 可用值示例：policy/product/process/general。
6. `metadata` 至少包含 generated_by 和 generation_date（YYYY-MM-DD）。
7. 不要输出 markdown，不要输出解释，只输出 JSON。
【文档片段】：
{context_chunk}
"""

ADVERSARIAL_PROMPT = """
你是一个 RAG 鲁棒性评估专家。请基于以下文档片段，生成 {num_questions} 个对抗性问题。
【目标】：
1. 问题必须“看似相关”，但文档中完全未提及答案。
2. 标准答案固定为：根据提供的上下文无法回答。
3. `ground_truth_contexts` 必须为 []（空数组），表示上下文中无可支持证据。
4. `difficulty` 固定为 hard。
5. 严格输出 JSON 数组，字段固定为：
   category, difficulty, question, ground_truth_contexts, ground_truth, source_document, metadata
6. 不要输出 markdown，不要输出解释，只输出 JSON。
【文档片段】：
{context_chunk}
"""

MIXED_PAIR_PROMPT = """
你是一个多跳检索评估专家。下面有两个文档片段（A 和 B），请基于它们生成 {num_questions} 个问题。
【强约束】：
1. 必须且仅生成 3 个问题：1 个 Easy、1 个 Medium、1 个 Hard。
2. Hard 必须结合片段 A 和片段 B 的信息才能回答，仅看其中一个片段无法完整回答。
3. 每个问题都要提供 `ground_truth_contexts`（字符串数组，摘录原文）与 `ground_truth`。
4. 严格输出 JSON 数组，字段固定为：
   category, difficulty, question, ground_truth_contexts, ground_truth, source_document, metadata
5. 不要输出 markdown，不要输出解释，只输出 JSON。
【片段 A】：
{chunk_a}

【片段 B】：
{chunk_b}
"""


class PromptRegistry:
    def __init__(self) -> None:
        # 可按需扩展更多 profile，调用侧只传 profile 名称。
        self._prompts = {
            "default": STANDARD_PROMPT,
            "standard": STANDARD_PROMPT,
            "adversarial": ADVERSARIAL_PROMPT,
            "mixed_pair": MIXED_PAIR_PROMPT,
            "strict_negative_first": """
你是 RAG 评估数据集专家。请先生成一个偏难问题，再补齐其余问题。
输出必须是 JSON 数组，字段固定为：
category, difficulty, question, ground_truth_contexts, ground_truth, source_document, metadata。
其中 `ground_truth_contexts` 必须是字符串数组，`metadata` 必须包含 generated_by 和 generation_date。
文档片段：{context_chunk}
需要生成数量：{num_questions}
""",
        }

    def get(self, profile: str) -> str:
        # 未命中时回退到 default，保证流程可运行。
        return self._prompts.get(profile, self._prompts["default"])

