# augmented/router.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from langchain_openai import ChatOpenAI
from src.core.config import settings
from src.utils.xml_parser import remove_think_and_n
# from .llm_router import get_llm_client # 假设你有一个获取 LLM 的工厂函数

# 1. 定义输出结构 (Schema)
class RouteDecision(BaseModel):
    """路由决策对象：给出意图、策略、置信度以及可选澄清问题。"""

    intent: Literal["CHIT_CHAT", "FACT_LOOKUP", "HOW_TO", "COMPARISON", "CODE_SEARCH", "UNKNOWN"] = Field(
        description="用户问题的意图分类"
    )
    confidence: float = Field(
        description="分类的置信度 (0.0 - 1.0)"
    )
    reasoning: str = Field(
        description="简短的分类理由，用于调试"
    )
    strategy: Literal["direct_reply", "fast_retrieval", "standard_retrieval", "deep_search", "code_search", "fallback", "clarify_needed"] = Field(
        description="建议的后续处理策略名称"
    )
    clarification_questions: Optional[List[str]] = Field(
        default=None,
        description="低置信度时用于澄清需求的问题列表；高置信度时为 null"
    )

# 2. 定义 Prompt
ROUTER_PROMPT_TEMPLATE = """
你是一个智能 RAG 系统的路由指挥官。你的任务是分析用户问题，输出严格的 JSON 对象以决定处理路径。

# 核心任务
. 识别用户意图 (intent)。
. 评估置信度 (confidence)。
. **关键逻辑**：
   - 如果问题清晰且置信度 >= 0.6：直接推荐执行策略 (如 fast_retrieval)。
   - 如果问题模糊、歧义或缺少关键上下文导致置信度 < 0.6：
     - 将 `strategy` 设为 `clarify_needed`。
     - **必须**在 `clarification_questions` 字段中生成 2-3 个具体的引导性问题或场景选项，帮助用户澄清需求。
     - 引导问题应覆盖最可能的几种情况，语气要友好且具指导性。

# 约束条件 (严格遵守)
. **输出格式**：必须且仅输出一个合法的 JSON 对象。
. **禁止项**：不要输出 Markdown 代码块标记 (如 ```json)，不要输出任何解释性文字。
. **字段限制**：
   - `intent`: ["CHIT_CHAT", "FACT_LOOKUP", "HOW_TO", "COMPARISON", "CODE_SEARCH", "UNKNOWN"]
   - `strategy`: ["direct_reply", "fast_retrieval", "standard_retrieval", "deep_search", "code_search", "fallback", "clarify_needed"]
   - `confidence`: 0.0 - 1.0 浮点数。
   - `clarification_questions`: 仅在低置信度时为非空列表，否则为 null。

# 意图与策略映射规则 (高置信度时)
- CHIT_CHAT -> direct_reply
- FACT_LOOKUP -> fast_retrieval
- HOW_TO -> standard_retrieval
- COMPARISON -> deep_search
- CODE_SEARCH -> code_search
- UNKNOWN -> fallback

# 低置信度处理示例 (Few-Shot)
用户输入: "那个怎么弄？"
JSON 输出:
{{
    "intent": "UNKNOWN",
    "confidence": 0.3,
    "reasoning": "代词'那个'指代不明，缺乏具体操作对象或上下文。",
    "strategy": "clarify_needed",
    "clarification_questions": [
        "您是指如何重置密码，还是如何申请报销？",
        "您是在操作手机 App 还是网页版时遇到的问题？",
        "能否提供具体的错误提示或您想实现的功能名称？"
    ]
}}

用户输入: "对比一下版本。"
JSON 输出:
{{
    "intent": "COMPARISON",
    "confidence": 0.4,
    "reasoning": "用户想要对比，但未指定对比的对象（如产品版本、方案A/B等）。",
    "strategy": "clarify_needed",
    "clarification_questions": [
        "您是想对比 v1.0 和 v2.0 版本的 API 差异吗？",
        "您是在对比我们的标准版和企业版服务方案吗？",
        "或者是想对比 Python 和 Java 在某个特定场景下的性能？"
    ]
}}

# 高置信度示例
用户输入: "Python 里怎么用 pandas 读取 csv？"
JSON 输出:
{{
    "intent": "CODE_SEARCH",
    "confidence": 0.98,
    "reasoning": "明确的编程库使用问题，意图清晰。",
    "strategy": "code_search",
    "clarification_questions": null
}}

# 用户问题
{question}

# 你的回答 (仅 JSON)
"""

class IntentRouter:
    """意图路由器：使用 LLM 将用户问题映射到可执行策略。"""

    def __init__(self, llm_model: str = "gpt-3.5-turbo"): 
        # 注意：路由任务很简单，可以用小模型 (gpt-3.5/haiku) 以节省成本和延迟
        self.llm = ChatOpenAI(
            base_url=settings.llm.base_url,
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            temperature=float(settings.llm.temperature),
        )

        def extract_and_clean(message):
            # message 可能是 AIMessage 对象，也可能是字符串 (取决于版本和配置)
            content = message.content if hasattr(message, 'content') else str(message)
            # 移除 think 和 n 标记
            return remove_think_and_n(content)

        clean_chain = RunnableLambda(extract_and_clean)
        self.parser = PydanticOutputParser(pydantic_object=RouteDecision)
        
        prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
        
        # 构建链：Prompt -> LLM -> clean_chain -> Parser
        self.chain = prompt | self.llm | clean_chain | self.parser

    async def route(self, question: str) -> RouteDecision:
        """
        对单个问题进行路由决策
        """
        try:
            decision = await self.chain.ainvoke({"question": question})
            return decision
        except Exception as e:
            # 解析失败时的降级策略：再次确认
            print(f"⚠️ 路由解析失败：{e}，默认降级为确认问题")
            # 降级时也可以返回一个特殊的 clarify_needed，或者直接走标准检索
            return RouteDecision(
                intent="UNKNOWN",
                confidence=0.0,
                reasoning="Parsing failed or model error",
                strategy="clarify_needed",
                clarification_questions=["您的问题似乎有些复杂，能再详细描述一下具体场景吗？", "您是想查询政策、操作流程还是技术文档？"]
            )
# 测试脚本
if __name__ == "__main__":
    # python -m src.agent.router
    import asyncio
    
    async def test_router():
        router = IntentRouter()
        queries = [
            "你好，在吗？",
            "公司的年假政策是怎样的？",
            "如何重置我的登录密码？",
            "对比一下 v1.0 和 v2.0 版本的 API 响应速度。",
            "Python 里怎么用 pandas 读取 csv？",
            "我有如下需求："
        ]
        
        for q in queries:
            print(f"\n❓ Q: {q}")
            res = await router.route(q)
            print(f"🎯 Intent: {res.intent}")
            print(f"🙋 Confidence: {res.confidence}")
            print(f"💡 Strategy: {res.strategy}")
            print(f"🧠 Reasoning: {res.reasoning}")

            if res.clarification_questions:
                print("❓ 需要澄清，建议询问用户:")
                for i, cq in enumerate(res.clarification_questions, 1):
                    print(f"   {i}. {cq}")
            else:
                print("✅ 意图清晰，直接执行策略。")
            print("-" * 30)
    asyncio.run(test_router())
