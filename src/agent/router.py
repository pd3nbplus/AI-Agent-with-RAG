# augmented/router.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from src.core.config import settings
from src.utils.xml_parser import remove_think_and_n
# from .llm_router import get_llm_client # 假设你有一个获取 LLM 的工厂函数

# 1. 定义输出结构 (Schema)
class RouteDecision(BaseModel):
    intent: Literal["CHIT_CHAT", "FACT_LOOKUP", "HOW_TO", "COMPARISON", "CODE_SEARCH", "UNKNOWN"] = Field(
        description="用户问题的意图分类"
    )
    confidence: float = Field(
        description="分类的置信度 (0.0 - 1.0)"
    )
    reasoning: str = Field(
        description="简短的分类理由，用于调试"
    )
    strategy: str = Field(
        description="建议的后续处理策略名称，如 'direct_reply', 'fast_retrieval', 'deep_search'"
    )

# 2. 定义 Prompt
ROUTER_PROMPT_TEMPLATE = """
你是一个智能 RAG 系统的路由指挥官。你的任务是分析用户问题，输出严格的 JSON 对象以决定处理路径。

# 任务目标
分析用户输入，确定其意图 (intent) 和对应的处理策略 (strategy)，并给出置信度和理由。

# 约束条件 (严格遵守)
1. **输出格式**：必须且仅输出一个合法的 JSON 对象。
2. **禁止项**：不要输出 Markdown 代码块标记 (如 ```json)，不要输出任何解释性文字、前缀或后缀。
3. **字段限制**：
   - `intent` 必须是以下之一：["CHIT_CHAT", "FACT_LOOKUP", "HOW_TO", "COMPARISON", "CODE_SEARCH", "UNKNOWN"]
   - `strategy` 必须是以下之一：["direct_reply", "fast_retrieval", "standard_retrieval", "deep_search", "code_search", "fallback"]
   - `confidence` 必须是 0.0 到 1.0 之间的浮点数。
   - `reasoning` 判断依据，应简洁明了，说明分类依据。

# 意图与策略映射规则
- **CHIT_CHAT**: 闲聊、问候。 -> 策略: `direct_reply`
- **FACT_LOOKUP**: 简单事实查询 (谁/什么/哪里/时间)。 -> 策略: `fast_retrieval`
- **HOW_TO**: 询问步骤、流程、操作方法。 -> 策略: `standard_retrieval`
- **COMPARISON**: 对比两个或多个事物。 -> 策略: `deep_search`
- **CODE_SEARCH**: 编程代码、API、报错分析。 -> 策略: `code_search`
- **UNKNOWN**: 无法理解或有害请求。 -> 策略: `fallback`

# 输出 JSON 结构示例
{{
    "intent": "HOW_TO",
    "confidence": 0.95,
    "reasoning": "用户询问如何进行报销，属于操作流程类问题。",
    "strategy": "standard_retrieval"
}}
{{
    "intent": "HOW_TO",
    "confidence": 0.88,
    "reasoning": "用户询问如何使用pytorch进行训练，属于编程代码类问题。",
    "strategy": "code_search"
}}

# 用户问题
{question}

# 你的回答 (仅 JSON)
"""

class IntentRouter:
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
            # 解析失败时的降级策略：默认走标准检索
            print(f"⚠️ 路由解析失败：{e}，默认降级为标准检索")
            return RouteDecision(
                intent="UNKNOWN",
                confidence=0.0,
                reasoning="Parsing failed",
                suggested_strategy="standard_retrieval"
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

    asyncio.run(test_router())