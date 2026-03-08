# src/rag/rewriter.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.core.config import settings
from src.core.prompt_registry import PROMPT_KEYS, core_prompt_registry
from src.utils.xml_parser import remove_think_and_n
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ==========================================
# 1. 策略配置中心 (Data-Driven)
# ==========================================
STRATEGY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "name": "StandardRewrite",
        "temperature": settings.llm.rewrite_temperature,
        "prompt_key": PROMPT_KEYS.RAG_REWRITER_STANDARD_SYSTEM,
    },
    "hyde": {
        "name": "HyDE (Hypothetical Document)",
        "temperature": settings.llm.hyde_temperature,
        "prompt_key": PROMPT_KEYS.RAG_REWRITER_HYDE_SYSTEM,
    },
    # 未来扩展示例：
    # "multi_query": { ... },
    # "step_back": { ... },
}

# ==========================================
# 2. 统一执行器 (Single Executor)
# ==========================================
class QueryRewriter:
    """
    统一查询重写器
    通过 strategy_name 动态加载配置，无重复代码
    """
    def __init__(self, strategy_name: str = "standard"):
        self.strategy_name = strategy_name.lower()
        
        # 获取配置，降级处理
        config = STRATEGY_CONFIGS.get(self.strategy_name, STRATEGY_CONFIGS["standard"])
        
        if self.strategy_name not in STRATEGY_CONFIGS:
            logger.warning(f"⚠️ 未知策略 '{self.strategy_name}'，降级使用 standard")
            self.strategy_name = "standard"
            config = STRATEGY_CONFIGS["standard"]

        self.name = config["name"]
        self.temperature = config["temperature"]
        
        # 初始化 LLM (每个策略独立实例以隔离温度配置)
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            temperature=self.temperature
        )
        
        # 构建 Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", core_prompt_registry.get(config["prompt_key"])),
            ("human", core_prompt_registry.get(PROMPT_KEYS.RAG_REWRITER_HUMAN_QUERY))
        ])
        
        logger.info(f"📝 初始化重写器：{self.name} (Temp={self.temperature})")

    def rewrite(self, original_query: str) -> str:
        """执行重写 """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"original_query": original_query})
            result = remove_think_and_n(response.content.strip())
            
            logger.debug(f"✨ [{self.name}] '{original_query}' -> '{result[:50]}...'")
            return result
        except Exception as e:
            logger.error(f"❌ [{self.name}] 失败，降级使用原查询：{e}")
            return original_query

# ==========================================
# 3. 工厂函数 (Factory Function)
# ==========================================
def get_rewriter(strategy_name: str = None) -> QueryRewriter:
    """工厂函数：根据配置返回重写器实例"""
    if not strategy_name:
        strategy_name = getattr(settings.search, 'rewritten_strategy', 'standard')
    return QueryRewriter(strategy_name=strategy_name)
