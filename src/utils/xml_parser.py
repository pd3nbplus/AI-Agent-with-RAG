import re,json
def extract_tool_calls_from_content(content: str):
    """从 content 中提取所有被<tool_call>包围的工具调用 JSON"""
    tool_calls = []
    matches = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", content, re.DOTALL)
    for i, match in enumerate(matches):
        try:
            data = json.loads(match.strip())
            name = data.get("name")
            arguments = data.get("arguments", {})
            if name:
                # 注意：LangChain 内部使用 'args' 而不是 'arguments'
                tool_calls.append({
                    "name": name,
                    "args": arguments,          # ⚠️ 关键：用 'args'
                    "id": f"call_{i}_{name}",   # 必须有唯一 id
                    "type": "tool_call"
                })
        except Exception as e:
            print(f"❌ 解析工具调用失败: {e}")
    return tool_calls

def remove_think_and_n(content: str) -> str:
    """移除 <think>...</think>，保留正文换行结构。"""
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()
