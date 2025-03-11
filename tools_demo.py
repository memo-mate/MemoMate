from langchain.agents import Tool
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: SecretStr
    LANGCHAIN_API_KEY: str
    LANGCHAIN_TRACING_V2: bool

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


chat_model = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
    base_url="https://api.siliconflow.cn/v1",
    model="Pro/deepseek-ai/DeepSeek-R1",
)


@tool
def send_email(recipient: str, content: str) -> str:
    """发送邮件到指定收件人，需要参数：收件人邮箱和邮件内容"""
    print(f"\n📧 模拟发送邮件到 {recipient}：\n{content}\n")
    return "邮件发送成功"


@tool
def search_web(query: str) -> str:
    """执行网络搜索，需要参数：搜索关键词"""
    print(f"\n🔍 模拟搜索：{query}\n")
    return "找到3条相关结果"


memory = MemorySaver()
tools = [
    Tool(
        name="send_email",
        description="发送邮件到指定收件人，需要参数：收件人邮箱和邮件内容",
        func=send_email,
    ),
    Tool(
        name="web_search",
        description="执行网络搜索，需要参数：搜索关键词",
        func=search_web,
    ),
]
agent_executor = create_react_agent(
    model=chat_model,
    tools=tools,
)

response = agent_executor.invoke(
    {
        "messages": [HumanMessage(content="请用中文通知张三（zhangsan@example.com）明天的项目会议改到下午3点")],
    }
)
print("\n最终结果：", response)
