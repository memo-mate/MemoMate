from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from rich import inspect

from app.configs import settings


def test_reasoning():
    history = []
    # 使用 OpenAI 的 API 进行推理
    llm = ChatOpenAI(
        model="Qwen/QwQ-32B",
        api_key=settings.openai_api_key,
        base_url="https://api.siliconflow.cn/v1",
        temperature=0,
        max_tokens=3200,
        timeout=None,
        max_retries=3,
    )
    history = [
        HumanMessage(content="你好"),
    ]
    response = llm.invoke(history)
    if isinstance(response, AIMessage):
        usage = response.usage_metadata
        input_tokens = usage.get("input_tokens", "unknown") if usage else "unknown"
        output_tokens = usage.get("output_tokens", "unknown") if usage else "unknown"
        print(f"流量计费 [Qwen/QwQ-32B]: [输入 token: {input_tokens}][输出 token: {output_tokens}]")

    # inspect(response)

    history += [
        HumanMessage(content="你好"),
        response,
        HumanMessage(content="请问今天是几月几号？"),
    ]
    response = llm.invoke(history)
    if isinstance(response, AIMessage):
        usage = response.usage_metadata
        input_tokens = usage.get("input_tokens", "unknown") if usage else "unknown"
        output_tokens = usage.get("output_tokens", "unknown") if usage else "unknown"
        print(f"流量计费 [Qwen/QwQ-32B]: [输入 token: {input_tokens}][输出 token: {output_tokens}]")

    inspect(response)
    print(response.content)
