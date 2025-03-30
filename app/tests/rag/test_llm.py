from unittest.mock import patch

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.core import settings
from app.core.log_adapter import logger
from app.rag.llm.completions import LLM, LLMParams, ModelAPIType, RAGLLMPrompt
from app.rag.llm.history import MemoMateMemory


def test_llm_chat() -> None:
    prompt = RAGLLMPrompt(context="", question="你好")
    params = LLMParams(
        api_type=ModelAPIType.OPENAI,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        streaming=False,
        stream_usage=False,
    )

    llm = LLM().generate(prompt, params)
    response: BaseMessage = llm.invoke(
        prompt.model_dump(exclude={"prompt"}),
        config=RunnableConfig(callbacks=[]),
    )
    print(f"content: {response.content}")
    print(f"usage_metadata: {getattr(response, 'usage_metadata', None)}")


"""
使用LangChain和OpenAI API协议进行对话的测试代码
"""


def test_simple_chat_completion() -> None:
    """
    简单的对话完成测试，使用LangChain的ChatOpenAI
    """
    # 创建一个ChatOpenAI实例
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # 可以替换为你想要的模型
        temperature=0,
    )

    # 发送消息
    messages = [HumanMessage(content="你好，请介绍一下自己")]
    response = llm.invoke(messages)

    logger.info("对话响应", response=response.content)

    # 断言响应不为空
    assert response.content
    assert isinstance(response, AIMessage)


def test_chat_with_template() -> None:
    """
    使用模板进行结构化对话测试
    """
    # 创建提示模板
    prompt = ChatPromptTemplate.from_template("""
    你是一个专业的助手，请根据以下信息回答问题：

    上下文: {context}
    问题: {question}

    请只回答与上下文相关的内容，如果不知道答案，请说"我不知道"。
    """)

    # 创建LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 创建链
    chain = prompt | llm | StrOutputParser()

    # 执行链
    response = chain.invoke(
        {"context": "张三是一名工程师，他擅长Python编程和机器学习。", "question": "张三是做什么的？"}
    )

    logger.info("链式响应", response=response)

    # 断言响应包含关键信息
    assert "工程师" in response


def test_using_app_llm_class() -> None:
    """
    使用应用中封装的LLM类进行测试
    """
    # 创建提示
    prompt = RAGLLMPrompt(
        context="人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。", question="什么是人工智能？"
    )

    # 创建参数
    params = LLMParams(
        api_type=ModelAPIType.OPENAI,
        model_name="gpt-3.5-turbo",
        streaming=False,
        stream_usage=False,
    )

    # 创建LLM并生成响应
    llm_chain = LLM().generate(prompt, params)
    response = llm_chain.invoke(
        prompt.model_dump(exclude={"prompt"}),
    )

    logger.info("应用LLM响应", content=response.content)

    # 断言响应不为空
    assert response.content
    assert "人工智能" in response.content


def test_streaming_chat() -> None:
    """
    测试流式输出对话
    """

    # 定义回调函数来收集流式输出
    class CollectStreamingOutput:
        def __init__(self):
            self.chunks = []

        def on_llm_new_token(self, token: str, **kwargs):
            self.chunks.append(token)
            logger.info("流式输出片段", token=token)

    # 创建回调处理器
    collect_handler = CollectStreamingOutput()

    # 创建流式LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        streaming=True,
        callbacks=[collect_handler],
    )

    # 发送消息
    messages = [
        HumanMessage(content="用一句话描述今天的天气"),
        AIMessage(content="今天天气晴朗，气温适宜，适合户外活动。"),
        HumanMessage(content="用一句话描述今天的天气"),
    ]
    response = llm.invoke(messages)

    logger.info("流式输出完整响应", response=response.content)

    # 断言流式输出被收集
    assert collect_handler.chunks
    assert response.content


def test_memory_chat() -> None:
    """
    测试记忆功能
    """
    # 创建提示
    prompt_text = """
问题: {question}
上下文: {context}"""

    # 创建系统消息
    system_message = """回答以下问题，基于提供的上下文信息。如果无法从上下文中找到答案，请从历史对话中找到答案，如果历史对话中也没有答案，请说"我不知道"。"""

    prompt_template = MemoMateMemory.create_prompt_template(prompt_text, system_message)
    # 创建参数
    params = LLMParams(
        api_type=ModelAPIType.OPENAI,
        model_name=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
        streaming=False,
        stream_usage=False,
    )

    def _get_session_history(session_id: str) -> BaseChatMessageHistory:
        logger.info("获取会话历史", session_id=session_id)
        if session_id not in MemoMateMemory.store:
            MemoMateMemory.store[session_id] = ChatMessageHistory(
                messages=[
                    HumanMessage(content="人工智能包括什么？"),
                    AIMessage(content="人工智能包括机器学习、自然语言处理、计算机视觉等。"),
                ]
            )
        return MemoMateMemory.store[session_id]

    # mock MemoMateMemory.get_session_history 替换为 _get_session_history
    with patch("app.rag.llm.history.MemoMateMemory.get_session_history", side_effect=_get_session_history):
        prompt = RAGLLMPrompt(
            prompt=prompt_template,
            context="人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
            question="什么是人工智能？",
        )

        # 创建LLM并生成响应
        llm_chain = LLM().generate(prompt, params)

        memory_chain = MemoMateMemory.gen_memory_chain(llm_chain)
        response = memory_chain.invoke(
            {"context": prompt.context, "question": prompt.question},
            config=RunnableConfig(session_id="test_session"),
        )

        logger.info("应用LLM响应", content=response)
