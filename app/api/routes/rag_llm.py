import uuid
from typing import Literal

from fastapi import APIRouter
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.core import logger, settings
from app.core.db import engine
from app.crud.history_message import add_history_message
from app.enums import HistoryMessageType
from app.rag.llm.completions import LLM, LLMParams, ModelAPIType, RAGLLMPrompt
from app.rag.llm.document_chain import RAGDocumentChain
from app.rag.llm.history import MemoMateMemory
from app.rag.retriever.base_retriever import BaseRetriever
from app.rag.retriever.history_aware_retriever import HistoryAwareRetriever
from app.schemas.llm import ChatResponse, RAGChatRequest

router = APIRouter()


class RAGWebSocketChatMessage(BaseModel):
    """RAG WebSocket聊天消息"""

    type: Literal["question", "stop", "ping"] = Field(default="question", description="消息类型: question, stop, ping")
    content: str | None = Field(default=None, description="消息内容")
    use_history: bool = Field(default=True, description="是否使用历史记录")
    retrieve_top_k: int = Field(default=5, description="检索文档数量")


# 创建检索器实例
retriever = BaseRetriever()
# 创建检索器和文档链实例
history_retriever = HistoryAwareRetriever(base_retriever=retriever)
document_chain = RAGDocumentChain()


@router.post("/chat", response_model=ChatResponse, description="RAG对话接口")
async def rag_chat(params: RAGChatRequest):
    """RAG对话接口"""
    session_id = params.session_id or str(uuid.uuid4())
    if params.session_id and params.history:
        MemoMateMemory.merge_history(session_id, params.history)

    # 获取上下文
    context = retriever.get_context(params.message, top_k=params.retrieve_top_k)

    # 创建提示
    prompt = RAGLLMPrompt(
        context=context,
        question=params.message,
    )

    # 创建参数
    llm_params = LLMParams(
        api_type=ModelAPIType.OPENAI,
        api_key=settings.OPENAI_API_KEY or None,
        model_name=settings.CHAT_MODEL,
        streaming=False,
        stream_usage=False,
    )

    # 创建LLM并生成响应
    llm_chain = LLM().generate(prompt, llm_params)

    if params.use_history:
        # 使用对话历史
        memory_chain = MemoMateMemory.gen_memory_chain(llm_chain)
        response = memory_chain.invoke(
            {"context": context, "question": params.message},
            config=RunnableConfig(callbacks=[], session_id=session_id),
        )
    else:
        response = llm_chain.invoke(prompt.model_dump(), RunnableConfig(callbacks=[]))

    # 保存用户问题和AI回答到数据库
    with Session(engine) as db_session:
        add_history_message(
            session=db_session, message=params.message, message_type=HistoryMessageType.HUMAN, session_id=session_id
        )

        add_history_message(
            session=db_session, message=response.content, message_type=HistoryMessageType.AI, session_id=session_id
        )
    new_history = list(params.history)
    new_history.append((params.message, response.content))
    logger.info("RAG响应", content=response.content, context_length=len(context), session_id=session_id)

    return ChatResponse(message=response.content, history=new_history, session_id=session_id)


@router.post("/chat/history_aware", response_model=ChatResponse, description="基于历史感知的RAG对话接口")
async def history_aware_rag_chat(params: RAGChatRequest):
    """使用历史感知检索技术的RAG对话接口

    此接口融合了对话历史上下文进行检索，能更准确理解用户意图：
    - 自动解析对话中的代词引用（如"它"、"这个"）
    - 考虑前几轮对话提供的背景信息
    - 保持连贯的多轮对话体验

    参数:
        - message: 当前用户问题
        - history: 历史对话记录，格式为[(用户问题1, AI回答1), ...]
        - retrieve_top_k: 检索的文档数量
        - use_history: 是否利用历史记录
        - session_id: 会话ID，用于维持对话连续性

    返回:
        包含AI回答和更新后历史记录的响应"""

    session_id = params.session_id or str(uuid.uuid4())

    # 如果有会话ID和历史记录，确保数据库与客户端历史同步
    if params.session_id and params.history:
        MemoMateMemory.merge_history(session_id, params.history)

    # 选择限制历史长度
    chat_history = params.history[-10:] if len(params.history) > 10 else params.history

    # 使用历史感知检索获取文档
    documents = history_retriever.retrieve_with_history(
        query=params.message, chat_history=chat_history, session_id=session_id
    )

    # 使用文档链生成回答
    response = document_chain.run(question=params.message, documents=documents, chat_history=chat_history)

    # 保存用户问题和AI回答到数据库
    with Session(engine) as db_session:
        # 保存用户问题
        add_history_message(
            session=db_session, message=params.message, message_type=HistoryMessageType.HUMAN, session_id=session_id
        )

        # 保存AI回答
        add_history_message(
            session=db_session, message=response, message_type=HistoryMessageType.AI, session_id=session_id
        )

    logger.info("历史感知检索技术的RAG响应", session_id=session_id, document_count=len(documents))

    # 更新历史
    new_history = list(params.history)
    new_history.append((params.message, response))
    if len(new_history) > 20:  # 限制返回历史长度
        new_history = new_history[-20:]

    return ChatResponse(message=response, history=new_history, session_id=session_id)
