import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

from fastapi import APIRouter, Request
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from sqlmodel import Session
from sse_starlette import EventSourceResponse

from app.api.demo.sse import SSEMessage
from app.api.deps import CurrentUser
from app.core import logger, settings
from app.core.db import engine
from app.crud.history_message import add_history_message
from app.enums import HistoryMessageType
from app.rag.llm.completions import LLM, LLMParams, ModelAPIType, RAGLLMPrompt
from app.rag.llm.history import MemoMateMemory
from app.schemas.llm import ChatResponse, RAGChatRequest

router = APIRouter()


class RAGWebSocketChatMessage(BaseModel):
    """RAG WebSocket聊天消息"""

    type: Literal["question", "stop", "ping"] = Field(default="question", description="消息类型: question, stop, ping")
    content: str | None = Field(default=None, description="消息内容")
    use_history: bool = Field(default=True, description="是否使用历史记录")
    retrieve_top_k: int = Field(default=5, description="检索文档数量")


# # 创建检索器实例
# retriever = BaseRetriever()
# # 将检索器转换为langchain检索器
# langchain_retriever = retriever.as_langchain_retriever()


@router.post("/chat", response_model=ChatResponse, description="RAG对话接口")
async def rag_chat(current_user: CurrentUser, params: RAGChatRequest):
    """RAG对话接口"""
    try:
        session_id = params.session_id or str(uuid.uuid4())
        is_new_session = params.session_id is None

        if params.session_id and params.history:
            MemoMateMemory.merge_history(session_id, params.history)

        llm_params = LLMParams(
            api_type=ModelAPIType.OPENAI,
            api_key=settings.OPENAI_API_KEY or None,
            model_name=settings.CHAT_MODEL,
            streaming=False,
            stream_usage=False,
        )
        llm = LLM().get_llm(llm_params)

        if is_new_session:
            context = retriever.get_context(params.message, top_k=params.retrieve_top_k)

            prompt = RAGLLMPrompt(
                prompt=ChatPromptTemplate.from_template("""{input}{context}"""),
                context=context,
                input=params.message,
            )

            llm_chain = prompt.prompt | llm
            response = llm_chain.invoke(prompt.model_dump(), RunnableConfig(callbacks=[]))
            new_history = []

        else:
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """根据聊天历史和用户最新问题，理解用户意图，重新表述为明确的查询。
                            如果问题中有代词引用（如"它"、"这个"），请将其替换为实际含义。
                            保持问题简洁清晰，不要解答问题，只需要重构问题。""",
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),  # 必须使用input作为变量名
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, langchain_retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """请基于上下文信息回答用户的问题。如果无法从上下文中找到答案，请基于历史对话回答。

                上下文信息:
                {context}""",
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

            llm_chain = create_retrieval_chain(
                retriever=history_aware_retriever, combine_docs_chain=question_answer_chain, output_key="output"
            )

            conversational_rag_chain = MemoMateMemory.gen_memory_chain(llm_chain)

            new_history = list(params.history)
            chain_response = conversational_rag_chain.invoke(
                {"input": params.message},
                config=RunnableConfig(callbacks=[], session_id=session_id),
            )

            response = chain_response["answer"]

        with Session(engine) as db_session:
            add_history_message(
                session=db_session,
                message=params.message,
                message_type=HistoryMessageType.HUMAN,
                session_id=session_id,
                user_id=current_user.id,
            )

            add_history_message(
                session=db_session,
                message=response.content if hasattr(response, "content") else response,
                message_type=HistoryMessageType.AI,
                session_id=session_id,
                user_id=current_user.id,
            )

        response_content = response.content if hasattr(response, "content") else response
        new_history.append((params.message, response_content))
        if len(new_history) > 20:  # 限制返回历史长度
            new_history = new_history[-20:]

        # 记录日志
        logger.info("RAG响应", content=response_content, session_id=session_id)

        return ChatResponse(message=response_content, history=new_history, session_id=session_id)

    except Exception as e:
        logger.exception("RAG对话处理异常", exc_info=e, session_id=params.session_id)
        return ChatResponse(
            message="很抱歉，处理您的请求时出现了问题。请稍后再试。",
            history=params.history,
            session_id=params.session_id,
        )


@router.post("/chat/sse", description="RAG对话流式接口")
async def rag_chat_sse(
    request: Request,
    current_user: CurrentUser,
    params: RAGChatRequest,
) -> Any:
    """RAG对话流式接口"""
    session_id = params.session_id or str(uuid.uuid4())
    is_new_session = params.session_id is None

    try:
        if params.session_id and params.history:
            MemoMateMemory.merge_history(session_id, params.history)

        llm_params = LLMParams(
            api_type=ModelAPIType.OPENAI,
            api_key=settings.OPENAI_API_KEY or None,
            model_name=settings.CHAT_MODEL,
            streaming=True,  # 开启流式响应
            stream_usage=True,
        )

        llm = LLM().get_llm(llm_params)

        if is_new_session:
            context = retriever.get_context(params.message, top_k=params.retrieve_top_k)

            prompt = RAGLLMPrompt(
                prompt=ChatPromptTemplate.from_template("""{input}{context}"""),
                context=context,
                input=params.message,
            )

            chain = prompt.prompt | llm

            input_data = {"input": params.message, "context": context}

        else:
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """根据聊天历史和用户最新问题，理解用户意图，重新表述为明确的查询。
                            如果问题中有代词引用（如"它"、"这个"），请将其替换为实际含义。
                            保持问题简洁清晰，不要解答问题，只需要重构问题。""",
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(
                llm=llm, retriever=langchain_retriever, contextualize_q_prompt=contextualize_q_prompt
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """请基于上下文信息回答用户的问题。如果无法从上下文中找到答案，请基于历史对话回答。
                上下文信息:
                {context}""",
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

            llm_chain = create_retrieval_chain(
                retriever=history_aware_retriever, combine_docs_chain=question_answer_chain, output_key="output"
            )

            chain = MemoMateMemory.gen_memory_chain(llm_chain)
            input_data = {"input": params.message}

        async def sse_generator() -> AsyncGenerator[str, None]:
            try:
                msg_id = str(uuid.uuid4())
                full_response = ""

                async for chunk in chain.astream(
                    input_data, config=RunnableConfig(callbacks=[], session_id=session_id)
                ):
                    if await request.is_disconnected():
                        logger.info(f"Disconnected from client {request.client}")
                        break

                    if isinstance(chunk, dict):
                        content = chunk.get("output", "") or chunk.get("answer", "")
                        if not content and "text" in chunk:
                            content = chunk["text"]
                    else:
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if not content or content.strip() == "" and not full_response:
                        continue
                    full_response += content
                    message = SSEMessage(event="chat_message", id=msg_id, retry=3000, data=content)
                    yield message.model_dump_json()

                message = SSEMessage(event="chat_complete", id=msg_id, retry=3000, data=full_response)
                yield message.model_dump_json()

                with Session(engine) as db_session:
                    add_history_message(
                        session=db_session,
                        message=params.message,
                        message_type=HistoryMessageType.HUMAN,
                        session_id=session_id,
                        user_id=current_user.id,
                    )
                    add_history_message(
                        session=db_session,
                        message=full_response,
                        message_type=HistoryMessageType.AI,
                        session_id=session_id,
                        user_id=current_user.id,
                    )
                logger.info(
                    "RAG流式响应完成",
                    content_length=len(full_response),
                    session_id=session_id,
                    is_new_session=is_new_session,
                )
            except asyncio.CancelledError:
                logger.info("sse_generator cancelled")
            except Exception as e:
                logger.exception("RAG流式响应错误", exc_info=e, session_id=session_id)
                error_message = SSEMessage(
                    event="chat_error", id=str(uuid.uuid4()), retry=3000, data="生成回答时出错，请稍后重试"
                )
                yield error_message.model_dump_json()

        return EventSourceResponse(sse_generator(), send_timeout=15)

    except Exception as e:
        logger.exception("RAG流式接口错误", exc_info=e, session_id=session_id)

        async def error_generator():
            message = SSEMessage(
                event="chat_error", id=str(uuid.uuid4()), retry=3000, data="处理请求时出错，请稍后重试"
            )
            yield message.model_dump_json()

        return EventSourceResponse(error_generator())
