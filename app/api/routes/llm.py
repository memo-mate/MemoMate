import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

import orjson
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from app.api.demo.sse import SSEMessage
from app.core import logger, settings
from app.rag.llm.completions import LLM, LLMParams, ModelAPIType, RAGLLMPrompt
from app.schemas.llm import ChatRequest, ChatResponse

router = APIRouter()  # dependencies=[Depends(get_current_user)]


class WebSocketChatMessage(BaseModel):
    type: Literal["question", "stop", "ping"] = Field(default="question", description="消息类型: question, stop, ping")
    content: str | None = Field(default=None, description="消息内容")


@router.post("/chat", response_model=ChatResponse, description="对话接口")
async def chat(params: ChatRequest) -> Any:
    """对话接口"""
    # 创建提示
    prompt = RAGLLMPrompt(
        prompt=ChatPromptTemplate.from_template("""{input}{context}"""),
        context="",
        input=params.message,
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
    response = llm_chain.invoke(prompt.model_dump(), RunnableConfig(callbacks=[]))

    logger.info("应用LLM响应", content=response.content)

    return ChatResponse(message=response.content, history=params.history)


@router.post("/chat/sse", description="对话流式接口")
async def chat_sse(
    request: Request,
    params: ChatRequest,
) -> Any:
    """对话流式接口"""
    logger.info("对话流式接口", params=params)

    # 创建提示
    prompt = RAGLLMPrompt(
        prompt=ChatPromptTemplate.from_template("""{input}{context}"""),
        context="",
        input=params.message,
    )

    # 创建参数
    llm_params = LLMParams(
        api_type=ModelAPIType.OPENAI,
        api_key=settings.OPENAI_API_KEY or None,
        model_name=settings.CHAT_MODEL,
        streaming=True,
        stream_usage=True,
    )

    # 创建LLM并生成响应
    llm_chain = LLM().generate(prompt, llm_params)

    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            msg_id = str(uuid.uuid4())  # 消息ID
            full_response = ""  # 完整的响应内容

            # 使用astream方法获取流式输出
            async for chunk in llm_chain.astream(prompt.model_dump(), RunnableConfig(callbacks=[])):
                # 客户端主动断开
                if await request.is_disconnected():
                    logger.info(f"Disconnected from client {request.client}")
                    break

                # 获取当前块的内容
                chunk_content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not chunk_content or chunk_content.strip() == "" and not full_response:
                    continue
                full_response += chunk_content

                # 创建SSE消息并发送
                message = SSEMessage(event="chat_message", id=msg_id, retry=3000, data=chunk_content)
                yield message.model_dump_json()

            # 发送完成事件
            message = SSEMessage(event="chat_complete", id=msg_id, retry=3000, data=full_response)
            yield message.model_dump_json()

            logger.info("流式响应完成", full_response=full_response)

        except asyncio.CancelledError:
            logger.info("sse_generator cancelled")
        except Exception as e:
            logger.exception("sse_generator error", exc_info=e)

    return EventSourceResponse(sse_generator(), send_timeout=15)


@router.websocket("/chat/ws")
async def chat_ws(websocket: WebSocket) -> Any:
    """websocket对话流式接口"""
    await websocket.accept()
    logger.info("WebSocket 连接已建立")

    # 创建 LLM 实例
    llm_instance = LLM()
    # 当前运行的任务
    current_task = None

    try:
        while True:
            # 接收消息
            message_text = await websocket.receive_text()
            logger.debug(f"收到消息: {message_text[:100]}...")

            try:
                text = orjson.loads(message_text)
                # 解析消息
                message = WebSocketChatMessage.model_validate(text)
                logger.debug(f"消息类型: {message.type}")

                # 处理 ping 消息
                if message.type == "ping":
                    logger.debug("处理 ping 消息")
                    await websocket.send_json({"type": "pong", "content": "pong"})
                    continue

                # 处理停止指令
                elif message.type == "stop":
                    logger.info("收到停止指令")
                    if current_task and not current_task.done():
                        current_task.cancel()
                    await websocket.send_json({"type": "info", "content": "已停止生成"})
                    continue

                # 处理问题
                elif message.type == "question":
                    if not message.content:
                        logger.warning("问题或上下文为空")
                        await websocket.send_json({"type": "error", "content": "问题或上下文不能为空"})
                        continue

                    # 发送消息
                    async def background_chunk_message(message: str, msg_id: str) -> None:
                        try:
                            # 准备 LLM 参数
                            llm_params = LLMParams(
                                streaming=True,
                                stream_usage=True,
                                api_type=ModelAPIType.OPENAI,
                                api_key=settings.OPENAI_API_KEY or None,
                                model_name=settings.CHAT_MODEL,
                            )

                            prompt = RAGLLMPrompt(
                                prompt=ChatPromptTemplate.from_template("""{input}{context}"""),
                                context="",
                                input=message,
                            )
                            # 获取生成链
                            chain = llm_instance.generate(prompt, llm_params)
                            full_response = ""
                            async for chunk in chain.astream(prompt.model_dump(), RunnableConfig(callbacks=[])):
                                if not chunk.content or chunk.content.strip() == "" and not full_response:
                                    continue
                                full_response += chunk.content
                                await websocket.send_json({"type": "token", "id": msg_id, "content": chunk.content})

                            await websocket.send_json({"type": "token", "id": msg_id, "content": full_response})
                            logger.info("对话完成", full_response=full_response)
                        except asyncio.CancelledError:
                            logger.info("对话取消")

                    msg_id = str(uuid.uuid4())
                    current_task = asyncio.create_task(background_chunk_message(message.content, msg_id))
                # 未知消息类型
                else:
                    logger.warning(f"未知消息类型: {message.type}")
                    await websocket.send_json({"type": "error", "content": f"未知消息类型: {message.type}"})

            except orjson.JSONDecodeError:
                logger.warning("无效的 JSON 格式")
                await websocket.send_json({"type": "error", "content": "无效的 JSON 格式"})
            except Exception as e:
                logger.exception("处理消息时出错", exc_info=e)
                await websocket.send_json({"type": "error", "content": f"处理消息时出错: {str(e)}"})

    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
        # 如果有正在运行的任务，取消它
        if current_task and not current_task.done():
            current_task.cancel()
    except Exception as e:
        logger.exception("WebSocket 处理时发生错误", exc_info=e)
