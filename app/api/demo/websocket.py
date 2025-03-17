import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from pydantic import BaseModel

from app.core.log_adapter import logger
from app.rag.llm.completions import LLM, LLMParams, ModelAPIType, RAGLLMPrompt

router = APIRouter()


class WebSocketMessage(BaseModel):
    type: str  # 消息类型: question, stop, ping
    content: str | None = None  # 消息内容
    context: str | None = None  # RAG 上下文


@router.websocket("/ws/rag")
async def websocket_rag_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket 连接已建立")

    # 创建 LLM 实例
    llm_instance = LLM()
    # 控制生成任务的标志
    cancel_generation = False
    # 当前运行的任务
    current_task = None

    try:
        while True:
            # 接收消息
            message_text = await websocket.receive_text()
            logger.debug(f"收到消息: {message_text[:100]}...")

            try:
                # 解析消息
                message = WebSocketMessage.model_validate(json.loads(message_text))
                logger.debug(f"消息类型: {message.type}")

                # 处理 ping 消息
                if message.type == "ping":
                    logger.debug("处理 ping 消息")
                    await websocket.send_json({"type": "pong", "content": "pong"})
                    continue

                # 处理停止指令
                elif message.type == "stop":
                    logger.info("收到停止指令")
                    cancel_generation = True
                    if current_task and not current_task.done():
                        current_task.cancel()
                    await websocket.send_json({"type": "info", "content": "已停止生成"})
                    cancel_generation = False
                    continue

                # 处理问题
                elif message.type == "question":
                    if not message.content or not message.context:
                        logger.warning("问题或上下文为空")
                        await websocket.send_json({"type": "error", "content": "问题或上下文不能为空"})
                        continue

                    # 重置标志
                    cancel_generation = False
                    logger.info(f"处理问题: {message.content[:50]}...")

                    # 创建回调处理器
                    callback = AsyncIteratorCallbackHandler()

                    # 准备 LLM 参数
                    params = LLMParams(
                        streaming=True,
                        api_type=ModelAPIType.OPENAI,
                    )

                    # 准备 RAG 提示
                    prompt = RAGLLMPrompt(context=message.context, question=message.content)

                    # 获取生成链
                    chain = llm_instance.generate(prompt, params)

                    # 开始异步生成
                    async def generate_response():
                        try:
                            await chain.ainvoke(prompt.model_dump(exclude={"prompt"}), config={"callbacks": [callback]})
                        except asyncio.CancelledError:
                            logger.info("生成已取消")
                            await websocket.send_json({"type": "info", "content": "\n[已停止生成]"})
                        except Exception as e:
                            logger.exception("生成回答时出错", exc_info=e)
                            await websocket.send_json({"type": "error", "content": f"生成回答时出错: {str(e)}"})

                    # 创建任务
                    current_task = asyncio.create_task(generate_response())

                    # 发送生成的内容
                    try:
                        async for token in callback.aiter():
                            if cancel_generation:
                                break

                            await websocket.send_json({"type": "token", "content": token})
                            logger.debug(f"发送 token: {token}")

                        # 发送完成消息
                        if not cancel_generation:
                            await websocket.send_json({"type": "done", "content": ""})
                            logger.info("回答生成完成")

                    except asyncio.CancelledError:
                        # 处理取消
                        logger.info("回答生成被取消")
                        pass

                # 未知消息类型
                else:
                    logger.warning(f"未知消息类型: {message.type}")
                    await websocket.send_json({"type": "error", "content": f"未知消息类型: {message.type}"})

            except json.JSONDecodeError:
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
