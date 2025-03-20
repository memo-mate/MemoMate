import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from app.api.deps import CurrentUser
from app.core.log_adapter import logger

router = APIRouter()


class SSEMessage(BaseModel):
    event: str  # 事件类型
    id: str  # 消息ID
    retry: int  # 重连间隔时间 milliseconds
    data: str  # 消息内容


@router.get("/sse", response_model=None, description="SSE消息Demo")
async def sse(request: Request, current_user: CurrentUser) -> EventSourceResponse:
    logger.info("SSE消息认证成功", user=current_user.username)

    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                # 客户端主动断开
                if await request.is_disconnected():
                    logger.info(f"Disconnected from client {request.client}")
                    break

                message = SSEMessage(event="message", id="1", retry=3000, data="Hello, world!")
                yield message.model_dump_json()
                await asyncio.sleep(1)
                message.id = "2"
                message.data = "Hello, world! 2"
                yield message.model_dump_json()
                await asyncio.sleep(1)
                message.id = "3"
                message.data = "Hello, world! 3"
                yield message.model_dump_json()
        except asyncio.CancelledError:
            print("sse_generator cancelled")
        except Exception as e:
            logger.exception("sse_generator error", exc_info=e)

    return EventSourceResponse(sse_generator(), send_timeout=5)
