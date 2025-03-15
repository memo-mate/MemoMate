import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from app.core.config import settings  # 导入设置以获取API前缀
from app.main import app

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AsyncIterator:
    """模拟异步迭代器用于测试"""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def mock_llm():
    """模拟LLM生成功能，返回预设响应"""
    with patch("app.api.demo.websocket.AsyncIteratorCallbackHandler") as mock_handler:
        # 设置模拟器返回预设的token
        tokens = ["这是", "一个", "模拟", "回答"]
        instance = mock_handler.return_value
        instance.aiter.return_value = AsyncIterator(tokens)

        # 模拟LLM.generate方法
        with patch("app.api.demo.websocket.LLM") as mock_llm:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock()

            mock_llm_instance = mock_llm.return_value
            mock_llm_instance.generate.return_value = mock_chain

            yield mock_llm_instance


# 添加对 ping/pong 功能的测试
@pytest.mark.asyncio
async def test_websocket_ping():
    """测试WebSocket ping/pong功能"""
    client = TestClient(app)
    websocket_url = f"{settings.API_V1_STR}/demo/ws/rag"
    logger.info(f"测试 WebSocket ping: {websocket_url}")

    with client.websocket_connect(websocket_url) as websocket:
        # 发送 ping 消息
        message = {"type": "ping"}
        websocket.send_text(json.dumps(message))
        logger.debug("已发送 ping 消息")

        # 接收 pong 响应
        response = websocket.receive_json()
        logger.debug(f"收到响应: {response}")

        assert response["type"] == "pong"
        assert response["content"] == "pong"


@pytest.mark.asyncio
async def test_websocket_rag_question(mock_llm):
    """测试WebSocket RAG问答功能"""
    client = TestClient(app)
    websocket_url = f"{settings.API_V1_STR}/demo/ws/rag"
    logger.info(f"测试 WebSocket RAG 问答: {websocket_url}")

    with client.websocket_connect(websocket_url) as websocket:
        # 发送问题
        message = {"type": "question", "content": "测试问题", "context": "测试上下文"}
        websocket.send_text(json.dumps(message))
        logger.debug(f"已发送问题: {message}")

        # 接收生成的token
        expected_tokens = ["这是", "一个", "模拟", "回答"]
        for expected_token in expected_tokens:
            response = websocket.receive_json()
            logger.debug(f"收到响应: {response}")
            assert response["type"] == "token"
            assert response["content"] == expected_token

        # 接收完成消息
        done_message = websocket.receive_json()
        logger.debug(f"收到完成消息: {done_message}")
        assert done_message["type"] == "done"


@pytest.mark.asyncio
async def test_websocket_rag_stop():
    """测试停止生成功能"""
    with patch("app.api.demo.websocket.AsyncIteratorCallbackHandler") as mock_handler:
        # 创建一个阻塞较长的异步迭代器
        async def mock_aiter():
            yield "开始生成"
            # 使用更长的延迟
            await asyncio.sleep(2.0)  # 增加延迟时间
            yield "这个不应该被返回"

        instance = mock_handler.return_value
        instance.aiter.return_value = mock_aiter()

        # 确保generate不会立即完成
        with patch("app.api.demo.websocket.LLM") as mock_llm:
            mock_chain = MagicMock()

            # 延迟调用完成
            async def delayed_invoke(*args, **kwargs):
                await asyncio.sleep(1.0)

            mock_chain.ainvoke = AsyncMock(side_effect=delayed_invoke)

            mock_llm_instance = mock_llm.return_value
            mock_llm_instance.generate.return_value = mock_chain

            client = TestClient(app)
            websocket_url = f"{settings.API_V1_STR}/demo/ws/rag"

            with client.websocket_connect(websocket_url) as websocket:
                # 发送问题
                question_message = {"type": "question", "content": "长回答的问题", "context": "测试上下文"}
                websocket.send_text(json.dumps(question_message))

                # 接收第一个token
                first_response = websocket.receive_json()
                assert first_response["type"] == "token"

                # 不要等待，立即发送停止指令
                stop_message = {"type": "stop"}
                websocket.send_text(json.dumps(stop_message))

                # 循环接收直到找到info消息
                for _ in range(5):  # 最多尝试5次
                    response = websocket.receive_json()
                    if response["type"] == "info":
                        assert "已停止生成" in response["content"]
                        return  # 测试通过

                # 如果没找到info消息，测试失败
                pytest.fail("未收到停止确认消息")


@pytest.mark.asyncio
async def test_websocket_invalid_message():
    """测试发送无效消息时的错误处理"""
    client = TestClient(app)
    websocket_url = f"{settings.API_V1_STR}/demo/ws/rag"
    logger.info(f"测试 WebSocket 无效消息处理: {websocket_url}")

    with client.websocket_connect(websocket_url) as websocket:
        # 发送无效JSON
        websocket.send_text("这不是JSON")
        logger.debug("已发送无效JSON")

        # 应该收到错误消息
        response = websocket.receive_json()
        logger.debug(f"收到响应: {response}")
        assert response["type"] == "error"
        assert "无效的 JSON 格式" in response["content"]

        # 发送缺少必填字段的消息
        websocket.send_text(json.dumps({"type": "question"}))
        logger.debug("已发送缺少必填字段的消息")

        # 应该收到错误消息
        response = websocket.receive_json()
        logger.debug(f"收到响应: {response}")
        assert response["type"] == "error"
        assert "问题或上下文不能为空" in response["content"]


@pytest.mark.asyncio
async def test_websocket_unknown_message_type():
    """测试发送未知消息类型时的错误处理"""
    client = TestClient(app)
    websocket_url = f"{settings.API_V1_STR}/demo/ws/rag"
    logger.info(f"测试 WebSocket 未知消息类型: {websocket_url}")

    with client.websocket_connect(websocket_url) as websocket:
        # 发送未知类型的消息
        message = {"type": "unknown_type", "content": "测试"}
        websocket.send_text(json.dumps(message))
        logger.debug(f"已发送未知类型消息: {message}")

        # 应该收到错误消息
        response = websocket.receive_json()
        logger.debug(f"收到响应: {response}")
        assert response["type"] == "error"
        assert "未知消息类型" in response["content"]


# 添加一个简单的连接测试
@pytest.mark.asyncio
async def test_simple_websocket_connection():
    """测试WebSocket连接是否可以建立"""
    client = TestClient(app)
    websocket_url = f"{settings.API_V1_STR}/demo/ws/rag"
    logger.info(f"测试简单 WebSocket 连接: {websocket_url}")

    try:
        with client.websocket_connect(websocket_url) as websocket:
            # 如果能够连接，测试通过
            assert websocket is not None
            logger.info(f"成功连接到 WebSocket: {websocket_url}")

            # 发送简单的 ping 消息测试连接
            simple_message = {"type": "ping"}
            websocket.send_text(json.dumps(simple_message))
            logger.debug("已发送 ping 消息")

            # 尝试接收回应
            try:
                response = websocket.receive_json()
                logger.info(f"收到响应: {response}")
                assert response["type"] == "pong"
            except Exception as e:
                logger.error(f"接收响应时出错: {e}")
                pytest.fail(f"接收响应失败: {e}")

    except WebSocketDisconnect as e:
        logger.error(f"WebSocket连接无法建立，服务器断开连接: {e}")
        pytest.fail(f"WebSocket连接无法建立，服务器断开连接: {e}")
    except Exception as e:
        logger.error(f"WebSocket连接测试失败，错误: {e}")
        pytest.fail(f"WebSocket连接测试失败，错误: {e}")
