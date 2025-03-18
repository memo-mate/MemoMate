"""重排序器测试模块"""

import logging
import unittest
from typing import Any, Dict, List, Optional, Sequence, Union
from unittest.mock import MagicMock, patch

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig

from app.core.log_adapter import logger
from app.rag.reranker.base import BaseReranker
from app.rag.reranker.cross_encoder import CrossEncoderReranker
from app.rag.reranker.llm_reranker import LLMReranker
from app.rag.reranker.reranking_retriever import RerankingRetriever


# 设置日志级别为DEBUG
logger.setLevel(logging.DEBUG)
# 添加控制台处理器
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# ... existing code ...
