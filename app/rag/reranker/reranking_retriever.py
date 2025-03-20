"""含重排序功能的检索器"""

from typing import Any, cast

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.runnables.config import run_in_executor
from pydantic import BaseModel, Field

from app.core.log_adapter import logger
from app.rag.reranker.base import BaseReranker


class RerankingRetriever(BaseRetriever, BaseModel):
    """包含重排序功能的检索器

    将基础检索器与重排序器结合，提高检索质量
    """

    base_retriever: BaseRetriever = Field(description="基础检索器")
    reranker: BaseReranker = Field(description="重排序器")
    fetch_k: int = Field(default=20, description="从基础检索器获取的文档数量")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """获取相关文档并应用重排序

        Args:
            query: 查询文本
            run_manager: 回调管理器

        Returns:
            重排序后的相关文档
        """
        # 使用基础检索器获取初始文档集
        logger.debug(f"使用基础检索器 {type(self.base_retriever).__name__} 获取初始文档")
        docs = self.base_retriever.get_relevant_documents(query)

        if not docs:
            logger.warning("基础检索器未返回任何文档")
            return []

        logger.debug(f"基础检索器返回了 {len(docs)} 个文档")

        # 如果需要，截取前fetch_k个文档
        if len(docs) > self.fetch_k:
            docs = docs[: self.fetch_k]
            logger.debug(f"截取了前 {self.fetch_k} 个文档用于重排序")

        # 应用重排序
        logger.debug(f"使用 {type(self.reranker).__name__} 进行重排序")
        reranked_docs = self.reranker.rerank(query, docs)

        return reranked_docs

    async def _aretrieve_relevant_docs(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """异步获取相关文档的底层实现

        Args:
            query: 查询文本
            run_manager: 异步回调管理器
            **kwargs: 其他参数

        Returns:
            重排序后的相关文档
        """
        # 使用run_in_executor执行同步方法
        sync_manager = run_manager.get_sync() if run_manager else None
        return await run_in_executor(
            None,  # 使用默认的executor
            self._get_relevant_documents,
            query,
            run_manager=sync_manager,
        )

    def invoke(
        self, input: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Document]:
        """获取相关文档并应用重排序

        Args:
            input: 输入，可以是字符串查询或字典
            config: 运行配置
            **kwargs: 其他参数

        Returns:
            重排序后的相关文档
        """
        config = ensure_config(config)

        # 解析输入
        if isinstance(input, str):
            query = input
        else:
            query = cast(dict[str, Any], input).get("query", "")

        # 使用基础检索器获取初始文档集
        logger.debug(f"使用基础检索器 {type(self.base_retriever).__name__} 获取初始文档")
        docs = self.base_retriever.invoke(input, config=config, **kwargs)

        if not docs:
            logger.warning("基础检索器未返回任何文档")
            return []

        logger.debug(f"基础检索器返回了 {len(docs)} 个文档")

        # 如果需要，截取前fetch_k个文档
        if len(docs) > self.fetch_k:
            docs = docs[: self.fetch_k]
            logger.debug(f"截取了前 {self.fetch_k} 个文档用于重排序")

        # 应用重排序
        logger.debug(f"使用 {type(self.reranker).__name__} 进行重排序")
        reranked_docs = self.reranker.rerank(query, docs)

        return reranked_docs

    async def ainvoke(
        self, input: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Document]:
        """异步获取相关文档并应用重排序

        Args:
            input: 输入，可以是字符串查询或字典
            config: 运行配置
            **kwargs: 其他参数

        Returns:
            重排序后的相关文档
        """
        config = ensure_config(config)

        # 解析输入
        if isinstance(input, str):
            query = input
        else:
            query = cast(dict[str, Any], input).get("query", "")

        # 使用基础检索器获取初始文档集
        logger.debug(f"[异步] 使用基础检索器 {type(self.base_retriever).__name__} 获取初始文档")
        docs = await self.base_retriever.ainvoke(input, config=config, **kwargs)

        if not docs:
            logger.warning("[异步] 基础检索器未返回任何文档")
            return []

        logger.debug(f"[异步] 基础检索器返回了 {len(docs)} 个文档")

        # 如果需要，截取前fetch_k个文档
        if len(docs) > self.fetch_k:
            docs = docs[: self.fetch_k]
            logger.debug(f"[异步] 截取了前 {self.fetch_k} 个文档用于重排序")

        # 应用重排序
        logger.debug(f"[异步] 使用 {type(self.reranker).__name__} 进行重排序")
        reranked_docs = self.reranker.rerank(query, docs)

        return reranked_docs
