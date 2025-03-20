"""重排序基类模块"""

from abc import ABC, abstractmethod

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.core.log_adapter import logger


class BaseReranker(BaseModel, ABC):
    """重排序器基类

    用于对检索出的文档进行重新排序，提高检索质量
    """

    top_k: int = Field(default=5, description="返回的文档数量")

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """重排序文档

        Args:
            query: 查询文本
            documents: 需要重排序的文档列表

        Returns:
            重排序后的文档列表
        """
        pass

    def filter_documents(self, documents: list[Document], scores: list[float] | None = None) -> list[Document]:
        """根据分数过滤文档

        Args:
            documents: 文档列表
            scores: 对应的分数列表

        Returns:
            按分数排序并限制数量后的文档列表
        """
        if scores:
            # 按分数排序
            sorted_docs = [
                doc for _, doc in sorted(zip(scores, documents, strict=False), key=lambda x: x[0], reverse=True)
            ]
            return sorted_docs[: self.top_k]

        # 如果没有分数，直接截取前top_k个
        return documents[: self.top_k]

    def log_rerank_info(self, query: str, original_docs: list[Document], reranked_docs: list[Document]) -> None:
        """记录重排序信息

        Args:
            query: 查询文本
            original_docs: 原始文档
            reranked_docs: 重排序后的文档
        """
        logger.debug(f"Reranking for query: {query}")
        logger.debug(f"Original documents: {len(original_docs)}")
        logger.debug(f"Reranked documents: {len(reranked_docs)}")
