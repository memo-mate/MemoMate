"""交叉编码器重排序器"""

from typing import Any, cast

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from pydantic import Field

from app.core.config import settings
from app.core.log_adapter import logger
from app.rag.reranker.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """基于交叉编码器的重排序器

    使用预训练的交叉编码器模型对检索结果进行重排序
    """

    model_name: str = Field(default=settings.RERANKER_MODEL_PATH, description="交叉编码器模型名称")
    model: CrossEncoder | None = None
    score_threshold: float = Field(default=0.0, description="分数阈值，低于该阈值的文档将被过滤")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        """初始化交叉编码器重排序器"""
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self) -> None:
        """加载交叉编码器模型"""
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"成功加载交叉编码器模型: {self.model_name}")
        except Exception as e:
            logger.error(f"加载交叉编码器模型失败: {e}")
            raise RuntimeError(f"无法加载交叉编码器模型: {e}")

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """使用交叉编码器重排序文档

        Args:
            query: 查询文本
            documents: 需要重排序的文档列表

        Returns:
            重排序后的文档列表
        """
        if not documents:
            logger.warning("没有文档需要重排序")
            return []

        if not self.model:
            logger.warning("交叉编码器模型未加载，正在尝试加载模型")
            self._load_model()

        # 确保模型已加载
        model = cast(CrossEncoder, self.model)

        # 准备模型输入
        model_inputs = [[query, doc.page_content] for doc in documents]

        # 对所有文档对进行打分
        logger.debug(f"使用交叉编码器对 {len(documents)} 个文档进行重排序")
        scores = model.predict(model_inputs)

        # 过滤低于阈值的文档
        filtered_docs_with_scores = [
            (doc, score) for doc, score in zip(documents, scores) if score >= self.score_threshold
        ]

        if not filtered_docs_with_scores:
            logger.warning(f"所有文档的相关性分数都低于阈值 {self.score_threshold}")
            return []

        # 分离文档和分数
        filtered_docs = [doc for doc, _ in filtered_docs_with_scores]
        filtered_scores = [score for _, score in filtered_docs_with_scores]

        # 将分数添加到文档的元数据中
        for doc, score in zip(filtered_docs, filtered_scores):
            doc.metadata["rerank_score"] = float(score)

        # 应用过滤
        result = self.filter_documents(filtered_docs, filtered_scores)
        self.log_rerank_info(query, documents, result)

        return result
