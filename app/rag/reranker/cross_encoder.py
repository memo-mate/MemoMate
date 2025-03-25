"""交叉编码器重排序器"""

from typing import Any, cast

import requests
from langchain_core.documents import Document

from app.core.config import settings
from app.core.log_adapter import logger
from app.rag.reranker.base import BaseReranker

__all__ = [
    "CrossEncoderReranker",
    "SiliconCloudCrossEncoderReranker",
]


class CrossEncoderReranker(BaseReranker):
    """基于交叉编码器的重排序器

    使用预训练的交叉编码器模型对检索结果进行重排序
    """

    def __init__(
        self, model_name: str = settings.RERANKER_MODEL_PATH, score_threshold: float = 0.0, top_k: int = 5
    ) -> None:
        """初始化交叉编码器重排序器

        Args:
            model_name: 交叉编码器模型名称
            score_threshold: 分数阈值，低于该阈值的文档将被过滤
            top_k: 返回的文档数量
        """
        super().__init__(top_k=top_k)
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载交叉编码器模型"""
        try:
            from sentence_transformers import CrossEncoder

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
        model = cast(Any, self.model)

        # 准备模型输入
        model_inputs = [[query, doc.page_content] for doc in documents]

        # 对所有文档对进行打分
        logger.debug(f"使用交叉编码器对 {len(documents)} 个文档进行重排序")
        scores = model.predict(model_inputs)

        # 过滤低于阈值的文档
        filtered_docs_with_scores = [
            (doc, score) for doc, score in zip(documents, scores, strict=False) if score >= self.score_threshold
        ]

        if not filtered_docs_with_scores:
            logger.warning(f"所有文档的相关性分数都低于阈值 {self.score_threshold}")
            return []

        # 分离文档和分数
        filtered_docs = [doc for doc, _ in filtered_docs_with_scores]
        filtered_scores = [score for _, score in filtered_docs_with_scores]

        # 将分数添加到文档的元数据中
        for doc, score in zip(filtered_docs, filtered_scores, strict=False):
            doc.metadata["rerank_score"] = float(score)

        # 应用过滤
        result = self.filter_documents(filtered_docs, filtered_scores)
        self.log_rerank_info(query, documents, result)

        return result


class SiliconCloudCrossEncoderReranker(BaseReranker):
    """基于硅基流动API的交叉编码器重排序器

    通过远程API调用，使用硅基流动的交叉编码器对检索结果进行重排序
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1/rerank",
        model_name: str = "BAAI/bge-reranker-v2-m3",
        score_threshold: float = 0.0,
        top_k: int = 5,
    ) -> None:
        """初始化硅基流动API交叉编码器重排序器

        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
            score_threshold: 分数阈值，低于该阈值的文档将被过滤
            top_k: 返回的文档数量
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.top_k = top_k
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        logger.info("SiliconCloudCrossEncoderReranker initialized", model=model_name, base_url=base_url)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """使用硅基流动API交叉编码器重排序文档

        Args:
            query: 查询文本
            documents: 需要重排序的文档列表

        Returns:
            重排序后的文档列表
        """
        if not documents:
            logger.warning("没有文档需要重排序")
            return []

        doc_texts = [doc.page_content for doc in documents]

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": self.top_k,
            "return_documents": False,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80,
        }

        try:
            logger.info(f"通过硅基流动API对 {len(documents)} 个文档进行重排序")
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                logger.error("硅基流动API响应中没有找到结果")
                return documents[: self.top_k]

            reranked_indices = []
            reranked_scores = []

            for result in results:
                index = result.get("index")
                score = result.get("relevance_score", 0.0)

                if score >= self.score_threshold:
                    reranked_indices.append(index)
                    reranked_scores.append(score)

            reranked_docs = [documents[idx] for idx in reranked_indices] if reranked_indices else []

            for doc, score in zip(reranked_docs, reranked_scores, strict=False):
                doc.metadata["rerank_score"] = float(score)

            if not reranked_docs:
                logger.warning(f"所有文档的相关性分数都低于阈值 {self.score_threshold}")
                return []

            self.log_rerank_info(query, documents, reranked_docs)
            return reranked_docs

        except requests.exceptions.RequestException as e:
            logger.error(f"硅基流动API请求失败: {str(e)}")
            return documents[: self.top_k]
        except Exception as e:
            logger.error(f"硅基流动API重排序过程中发生错误: {str(e)}")
            return documents[: self.top_k]
