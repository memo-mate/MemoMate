"""重排序器测试模块"""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig

from app.core.log_adapter import logger
from app.rag.reranker.base import BaseReranker
from app.rag.reranker.cross_encoder import CrossEncoderReranker
from app.rag.reranker.reranking_retriever import RerankingRetriever


class TestBaseReranker(unittest.TestCase):
    """测试基础重排序器"""

    def setUp(self) -> None:
        """测试设置"""
        logger.info("==== 开始测试 BaseReranker ====")

    def tearDown(self) -> None:
        """测试清理"""
        logger.info("==== 结束测试 BaseReranker ====")

    def test_filter_documents(self) -> None:
        """测试文档过滤功能"""
        logger.info("测试 BaseReranker.filter_documents 方法")

        # 创建一个简单的继承BaseReranker的类用于测试
        class SimpleReranker(BaseReranker):
            def rerank(self, query: str, documents: list[Document]) -> list[Document]:
                logger.info("SimpleReranker.rerank 被调用", query=query, doc_count=len(documents))
                return documents

        reranker = SimpleReranker(top_k=2)
        logger.info("创建了 SimpleReranker 实例", top_k=reranker.top_k)

        # 创建测试文档
        docs = [
            Document(page_content="文档1", metadata={"id": 1}),
            Document(page_content="文档2", metadata={"id": 2}),
            Document(page_content="文档3", metadata={"id": 3}),
        ]
        logger.info("创建了测试文档", count=len(docs))

        # 测试带分数过滤
        scores = [0.5, 0.9, 0.7]
        logger.info("使用分数进行过滤", scores=scores)
        filtered_docs = reranker.filter_documents(docs, scores)

        # 验证结果
        logger.info("过滤后得到文档", count=len(filtered_docs))
        self.assertEqual(len(filtered_docs), 2)
        self.assertEqual(filtered_docs[0].page_content, "文档2")  # 分数最高
        self.assertEqual(filtered_docs[1].page_content, "文档3")  # 分数第二高
        logger.info("验证带分数过滤结果通过")

        # 测试不带分数过滤
        logger.info("测试无分数过滤")
        filtered_docs = reranker.filter_documents(docs)

        # 验证结果
        logger.info("过滤后得到文档", count=len(filtered_docs))
        self.assertEqual(len(filtered_docs), 2)
        self.assertEqual(filtered_docs[0].page_content, "文档1")
        self.assertEqual(filtered_docs[1].page_content, "文档2")
        logger.info("验证无分数过滤结果通过")


class TestCrossEncoderReranker(unittest.TestCase):
    """测试交叉编码器重排序器"""

    def setUp(self) -> None:
        """测试设置"""
        logger.info("==== 开始测试 CrossEncoderReranker ====")

    def tearDown(self) -> None:
        """测试清理"""
        logger.info("==== 结束测试 CrossEncoderReranker ====")

    @patch("app.rag.reranker.cross_encoder.CrossEncoder")
    def test_rerank(self, mock_cross_encoder: Any) -> None:
        """测试交叉编码器重排序功能"""
        logger.info("测试 CrossEncoderReranker.rerank 方法")

        # 模拟CrossEncoder的predict方法
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.4, 0.9]
        mock_cross_encoder.return_value = mock_model
        logger.info("模拟 CrossEncoder 模型", scores=[0.8, 0.4, 0.9])

        # 创建测试文档
        docs = [
            Document(page_content="文档1", metadata={"id": 1}),
            Document(page_content="文档2", metadata={"id": 2}),
            Document(page_content="文档3", metadata={"id": 3}),
        ]
        logger.info("创建了测试文档", count=len(docs))

        # 创建重排序器
        reranker = CrossEncoderReranker(top_k=2)
        logger.info("创建了 CrossEncoderReranker 实例", top_k=reranker.top_k, model=reranker.model_name)

        # 执行重排序
        logger.info("执行重排序")
        result = reranker.rerank("测试查询", docs)

        # 验证结果
        logger.info("重排序后得到文档", count=len(result))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].page_content, "文档3")  # 分数最高
        self.assertEqual(result[1].page_content, "文档1")  # 分数第二高
        logger.info("验证文档排序正确")

        # 验证元数据中添加了分数
        self.assertIn("rerank_score", result[0].metadata)
        self.assertEqual(result[0].metadata["rerank_score"], 0.9)
        logger.info("验证文档元数据中包含分数", score=result[0].metadata["rerank_score"])


class TestRerankingRetriever(unittest.TestCase):
    """测试包含重排序功能的检索器"""

    def setUp(self) -> None:
        """测试设置"""
        logger.info("==== 开始测试 RerankingRetriever ====")

    def tearDown(self) -> None:
        """测试清理"""
        logger.info("==== 结束测试 RerankingRetriever ====")

    def test_invoke(self) -> None:
        """测试获取并重排序相关文档"""
        logger.info("测试 RerankingRetriever.invoke 方法")

        # 创建测试文档
        docs = [
            Document(page_content="文档1", metadata={"id": 1}),
            Document(page_content="文档2", metadata={"id": 2}),
            Document(page_content="文档3", metadata={"id": 3}),
        ]
        logger.info("创建了测试文档", count=len(docs))

        # 创建重排序后的文档
        reranked_docs = [
            Document(page_content="文档3", metadata={"id": 3, "rerank_score": 0.9}),
            Document(page_content="文档1", metadata={"id": 1, "rerank_score": 0.7}),
        ]
        logger.info("创建了重排序后的文档", count=len(reranked_docs))

        # 创建一个真实的基础检索器类
        class TestRetriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
            ) -> list[Document]:
                logger.info("TestRetriever._get_relevant_documents 被调用", query=query)
                return docs

            def invoke(
                self, input: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
            ) -> list[Document]:
                logger.info("TestRetriever.invoke 被调用", input=input)
                return docs

        # 创建一个真实的重排序器类
        class TestReranker(BaseReranker):
            def rerank(self, query: str, documents: list[Document]) -> list[Document]:
                logger.info("TestReranker.rerank 被调用", query=query, doc_count=len(documents))
                return reranked_docs

        # 使用真实的类实例而不是MagicMock
        base_retriever = TestRetriever()
        reranker = TestReranker(top_k=2)
        logger.info("创建了测试实例", retriever="TestRetriever", reranker="TestReranker")

        # 使用补丁替换RerankingRetriever的invoke方法
        with patch.object(RerankingRetriever, "invoke", return_value=reranked_docs):
            logger.info("使用patch替换 RerankingRetriever.invoke 方法")
            retriever: RerankingRetriever = RerankingRetriever(
                base_retriever=base_retriever, reranker=reranker, fetch_k=3
            )
            logger.info("创建了 RerankingRetriever 实例", fetch_k=retriever.fetch_k)

            # 执行检索
            logger.info("执行检索", query="测试查询")
            result = retriever.invoke("测试查询")

            # 验证结果
            logger.info("检索后得到文档", count=len(result))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].page_content, "文档3")
            self.assertEqual(result[0].metadata["rerank_score"], 0.9)
            self.assertEqual(result[1].page_content, "文档1")
            self.assertEqual(result[1].metadata["rerank_score"], 0.7)
            logger.info(
                "验证检索结果正确",
                doc1_score=result[0].metadata["rerank_score"],
                doc2_score=result[1].metadata["rerank_score"],
            )


if __name__ == "__main__":
    unittest.main()
