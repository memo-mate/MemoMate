from langchain_core.documents import Document

from app.core import settings
from app.core.log_adapter import logger
from app.rag.embedding.embedding_db import QdrantVectorStore
from app.rag.embedding.embeeding_model import MemoMateEmbeddings


class BaseRetriever:
    """基础检索器类"""

    def __init__(
        self,
        collection_name: str = settings.QDRANT_COLLECTION,
        embedding_model=None,
        vector_store_url: str | None = None,
        vector_store_api_key: str | None = None,
        vector_store_path: str | None = settings.QDRANT_PATH,
    ):
        # 初始化嵌入模型
        self.embeddings = embedding_model or MemoMateEmbeddings.openai_embedding(
            api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE
        )
        # 初始化向量存储
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            embeddings=self.embeddings,
            url=vector_store_url,
            api_key=vector_store_api_key,
            path=vector_store_path,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """检索与查询相关的文档"""
        try:
            results = self.vector_store.similarity_search(query=query, k=top_k)

            logger.info(f"向量检索结果数量: {len(results)}")

            if not results:
                logger.warning("未找到匹配的文档")
                return []

            return results
        except Exception as e:
            logger.exception("文档检索失败", exc_info=e)
            return []

    def get_context(self, query: str, top_k: int = 5) -> str:
        """获取格式化的上下文内容"""
        documents = self.retrieve(query, top_k=top_k)

        if not documents:
            return ""

        # 将文档内容拼接为上下文字符串
        context_parts = []
        for i, doc in enumerate(documents):
            # 添加文档内容和元数据
            source = doc.metadata.get("filename", "未知来源")
            context_parts.append(f"[文档{i + 1}] 来源: {source}\n{doc.page_content}\n")

        return "\n".join(context_parts)

    def as_langchain_retriever(self, **kwargs):
        """将检索器转换为langchain检索器"""
        return self.vector_store.as_retriever(**kwargs)
