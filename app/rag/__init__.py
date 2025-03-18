"""RAG模块

该模块提供了检索增强生成相关功能，包括：
- 嵌入和向量存储
- 文档检索
- 大语言模型集成
- 重排序功能
"""

from app.rag.embedding import (
    QdrantStore,
    DocumentProcessor,
    DirectoryProcessor,
    QdrantRetriever,
    MultiQueryRetriever,
    HybridRetriever,
    IndexManager,
    RetrievalEvaluator,
    RAGEvaluator,
)

from app.rag.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    LLMReranker,
    RerankingRetriever,
)

__all__ = [
    # 嵌入模块
    "QdrantStore",
    "DocumentProcessor",
    "DirectoryProcessor",
    "QdrantRetriever",
    "MultiQueryRetriever",
    "HybridRetriever",
    "IndexManager",
    "RetrievalEvaluator",
    "RAGEvaluator",
    # 重排序模块
    "BaseReranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "RerankingRetriever",
]
