from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core import settings
from app.core.log_adapter import logger
from app.enums import EmbeddingDriverEnum
from app.rag.embedding.embedding_db import QdrantVectorStore
from app.rag.embedding.embeeding_model import MemoMateEmbeddings

# 本地启动 qdrant 服务
# docker run -d --name qdrant-server -p 6333:6333 -e QDRANT__API__HTTP_ENABLED=true -e QDRANT__API__HTTP_API_KEY=memo.fastapi qdrant/qdrant


def get_qdrant_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        collection_name="test",
        embeddings=MemoMateEmbeddings.local_embedding(driver=EmbeddingDriverEnum.MAC),
        path=settings.QDRANT_PATH,
    )


def test_local_embedding():
    embedding: Embeddings = MemoMateEmbeddings.local_embedding()
    embedding = embedding.embed_documents(["Hello, world!"])
    assert len(embedding) == 1


def test_openai_embedding():
    embedding: Embeddings = MemoMateEmbeddings.openai_embedding(model_name="")
    embedding = embedding.embed_documents(["Hello, world!"])
    assert len(embedding) == 1
    assert len(embedding[0]) == 1024


def test_qdrant_instance():
    VS = get_qdrant_vector_store()
    logger.info(f"QdrantDB 客户端: {VS.client}")
    assert VS.client is not None, "QdrantDB 客户端未初始化"


def test_qdrant_list_collections():
    VS = get_qdrant_vector_store()
    collections = VS.list_collections()
    logger.info(f"集合列表: {collections}")
    assert len(collections) > 0, "集合列表为空"


def test_qdrant_add_documents():
    VS = get_qdrant_vector_store()
    texts = [
        "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的系统。",
        "机器学习是人工智能的一个子领域，专注于让计算机系统从数据中学习。",
        "深度学习是机器学习的一种方法，使用神经网络进行学习。",
        "自然语言处理是人工智能的一个分支，专注于让计算机理解和生成人类语言。",
        "计算机视觉是人工智能的一个分支，专注于让计算机理解和处理图像和视频。",
    ]
    metadatas = [{"text": text, "source": "AI介绍", "author": "示例作者"} for text in texts]
    documents = [
        Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas, strict=True)
    ]

    VS.add_documents(collection_name="test", documents=documents)
    logger.info("添加文档: 完成")


def test_qdrant_vertor_store_search():
    VS = get_qdrant_vector_store()
    results = VS.search(query="什么是机器学习？", search_type="similarity_score_threshold", score_threshold=0.5)
    logger.info(f"搜索结果: {results}")


def test_qdrant_vertor_store_search_by_metadata():
    VS = get_qdrant_vector_store()
    results = VS.search_by_metadata(metadatas={"source": "AI介绍", "author": "示例作者"})
    logger.info(f"通过元数据搜索结果: {results}")


def test_qdrant_vertor_store_update_vector():
    VS = get_qdrant_vector_store()
    results = VS.search_by_metadata(metadatas={"source": "AI介绍", "author": "示例作者"})
    VS.update_vector(
        collection_name="test",
        id=results[0]["id"],
        text="什么是深度学习？",
        metadata={
            "text": "深度学习是机器学习的二种方法，使用神经网络进行学习。",
            "source": "AI不介绍",
            "author": "示例作者",
        },
    )
    logger.info("更新向量: 完成")


def test_qdrant_vertor_store_delete_collection():
    VS = get_qdrant_vector_store()
    VS.delete_collection(collection_name="test")
    logger.info("删除集合: test")


def test_qdrant_vertor_store_as_retriever():
    VS = get_qdrant_vector_store()
    retriever = VS.as_retriever()
    logger.info(f"检索器: {retriever}")
    results = retriever.get_relevant_documents(query="什么是机器学习？")
    logger.info(f"检索结果: {results}")
