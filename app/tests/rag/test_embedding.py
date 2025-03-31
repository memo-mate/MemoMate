from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core import settings
from app.core.log_adapter import logger
from app.enums import EmbeddingDriverEnum
from app.rag.embedding.embedding_db import QdrantDB
from app.rag.embedding.embeeding_model import MemoMateEmbeddings


def test_local_embedding():
    embedding: Embeddings = MemoMateEmbeddings.local_embedding()
    embedding = embedding.embed_documents(["Hello, world!"])
    assert len(embedding) == 1
    assert len(embedding[0]) == 1024


def test_openai_embedding():
    embedding: Embeddings = MemoMateEmbeddings.openai_embedding(model_name="")
    embedding = embedding.embed_documents(["Hello, world!"])
    assert len(embedding) == 1
    assert len(embedding[0]) == 1024


def test_vertor_db():
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

    embedding: Embeddings = MemoMateEmbeddings.local_embedding(driver=EmbeddingDriverEnum.MAC)
    db = QdrantDB(embeddings=embedding, url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    collections = db.list_collections()

    logger.info(f"集合列表: {collections}")

    db.create_collection(collection_name="test", vector_size=1024)
    logger.info("创建集合: test")

    db.add_documents(collection_name="test", documents=documents)
    logger.info("添加文档: 完成")

    results = db.search(collection_name="test", query_text="什么是机器学习？")
    logger.info(f"搜索结果: {results}")

    results = db.search_by_metadata(collection_name="test", metadatas={"source": "AI介绍", "author": "示例作者"})
    logger.info(f"通过元数据搜索结果: {results}")

    db.update_vector(
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

    results = db.search(collection_name="test", query_text="什么是机器学习？")
    logger.info(f"重新搜索结果: {results}")

    db.delete_collection(collection_name="test")
    logger.info("删除集合: test")
