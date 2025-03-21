from qdrant_client.http.models import Distance
from rich import print

from app.core.config import settings
from app.core.log_adapter import logger


def example_embed_db() -> None:
    """嵌入数据库示例"""
    logger.info("=== 嵌入数据库示例 ===")

    from app.rag.embedding.embed_db import QdrantDB
    from app.rag.embedding.embeeding_model import SiliconCloudEmbedding

    embedding_model = SiliconCloudEmbedding(
        key=settings.OPENAI_API_KEY,
        model_name=settings.EMBEDDING_MODEL,
        base_url="https://api.siliconflow.cn/v1/embeddings",
    )

    texts = [
        "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的系统。",
        "机器学习是人工智能的一个子领域，专注于让计算机系统从数据中学习。",
        "深度学习是机器学习的一种方法，使用神经网络进行学习。",
        "自然语言处理是人工智能的一个分支，专注于让计算机理解和生成人类语言。",
        "计算机视觉是人工智能的一个分支，专注于让计算机理解和处理图像和视频。",
    ]

    db = QdrantDB(url=settings.QDRANT_URL)

    logger.info("=== 集合列表 ===")
    print(db.list_collections())

    db.delete_collection(collection_name="test")
    db.create_collection(collection_name="test", vector_size=1024, distance=Distance.COSINE)

    metadatas = [{"text": text, "source": "AI介绍", "author": "示例作者"} for text in texts]

    db.add_vectors(collection_name="test", vectors=embedding_model.embed_documents(texts), metadatas=metadatas)

    logger.info("=== 向量查询 ===")
    results = db.search_by_vector(
        collection_name="test", query_vector=embedding_model.embed_query("什么是机器学习？"), limit=3
    )

    print(results)

    logger.info("=== 通过元数据查询 ===")
    results = db.search_by_metadata(collection_name="test", metadatas={"source": "AI介绍", "author": "示例作者"})
    print(results)

    logger.info("=== 更改向量 ===")
    db.update_vector(
        collection_name="test",
        id=results[0]["id"],
        vector=embedding_model.embed_query("什么是深度学习？"),
        metadata={
            "text": "深度学习是机器学习的二种方法，使用神经网络进行学习。",
            "source": "AI不介绍",
            "author": "示例作者",
        },
    )
    logger.info("=== 重新查询 ===")
    results = db.search_by_vector(
        collection_name="test", query_vector=embedding_model.embed_query("什么是机器学习？"), limit=3
    )
    print(results)


def run_examples() -> None:
    """运行所有示例"""
    example_embed_db()


if __name__ == "__main__":
    run_examples()
