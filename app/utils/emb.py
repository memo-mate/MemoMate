from fastapi import HTTPException, status
from qdrant_client.http.models import Filter

from app.core.config import settings
from app.core.log_adapter import logger
from app.rag.embedding.embedding_db import QdrantVectorStore
from app.rag.embedding.embeeding_model import EmbeddingFactory


def get_vector_store(collection_name: str) -> QdrantVectorStore:
    """获取向量存储实例

    Args:
        collection_name: 集合名称

    Returns:
        QdrantVectorStore: 向量存储实例

    Raises:
        HTTPException: 创建向量存储实例失败时抛出
    """
    try:
        embedding_model = EmbeddingFactory.get()

        vector_store = QdrantVectorStore(
            collection_name=collection_name,
            embeddings=embedding_model,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        return vector_store
    except Exception as e:
        logger.exception("创建向量存储实例失败", exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"创建向量存储实例失败: {str(e)}")


def convert_dict_to_qdrant_filter(filter_dict: dict | None) -> Filter | None:
    """将字典格式的过滤条件转换为Qdrant的Filter对象

    Args:
        filter_dict: 字典格式的过滤条件

    Returns:
        Filter | None: Qdrant的Filter对象，或者None
    """
    if not filter_dict:
        return None

    try:
        # 这里需要根据实际情况实现转换逻辑
        # 这是一个简化的示例，实际应用中可能需要更复杂的转换
        return Filter(**filter_dict)
    except Exception as e:
        logger.warning(f"转换过滤条件失败: {str(e)}", filter_dict=filter_dict)
        return None
