"""嵌入模块测试脚本"""

import os
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from rich import print

from app.core.config import settings
from app.rag.embedding.document_processor import DocumentProcessor
from app.rag.embedding.retriever import QdrantRetriever
from app.rag.embedding.vector_search import HuggingFaceEmbeddings
from app.rag.embedding.vector_store import QdrantStore


def get_unique_test_path(prefix: str) -> str:
    """生成唯一的测试路径"""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(settings.VECTOR_STORE_PATH, f"{prefix}_{timestamp}_{unique_id}")


def test_embedding() -> bool:
    """测试嵌入功能"""
    print("=== 测试嵌入功能 ===")
    print(f"使用的嵌入模型路径: {settings.EMBEDDING_MODEL_PATH}")

    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

    # 测试文本
    text = "这是一个测试文本，用于测试嵌入功能。"

    # 计时开始
    start_time = time.time()

    # 嵌入文本
    embedding = embedding_model.embed_query(text)

    # 计时结束
    end_time = time.time()

    print(f"文本: {text}")
    print(f"嵌入维度: {len(embedding)}")
    print(f"嵌入耗时: {end_time - start_time:.4f} 秒")

    return True


def test_vector_store() -> bool:
    """测试向量存储功能"""
    print("\n=== 测试向量存储功能 ===")

    try:
        # 初始化嵌入模型
        embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

        print(f"嵌入模型初始化成功: {embedding_model.model_name}")

        # 初始化向量存储 - 使用内存模式
        vector_store = QdrantStore(
            path=None, url=None, collection_name=settings.DEFAULT_COLLECTION_NAME, embedding_model=embedding_model
        )

        # 测试文本
        texts = [
            "北京是中国的首都，是政治、文化、国际交往中心。",
            "上海是中国最大的城市，是经济、金融、贸易、航运中心。",
            "广州是中国南方的重要城市，是华南地区的商业中心。",
            "深圳是中国改革开放的窗口，是创新创业的热土。",
        ]

        # 添加文本
        ids = vector_store.add_texts(texts)

        print(f"添加的文本数量: {len(texts)}")
        print(f"生成的ID: {ids}")

        # 相似度搜索
        query = "中国的经济中心"
        results = vector_store.similarity_search(query, k=2)

        print(f"查询: {query}")
        print(f"结果数量: {len(results)}")

        for i, doc in enumerate(results):
            print(f"结果 {i+1}: {doc.page_content}")

        # 删除集合
        vector_store.delete_collection()
        print("已删除集合")

        return True
    except Exception as e:
        print(f"测试向量存储时出错: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_retriever() -> bool:
    """测试检索器功能"""
    print("\n=== 测试检索器功能 ===")

    try:
        # 初始化嵌入模型
        embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

        print(f"嵌入模型初始化成功: {embedding_model.model_name}")

        # 初始化Qdrant客户端 - 使用内存模式
        client = QdrantClient(location=":memory:")

        # 手动创建集合
        collection_name = settings.DEFAULT_COLLECTION_NAME
        dimension = len(embedding_model.embed_query("测试查询"))

        # 检查集合是否存在
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            # 创建集合
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=dimension,
                    distance=rest.Distance.COSINE,
                ),
            )
            print(f"已手动创建集合: {collection_name}")

        # 测试文本
        texts = [
            "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的系统。",
            "机器学习是人工智能的一个子领域，专注于让计算机系统从数据中学习。",
            "深度学习是机器学习的一种方法，使用神经网络进行学习。",
            "自然语言处理是人工智能的一个分支，专注于让计算机理解和生成人类语言。",
            "计算机视觉是人工智能的一个分支，专注于让计算机理解和处理图像和视频。",
        ]

        # 生成嵌入
        embeddings = embedding_model.embed_documents(texts)

        # 准备点
        import json
        import uuid

        points = [
            rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": json.dumps({"source": "测试"}),
                },
            )
            for text, embedding in zip(texts, embeddings, strict=False)
        ]

        # 添加点
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        print(f"已添加 {len(texts)} 条文本到集合")

        # 初始化检索器 - 直接传递嵌入模型
        retriever = QdrantRetriever(
            client=client,
            collection_name=collection_name,
            embedding_model=embedding_model,  # 直接传递嵌入模型对象
            top_k=settings.RETRIEVER_TOP_K,
        )

        # 测试查询
        query = "什么是机器学习？"

        # 计时开始
        start_time = time.time()

        # 检索文档
        docs = retriever.invoke(query)

        # 计时结束
        end_time = time.time()

        print(f"查询: {query}")
        print(f"结果数量: {len(docs)}")
        print(f"检索耗时: {end_time - start_time:.4f} 秒")

        for i, doc in enumerate(docs):
            print(f"结果 {i+1}: {doc.page_content}")

        # 删除集合
        client.delete_collection(collection_name=collection_name)
        print("已删除集合")

        return True
    except Exception as e:
        print(f"测试检索器时出错: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_document_processor() -> bool:
    """测试文档处理器功能"""
    print("\n=== 测试文档处理器功能 ===")

    # 初始化文档处理器
    processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # 测试文本
    text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    致力于开发能够模拟人类智能的系统。AI技术包括机器学习、深度学习、
    自然语言处理、计算机视觉等多个领域。
    
    机器学习是AI的核心技术之一，它使计算机系统能够从数据中学习，
    而无需显式编程。深度学习是机器学习的一种方法，使用多层神经网络进行学习。
    
    自然语言处理（NLP）使计算机能够理解、解释和生成人类语言。
    计算机视觉则使计算机能够从图像或视频中获取信息并理解视觉世界。
    
    人工智能在医疗、金融、教育、交通等多个领域有广泛应用，
    并且随着技术的发展，其应用范围还在不断扩大。
    """

    # 计时开始
    start_time = time.time()

    # 处理文本
    chunks = processor.split_text(text, {"source": "测试文本"})

    # 计时结束
    end_time = time.time()

    print(f"原始文本长度: {len(text)}")
    print(f"分块数量: {len(chunks)}")
    print(f"处理耗时: {end_time - start_time:.4f} 秒")

    for i, chunk in enumerate(chunks):
        print(f"块 {i+1}: {chunk.page_content[:50]}...")

    return True


def run_tests() -> None:
    """运行所有测试"""
    # 创建数据库目录
    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)

    # 运行测试
    tests = [
        test_embedding,
        test_vector_store,
        test_retriever,
        test_document_processor,
    ]

    results = []

    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"测试 {test.__name__} 失败: {str(e)}")
            results.append((test.__name__, False))

    # 打印测试结果
    print("\n=== 测试结果 ===")
    for name, result in results:
        status = "通过" if result else "失败"
        print(f"{name}: {status}")


if __name__ == "__main__":
    run_tests()
