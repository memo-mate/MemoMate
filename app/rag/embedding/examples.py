import json
import os
import uuid

# 禁用LangSmith追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "not-needed"

from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rich import print

from app.core.config import settings

# from app.rag.embedding.evaluation import RetrievalEvaluator
# from app.rag.embedding.index_manager import IndexManager
# from app.rag.embedding.retriever import MultiQueryRetriever, QdrantRetriever
# from app.rag.embedding.vector_search import HuggingFaceEmbeddings
# from app.rag.embedding.vector_store import QdrantStore


def example_embedding() -> None:
    """嵌入示例"""
    print("=== 嵌入示例 ===")

    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

    # 嵌入文本
    text = "这是一个测试文本，用于演示嵌入功能。"
    embedding = embedding_model.embed_query(text)

    print(f"文本: {text}")
    print(f"嵌入维度: {len(embedding)}")

    # 嵌入多个文本
    texts = [
        "这是第一个测试文本。",
        "这是第二个测试文本。",
        "这是第三个测试文本。",
    ]
    embeddings = embedding_model.embed_documents(texts)

    print(f"文本数量: {len(texts)}")
    print(f"嵌入数量: {len(embeddings)}")
    print(f"每个嵌入的维度: {len(embeddings[0])}")


def example_document_processing() -> None:
    """文档处理示例"""
    print("\n=== 文档处理示例 ===")

    # 初始化文档处理器
    processor = DocumentProcessor(
        chunk_size=100,
        chunk_overlap=20,
    )

    # 处理文本
    text = """
    这是一个长文本示例，用于演示文档处理功能。
    文档处理器可以将长文本分割成小块，以便于后续处理。
    每个块都有一定的重叠，以保持上下文连贯性。
    这对于大型文档的处理非常有用。
    """

    chunks = processor.split_text(text, {"source": "示例文本"})

    print(f"原始文本长度: {len(text)}")
    print(f"分块数量: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"块 {i + 1}:")
        print(f"内容: {chunk.page_content}")
        print(f"元数据: {chunk.metadata}")
        print()


def example_vector_store() -> None:
    """向量存储示例"""
    print("\n=== 向量存储示例 ===")

    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

    # 初始化向量存储
    vector_store = QdrantStore(
        path=settings.VECTOR_STORE_PATH,
        collection_name=settings.DEFAULT_COLLECTION_NAME,
        embeddings=embedding_model,
    )

    # 添加文本
    texts = [
        "北京是中国的首都，是政治、文化、国际交往中心。",
        "上海是中国最大的城市，是经济、金融、贸易、航运中心。",
        "广州是中国南方的重要城市，是华南地区的商业中心。",
        "深圳是中国改革开放的窗口，是创新创业的热土。",
    ]

    metadatas = [
        {"city": "北京", "type": "capital"},
        {"city": "上海", "type": "economic"},
        {"city": "广州", "type": "commercial"},
        {"city": "深圳", "type": "innovation"},
    ]

    ids = vector_store.add_texts(texts, metadatas)

    print(f"添加的文本数量: {len(texts)}")
    print(f"生成的ID: {ids[:5]}")

    # 相似度搜索
    query = "中国的经济中心"
    results = vector_store.similarity_search(query, k=2)

    print(f"查询: {query}")
    print(f"结果数量: {len(results)}")

    for i, doc in enumerate(results):
        print(f"结果 {i + 1}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
        print()

    # 带分数的相似度搜索
    results_with_scores = vector_store.similarity_search_with_score(query, k=2)

    print(f"带分数的结果数量: {len(results_with_scores)}")

    for i, (doc, score) in enumerate(results_with_scores):
        print(f"结果 {i + 1}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
        print(f"分数: {score}")
        print()

    # 删除集合
    vector_store.delete_collection()
    print("已删除集合")


def example_retriever() -> None:
    """检索器示例"""
    print("\n=== 检索器示例 ===")

    # 初始化Qdrant客户端 - 使用内存模式
    client = QdrantClient(location=":memory:", prefer_grpc=True)
    print(f"客户端ID: {id(client)}")

    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

    # 测试文本
    texts = [
        "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的系统。",
        "机器学习是人工智能的一个子领域，专注于让计算机系统从数据中学习。",
        "深度学习是机器学习的一种方法，使用神经网络进行学习。",
        "自然语言处理是人工智能的一个分支，专注于让计算机理解和生成人类语言。",
        "计算机视觉是人工智能的一个分支，专注于让计算机理解和处理图像和视频。",
    ]

    metadatas = [
        {"topic": "AI", "level": "general"},
        {"topic": "ML", "level": "intermediate"},
        {"topic": "DL", "level": "advanced"},
        {"topic": "NLP", "level": "intermediate"},
        {"topic": "CV", "level": "intermediate"},
    ]

    # 直接使用Qdrant客户端API创建集合和添加数据
    import json
    import uuid

    from qdrant_client.http import models as rest

    collection_name = settings.DEFAULT_COLLECTION_NAME
    vector_size = len(embedding_model.embed_query("测试查询"))

    # 创建集合
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )
        print(f"已创建集合: {collection_name}")
    except Exception as e:
        print(f"创建集合时出错: {str(e)}")
        # 如果集合已存在，则删除并重新创建
        try:
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )
            print(f"已删除并重新创建集合: {collection_name}")
        except Exception as e:
            print(f"删除并重新创建集合时出错: {str(e)}")

    # 生成嵌入
    embeddings = embedding_model.embed_documents(texts)

    # 准备点
    points = []
    for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings, strict=False)):
        point = rest.PointStruct(
            id=str(uuid.uuid4()),  # 使用UUID作为ID
            vector=embedding,
            payload={"text": text, "metadata": json.dumps(metadata)},
        )
        points.append(point)

    # 添加点
    try:
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        print(f"已添加 {len(points)} 个点到集合")
    except Exception as e:
        print(f"添加点时出错: {str(e)}")

    # 打印集合信息
    try:
        collection_info = client.get_collection(collection_name)
        print(f"集合信息: {collection_info}")
    except Exception as e:
        print(f"获取集合信息时出错: {str(e)}")

    # 初始化Qdrant检索器 - 使用相同的客户端实例
    retriever = QdrantRetriever(
        client=client,  # 使用相同的客户端实例
        collection_name=collection_name,
        embedding_model=embedding_model,
        top_k=3,
    )

    # 检索文档
    query = "什么是机器学习？"
    docs = retriever.invoke(query)

    print(f"查询: {query}")
    print(f"结果数量: {len(docs)}")

    for i, doc in enumerate(docs):
        print(f"结果 {i + 1}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
        print()

    # 初始化多查询检索器
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
        temperature=0,
    )
    print("多查询检索测试")

    # 初始化多查询检索器 - 使用相同的检索器
    multi_retriever = MultiQueryRetriever(
        client=client,  # 使用相同的客户端实例
        retriever=retriever,
        llm=llm,
        query_count=2,
    )

    # 使用多查询检索器
    query = "AI技术的应用"
    docs = multi_retriever.invoke(query)

    print(f"多查询: {query}")
    print(f"结果数量: {len(docs)}")

    for i, doc in enumerate(docs):
        print(f"结果 {i + 1}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
        print()

    # 清理资源
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"已删除集合 {collection_name}")
    finally:
        client.close()


def example_index_manager() -> None:
    """索引管理器示例"""
    print("\n=== 索引管理器示例 ===")

    # 初始化索引管理器
    index_manager = IndexManager(
        vector_store_path=settings.VECTOR_STORE_PATH,
        embedding_model_path=settings.EMBEDDING_MODEL_PATH,
        collection_name=settings.DEFAULT_COLLECTION_NAME,
    )

    # 索引文本
    text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    致力于开发能够模拟人类智能的系统。AI技术包括机器学习、深度学习、
    自然语言处理、计算机视觉等多个领域。

    机器学习是AI的核心技术之一，它使计算机系统能够从数据中学习，
    而无需显式编程。深度学习是机器学习的一种方法，使用多层神经网络进行学习。
    """

    metadata = {"source": "AI介绍", "author": "示例作者"}

    result = index_manager.index_text(text, metadata)

    print("索引文本结果:")
    print(result)

    # 获取索引统计信息
    stats = index_manager.get_index_stats()

    print("\n索引统计信息:")
    print(stats)

    # 删除索引
    delete_result = index_manager.delete_index()

    print("\n删除索引结果:")
    print(delete_result)


def example_evaluation() -> None:
    """评估示例"""
    print("\n=== 评估示例 ===")

    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

    # 初始化向量存储
    vector_store = QdrantStore(
        path=settings.VECTOR_STORE_PATH,
        collection_name=settings.DEFAULT_COLLECTION_NAME,
        embeddings=embedding_model,
    )

    # 添加文本
    texts = [
        "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的系统。",
        "机器学习是人工智能的一个子领域，专注于让计算机系统从数据中学习。",
        "深度学习是机器学习的一种方法，使用神经网络进行学习。",
        "自然语言处理是人工智能的一个分支，专注于让计算机理解和生成人类语言。",
        "计算机视觉是人工智能的一个分支，专注于让计算机理解和处理图像和视频。",
    ]

    metadatas = [
        {"topic": "AI", "level": "general"},
        {"topic": "ML", "level": "intermediate"},
        {"topic": "DL", "level": "advanced"},
        {"topic": "NLP", "level": "intermediate"},
        {"topic": "CV", "level": "intermediate"},
    ]

    vector_store.add_texts(texts, metadatas)

    # 生成唯一集合名称
    collection_name = f"example_{uuid.uuid4().hex[:8]}"
    client = QdrantClient(location=":memory:", prefer_grpc=True)

    # 创建集合并添加数据
    client.create_collection(
        collection_name=collection_name,
        vectors_config={"size": 1024, "distance": "Cosine"},
    )

    # 将向量添加到内存中的集合
    embeddings = embedding_model.embed_documents(texts)
    points = []
    for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings, strict=False)):
        point = models.PointStruct(id=i, vector=embedding, payload={"text": text, "metadata": json.dumps(metadata)})
        points.append(point)

    client.upload_points(
        collection_name=collection_name,
        points=points,
    )

    # 初始化检索器
    retriever = QdrantRetriever(
        client=client,
        collection_name=collection_name,
        top_k=3,
    )

    # 初始化评估器
    evaluator = RetrievalEvaluator(retriever=retriever)

    # 评估查询
    queries = [
        "什么是机器学习？",
        "深度学习和神经网络有什么关系？",
    ]

    relevant_docs = [
        [texts[1], texts[2]],  # 与"什么是机器学习？"相关的文档
        [texts[2]],  # 与"深度学习和神经网络有什么关系？"相关的文档
    ]

    # 评估精确率
    precision_results = evaluator.evaluate_precision(queries, relevant_docs)

    print("精确率评估结果:")
    print(precision_results)

    # 评估召回率
    recall_results = evaluator.evaluate_recall(queries, relevant_docs)

    print("\n召回率评估结果:")
    print(recall_results)

    # 评估延迟
    latency_results = evaluator.evaluate_latency(queries, runs=2)

    print("\n延迟评估结果:")
    print(latency_results)

    # 评估所有指标
    all_results = evaluator.evaluate_all(queries, relevant_docs, runs=2)

    print("\n所有指标评估结果:")
    print(all_results)

    # 删除集合
    vector_store.delete_collection()
    print("\n已删除集合")
    client.close()


def example_embed_db() -> None:
    """嵌入数据库示例"""
    print("\n=== 嵌入数据库示例 ===")

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

    db = QdrantDB()
    db.create_collection(collection_name="test", vector_size=1024, distance="Cosine")

    embeddings = embedding_model.embed_documents(texts)
    metadatas = [{"text": text, "source": "AI介绍", "author": "示例作者"} for text in texts]

    db.add_vectors(collection_name="test", vectors=embeddings, metadatas=metadatas)

    results = db.search_by_vector(
        collection_name="test", query_vector=embedding_model.embed_query("什么是机器学习？"), limit=3
    )

    print(results)


def run_examples() -> None:
    """运行所有示例"""
    example_embed_db()

    # 运行示例
    # example_embedding()
    # example_document_processing()
    # example_vector_store()
    # example_retriever()
    # example_index_manager()

    # # 添加延迟，确保资源被释放
    # time.sleep(1)

    # example_evaluation()


if __name__ == "__main__":
    run_examples()
