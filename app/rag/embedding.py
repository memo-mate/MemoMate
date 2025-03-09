import uuid
from collections.abc import Generator
from io import BytesIO

from chromadb import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter, MarkdownTextSplitter  # noqa
from rich import inspect, print  # noqa

from app.configs import settings

# 使用模型名称，HuggingFace会自动处理下载和缓存
embeddings = HuggingFaceEmbeddings(
    model_name="/Users/datagrand/Code/agent-demo/bge-large-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"device": "mps"},
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./data/memo_db",
    client_settings=Settings(anonymized_telemetry=False),
)

# collection = vector_store.get_or_create_collection("example_collection")


# 测试embedding模型是否生效
def test_embedding_model():
    # 测试文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的一个子领域",
        "自然语言处理是机器学习的应用",
        "今天天气真不错，我想去公园散步",
    ]

    # 生成文本的向量表示
    embeddings_result = embeddings.embed_documents(texts)

    # 打印结果信息
    print(f"生成的向量数量: {len(embeddings_result)}")
    print(f"每个向量的维度: {len(embeddings_result[0])}")

    # 测试单个文本的向量化
    single_text = "这是一个用于测试的句子"
    single_embedding = embeddings.embed_query(single_text)
    print(f"单个文本的向量维度: {len(single_embedding)}")

    # 可以进一步测试语义相似度
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # 计算文本之间的余弦相似度
    similarities = cosine_similarity(embeddings_result)

    # 打印相似度矩阵
    print("\n文本相似度矩阵:")
    for i in range(len(texts)):
        for j in range(len(texts)):
            print(f"文本 {i + 1} 和文本 {j + 1} 的相似度: {similarities[i][j]:.4f}")

    # 我们期望前三个文本的相似度应该比它们与第四个文本的相似度高
    print("\n验证语义相似度:")
    avg_similarity_first_three = np.mean([similarities[0][1], similarities[0][2], similarities[1][2]])
    avg_similarity_with_fourth = np.mean([similarities[0][3], similarities[1][3], similarities[2][3]])
    print(f"前三个相关文本之间的平均相似度: {avg_similarity_first_three:.4f}")
    print(f"前三个文本与第四个不相关文本的平均相似度: {avg_similarity_with_fourth:.4f}")

    if avg_similarity_first_three > avg_similarity_with_fourth:
        print("验证通过！相关文本的相似度确实更高。")
        return True
    else:
        print("验证失败。请检查模型是否正确加载。")
        return False


def load_documents(dir: str) -> Generator[Document, None, None]:
    loader = DirectoryLoader(dir, glob="**/*.md")
    documents = loader.load()
    if not documents:
        raise ValueError("文档目录为空")
    # 使用 MarkdownTextSplitter 分割文档，chunk 大小为 3200，重叠 30
    text_spliter = MarkdownTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    # 分割文档
    split_docs = text_spliter.split_documents(documents)
    for doc in split_docs:
        # 添加 doc_id
        doc.metadata["doc_id"] = str(uuid.uuid4())
        yield doc


def load_documents_from_io(file: BytesIO) -> Document:
    return Document(
        page_content=file.read(),
        metadata={
            "source": file.name,
            "doc_id": str(uuid.uuid4()),
        },
    )


def main() -> None:
    # need unstructured[md]
    documents = load_documents("./data/Miner2PdfAndWord_Markitdown2Excel")
    print(documents)

    # 向量存储
    vector_store.add_documents(documents)


if __name__ == "__main__":
    main()
