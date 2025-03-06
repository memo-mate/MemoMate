import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise


# 测试embedding模型是否生效
def test_embedding_model():
    # 使用模型名称，HuggingFace会自动处理下载和缓存
    embeddings = HuggingFaceEmbeddings(
        model_name="/Users/datagrand/Code/agent-demo/bge-large-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "mps"},
        # 注意：HuggingFaceEmbeddings不支持model_path参数
        # 如果您已下载模型，确保它位于默认的HuggingFace缓存路径中
        # 通常是~/.cache/huggingface/hub/
    )

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

    # 计算文本之间的余弦相似度
    similarities = pairwise.cosine_similarity(embeddings_result)

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
