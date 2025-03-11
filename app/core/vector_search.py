from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
import lancedb
import json
from langchain_core.documents import Document
from app.core.config import MODEL_PATH, DB_PATH, TABLE_NAME, EMBEDDING_CONFIG


class VectorSearch:
    def __init__(self):
        # 初始化嵌入模型
        self.embedding = HuggingFaceEmbeddings(
            model_name=str(MODEL_PATH), **EMBEDDING_CONFIG
        )

        # 连接数据库
        self.connection = lancedb.connect(DB_PATH)

        # 创建向量库
        try:
            # 检查表是否存在
            if TABLE_NAME not in self.connection.table_names():
                # 创建空表
                empty_data = [
                    {
                        "vector": self.embedding.embed_query("init"),
                        "text": "init",
                        "id": "0",
                    }
                ]
                self.connection.create_table(TABLE_NAME, data=empty_data)

            # 获取表引用
            self.table = self.connection.open_table(TABLE_NAME)

            self.vector_store = LanceDB(
                connection=self.connection,
                embedding=self.embedding,
                table_name=TABLE_NAME,
            )
        except Exception as e:
            raise RuntimeError(f"数据库初始化失败: {str(e)}")

    def _process_results(self, results):
        """处理搜索结果,将JSON字符串解析为字典"""
        processed_results = []
        for item in results:
            text = item["text"]
            # 将距离转换为相似度：1/(1+distance)，这样距离越小，相似度越接近1
            distance = item["_distance"]
            similarity = 1 / (1 + distance)  # 转换为0-1之间的相似度值
            metadata = {"source": item["source"], "id": item["id"]}

            # 如果metadata字段是JSON字符串，解析它
            if "metadata" in item and item["metadata"]:
                try:
                    metadata_dict = json.loads(item["metadata"])
                    metadata.update(metadata_dict)
                except Exception:
                    # 如果解析失败，保留原始字符串
                    metadata["metadata"] = item["metadata"]

            doc = Document(page_content=text, metadata=metadata)
            processed_results.append((doc, similarity))

        return processed_results

    def as_retriever(self, k=5):
        """
        返回一个retriever对象，包含get_relevant_documents方法

        Args:
            k: 返回的结果数量

        Returns:
            包含get_relevant_documents方法的retriever对象
        """

        class VectorSearchRetriever:
            def __init__(self, vector_search, k):
                self.vector_search = vector_search
                self.k = k

            def get_relevant_documents(self, query):
                results = self.vector_search.similarity_search(query, k=self.k)
                # 只返回文档部分，不返回相似度分数
                return [doc for doc, _ in results]

        return VectorSearchRetriever(self, k)

    def vector_exists(self, text: str) -> bool:
        results = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 1, "score_threshold": 0.99}
        ).invoke(text)
        return len(results) > 0

    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.2):
        # 使用embedding模型获取查询向量
        query_vector = self.embedding.embed_query(query)

        # 直接使用LanceDB的原生API进行搜索
        results = self.table.search(query_vector).limit(k).to_pandas()

        # 处理结果
        return self._process_results(results.to_dict("records"))


# 使用示例
if __name__ == "__main__":
    searcher = VectorSearch()
    query = "CTE导致节点宕机"
    results = searcher.similarity_search(query)

    print(f"提问：{query}")
    for i, (doc, similarity) in enumerate(results):
        print(f"结果{i+1}：  （相似度：{similarity:.3f}）:")
        print(f"内容：  {doc.page_content}")
        print(f'来源：  {doc.metadata["source"]}')
        if "page" in doc.metadata:
            print(f'页码：  {doc.metadata["page"]}')
        print("-" * 100)
