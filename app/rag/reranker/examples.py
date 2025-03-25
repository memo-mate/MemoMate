"""重排序器使用示例"""

from langchain_core.documents import Document

from app.core.config import settings
from app.core.log_adapter import logger
from app.rag.reranker.cross_encoder import SiliconCloudCrossEncoderReranker


def main() -> None:
    """运行重排序器示例"""
    logger.info("开始运行重排序器示例")

    # 创建示例文档
    documents = [
        Document(page_content="北京是中国的首都，有故宫、长城等著名景点。", metadata={"source": "cities", "id": 1}),
        Document(
            page_content="上海是中国的经济中心，有东方明珠、外滩等地标建筑。", metadata={"source": "cities", "id": 2}
        ),
        Document(page_content="广州是中国南方的重要城市，有白云山和珠江风光。", metadata={"source": "cities", "id": 3}),
        Document(
            page_content="深圳是中国改革开放的窗口，是重要的科技创新中心。", metadata={"source": "cities", "id": 4}
        ),
        Document(page_content="成都是四川省的省会，有大熊猫基地和宽窄巷子。", metadata={"source": "cities", "id": 5}),
    ]
    logger.info("创建了示例文档", count=len(documents))

    # 设置查询
    query = "中国的经济中心在哪里？"
    logger.info("设置查询", query=query)

    # 测试API交叉编码器重排序器
    logger.info("测试硅基流动API交叉编码器重排序器")
    silicon_cloud_cross_encoder = SiliconCloudCrossEncoderReranker(
        api_key=settings.OPENAI_API_KEY,
        base_url="https://api.siliconflow.cn/v1/rerank",
        model_name="BAAI/bge-reranker-v2-m3",
        top_k=3,
    )

    silicon_cloud_reranked_results = silicon_cloud_cross_encoder.rerank(query, documents)

    logger.info("硅基流动API交叉编码器重排序结果")
    for i, doc in enumerate(silicon_cloud_reranked_results, 1):
        logger.info(f"结果 {i}", content=doc.page_content, score=doc.metadata.get("rerank_score", "无分数"))

    logger.info("示例运行完成")


if __name__ == "__main__":
    main()
