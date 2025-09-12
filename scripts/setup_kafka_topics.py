from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import KafkaError
from app.core import settings
from app.enums.queue import QueueTopic
from app.core.log_adapter import logger
import sys


def create_topics():
    """创建所需的 Kafka topics"""

    # 创建 AdminClient
    admin_client = AdminClient({
        "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS
    })

    # 定义要创建的 topics
    topics_to_create = []

    # 主要的 topics
    for topic in QueueTopic:
        topics_to_create.append(
            NewTopic(
                topic=topic.value,
                num_partitions=1,
                replication_factor=1
            )
        )

        # 为每个 topic 创建对应的 DLQ (Dead Letter Queue)
        topics_to_create.append(
            NewTopic(
                topic=f"{topic.value}.dlq",
                num_partitions=1,
                replication_factor=1
            )
        )

    # 创建 topics
    topic_futures = admin_client.create_topics(topics_to_create)

    # 等待创建完成并检查结果
    for topic_name, future in topic_futures.items():
        try:
            future.result()  # 等待创建完成
            logger.info(f"Topic '{topic_name}' 创建成功")
        except Exception as e:
            if "Topic already exists" in str(e) or "TOPIC_ALREADY_EXISTS" in str(e):
                logger.info(f"Topic '{topic_name}' 已存在，跳过创建")
            else:
                logger.error(f"创建 topic '{topic_name}' 失败: {e}")
                return False

    return True


def check_kafka_connection():
    """检查 Kafka 连接"""
    try:
        admin_client = AdminClient({
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS
        })

        # 尝试获取集群元数据
        metadata = admin_client.list_topics(timeout=10)
        logger.info(f"成功连接到 Kafka，broker: {settings.KAFKA_BOOTSTRAP_SERVERS}")
        logger.info(f"现有 topics: {list(metadata.topics.keys())}")
        return True

    except Exception as e:
        logger.error(f"无法连接到 Kafka: {e}")
        logger.error(f"请确保 Kafka 服务正在运行：docker-compose -f kafka.yml up -d")
        return False


def main():
    """主函数"""
    logger.info("开始设置 Kafka topics...")

    # 检查连接
    if not check_kafka_connection():
        sys.exit(1)

    # 创建 topics
    if create_topics():
        logger.info("所有 Kafka topics 设置完成")

        # 列出创建的 topics
        logger.info("已创建的 topics:")
        for topic in QueueTopic:
            logger.info(f"  - {topic.value}")
            logger.info(f"  - {topic.value}.dlq")
    else:
        logger.error("创建 topics 时出现错误")
        sys.exit(1)


if __name__ == "__main__":
    main()
