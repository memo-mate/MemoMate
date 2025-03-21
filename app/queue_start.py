import orjson
import rich
from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord

from app.core import consts
from app.core.config import settings
from app.enums.queue import QueueTopic


def paser_consumer() -> None:
    consumer = KafkaConsumer(
        QueueTopic.FILE_PARSING_TASK,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda x: orjson.loads(x) if x else None,
        auto_offset_reset="earliest",  # 从最早的消息开始读取
        enable_auto_commit=True,
        group_id=consts.KAFKA_CONSUMER_PARSER_GROUP_ID,  # 确保设置了消费者组
    )

    try:
        # 设置超时，这样如果没有消息也不会一直阻塞
        for message in consumer:
            message: ConsumerRecord
            # 确保消息值不为空
            if message.value is not None:
                rich.print(f"收到消息: {message.value}")
                rich.print(f"Topic: {message.topic}, Partition: {message.partition}, Offset: {message.offset}")

    except KeyboardInterrupt:
        rich.print("正在优雅关闭消费者...")
    finally:
        consumer.close()
        rich.print("消费者已关闭")
