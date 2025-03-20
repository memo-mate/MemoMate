import orjson
import rich
from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord


def producer_demo() -> None:
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda x: orjson.dumps(x),
    )
    for _ in range(100):
        producer.send("foobar", {"message": "hello kafka"})


def producer_demo2() -> None:
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
    )

    for _ in range(100):
        producer.send("foobar", b"hello kafka")


def consumer_demo() -> None:
    consumer = KafkaConsumer(
        "foobar",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda x: orjson.loads(x) if x else None,
        auto_offset_reset="earliest",  # 从最早的消息开始读取
        enable_auto_commit=True,
        group_id="test",  # 确保设置了消费者组
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


if __name__ == "__main__":
    producer_demo()
    consumer_demo()
