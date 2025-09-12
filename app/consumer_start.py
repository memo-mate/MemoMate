from typing import Self, TypedDict

import orjson
import rich
from confluent_kafka import Consumer, Message, Producer

from app.core import consts, settings
from app.core.log_adapter import logger
from app.document_parsing.markdown_parser import parse_markdown
from app.enums import QueueTopic
from app.enums.embedding import EmbeddingDriverEnum
from app.rag.embedding.embeeding_model import EmbeddingFactory


class DocumentParserDict(TypedDict):
    id: str
    file_path: str
    task_type: QueueTopic
    retry_count: int


class ConsumerFactory:
    def __init__(self):
        self.group_id = consts.KAFKA_CONSUMER_PARSER_GROUP_ID
        self.topics = [QueueTopic.FILE_PARSING_TASK]
        self.dlq_topics = [f"{topic}.dlq" for topic in self.topics]
        self.consumer = Consumer(
            {
                "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
                "group.id": self.group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": "true",
            }
        )
        self.producer = Producer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})

    def __enter__(self) -> Self:
        # 检查 kafka 是否启动
        try:
            metadata = self.consumer.list_topics(timeout=10)
            logger.info("成功连接到 kafka", topics=list(metadata.topics.keys()))

            # 检查目标 topic 是否存在
            topics_not_found = list(set(self.topics) - set(metadata.topics.keys()))
            if topics_not_found:
                logger.error(f"Topic '{topics_not_found}' 不存在！")
                logger.error("请先运行: uv run python scripts/setup_kafka_topics.py")
                raise Exception(f"Topic '{topics_not_found}' does not exist")

        except Exception as e:
            logger.error(f"Kafka 连接或 topic 检查失败: {e}")
            logger.error("请确保：")
            logger.error("1. Kafka 服务正在运行: docker-compose -f kafka.yml up -d")
            logger.error("2. Topics 已创建: uv run python scripts/setup_kafka_topics.py")
            raise

        # 订阅 topic
        self.consumer.subscribe(self.topics)

        if not consts.MD_SOURCE_DIR.exists():
            consts.MD_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "about to start consuming messages",
            group_id=self.group_id,
            topics=[topic.value for topic in self.topics],
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.consumer.close()
        logger.info(f"consumer {self.consumer} closed")

    def decode_message(self, msg: Message, topic: QueueTopic) -> dict:
        data = msg.value()
        try:
            data = orjson.loads(data)
        except orjson.JSONDecodeError:
            logger.error(f"Invalid JSON: {data}")
            self.producer.produce(f"{topic}.dlq", msg.value())
            return None
        return data

    def handle_task(self, data: DocumentParserDict, topic: QueueTopic) -> bool:
        success = False
        try:
            task_type = data["task_type"]
            match task_type:
                case QueueTopic.FILE_PARSING_TASK:
                    parse_markdown(data["file_path"])
                case QueueTopic.SUMMARY_GENERATOR:
                    # TODO: process summary generator
                    pass
                case QueueTopic.EXTERNAL_CRAWLER:
                    # TODO: process external crawler
                    pass
                case _:
                    logger.error(f"Unknown task type: {task_type}")
            success = True
        except Exception as e:
            logger.error(f"Error processing task: {e}")

        if not success:
            # send to dlq if retry_count is 0
            retry_count = data["retry_count"]
            if retry_count == 0:
                self.producer.produce(f"{topic}.dlq", orjson.dumps(data))
            else:
                data["retry_count"] = retry_count - 1
                self.producer.produce(topic, orjson.dumps(data))
        return success

    def run(self) -> None:
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    pass
                elif msg.error():
                    logger.error(f"ERROR: {msg.error()}")
                else:
                    data = self.decode_message(msg, msg.topic())
                    if data is None:
                        continue
                    else:
                        rich.print(f"Processing message: {data}")

                    success = self.handle_task(data, msg.topic())
                    logger.info(f"Task {data['id']} processed: {success}")
                    # 打印消息的 topic / partition / offset / key / value
                    # rich.print(
                    #     f"Received message: "
                    #     f"topic={msg.topic()} "
                    #     f"partition={msg.partition()} "
                    #     f"offset={msg.offset()} "
                    #     f"key={msg.key()} "
                    #     f"value={msg.value().decode('utf-8')}"
                    # )
                    self.consumer.commit()

        except Exception as e:
            logger.info(f"consumer ending: {e}")


def main() -> None:
    EmbeddingFactory.init(
        {
            "provider": "huggingface",
            "model": "BAAI/bge-m3",
            "driver": EmbeddingDriverEnum.MAC,
        }
    )
    with ConsumerFactory() as consumer:
        consumer.run()


if __name__ == "__main__":
    main()
