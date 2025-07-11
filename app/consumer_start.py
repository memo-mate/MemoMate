from typing import Literal, TypedDict

import orjson
import rich
from confluent_kafka import Consumer, Message, Producer

from app.core import settings
from app.core.log_adapter import logger
from app.document_parsing.markdown_parser import parse_markdown


class DocumentParserDict(TypedDict):
    id: str
    file_path: str
    task_type: Literal["document_parser", "summary_generator", "external_crawler"]
    retry_count: int


class ConsumerFactory:
    def __init__(self, topic: str):
        self.group_id = "document-parser-group"
        self.topic = topic
        self.dlq_topic = f"{topic}.dlq"
        self.consumer = Consumer(
            {
                "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
                "group.id": self.group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": "false",  # this allows to easily replay the same events in development
            }
        )
        self.producer = Producer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})
        self.consumer.subscribe([topic])

    def __enter__(self) -> "Consumer":
        logger.info(f"about to start {self.consumer} consuming messages from {self.topic} in group {self.group_id}")
        return self

    def __del__(self):
        self.consumer.close()
        logger.info(f"consumer {self.consumer} closed")

    def decode_message(self, msg: Message) -> dict:
        data = msg.value()
        try:
            data = orjson.loads(data)
        except orjson.JSONDecodeError:
            logger.error(f"Invalid JSON: {data}")
            self.producer.produce(self.dlq_topic, msg.value())
            return None
        return data

    def handle_task(self, data: DocumentParserDict) -> bool:
        success = False
        try:
            task_type = data["task_type"]
            match task_type:
                case "document_parser":
                    parse_markdown(data["file_path"])
                case "summary_generator":
                    # TODO: process summary generator
                    pass
                case "external_crawler":
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
                self.producer.produce(self.dlq_topic, orjson.dumps(data))
            else:
                data["retry_count"] = retry_count - 1
                self.producer.produce(self.topic, orjson.dumps(data))
        return success

    def run(self) -> None:
        logger.info("about to start consuming messages")

        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    pass
                elif msg.error():
                    logger.error(f"ERROR: {msg.error()}")
                else:
                    data = self.decode_message(msg)
                    if data is None:
                        continue
                    else:
                        rich.print(f"Processing message: {data}")

                    success = self.handle_task(data)
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
