import logging

import orjson
import rich
import tenacity
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import TopicPartition

from app.core.config import settings
from app.core.log_adapter import logger
from app.enums.queue import QueueTopic


def get_topic_lag(topic: QueueTopic, groups: list[str]) -> int:
    """获取Topic的Lag"""
    if not groups:
        # 如果未指定消费者组，则返回0
        logger.warning("未指定消费者组，无法计算滞后量")
        return 0

    total_lag = 0
    try:
        # 创建消费者来获取主题的分区信息
        consumer = KafkaConsumer(bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS)

        # 获取主题的分区
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            logger.warning(f"主题 {topic} 不存在或没有分区")
            return 0

        # 为每个分区创建TopicPartition对象
        topic_partitions = [TopicPartition(topic=topic, partition=p) for p in partitions]

        # 获取每个分区的开始和结束偏移量
        beginning_offsets = consumer.beginning_offsets(topic_partitions)  # 最早可用偏移量
        end_offsets = consumer.end_offsets(topic_partitions)  # 最新偏移量

        # 对每个消费者组进行计算
        for group in groups:
            # 创建消费者来获取特定组的偏移量
            group_consumer = KafkaConsumer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS, group_id=group, request_timeout_ms=30000
            )

            # 获取消费者组当前的偏移量
            committed_offsets = {}
            group_exists = True

            try:
                # 尝试获取已提交的偏移量
                for tp in topic_partitions:
                    offset = group_consumer.committed(tp)
                    committed_offsets[tp] = offset if offset is not None else beginning_offsets[tp]
            except Exception as e:
                # 如果出错（例如消费者组不存在），则假设消费者组从未消费过任何消息
                logger.warning(f"无法获取消费者组 {group} 的提交偏移量，假设从未消费", group=group, exc_info=e)
                group_exists = False
                for tp in topic_partitions:
                    committed_offsets[tp] = beginning_offsets[tp]  # 使用最早的偏移量

            # 计算每个分区的lag并求和
            group_lag = 0
            for tp in topic_partitions:
                lag = end_offsets[tp] - committed_offsets[tp]
                group_lag += lag
                logger.debug(f"分区 {tp.partition} 的滞后量: {lag}", partition=tp.partition, lag=lag, group=group)

            # 如果消费者组不存在，添加提示信息
            if not group_exists:
                logger.info(
                    f"消费者组 {group} 可能不存在或未曾连接，总滞后量: {group_lag}",
                    group=group,
                    total_lag=group_lag,
                )

            total_lag += group_lag
            group_consumer.close()

    except Exception as e:
        logger.exception("计算主题滞后量时出错", topic=topic, exc_info=e)
    finally:
        consumer.close()

    return total_lag


def get_topic_lag_by_group(topic: QueueTopic, group: str) -> int:
    """获取指定消费者组的滞后量"""
    return get_topic_lag(topic, [group])


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
def send_message_to_topic(topic: QueueTopic, message: dict) -> None:
    """发送消息到指定主题"""
    producer = KafkaProducer(
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda x: orjson.dumps(x),
    )
    producer.send(topic, message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    rich.print(f"🚀 [red]滞后量: {get_topic_lag('foobar', ['test'])}[/red]")
