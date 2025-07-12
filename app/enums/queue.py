from enum import StrEnum


class QueueTopic(StrEnum):
    """
    队列Topic
    """

    FILE_PARSING_TASK = "document_parser"
    SUMMARY_GENERATOR = "summary_generator"
    EXTERNAL_CRAWLER = "external_crawler"
