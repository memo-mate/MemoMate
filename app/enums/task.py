from enum import IntEnum, StrEnum
from typing import Self


class FileParsingTaskState(IntEnum):
    """文件解析任务状态"""

    # 等待中
    pedding = 0
    # 开始
    started = 1
    # 转换中
    converting = 2
    # 转换完成
    converted = 3
    # 转换失败
    converting_failed = 4
    # 解析中
    parsing = 5
    # 解析成功
    parsed = 6
    # 解析失败
    parsing_failed = 7
    # 成功
    success = 8


class DocumentFileTaskType(StrEnum):
    """文档文件任务类型"""

    txt = ".txt"
    csv = ".csv"
    pdf = ".pdf"
    docx = ".docx"
    doc = ".doc"
    xlsx = ".xlsx"
    xls = ".xls"
    pptx = ".pptx"
    mp3 = ".mp3"
    wav = ".wav"
    mp4 = ".mp4"
    avi = ".avi"
    json = ".json"
    jsonl = ".jsonl"

    @classmethod
    def get_file_type(cls, file_name: str) -> Self:
        """获取文件类型"""
        suffix = "." + file_name.rsplit(".", 1)[-1].lower()

        # 动态查找匹配的枚举值
        for file_type in cls:
            if file_type.value == suffix:
                return file_type

        # 如果没有找到匹配的文件类型，抛出异常
        raise ValueError(f"不支持的文件类型: {suffix}")
