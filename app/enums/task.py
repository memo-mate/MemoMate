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

    @classmethod
    def get_file_type(cls, file_name: str) -> Self:
        """获取文件类型"""
        suffix = "." + file_name.rsplit(".", 1)[-1].lower()
        match suffix:
            case ".txt":
                return cls.txt
            case ".csv":
                return cls.csv
            case ".pdf":
                return cls.pdf
            case ".docx":
                return cls.docx
            case ".doc":
                return cls.doc
            case ".xlsx":
                return cls.xlsx
            case ".xls":
                return cls.xls
            case ".pptx":
                return cls.pptx
            case ".mp3":
                return cls.mp3
            case ".wav":
                return cls.wav
            case ".mp4":
                return cls.mp4
            case ".avi":
                return cls.avi
            case _:
                raise ValueError(f"Unsupported file type: {suffix}")
