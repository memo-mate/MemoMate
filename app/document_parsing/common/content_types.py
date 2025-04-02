from enum import Enum


class DocumentContentType(str, Enum):
    """文档内容类型枚举"""

    TEXT = "text"  # 普通文本
    TITLE = "title"  # 标题
    TABLE = "table"  # 表格
    IMAGE = "image"  # 图片
    FORMULA = "formula"  # 公式
    CODE = "code"  # 代码
