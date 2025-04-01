from dataclasses import dataclass
from enum import Enum


class PDFContentType(str, Enum):
    """PDF内容类型枚举"""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    CODE = "code"


@dataclass
class PDFSection:
    """PDF文档章节数据类"""

    title: str
    level: int
    page_number: int
    content: str
    metadata: dict = None


@dataclass
class TOCEntry:
    """目录条目"""

    title: str
    level: int
    page_number: int


@dataclass
class PdfParserConfig:
    """PDF解析器配置参数"""

    # 分块策略选项
    STRATEGY_BASIC = "basic"  # 基本分块
    STRATEGY_STRUCTURE = "structure"  # 结构化分块

    strategy: str = "structure"
    heading_levels: list[int] = None
    preserve_page_info: bool = True

    def __post_init__(self):
        # 设置默认值
        if self.heading_levels is None:
            self.heading_levels = [1, 2, 3, 4]


@dataclass
class PDFTable:
    data: list[list[str]]  # 表格数据
    caption: str = ""  # 表格标题
    page_number: int = 0  # 页码
    position: tuple = None  # 表格在页面中的位置


@dataclass
class PDFImage:
    image_data: bytes
    caption: str = ""
    page_number: int = 0
    position: tuple[float, float, float, float] | None = None
    format: str = "png"
