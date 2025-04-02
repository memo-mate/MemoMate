from dataclasses import dataclass


@dataclass
class DocumentSection:
    """文档章节数据类"""

    title: str
    level: int
    content: str
    metadata: dict = None
    page_number: int = None


@dataclass
class TOCEntry:
    """目录条目"""

    title: str
    level: int
    page_number: int = None


@dataclass
class ParserConfig:
    """解析器配置参数"""

    # 分块策略选项
    STRATEGY_BASIC = "basic"  # 基本分块
    STRATEGY_STRUCTURE = "structure"  # 结构化分块

    strategy: str = "structure"
    heading_levels: list[int] = None

    def __post_init__(self):
        # 设置默认值
        if self.heading_levels is None:
            self.heading_levels = [1, 2, 3, 4]
