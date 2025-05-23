# 中文数字映射
CHINESE_NUMERALS: dict[str, int] = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
}

# 标题识别模式 - (正则模式, 标题级别)
TITLE_PATTERNS: list[tuple[str, int]] = [
    (r"^(\d+)\.\s+(.+)$", 1),  # 1. 标题
    (r"^(\d+\.\d+)\s+(.+)$", 2),  # 1.1 标题
    (r"^(\d+\.\d+\.\d+)\s+(.+)$", 3),  # 1.1.1 标题
    (r"^(\d+\.\d+\.\d+\.\d+)\s+(.+)$", 4),  # 1.1.1.1 标题
    (r"^第(\d+)章\s*[:：]?\s*(.+)$", 1),  # 第1章
    (r"^第(\d+)节\s*[:：]?\s*(.+)$", 2),  # 第1节
    (r"^第(\d+)条\s*[:：]?\s*(.+)$", 2),  # 第1条
    (r"^第([一二三四五六七八九十百]+)章\s*[:：]?\s*(.+)$", 1),  # 第一章
    (r"^第([一二三四五六七八九十百]+)节\s*[:：]?\s*(.+)$", 2),  # 第一节
    (r"^第([一二三四五六七八九十百]+)条\s*[:：]?\s*(.+)$", 2),  # 第一条
    (r"^([一二三四五六七八九十]+)、\s*(.+)$", 2),  # 一、
    (r"^（([一二三四五六七八九十]+)）\s*(.+)$", 3),  # （一）
    (r"^\(([一二三四五六七八九十]+)\)\s*(.+)$", 3),  # (一)
    (r"^(\d+)、\s*(.+)$", 2),  # 1、标题
    (r"^（(\d+)）\s*(.+)$", 3),  # （1）标题
    (r"^\((\d+)\)\s*(.+)$", 3),  # (1)标题
]

# 常见的无编号标题关键词
COMMON_TITLE_KEYWORDS = [
    "简介",
    "概述",
    "介绍",
    "说明",
    "总结",
    "结论",
    "摘要",
    "目录",
    "前言",
    "背景",
    "附录",
    "参考文献",
    "致谢",
    "后记",
    "序言",
]
