import math

import rich

from app.enums.embedding import EmbeddingDriverEnum
from app.rag.embedding.embeeding_model import EmbeddingFactory
from app.tool_searcher.tool_library import ToolLibrary


class TrigonometryCalculator:
    @staticmethod
    def add(a: float, b: float):
        """
        Adds two numbers.
        """
        return a + b

    @staticmethod
    def sine(x: float):
        """
        Calculates the sine of an angle in radians.

        :param x: The angle in radians.
        :return: The sine of x.
        """
        return math.sin(x)

    @staticmethod
    def cosine(x: float):
        """
        Calculates the cosine of an angle in radians.

        :param x: The angle in radians.
        :return: The cosine of x.
        """
        return math.cos(x)

    @staticmethod
    def tangent(x: float):
        """
        Calculates the tangent of an angle in radians.

        :param x: The angle in radians.
        :return: The tangent of x.
        """
        return math.tan(x)

    @staticmethod
    def demo(query: str | None = None):
        """
        A demo function.
        """
        return query


async def test_tool_library() -> None:
    # 使用huggingface的bge-m3模型
    EmbeddingFactory.init(
        {
            "provider": "huggingface",
            "model": "BAAI/bge-m3",
            "driver": EmbeddingDriverEnum.MAC,
        }
    )
    # 使用openai兼容的api模型
    # EmbeddingFactory.init(
    #     {
    #         "provider": "openai",
    #         "model": "Qwen/Qwen3-Embedding-8B",
    #         "api_key": settings.SILICONFLOW_API_KEY,
    #         "base_url": settings.SILICONFLOW_API_BASE,
    #     }
    # )
    TEST_QUESTIONS = {
        "simple": [
            "查询苹果的最新股价和股息数据",
            "分别查询苹果的 1min、5min、15min、1h、4h 股票数据",
            "苹果目前的资产负债表和现金流量表怎么样？",
            "总结一下最近 10 条股票新闻",
            "从新闻角度分析特朗普币还值得买入吗？",
        ],
        "standard": [
            "Please extract net profit, total assets and shareholders' equity data from Netflix's financial report, calculate its ROA and ROE, and analyze the impact of its video content capitalization policy on these indicators.",
            "Please construct a rolling 12-month EBITDA chart based on the quarterly financial data in Meta's latest annual report and mark the YoY growth rate inflection point.",
            "Please analyze the changes in Apple Inc.'s (AAPL) capital structure over the past three years and calculate the changing trend of its weighted average cost of capital (WACC).",
            "We extract R&D spending data from Tesla's (TSLA) 10-K reports and analyze its correlation with revenue growth.",
            "Analyze the impact of Microsoft's (MSFT) merger and acquisition activities in the past five years on its financial statements, especially the changes in goodwill and intangible assets.",
        ],
        "complex": [
            "找出今天 10 支大资金流动的股票/期货。",
        ],
    }

    tulip = ToolLibrary(
        instance_imports=[TrigonometryCalculator],  # 从类导入tools
    )
    await tulip.ainit()

    funciton_list = []
    for task in TEST_QUESTIONS["standard"]:
        print(f"{task=}")
        res = tulip.search(task, top_k=4, similarity_threshold=0.8)
        # print(f"{res=}")
        funcs = [r.name for r in res]
        print(f"{funcs=}")
        funciton_list.extend(r.model_dump() for r in res)


async def test_tool_library_openai_definition() -> None:
    # 使用huggingface的bge-m3模型
    EmbeddingFactory.init(
        {
            "provider": "huggingface",
            "model": "BAAI/bge-m3",
            "driver": EmbeddingDriverEnum.MAC,
        }
    )
    tulip = ToolLibrary(
        # file_imports=[    # 从模块导入tools 函数
        #     ("tools.calendar", []),
        #     ("logics.copilot.tools.stock_tool", []),
        # ],
        instance_imports=[TrigonometryCalculator],  # 从类导入tools
    )
    await tulip.ainit()
    # rich.inspect(tulip.tools["demo"])
    rich.print(tulip.tools)
