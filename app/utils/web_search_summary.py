import asyncio
import operator
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core import settings
from app.core.log_adapter import logger

# ==================== 异常定义 ====================


class HumanReviewRequiredException(Exception):
    """需要人工审核的异常"""

    def __init__(self, message: str, summary_preview: str = "", state: dict = None):
        super().__init__(message)
        self.summary_preview = summary_preview
        self.state = state


# ==================== 配置和枚举 ====================


class SummaryStrategy(str, Enum):
    """总结策略枚举"""

    CONCISE = "concise"  # 简洁摘要
    DETAILED = "detailed"  # 详细摘要
    BULLET_POINTS = "bullet_points"  # 要点摘要
    TECHNICAL = "technical"  # 技术摘要


class DocumentFormat(str, Enum):
    """支持的文档格式"""

    WEB = "web"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "markdown"
    CSV = "csv"
    JSON = "json"


class ProcessingConfig(BaseModel):
    """处理配置"""

    token_max: int = Field(default=3000, description="最大token限制")
    chunk_size: int = Field(default=1000, description="文档分块大小")
    chunk_overlap: int = Field(default=200, description="分块重叠大小")
    temperature: float = Field(default=0, description="LLM温度参数")
    recursion_limit: int = Field(default=10, description="递归限制")
    enable_cache: bool = Field(default=True, description="是否启用缓存")
    enable_human_review: bool = Field(default=False, description="是否需要人工审核")
    interactive_review: bool = Field(default=False, description="是否启用交互式审核（命令行输入）")


# ==================== LLM 初始化 ====================


def create_llm(config: ProcessingConfig):
    """创建LLM实例"""
    try:
        return init_chat_model(
            "deepseek-ai/DeepSeek-R1",
            model_provider="openai",
            temperature=config.temperature,
            api_key=settings.SILICONFLOW_API_KEY,
            base_url=settings.SILICONFLOW_API_BASE,
        )
    except Exception as e:
        logger.exception("初始化LLM失败", exc_info=e)
        raise


# ==================== 提示模板策略 ====================


class PromptTemplateStrategy(ABC):
    """提示模板策略基类"""

    @abstractmethod
    def get_map_template(self) -> str:
        """获取Map阶段的提示模板"""
        pass

    @abstractmethod
    def get_reduce_template(self) -> str:
        """获取Reduce阶段的提示模板"""
        pass


class ConciseStrategy(PromptTemplateStrategy):
    """简洁摘要策略"""

    def get_map_template(self) -> str:
        return "请为以下文本生成简洁的摘要，突出核心要点：\n{context}"

    def get_reduce_template(self) -> str:
        return """
以下是一组摘要：
{docs}

请将这些摘要整合成一个最终的简洁摘要，突出主要主题和关键点。
回答请使用中文。
"""


class DetailedStrategy(PromptTemplateStrategy):
    """详细摘要策略"""

    def get_map_template(self) -> str:
        return "请为以下文本生成详细的摘要，包含重要细节和背景信息：\n{context}"

    def get_reduce_template(self) -> str:
        return """
以下是一组详细摘要：
{docs}

请将这些摘要整合成一个全面的最终摘要，保留重要细节和背景信息。
回答请使用中文。
"""


class BulletPointsStrategy(PromptTemplateStrategy):
    """要点摘要策略"""

    def get_map_template(self) -> str:
        return "请将以下文本的主要内容整理成要点列表：\n{context}"

    def get_reduce_template(self) -> str:
        return """
以下是一组要点摘要：
{docs}

请将这些要点整合并去重，形成一个结构化的最终要点列表。
回答请使用中文，使用项目符号格式。
"""


class TechnicalStrategy(PromptTemplateStrategy):
    """技术摘要策略"""

    def get_map_template(self) -> str:
        return "请为以下技术文档生成摘要，重点关注技术细节、方法和实现：\n{context}"

    def get_reduce_template(self) -> str:
        return """
以下是一组技术摘要：
{docs}

请将这些技术摘要整合成一个全面的技术总结，保留关键的技术细节和方法。
回答请使用中文。
"""


# ==================== 文档加载器工厂 ====================


class DocumentLoaderFactory:
    """文档加载器工厂"""

    @staticmethod
    def create_loader(source: str | Path, doc_format: DocumentFormat):
        """根据格式创建对应的文档加载器"""
        loaders = {
            DocumentFormat.WEB: lambda: WebBaseLoader(str(source)),
            DocumentFormat.PDF: lambda: PyPDFLoader(str(source)),
            DocumentFormat.DOCX: lambda: Docx2txtLoader(str(source)),
            DocumentFormat.TXT: lambda: TextLoader(str(source)),
            DocumentFormat.MD: lambda: UnstructuredMarkdownLoader(str(source)),
            DocumentFormat.CSV: lambda: CSVLoader(str(source)),
            DocumentFormat.JSON: lambda: JSONLoader(str(source)),
        }

        if doc_format not in loaders:
            raise ValueError(f"不支持的文档格式: {doc_format}")

        return loaders[doc_format]()


# ==================== 状态定义 ====================


class OverallState(TypedDict):
    """主图的整体状态"""

    contents: list[str]  # 输入的文档内容
    summaries: Annotated[list[str], operator.add]  # 生成的摘要列表
    collapsed_summaries: list[Document]  # 折叠后的摘要
    final_summary: str  # 最终摘要
    config: ProcessingConfig  # 处理配置
    strategy: str  # 摘要策略
    metadata: dict[str, Any]  # 元数据
    human_review_required: bool  # 是否需要人工审核
    processing_steps: list[str]  # 处理步骤记录


class SummaryState(TypedDict):
    """单个摘要节点的状态"""

    content: str
    config: ProcessingConfig
    strategy: str


class HumanReviewState(TypedDict):
    """人工审核状态"""

    summary: str
    approved: bool
    feedback: str


# ==================== 核心处理类 ====================


class WebSearchProcessor:
    """Web搜索和文档处理器"""

    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.llm = create_llm(self.config)
        self.strategy_map = {
            SummaryStrategy.CONCISE: ConciseStrategy(),
            SummaryStrategy.DETAILED: DetailedStrategy(),
            SummaryStrategy.BULLET_POINTS: BulletPointsStrategy(),
            SummaryStrategy.TECHNICAL: TechnicalStrategy(),
        }
        self.memory = MemorySaver() if self.config.enable_cache else None
        self.human_review_handler = None  # 人工审核处理器

    def get_strategy(self, strategy_name: str) -> PromptTemplateStrategy:
        """获取摘要策略"""
        strategy = SummaryStrategy(strategy_name)
        return self.strategy_map[strategy]

    def set_human_review_handler(self, handler):
        """设置人工审核处理器

        Args:
            handler: 异步函数，接收state参数，返回包含approved和feedback字段的字典
        """
        self.human_review_handler = handler

    async def _interactive_review(self, state: OverallState) -> dict:
        """内置的交互式审核功能"""
        summaries = state["collapsed_summaries"]
        metadata = state.get("metadata", {})

        print("\n" + "=" * 60)
        print("📋 需要人工审核的内容")
        print("=" * 60)

        print("📊 基本信息:")
        print(f"   - 文档数量: {len(summaries)}")
        print(f"   - 元数据: {metadata}")

        print("\n📄 内容详情:")
        for i, doc in enumerate(summaries, 1):
            content = doc.page_content
            preview = content if len(content) <= 100 else content[:100] + "..."
            print(f"   {i}. {preview}")
            print(f"      (完整长度: {len(content)} 字符)")

        # 内容分析
        all_content = " ".join([doc.page_content for doc in summaries])
        total_length = len(all_content)

        print("\n🔍 内容分析:")
        print(f"   - 总字符数: {total_length}")

        # 敏感词检测
        sensitive_words = ["财务", "敏感", "机密", "内部", "密码", "账号", "私有"]
        found_sensitive = [word for word in sensitive_words if word in all_content]
        if found_sensitive:
            print(f"   - ⚠️  发现敏感词: {found_sensitive}")
        else:
            print("   - ✅ 未发现敏感词")

        # 质量指标
        quality_words = ["报告", "分析", "评估", "规划", "总结", "建议", "方案", "策略"]
        quality_count = sum(1 for word in quality_words if word in all_content)
        print(f"   - 📈 质量指标: {quality_count}/{len(quality_words)}")

        print("=" * 60)

        # 获取用户决策
        while True:
            print("\n🤔 请做出审核决策:")
            print("   1. 通过 (输入 'y', 'yes', '1', '通过')")
            print("   2. 拒绝 (输入 'n', 'no', '2', '拒绝')")
            print("   3. 查看完整内容 (输入 'view', 'v', '查看')")
            print("   4. 退出程序 (输入 'quit', 'q', '退出')")

            try:
                choice = input("\n👉 请输入你的选择: ").strip().lower()

                if choice in ["y", "yes", "1", "通过"]:
                    feedback = input("💬 请输入通过理由 (可选): ").strip()
                    return {
                        "approved": True,
                        "feedback": feedback or "人工审核通过",
                        "reviewer": "interactive_human",
                        "review_details": {
                            "sensitive_words": found_sensitive,
                            "quality_score": quality_count,
                            "total_length": total_length,
                        },
                    }

                elif choice in ["n", "no", "2", "拒绝"]:
                    feedback = input("💬 请输入拒绝理由 (必填): ").strip()
                    if not feedback:
                        print("❌ 拒绝时必须提供理由，请重新输入")
                        continue
                    return {
                        "approved": False,
                        "feedback": feedback,
                        "reviewer": "interactive_human",
                        "review_details": {
                            "sensitive_words": found_sensitive,
                            "quality_score": quality_count,
                            "total_length": total_length,
                        },
                    }

                elif choice in ["view", "v", "查看"]:
                    print("\n" + "=" * 60)
                    print("📋 完整内容详情")
                    print("=" * 60)

                    for i, doc in enumerate(summaries, 1):
                        print(f"\n📄 文档 {i}:")
                        print(f"   {doc.page_content}")
                        if hasattr(doc, "metadata") and doc.metadata:
                            print(f"   元数据: {doc.metadata}")

                    print("\n按回车键继续...")
                    input()
                    continue

                elif choice in ["quit", "q", "退出"]:
                    print("👋 退出程序")
                    import sys

                    sys.exit(0)

                else:
                    print("❌ 无效输入，请重新选择")
                    continue

            except KeyboardInterrupt:
                print("\n👋 程序被中断")
                import sys

                sys.exit(0)
            except EOFError:
                print("\n👋 输入结束，退出程序")
                import sys

                sys.exit(0)

    def length_function(self, documents: list[Document]) -> int:
        """计算文档列表的token数量"""
        try:
            return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)
        except Exception as e:
            logger.exception("计算token数量失败", exc_info=e)
            # fallback: 估算token数量
            return sum(len(doc.page_content.split()) * 1.3 for doc in documents)

    async def generate_summary(self, state: SummaryState) -> dict:
        """生成单个文档的摘要"""
        try:
            strategy = self.get_strategy(state["strategy"])
            map_template = strategy.get_map_template()
            map_prompt = ChatPromptTemplate([("human", map_template)])
            map_chain = map_prompt | self.llm | StrOutputParser()

            response = await map_chain.ainvoke({"context": state["content"]})
            logger.info("生成单个摘要成功", content_length=len(state["content"]))
            return {"summaries": [response]}
        except Exception as e:
            logger.exception("生成摘要失败", exc_info=e)
            return {"summaries": [f"摘要生成失败: {str(e)}"]}

    def map_summaries(self, state: OverallState) -> list[Send]:
        """映射函数：为每个文档内容创建Send对象"""
        return [
            Send("generate_summary", {"content": content, "config": state["config"], "strategy": state["strategy"]})
            for content in state["contents"]
        ]

    def collect_summaries(self, state: OverallState) -> dict:
        """收集所有摘要并转换为Document对象"""
        collapsed_summaries = [Document(page_content=summary) for summary in state["summaries"]]
        processing_steps = state.get("processing_steps", [])
        processing_steps.append(f"收集了 {len(state['summaries'])} 个摘要")

        return {"collapsed_summaries": collapsed_summaries, "processing_steps": processing_steps}

    async def collapse_summaries(self, state: OverallState) -> dict:
        """折叠摘要：将过长的摘要列表进一步总结"""
        try:
            strategy = self.get_strategy(state["strategy"])
            reduce_template = strategy.get_reduce_template()
            reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
            reduce_chain = reduce_prompt | self.llm | StrOutputParser()

            # 根据token限制分割文档列表
            doc_lists = split_list_of_docs(
                state["collapsed_summaries"], self.length_function, state["config"].token_max
            )

            results = []
            for doc_list in doc_lists:
                collapsed_doc = await acollapse_docs(doc_list, reduce_chain.ainvoke)
                results.append(collapsed_doc)

            processing_steps = state.get("processing_steps", [])
            processing_steps.append(f"折叠摘要: {len(state['collapsed_summaries'])} -> {len(results)}")

            logger.info("折叠摘要成功", original_count=len(state["collapsed_summaries"]), collapsed_count=len(results))

            return {"collapsed_summaries": results, "processing_steps": processing_steps}
        except Exception as e:
            logger.exception("折叠摘要失败", exc_info=e)
            return {"collapsed_summaries": state["collapsed_summaries"]}

    def should_collapse(
        self, state: OverallState
    ) -> Literal["collapse_summaries", "human_review", "generate_final_summary"]:
        """决定下一步操作"""
        num_tokens = self.length_function(state["collapsed_summaries"])

        if num_tokens > state["config"].token_max:
            return "collapse_summaries"
        elif state["config"].enable_human_review and not state.get("human_review_required", False):
            return "human_review"
        else:
            return "generate_final_summary"

    async def human_review(self, state: OverallState) -> dict:
        """人工审核节点"""
        summary_preview = "\n".join([doc.page_content[:200] + "..." for doc in state["collapsed_summaries"]])

        logger.info("需要人工审核", summary_preview=summary_preview[:500])

        # 检查是否启用交互式审核
        if self.config.interactive_review:
            try:
                # 使用内置的交互式审核
                review_result = await self._interactive_review(state)

                return {
                    "human_review_required": True,
                    "metadata": {
                        **state.get("metadata", {}),
                        "human_review_status": "completed",
                        "review_timestamp": asyncio.get_event_loop().time(),
                        "review_approved": review_result.get("approved", True),
                        "review_feedback": review_result.get("feedback", ""),
                        "reviewer": review_result.get("reviewer", "interactive_human"),
                        "review_details": review_result.get("review_details", {}),
                    },
                }
            except Exception as e:
                logger.exception("交互式审核执行失败", exc_info=e)
                # 如果交互式审核失败，抛出异常
                raise HumanReviewRequiredException(
                    "需要人工审核，但交互式审核执行失败", summary_preview=summary_preview, state=state
                )

        # 检查是否有自定义的审核处理器
        if hasattr(self, "human_review_handler") and self.human_review_handler:
            try:
                # 调用自定义的人工审核处理器
                review_result = await self.human_review_handler(state)

                return {
                    "human_review_required": True,
                    "metadata": {
                        **state.get("metadata", {}),
                        "human_review_status": "completed",
                        "review_timestamp": asyncio.get_event_loop().time(),
                        "review_approved": review_result.get("approved", True),
                        "review_feedback": review_result.get("feedback", ""),
                        "reviewer": review_result.get("reviewer", "custom_handler"),
                        "review_details": review_result.get("review_details", {}),
                    },
                }
            except Exception as e:
                logger.exception("人工审核处理器执行失败", exc_info=e)
                # 如果审核处理器失败，抛出异常要求手动处理
                raise HumanReviewRequiredException(
                    "需要人工审核，但审核处理器执行失败", summary_preview=summary_preview, state=state
                )

        # 如果没有自定义处理器且未启用交互式审核，抛出异常要求人工介入
        raise HumanReviewRequiredException("需要人工审核，请处理后继续", summary_preview=summary_preview, state=state)

    async def generate_final_summary(self, state: OverallState) -> dict:
        """生成最终摘要"""
        try:
            strategy = self.get_strategy(state["strategy"])
            reduce_template = strategy.get_reduce_template()
            reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
            reduce_chain = reduce_prompt | self.llm | StrOutputParser()

            summary_texts = [doc.page_content for doc in state["collapsed_summaries"]]
            response = await reduce_chain.ainvoke({"docs": "\n".join(summary_texts)})

            processing_steps = state.get("processing_steps", [])
            processing_steps.append("生成最终摘要完成")

            logger.info("生成最终摘要成功", summary_length=len(response))

            return {
                "final_summary": response,
                "processing_steps": processing_steps,
                "metadata": {
                    **state.get("metadata", {}),
                    "final_summary_length": len(response),
                    "total_processing_steps": len(processing_steps),
                },
            }
        except Exception as e:
            logger.exception("生成最终摘要失败", exc_info=e)
            return {"final_summary": f"最终摘要生成失败: {str(e)}"}

    def create_map_reduce_graph(self):
        """创建优化的Map-Reduce图"""
        graph = StateGraph(OverallState)

        # 添加节点
        graph.add_node("generate_summary", self.generate_summary)
        graph.add_node("collect_summaries", self.collect_summaries)
        graph.add_node("collapse_summaries", self.collapse_summaries)
        graph.add_node("human_review", self.human_review)
        graph.add_node("generate_final_summary", self.generate_final_summary)

        # 添加边
        graph.add_conditional_edges(START, self.map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self.should_collapse)
        graph.add_conditional_edges("collapse_summaries", self.should_collapse)
        graph.add_edge("human_review", "generate_final_summary")
        graph.add_edge("generate_final_summary", END)

        # 如果启用缓存，添加检查点
        compile_config = {}
        if self.memory:
            compile_config["checkpointer"] = self.memory

        return graph.compile(**compile_config)

    async def process_documents(
        self,
        source: str | Path | list[str],
        doc_format: DocumentFormat = DocumentFormat.WEB,
        strategy: SummaryStrategy = SummaryStrategy.CONCISE,
        custom_config: ProcessingConfig | None = None,
    ) -> dict[str, Any]:
        """处理文档的主要接口"""
        config = custom_config or self.config

        try:
            # 加载文档
            if isinstance(source, list):
                contents = source
            else:
                loader = DocumentLoaderFactory.create_loader(source, doc_format)
                documents = loader.load()

                # 分割文档
                text_splitter = TokenTextSplitter.from_tiktoken_encoder(
                    chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
                )
                split_docs = text_splitter.split_documents(documents)
                contents = [doc.page_content for doc in split_docs]

            logger.info(
                "开始处理文档", source=str(source), doc_format=doc_format, strategy=strategy, chunk_count=len(contents)
            )

            # 创建处理图
            app = self.create_map_reduce_graph()

            # 初始状态
            initial_state = {
                "contents": contents,
                "config": config,
                "strategy": strategy.value,
                "metadata": {
                    "source": str(source),
                    "doc_format": doc_format.value,
                    "strategy": strategy.value,
                    "chunk_count": len(contents),
                    "start_time": asyncio.get_event_loop().time(),
                },
                "processing_steps": [f"开始处理 {len(contents)} 个文档块"],
            }

            # 执行处理
            final_state = None
            async for step in app.astream(initial_state, {"recursion_limit": config.recursion_limit}):
                step_name = list(step.keys())[0]
                logger.info("执行处理步骤", step_name=step_name)
                final_state = step[step_name]

            # 添加完成时间
            if final_state and "metadata" in final_state:
                final_state["metadata"]["end_time"] = asyncio.get_event_loop().time()
                final_state["metadata"]["processing_duration"] = (
                    final_state["metadata"]["end_time"] - final_state["metadata"]["start_time"]
                )

            logger.info(
                "文档处理完成",
                processing_duration=final_state["metadata"].get("processing_duration"),
                final_summary_length=len(final_state.get("final_summary", "")),
            )

            return final_state

        except Exception as e:
            logger.exception("文档处理失败", source=str(source), exc_info=e)
            raise


# ==================== 便捷函数 ====================


async def summarize_web_content(
    url: str, strategy: SummaryStrategy = SummaryStrategy.CONCISE, config: ProcessingConfig | None = None
) -> str:
    """便捷函数：总结网页内容"""
    processor = WebSearchProcessor(config)
    result = await processor.process_documents(url, DocumentFormat.WEB, strategy)
    return result.get("final_summary", "")


async def summarize_document(
    file_path: str | Path,
    doc_format: DocumentFormat,
    strategy: SummaryStrategy = SummaryStrategy.CONCISE,
    config: ProcessingConfig | None = None,
) -> str:
    """便捷函数：总结文档"""
    processor = WebSearchProcessor(config)
    result = await processor.process_documents(file_path, doc_format, strategy)
    return result.get("final_summary", "")


async def summarize_text_list(
    texts: list[str], strategy: SummaryStrategy = SummaryStrategy.CONCISE, config: ProcessingConfig | None = None
) -> str:
    """便捷函数：总结文本列表"""
    processor = WebSearchProcessor(config)
    result = await processor.process_documents(texts, strategy=strategy)
    return result.get("final_summary", "")


# ==================== 示例和测试 ====================


async def run_examples():
    """运行示例"""
    logger.info("开始运行Web搜索处理示例")

    # 示例1: 网页内容总结
    try:
        logger.info("示例1: 网页内容总结")
        config = ProcessingConfig(token_max=2000, enable_human_review=False, enable_cache=True)

        summary = await summarize_web_content(
            "https://lilianweng.github.io/posts/2023-06-23-agent/", SummaryStrategy.CONCISE, config
        )
        logger.info("网页摘要生成成功", summary_length=len(summary))
        print(f"\n网页摘要:\n{summary}\n")

    except Exception as e:
        logger.exception("网页内容总结失败", exc_info=e)

    # 示例2: 多种策略对比
    try:
        logger.info("示例2: 多种策略对比")
        test_texts = [
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "机器学习是人工智能的一个重要分支，通过算法让计算机系统能够自动从数据中学习和改进。",
            "深度学习是机器学习的子集，使用多层神经网络来模拟人脑的工作方式。",
        ]

        strategies = [SummaryStrategy.CONCISE, SummaryStrategy.DETAILED, SummaryStrategy.BULLET_POINTS]

        for strategy in strategies:
            summary = await summarize_text_list(test_texts, strategy)
            logger.info(f"{strategy.value}策略摘要生成成功", summary_length=len(summary))
            print(f"\n{strategy.value}摘要:\n{summary}\n")

    except Exception as e:
        logger.exception("多策略对比失败", exc_info=e)


def visualize_graph():
    """可视化图结构"""
    try:
        processor = WebSearchProcessor()
        app = processor.create_map_reduce_graph()

        # 尝试生成图的可视化
        try:
            from IPython.display import Image, display

            display(Image(app.get_graph().draw_mermaid_png()))
        except ImportError:
            logger.info("IPython未安装，跳过图像显示")

        logger.info("优化后的图结构包含以下节点:")
        logger.info("- generate_summary: 生成单个文档摘要")
        logger.info("- collect_summaries: 收集所有摘要")
        logger.info("- collapse_summaries: 折叠过长的摘要")
        logger.info("- human_review: 人工审核节点")
        logger.info("- generate_final_summary: 生成最终摘要")

    except Exception as e:
        logger.exception("可视化失败", exc_info=e)


# ==================== 主函数 ====================


async def main():
    """主函数"""
    logger.info("启动优化版Web搜索处理器")
    logger.info("=" * 50)

    # 可视化图结构
    visualize_graph()

    # 运行示例
    # await run_examples()


if __name__ == "__main__":
    asyncio.run(main())
