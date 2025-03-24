from pathlib import Path
from typing import Any

from langchain_community.document_loaders import Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from app.core.config import settings
from app.core.log_adapter import logger


class MarkdownParser:
    def __call__(self, file_path: str, *args: Any, **kwds: Any) -> list[Document]:
        """处理文档并返回分块后的文档列表"""
        return self.chunk(file_path)

    def chunk(self, file_path: str | None = None) -> list[Document]:
        """
        将文档分割成多个块（支持Markdown和Word文档）

        Args:
            file_path: 文档文件路径，如果为None则使用默认路径

        Returns:
            list[Document]: 分割后的文档块列表
        """
        # 使用提供的文件路径或默认路径
        doc_path = file_path or r"/Users/datagrand/Documents/articles/SQLAlchemy简单用法.md"

        # 检查文件类型并使用相应的加载器
        file_suffix = Path(doc_path).suffix.lower()
        documents = []

        try:
            if file_suffix == ".md":
                logger.info("检测到Markdown文档", file_path=doc_path)
                loader = UnstructuredMarkdownLoader(doc_path)
                documents = loader.load()
            elif file_suffix in [".docx", ".doc"]:
                logger.info("检测到Word文档", file_path=doc_path)
                loader = Docx2txtLoader(doc_path)
                documents = loader.load()
            else:
                supported_types = [".md", ".docx", ".doc"]
                logger.error(
                    "不支持的文件类型", file_path=doc_path, file_type=file_suffix, supported_types=supported_types
                )
                raise ValueError(f"不支持的文件类型: {file_suffix}，仅支持: {supported_types}")
        except Exception as e:
            logger.exception("加载文档失败", exc_info=e, file_path=doc_path)
            raise

        logger.info("文档加载成功", file_path=doc_path, document_count=len(documents))

        # 创建文本分割器
        text_spliter = MarkdownTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        logger.info("开始分割文档", chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

        # 分割文档
        split_docs = text_spliter.split_documents(documents)
        logger.info("文档分割完成", chunk_count=len(split_docs))

        return split_docs

    def chunk_by_headers(self, file_path: str | None = None) -> list[Document]:
        """
        按标题结构分割Markdown文档，保持文档结构

        Args:
            file_path: Markdown文件路径，如果为None则使用默认路径

        Returns:
            list[Document]: 分割后的文档块列表，按标题结构组织
        """
        # 使用提供的文件路径或默认路径
        doc_path = file_path or r"/Users/datagrand/Documents/articles/SQLAlchemy简单用法.md"

        if Path(doc_path).suffix.lower() != ".md":
            logger.warning("该方法仅适用于Markdown文档", file_path=doc_path)
            return self.chunk(file_path)

        try:
            # 读取文档内容
            with open(doc_path, encoding="utf-8") as f:
                markdown_text = f.read()

            # 定义要分割的标题级别
            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5"),
                ("######", "H6"),
            ]

            # 创建标题分割器
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False,  # 保留标题，确保Markdown结构完整
            )

            # 按标题分割文档
            docs = markdown_splitter.split_text(markdown_text)

            # 添加文件路径信息到元数据
            for doc in docs:
                doc.metadata["source"] = doc_path

            logger.info("按标题结构分割文档完成", file_path=doc_path, chunk_count=len(docs))
            return docs

        except Exception as e:
            logger.exception("按标题分割文档失败", exc_info=e, file_path=doc_path)
            # 如果按标题分割失败，回退到普通分割方法
            logger.info("回退到普通分割方法")
            return self.chunk(file_path)

    def structure_preserving_chunk(self, file_path: str | None = None) -> list[Document]:
        """
        结构保留分块方法，确保每个块的Markdown语法结构完整性

        Args:
            file_path: 文档文件路径，如果为None则使用默认路径

        Returns:
            list[Document]: 分割后的文档块列表，每个块保持结构完整性
        """
        # 使用提供的文件路径或默认路径
        doc_path = file_path or r"/Users/datagrand/Documents/articles/SQLAlchemy简单用法.md"

        # 检查文件类型
        file_suffix = Path(doc_path).suffix.lower()
        if file_suffix == ".md":
            # 对Markdown文件，优先使用按标题分割
            return self.chunk_by_headers(doc_path)
        else:
            # 对其他文件使用普通分割
            return self.chunk(file_path)

    def _fix_markdown_syntax(self, chunks: list[Document]) -> list[Document]:
        """
        修复Markdown语法，确保每个块的Markdown语法完整

        Args:
            chunks: 分割后的文档块列表

        Returns:
            List[Document]: 修复后的文档块列表
        """
        fixed_chunks = []

        for chunk in chunks:
            content = chunk.page_content
            metadata = chunk.metadata.copy()

            # 检测未闭合的代码块
            if "```" in content:
                # 计算代码块开始和结束标记的数量
                open_marks = content.count("```")

                # 如果开始标记数量是奇数(未闭合)
                if open_marks % 2 != 0:
                    # 追加结束标记
                    content += "\n```"
                    metadata["fixed_code_block"] = True

            # 检测未闭合的表格
            if "|" in content and "---" in content:
                lines = content.split("\n")
                has_table_header = False
                has_table_separator = False

                for line in lines:
                    if "|" in line and not has_table_header:
                        has_table_header = True
                    if "|" in line and "---" in line:
                        has_table_separator = True

                # 如果有表头但没有分隔符，添加分隔符
                if has_table_header and not has_table_separator:
                    # 在第一个表头后添加分隔符
                    lines_with_separator = []
                    header_found = False

                    for line in lines:
                        lines_with_separator.append(line)
                        if "|" in line and not header_found:
                            # 计算分隔符
                            separator = "|"
                            for _cell in (
                                line.split("|")[1:-1]
                                if line.startswith("|") and line.endswith("|")
                                else line.split("|")
                            ):
                                separator += " --- |"
                            lines_with_separator.append(separator)
                            header_found = True

                    content = "\n".join(lines_with_separator)
                    metadata["fixed_table"] = True

            # 将修复后的内容添加到结果
            fixed_chunks.append(Document(page_content=content, metadata=metadata))

        return fixed_chunks

    def preview_chunks(self, file_path: str | None = None) -> None:
        """
        使用Rich实时预览分割后的文档块

        Args:
            file_path: 文档文件路径，如果为None则使用默认路径
        """
        try:
            # 获取分块结果
            chunks = self.chunk(file_path)

            # 创建控制台
            console = Console()

            with Live(refresh_per_second=1) as live:
                # 显示处理信息
                file_name = Path(file_path or "默认文档").name
                live.update(Panel(f"正在处理文件: {file_name}\n共有 {len(chunks)} 个文档块", style="bold cyan"))
                import time

                time.sleep(1)

                for i, chunk in enumerate(chunks):
                    # 创建美化的面板，显示块内容和元数据
                    chunk_text = Text(chunk.page_content, style="white")
                    metadata_text = Text(f"\n\n元数据: {chunk.metadata}", style="dim cyan")

                    # 合并文本
                    combined_text = Text()
                    combined_text.append(chunk_text)
                    combined_text.append(metadata_text)

                    # 创建面板
                    panel = Panel(
                        combined_text,
                        title=f"块 #{i + 1}/{len(chunks)}",
                        title_align="left",
                        border_style="blue",
                        padding=(1, 2),
                    )

                    # 更新预览
                    live.update(panel)

                    # 暂停一会儿以便查看
                    time.sleep(1.5)

                # 显示完成信息
                live.update(
                    Panel(f"文档 {file_name} 分块预览完成！\n共处理 {len(chunks)} 个文档块", style="bold green")
                )
        except Exception as e:
            logger.exception("预览文档块失败", exc_info=e, file_path=file_path)
            console = Console()
            console.print(f"[bold red]处理失败:[/bold red] {str(e)}")

    def preview_all_chunks(self, file_path: str | None = None, preserve_structure: bool = False) -> None:
        """
        使用Rich一次性预览所有文档块，以Panel + Markdown格式展示

        Args:
            file_path: 文档文件路径，如果为None则使用默认路径
            preserve_structure: 是否保留文档结构，True则使用结构保留分块
        """
        try:
            # 获取分块结果，选择是否使用保留结构的分块方法
            if preserve_structure:
                chunks = self.structure_preserving_chunk(file_path)
            else:
                chunks = self.chunk(file_path)

            file_name = Path(file_path or "默认文档").name

            # 创建控制台
            console = Console()

            # 显示处理信息和分块方法
            method_text = "结构保留分块" if preserve_structure else "标准分块"
            console.print(
                Panel(f"[bold cyan]文档 {file_name} 已分割成 {len(chunks)} 个块 (使用{method_text})[/bold cyan]")
            )

            # 遍历展示所有块
            for i, chunk in enumerate(chunks):
                # 尝试将内容格式化为Markdown
                try:
                    # 总是使用Markdown渲染器来尝试渲染内容
                    # 这使得块能够保持Markdown格式
                    content_renderable = Markdown(chunk.page_content)

                    # 创建元数据文本
                    metadata_text = Text("\n\n[元数据]\n", style="bold cyan")
                    for key, value in chunk.metadata.items():
                        metadata_text.append(f"{key}: ", style="cyan")
                        metadata_text.append(f"{value}\n", style="white")

                    # 使用Group组件组合内容和元数据
                    group = Group(content_renderable, metadata_text)

                    # 创建面板
                    panel = Panel(
                        group,
                        title=f"块 #{i + 1}/{len(chunks)}"
                        + (
                            " (结构已修复)"
                            if chunk.metadata.get("fixed_code_block") or chunk.metadata.get("fixed_table")
                            else ""
                        ),
                        title_align="left",
                        border_style="blue",
                        padding=(1, 2),
                        expand=False,
                    )

                    console.print(panel)

                    # 在块之间添加分隔线（除了最后一个块）
                    if i < len(chunks) - 1:
                        console.print("─" * 80, style="dim")

                except Exception as e:
                    logger.warning(f"渲染块 #{i + 1} 时出错", exc_info=e)
                    # 当块无法渲染时，使用普通文本显示
                    console.print(
                        Panel(
                            Text(f"[块内容]\n{chunk.page_content}\n\n[元数据]\n{chunk.metadata}", style="red"),
                            title=f"块 #{i + 1} - 渲染错误",
                            title_align="left",
                            border_style="red",
                        )
                    )
                    if i < len(chunks) - 1:
                        console.print("─" * 80, style="dim")

            # 显示完成信息
            console.print(Panel(f"[bold green]文档 {file_name} 所有块预览完成！[/bold green]"))

        except Exception as e:
            logger.exception("预览所有文档块失败", exc_info=e, file_path=file_path)
            console = Console()
            console.print(f"[bold red]处理失败:[/bold red] {str(e)}")


if __name__ == "__main__":
    parser = MarkdownParser()
    # 使用rich预览分块结果
    # parser.preview_chunks()
    # 使用结构保留分块方法预览
    parser.preview_all_chunks(preserve_structure=True)
