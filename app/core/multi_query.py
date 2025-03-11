from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app.core.config import *  # noqa: F403
from app.core.vector_search import VectorSearch
from langchain_core.documents import Document
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
import os
from typing import List, Tuple
from langchain_core.prompts import PromptTemplate

load_dotenv()


class MultiQuerySearch:
    """多查询检索类，通过LLM生成多个相关查询，提高检索效果"""

    def __init__(self):
        # 初始化本地向量搜索
        self.vector_search = VectorSearch()

        # 设置OpenAI API参数
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv(
            "MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        )

        # 初始化LLM
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.api_base,
                temperature=0.3,
                max_tokens=4096,
                timeout=30,
                max_retries=3,
            )
            print("[green]成功初始化LLM[/green]")
        except Exception as e:
            print(f"[red]初始化LLM失败: {str(e)}[/red]")
            print("[yellow]将使用基本检索模式[/yellow]")
            self.llm = None

    def search(
        self, query: str, k: int = 5, use_multi_query: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        执行搜索，可选是否使用多查询增强

        Args:
            query: 用户查询
            k: 返回的结果数量
            use_multi_query: 是否使用多查询增强

        Returns:
            检索结果列表，每项包含文档和相似度分数
        """
        if not use_multi_query or self.llm is None:
            # 如果不使用多查询或LLM初始化失败，直接使用基本检索
            print("[yellow]使用基本检索模式[/yellow]")
            return self.vector_search.similarity_search(query, k=k)

        try:
            # 生成多个查询
            queries = self._generate_queries(query)

            # 显示生成的查询
            print(
                Panel(
                    Markdown("\n".join([f"- {q}" for q in queries])),
                    title="多查询增强",
                    border_style="blue",
                )
            )

            # 对每个查询执行搜索
            all_results = []
            seen_texts = set()

            for q in queries:
                results = self.vector_search.similarity_search(q, k=k)
                for doc, score in results:
                    if doc.page_content not in seen_texts:
                        all_results.append((doc, score))
                        seen_texts.add(doc.page_content)

            # 按相似度排序并限制结果数量
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[:k]

        except Exception as e:
            print(f"[red]多查询检索失败: {str(e)}[/red]")
            print("[yellow]回退到基本检索模式[/yellow]")
            return self.vector_search.similarity_search(query, k=k)

    def _generate_queries(self, query: str) -> List[str]:
        """生成多个相关查询"""
        if not self.llm:
            return [query]  # 如果LLM不可用，只返回原始查询

        try:
            # 创建提示
            prompt = PromptTemplate(
                input_variables=["question"],
                template="""你是一个AI助手，你的任务是生成3个不同的搜索查询，这些查询与原始查询的含义相似，但使用不同的词语和表达方式。
                这些查询将用于检索文档，所以它们应该是独立的、多样化的，并且能够捕捉原始查询的不同方面。
                请直接返回这些查询，每行一个，不要有编号或其他文本。
                
                原始查询: {question}
                """,
            )

            # 调用LLM生成查询
            response = self.llm.invoke(prompt.format(question=query))

            # 解析响应
            if hasattr(response, "content"):
                # 对于ChatOpenAI，响应是一个消息对象
                content = response.content
            else:
                # 对于其他LLM，响应可能是字符串
                content = str(response)

            # 分割成多行并过滤空行
            queries = [line.strip() for line in content.split("\n") if line.strip()]

            # 确保至少有一个查询（原始查询）
            if not queries:
                return [query]

            # 添加原始查询
            if query not in queries:
                queries.append(query)

            return queries

        except Exception as e:
            print(f"[red]生成查询失败: {str(e)}[/red]")
            return [query]  # 返回原始查询

    def format_results(self, results: List[Tuple[Document, float]]) -> str:
        """格式化检索结果为可读文本"""
        context_texts = []
        for i, (doc, similarity) in enumerate(results):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            page_info = f"(第{page}页)" if page else ""

            # 添加工作表信息（如果有）
            sheet_name = doc.metadata.get("sheet_name", "")
            sheet_info = f"(工作表: {sheet_name})" if sheet_name else ""

            # 添加段落信息（如果有）
            paragraph = doc.metadata.get("paragraph", "")
            paragraph_info = f"(段落: {paragraph})" if paragraph else ""

            # 添加相似度信息
            similarity_info = f"[相似度: {similarity:.3f}]"

            context_text = f"[文档{i+1}] {similarity_info} {source}{page_info}{sheet_info}{paragraph_info}\n{doc.page_content}\n"
            context_texts.append(context_text)

        return "\n".join(context_texts)


# 使用示例
if __name__ == "__main__":
    multi_query = MultiQuerySearch()

    while True:
        question = input("\n请输入您的问题 (输入'exit'退出): ")
        if question.lower() in ("exit", "quit", "q"):
            break

        # 执行多查询检索
        results = multi_query.search(question, k=5, use_multi_query=True)

        # 格式化并打印结果
        formatted_results = multi_query.format_results(results)
        print(
            Panel(
                Markdown(formatted_results),
                title=f"检索结果 (共{len(results)}条)",
                border_style="green",
            )
        )
