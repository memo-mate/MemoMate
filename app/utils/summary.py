import asyncio
import operator
from typing import Annotated, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.core import settings
from app.rag.llm.completions import LLM, LLMParams

# ==================== 配置 ====================

# 初始化LLM
llm = LLM().get_llm(LLMParams(model_name=settings.CHAT_MODEL))

# ==================== 提示模板 ====================
map_template = "Write a concise summary of the following text: {context}"
reduce_template = """
The following is a set of summaries:
{docs}

Take these summaries and distill them into a final, consolidated summary
of the main themes and key points.Answer in Chinese.
"""

map_prompt = ChatPromptTemplate([("human", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

# 创建链
map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()


# ==================== 状态定义 ====================
class OverallState(TypedDict):
    """主图的整体状态"""

    contents: list[str]  # 输入的文档内容
    summaries: Annotated[list[str], operator.add]  # 生成的摘要列表
    collapsed_summaries: list[Document]  # 折叠后的摘要
    final_summary: str  # 最终摘要


class SummaryState(TypedDict):
    """单个摘要节点的状态"""

    content: str


# ==================== 辅助函数 ====================
def length_function(documents: list[Document]) -> int:
    """计算文档列表的token数量"""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


token_max = 3000  # 最大token限制


# ==================== 节点函数 ====================
async def generate_summary(state: SummaryState) -> dict:
    """生成单个文档的摘要"""
    response = await map_chain.ainvoke({"context": state["content"]})
    return {"summaries": [response]}


def map_summaries(state: OverallState) -> list[Send]:
    """映射函数：为每个文档内容创建Send对象"""
    return [Send("generate_summary", {"content": content}) for content in state["contents"]]


def collect_summaries(state: OverallState) -> dict:
    """收集所有摘要并转换为Document对象"""
    return {"collapsed_summaries": [Document(page_content=summary) for summary in state["summaries"]]}


async def collapse_summaries(state: OverallState) -> dict:
    """折叠摘要：将过长的摘要列表进一步总结"""
    # 根据token限制分割文档列表
    doc_lists = split_list_of_docs(state["collapsed_summaries"], length_function, token_max)

    results = []
    for doc_list in doc_lists:
        # 对每个分组进行总结
        collapsed_doc = await acollapse_docs(doc_list, reduce_chain.ainvoke)
        results.append(collapsed_doc)

    return {"collapsed_summaries": results}


def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
    """决定是否需要继续折叠"""
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


async def generate_final_summary(state: OverallState) -> dict:
    """生成最终摘要"""
    # 将折叠的摘要转换为字符串格式
    summary_texts = [doc.page_content for doc in state["collapsed_summaries"]]
    response = await reduce_chain.ainvoke({"docs": "\n".join(summary_texts)})
    return {"final_summary": response}


# ==================== 构建图 ====================
def create_map_reduce_graph():
    """创建Map-Reduce图"""
    graph = StateGraph(OverallState)

    # 添加节点
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # 添加边
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    return graph.compile()


# ==================== 示例使用 ====================


async def run_long_document_example():
    """运行长文档示例"""
    print("\n=== 长文档示例：网页内容总结 ===")

    # 加载网页内容（这里使用一个公开的博客文章）
    try:
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        documents = loader.load()

        # 分割文档
        text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        print(f"分割后的文档数量: {len(split_docs)}")

        app = create_map_reduce_graph()

        # 执行处理
        async for step in app.astream({"contents": [doc.page_content for doc in split_docs]}, {"recursion_limit": 10}):
            step_name = list(step.keys())[0]
            print(f"执行步骤: {step_name}")

            if step_name == "generate_final_summary":
                print(f"\n长文档最终摘要:\n{step[step_name]['final_summary']}")

    except Exception as e:
        print(f"加载网页内容失败: {e}")
        print("使用本地长文档示例...")


def visualize_graph():
    """可视化图结构（需要安装相关依赖）"""
    try:
        app = create_map_reduce_graph()
        # 如果安装了相关依赖，可以生成图的可视化
        from IPython.display import Image

        Image(app.get_graph().draw_mermaid_png())

        print("图结构已创建，包含以下节点:")
        print("- generate_summary: 生成单个文档摘要")
        print("- collect_summaries: 收集所有摘要")
        print("- collapse_summaries: 折叠过长的摘要")
        print("- generate_final_summary: 生成最终摘要")
    except Exception as e:
        print(f"可视化失败: {e}")


# ==================== 主函数 ====================
async def main():
    """主函数"""
    print("LangGraph Map-Reduce 演示")
    print("=" * 50)
    # 可视化图结构
    # visualize_graph()
    # 运行长文档示例
    await run_long_document_example()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
