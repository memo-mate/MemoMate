"""XML and JSON RAG Agent.

具备针对结构化数据和非结构化数据的查询和分析能力。同时具备图表生成、数据库查询、数据分析等能力。
"""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_handoff_tool, create_supervisor

from app import prompts
from app.agents.rag_agent import create_time_sensitive_rag_graph
from app.agents.sub_agent import build_echart_agent
from app.core import logger, settings
from app.rag.llm.completions import LLM, LLMParams

# AI 和工具最多交互轮数
MAX_INTERACTION = 10

__TOOLS = None


class DuckDBState(TypedDict):
    """Enhanced DuckDB state with error tracking and optimization."""

    messages: Annotated[list[BaseMessage], add_messages]
    recursion_times: NotRequired[int]
    # 错误追踪和优化
    error_history: NotRequired[list[dict]]  # 错误历史 [{error: str, attempt: int, timestamp: str}]
    context7_topics_queried: NotRequired[set[str]]  # 已查询的 context7 主题，避免重复
    last_sql_error: NotRequired[str]  # 上次 SQL 错误信息，用于判断是否重复错误


def custom_tool_error_handler(error: Exception) -> str:
    """Custom error handler that provides detailed SQL error information.

    This handler extracts key error information and guides the agent
    to query context7 for the correct syntax.
    """
    error_msg = str(error)

    # SQL Parser Errors
    if "Parser Error" in error_msg or "unterminated" in error_msg:
        return (
            f"🔍 **SQL语法错误检测**\n"
            f"错误信息: {error_msg}\n\n"
            f"⚠️ **必须执行**: 立即调用 context7 查询正确语法！\n"
            f"推荐查询: 'json operators', 'json path syntax'\n"
            f"禁止盲目修改SQL，必须基于官方文档。"
        )

    # Function not found
    elif "function does not exist" in error_msg or "No function matches" in error_msg:
        return (
            f"🔍 **函数不存在错误**\n"
            f"错误信息: {error_msg}\n\n"
            f"⚠️ **必须执行**: 查询 context7 获取函数列表\n"
            f"推荐查询: 'functions list', 'json functions'"
        )

    # Type conversion errors
    elif "could not convert" in error_msg or "cast" in error_msg.lower():
        return (
            f"🔍 **类型转换错误**\n"
            f"错误信息: {error_msg}\n\n"
            f"⚠️ **必须执行**: 查询 context7\n"
            f"推荐查询: 'type casting', 'data types'"
        )

    # Generic database error
    elif "Database error" in error_msg:
        return (
            f"🔍 **数据库执行错误**\n"
            f"错误信息: {error_msg}\n\n"
            f"⚠️ **必须执行**: 分析错误类型，查询 context7 获取正确语法\n"
            f"参考'错误处理强制规则'中的错误类型表。"
        )

    # Other errors
    return f"❌ 工具执行错误: {error_msg}\n请检查参数或查询 context7 文档。"


def should_continue(state: DuckDBState, config: RunnableConfig) -> str:
    """判断是否继续执行."""
    # 使用 tools_condition 判断是否有工具调用
    return tools_condition(state)


async def build_duck_graph() -> CompiledStateGraph:
    """Build the DuckDB graph."""
    global __TOOLS
    if not __TOOLS:
        # Get the MCP tools for DuckDB and Context7
        client = MultiServerMCPClient(
            {
                "duckdb": {
                    "command": "uvx",
                    "args": ["mcp-server-duckdb", "--db-path", settings.XML_DB_PATH, "--readonly"],
                    "transport": "stdio",
                },
                "context7": {
                    "url": "https://mcp.context7.com/mcp",
                    "transport": "streamable_http",
                },
                "fetch": {
                    "command": "uvx",
                    "args": ["mcp-server-fetch"],
                    "transport": "stdio",
                },
            }
        )
        __TOOLS = await client.get_tools()

    tool_node = ToolNode(
        tools=__TOOLS,
        handle_tool_errors=custom_tool_error_handler,  # 自定义错误处理，提供详细的SQL错误指引
    )
    llm = LLM().get_llm(LLMParams(model_name=settings.CHAT_MODEL))
    llm_with_tools = llm.bind_tools(__TOOLS)

    async def agent_node(state: DuckDBState, config: RunnableConfig) -> DuckDBState:
        """Agent 节点：处理用户消息并决定是否调用工具."""
        messages = state["messages"]
        max_interaction = config.get("configurable", {}).get("max_interaction", MAX_INTERACTION)
        current_times = state.get("recursion_times", 0)

        # 在调用 LLM 前检查递归次数，避免生成无法完成的工具调用
        if current_times >= max_interaction:
            from langchain_core.messages import AIMessage

            logger.warning(f"⚠️ 达到最大交互次数 {max_interaction}，停止执行")
            stop_message = AIMessage(
                content=f"已达到最大交互次数限制（{max_interaction}次），无法继续执行。请精简查询或提高交互次数限制。"
            )
            return {"messages": [stop_message], "recursion_times": current_times}

        # 构建增强的系统提示（融入最佳实践）
        enhanced_prompt = prompts.XML_DATA_ANALYST_SYSTEM_PROMPT

        sys_msg = SystemMessage(content=enhanced_prompt)
        full_messages = [sys_msg] + messages

        # 调用 LLM
        response = await llm_with_tools.ainvoke(full_messages)

        # 增加递归计数
        new_recursion_times = current_times + 1

        return {"messages": [response], "recursion_times": new_recursion_times}

    workflow = StateGraph(DuckDBState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    # 添加边
    # 流程：START -> Agent -> 工具调用 -> Agent -> END
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    workflow.add_edge("tools", "agent")

    # 编译图
    duck_graph = workflow.compile(name="duckdb_agent")
    return duck_graph


class ChartRAGState(AgentState):
    """Chart RAG State combining RAG and Echarts capabilities."""

    step_count: NotRequired[int]
    max_steps: NotRequired[int]
    context: NotRequired[list[Document]]
    retrieved_docs: list[Document]


async def build_xml_agent() -> CompiledStateGraph:
    """Build the chart RAG agent with a supervisor."""
    rag_agent = create_time_sensitive_rag_graph(name="xml_rag_agent")
    echart_agent = await build_echart_agent()
    duckdb_agent = await build_duck_graph()

    assistant_agent = create_supervisor(
        agents=[rag_agent, echart_agent, duckdb_agent],
        model=LLM().get_llm(LLMParams(model_name=settings.CHAT_MODEL)),
        supervisor_name="supervisor_xml_agent",
        state_schema=ChartRAGState,
        tools=[
            create_handoff_tool(
                agent_name="xml_rag_agent",
                name="assign_to_rag_agent",
                description="Transfer to RAG agent to search and retrieve information from the knowledge base. Use this for questions about projects, research topics, technical documents, specifications, budgets, and any other knowledge-based queries.",
            ),
            create_handoff_tool(
                agent_name="echart_agent",
                name="assign_to_echart_agent",
                description="Transfer to Echart agent to generate charts and data visualizations. Use this when user explicitly requests charts, graphs, or visual representations of data.",
            ),
            create_handoff_tool(
                agent_name="duckdb_agent",
                name="assign_to_duckdb_agent",
                description="Transfer to DuckDB agent to query and analyze structured data. Use this for: statistical queries (counts, sums, averages), data aggregation and grouping, numerical calculations, filtering and sorting data, or any questions requiring database queries. Examples: '有多少个检验批', '统计各类型的数量', '计算总金额'.",
            ),
        ],
        prompt=prompts.XML_SUPERVISOR_PROMPT_V1,
        output_mode="last_message",
        add_handoff_messages=False,
        add_handoff_back_messages=False,
    ).compile()
    return assistant_agent
