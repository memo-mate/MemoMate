"""Echarts ReAct Agent.

Provide an Echarts ReAct Agent that can generate Echarts charts based on user's description.
"""

import uuid
from typing import Annotated, Literal, NotRequired

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import ImageContent
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState

from app.core import settings
from app.rag.llm.completions import LLM, LLMParams
from app.utils.common_llm_tools import clear_image_history


class EchartsGraphState(AgentState):
    """Echarts Graph State."""

    messages: Annotated[list[BaseMessage], add_messages]
    step_count: NotRequired[int]
    max_steps: NotRequired[int]


async def build_echart_agent():
    """Build an Echarts ReAct Agent."""
    max_steps = 5
    # 初始化工具客户端
    client = MultiServerMCPClient(
        {
            "mcp-echarts": {
                "command": "npx",
                "args": ["-y", "mcp-echarts"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()

    model = LLM().get_llm(LLMParams(model_name=settings.CHAT_MODEL))
    model = model.bind_tools(tools)

    # Step 1: LLM 节点（带fallback与步数控制）
    async def llm_node(state: EchartsGraphState):
        state["messages"] = clear_image_history(state)
        messages = state.get("messages", [])
        step_count = state.get("step_count", 0)
        max_allowed = state.get("max_steps", max_steps)

        # 如果已经达到上限，直接返回终止消息
        if step_count > max_allowed:
            return {
                "messages": [AIMessage(content=f"⚠️ 已达到最大执行步数 {max_allowed}，停止本次任务。")],
                "step_count": step_count,
            }

        _rules = [
            "Don't return any result, just call the tools",
            # "Mandatory requirement for input parameter outputType='option'",
        ]
        _rule_str = "\n".join([f"- {rule}" for rule in _rules])
        system_prompt = f"""\
# Role
You are an ECharts chart generation assistant specialized in creating correct chart configurations.

# Task
Use the Echarts tool exactly once to generate a valid chart configuration according to the user’s description.

# Behavior Guidelines
- Think carefully before calling the tool.
- Only call the tool once per request.
- Do not repeat or retry the tool call unless the user explicitly requests a change.
- Stop immediately after the chart is generated.
- If uncertain, ask for clarification instead of multiple calls.

# Anti-Loop Rule
If you already executed the Echarts tool, do not execute it again within this turn.

# Core Principles
{_rule_str}
"""
        try:
            messages = [HumanMessage(content=system_prompt)] + state["messages"]
            response = await model.ainvoke(messages)
            return {"messages": [response], "step_count": step_count + 1}

        except Exception as e:
            # 模型调用异常
            return {
                "messages": [AIMessage(content=f"❌ 模型调用失败：{e}，已停止执行。")],
                "step_count": step_count + 1,
            }

    def clean_tool_calls(state: EchartsGraphState):
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None

        # 提取所有图表数据
        charts = []
        tool_messages: filter[ToolMessage] = filter(lambda msg: isinstance(msg, ToolMessage) and msg.artifact, messages)

        chart_index = 0
        for msg in tool_messages:
            for art in msg.artifact:
                if isinstance(art, ImageContent):
                    # 构建符合 ChartImage 接口的数据
                    charts.append(
                        {
                            "id": f"chart-{chart_index}",
                            "data": art.data,
                            "mimeType": art.mimeType or "image/png",
                            "title": f"图表 {chart_index + 1}",
                        }
                    )
                    chart_index += 1

        # 清空工具调用
        if last_msg:
            last_msg.tool_calls = []
            if isinstance(last_msg.content, str):
                _content = last_msg.content.strip()
                last_msg.content = [{"type": "text", "text": _content}] if _content else []

                # 添加图片到消息内容（用于非 UI 渲染）
                # for chart in charts:
                #     last_msg.content.append(
                #         {
                #             "type": "image",
                #             "data": chart["data"],
                #             "source_type": "base64",
                #             "mime_type": chart["mimeType"],
                #         }
                #     )

        if charts:
            # 创建带组件配置的消息
            imgs = []
            for chart in charts:
                imgs.append(
                    {
                        "url": f"data:{chart['mimeType']};base64,{chart['data']}",
                        "alt": chart["title"],
                    }
                )
            final_message = AIMessage(
                id=str(uuid.uuid4()),
                content="这是一个图表可视化结果",
                additional_kwargs={
                    "type": "image",
                    "data": {
                        "layout": "carousel",
                        "images": imgs,
                        "caption": "图表可视化结果",
                    },
                },
            )
            state["messages"] = [*messages, final_message]
        state["step_count"] = 0

        return state

    # Step 3: 边逻辑（判断是否继续循环）
    def next_step(state: EchartsGraphState):
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        step_count = state.get("step_count", 0)
        max_allowed = state.get("max_steps", max_steps)

        _router: Literal["clean_tool_calls", "tools"] = "clean_tool_calls"

        # 达到步数上限
        if step_count > max_allowed:
            _router = "clean_tool_calls"

        # 如果是工具调用则循环回 llm
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            _router = "tools"

        match _router:
            case "clean_tool_calls":
                return "clean_tool_calls"
            case "tools":
                return "tools"
            case _:
                raise ValueError(f"Invalid router: {_router}")

    # ========== Graph 定义 ==========
    builder = StateGraph(EchartsGraphState)
    builder.add_node("llm", llm_node)
    # Step 2: 工具执行节点
    tool_node = ToolNode(tools)
    builder.add_node("tools", tool_node)
    builder.add_node("clean_tool_calls", clean_tool_calls)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", next_step, {"tools": "tools", "clean_tool_calls": "clean_tool_calls"})
    builder.add_edge("tools", "llm")
    builder.add_edge("clean_tool_calls", END)

    # 编译成可执行图
    graph = builder.compile(name="echart_agent")
    return graph
