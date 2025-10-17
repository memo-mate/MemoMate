"""Writer agent.用来演示如何使用LangGraph+Agent-Chat-UI的Web Component UI组件."""

import uuid
from collections.abc import Sequence
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.ui import AnyUIMessage, push_ui_message, ui_message_reducer


class AgentState(TypedDict):  # noqa: D101
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    ui_id: NotRequired[str]


class CreateTextDocument(TypedDict):
    """Prepare a document heading for the user."""

    title: str


async def writer_node(state: AgentState):
    """Writer node."""
    ui_id = str(uuid.uuid4())
    message = AIMessage(id=ui_id, content="Creating a document with the title: writer_node")
    push_ui_message(
        name="chart-preview",
        props={"title": "图表生成失败", "charts": [], "isLoading": True},
        id=ui_id,
        message=message,
        merge=True,
    )

    return {"messages": [message], "ui_id": ui_id}


async def writer_node_2(state: AgentState):
    """Writer node 2."""
    ui_id = state.get("ui_id")
    message = AIMessage(id=ui_id, content="Creating a document with the title: writer_node")
    push_ui_message(
        name="chart-preview",
        props={"title": "图表生成成功", "charts": [], "isLoading": False},
        id=ui_id,
        message=message,
        merge=True,
    )
    return {"messages": [message]}


workflow = StateGraph(AgentState)
workflow.add_node(writer_node)
workflow.add_node(writer_node_2)
workflow.add_edge("writer_node", "writer_node_2")
workflow.add_edge("writer_node_2", END)
workflow.add_edge("__start__", "writer_node")
writer_graph = workflow.compile()
