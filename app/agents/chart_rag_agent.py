from typing import NotRequired

from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_handoff_tool, create_supervisor

from app.agents.rag_agent import create_time_sensitive_rag_graph
from app.agents.sub_agent.echart_agent import build_echart_agent
from app.core import settings
from app.rag.llm.completions import LLM, LLMParams


class ChartRAGState(AgentState):
    """Chart RAG State combining RAG and Echarts capabilities."""

    step_count: NotRequired[int]
    max_steps: NotRequired[int]
    context: NotRequired[list[Document]]
    retrieved_docs: list[Document]


async def build_chart_rag_agent() -> CompiledStateGraph:
    """Build the chart RAG agent with a supervisor."""
    rag_agent = create_time_sensitive_rag_graph(name="rag_agent")
    echart_agent = await build_echart_agent()

    assistant_agent = create_supervisor(
        agents=[rag_agent, echart_agent],
        model=LLM().get_llm(LLMParams(model_name=settings.CHAT_MODEL)),
        supervisor_name="assistant_rag_agent",
        state_schema=ChartRAGState,
        tools=[
            create_handoff_tool(
                agent_name="rag_agent",
                name="assign_to_rag_agent",
                description="Transfer to RAG agent to search and retrieve information from the knowledge base. Use this for questions about projects, research topics, technical documents, specifications, budgets, and any other knowledge-based queries.",
            ),
            create_handoff_tool(
                agent_name="echart_agent",
                name="assign_to_echart_agent",
                description="Transfer to Echart agent to generate charts and data visualizations. Use this when user explicitly requests charts, graphs, or visual representations of data.",
            ),
        ],
        prompt="""\
You are a supervisor that coordinates between specialized agents to answer user questions.

Your role is to:
1. Analyze the user's question
2. Delegate to the appropriate agent - DO NOT answer directly
3. Return the agent's response to the user

Agent capabilities:
- rag_agent: Searches and retrieves information from the knowledge base. Use this for ANY question that might be answered from stored documents, including questions about projects, research topics, technical specifications, etc.
- echart_agent: Generates charts and visualizations based on data. Use this when the user explicitly asks for charts, graphs, or data visualization.

IMPORTANT: Always try rag_agent FIRST for knowledge-based questions. Only respond directly if the question is purely conversational (like greetings) or requires no information retrieval.
    """,
        output_mode="full_history",
        add_handoff_messages=False,
        add_handoff_back_messages=False,
    ).compile()
    return assistant_agent
