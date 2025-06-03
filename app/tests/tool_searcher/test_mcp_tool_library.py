import asyncio

from app.core.config import settings
from app.tool_searcher.mcp_tool_library import MCPToolLibrary
from app.utils.contextlib_tools import temporary_no_proxy


# 使用示例
async def test_mcp_library():
    # 初始化MCP工具库
    mcp_servers = {
        "crawl4-mcp": {"transport": "sse", "url": "http://localhost:8051/sse"},
    }

    mcp_lib = MCPToolLibrary(mcp_servers=mcp_servers, update_interval=60)

    # 加载MCP工具
    tools = await mcp_lib.load_tools()
    print(tools)

    # 根据问题描述找到相关工具
    relevant_tools = await mcp_lib.search_langchain_tools(
        "检查服务是否正常运行的工具", top_k=3, similarity_threshold=1.5
    )

    # 创建Agent
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    model = ChatOpenAI(model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE)
    agent = create_react_agent(model, relevant_tools)

    # 使用Agent回答问题
    with temporary_no_proxy():
        response = await agent.ainvoke({"messages": "check crawl4-mcp server is running normally"})

    print(response)


if __name__ == "__main__":
    asyncio.run(test_mcp_library())
