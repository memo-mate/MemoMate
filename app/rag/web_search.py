import uuid

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import ToolMessage
from pydantic import BaseModel


class WebSearchResult(BaseModel):
    url: str
    content: str


class WebSearch:
    def search(self, query: str) -> ToolMessage:
        tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            # include_domains=[...],
            # exclude_domains=[...],
            # name="...",            # overwrite default tool name
            # description="...",     # overwrite default tool description
            # args_schema=...,       # overwrite default args_schema: BaseModel
        )

        model_generated_tool_call = {
            "args": {"query": query},
            "id": str(uuid.uuid4()),
            "name": "tavily",
            "type": "tool_call",
        }

        result: ToolMessage = tool.invoke(model_generated_tool_call)
        return result


web_search = WebSearch()
