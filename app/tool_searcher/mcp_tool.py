import json
from typing import Any

from langchain_core.tools import BaseTool

from app.core.log_adapter import logger
from app.utils.contextlib_tools import temporary_no_proxy

PREFIX = "mcp_tool_"


class MCPTool:
    def __init__(self, langchain_tool: BaseTool):
        self.langchain_tool = langchain_tool
        self.name = langchain_tool.name
        self.description = langchain_tool.description
        self.parameters = langchain_tool.args_schema if langchain_tool.args_schema else {}
        self.unique_id = f"{PREFIX}{self.name}"

    async def execute(self, **kwargs) -> tuple[Any, bool]:
        try:
            with temporary_no_proxy():
                if hasattr(self.langchain_tool, "_run") or hasattr(self.langchain_tool, "_arun"):
                    result = await self.langchain_tool.ainvoke(input=kwargs)
                else:
                    result = await self.langchain_tool.ainvoke(kwargs)
                return (result, False)
        except Exception as e:
            logger.exception(f"Error executing MCP tool {self.name}", exc_info=e)
            return (f"Error executing tool: {str(e)}", True)

    @property
    def definition(self) -> dict:
        return {
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def format_for_chroma(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": json.dumps(self.parameters),
            "type": "mcp_tool",
        }
