from collections import Counter
from pathlib import Path

import chromadb
import orjson
from fastmcp import Client as FastMCPClient
from fastmcp import FastMCP
from fastmcp.client.client import CallToolResult
from fastmcp.tools import Tool
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.core.consts import DATA_DIR
from app.core.log_adapter import logger
from app.rag.embedding.embeeding_model import EmbeddingFactory


class ToolLibrary:
    def __init__(
        self,
        instance_imports: list[object] | None = None,
        data_dir: str = DATA_DIR / "chroma",
        description: str | None = None,
        default_timeout: int = 60,
        timeout_settings: dict | None = None,
        reload_collection: bool = True,
    ) -> None:
        """
        Initialize the tool library: set up the vector store and load the tool information.

        :param instance_imports: List of instances of classes from which to load tools.
        :param description: Natural language description of the tool library.
        :param default_timeout: Execution timeout for tools.
        :param timeout_settings: Tool-specific timeout settings.
        :param reload_collection: Whether to reload the collection.
        """
        self.description = description
        self.embedding_model: Embeddings = EmbeddingFactory.get()
        self.tool_library_mcp = FastMCP("memomate-tools")
        self.internal_mcp_client = FastMCPClient(self.tool_library_mcp)
        self.external_mcp_client = FastMCPClient(settings.MCP_CONFIG)
        self.__tools: dict[str, Tool] = {}
        self.__internal_tools: set[str] = set()
        self.__external_tools: set[str] = set()
        self.reload_collection = reload_collection

        # timeout settings
        self.default_timeout = default_timeout
        self.timeout_settings = timeout_settings if timeout_settings else {}

        # set up directory
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        # vector store
        self.chroma_client = chromadb.PersistentClient(path=data_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="memomate-tools")
        # load new tools from instances
        if not instance_imports:
            return
        # load existing tools from vector store and remove unspecified ones from vector store
        class_counts = Counter(type(instance).__name__ for instance in instance_imports) if instance_imports else {}
        duplicates = [t for t, count in class_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate instances detected for classes: {', '.join(duplicates)}")

        for instance_import in instance_imports:
            for name, func in instance_import.__dict__.items():
                if name.startswith("_"):
                    continue
                if isinstance(func, staticmethod) or isinstance(func, classmethod):
                    func = func.__func__
                self.tool_library_mcp.tool(func)

    @property
    def tools(self) -> dict[str, Tool]:
        if not self.__tools:
            raise RuntimeError("Tools are not loaded. Please call ainit() first.")
        return self.__tools

    @tools.setter
    def tools(self, tools: dict[str, Tool]) -> None:
        self.__tools = tools

    async def ainit(self) -> None:
        """
        Initialize the tool library: set up the vector store and load the tool information.
        """
        await self.reload_tools()

        # 获取vector store中的tools
        stored_tools = self.collection.get(include=["metadatas"])
        stored_tools_ids = stored_tools["ids"]
        new_tools = []
        if self.reload_collection:
            if stored_tools_ids:
                self.collection.delete(ids=stored_tools_ids)
                logger.info(f"Clear {len(stored_tools_ids)} tools from collection {self.collection.name}.")
            new_tools = list(self.tools.values())
        else:
            to_be_removed = [tool_id for tool_id in stored_tools_ids if tool_id not in self.tools]
            if to_be_removed:
                self.collection.delete(ids=to_be_removed)
            logger.info(f"Removed {len(to_be_removed)} tools from collection {self.collection.name}.")
            new_tools = [self.tools[tool_id] for tool_id in self.tools.keys() if tool_id not in stored_tools_ids]

        self._save_to_vector_store(new_tools)
        logger.info(f"Loaded {len(self.tools)} tools from collection {self.collection.name}.")

    def save_embeddings(self, text: str) -> list[float]:
        """
        Save the embeddings of the text.
        """
        embeddings = self.embedding_model.embed_documents([text])
        return embeddings[0]

    def query_embeddings(self, text: str) -> list[float]:
        """
        Query the embeddings of the text.
        """
        embeddings = self.embedding_model.embed_query(text)
        return embeddings

    def _save_to_vector_store(self, tools: list[Tool]) -> None:
        logger.info(f"Adding tools to collection {self.collection}: {self.tools.keys()}")

        # 将复杂的 metadata 转换为 ChromaDB 支持的简单类型
        simplified_metadatas = []
        for tool in self.tools.values():
            tool_data = tool.model_dump()
            simplified_metadata = {
                "name": tool_data.get("name", ""),
                "description": tool_data.get("description", ""),
                # 将复杂对象转换为 JSON 字符串
                "inputSchema": tool_data.get("inputSchema") and orjson.dumps(tool_data["inputSchema"]).decode(),
                # 添加其他简单字段
                "outputSchema": tool_data.get("outputSchema") and orjson.dumps(tool_data["outputSchema"]).decode(),
            }
            # 过滤掉 None 值和空字符串
            simplified_metadata = {k: v for k, v in simplified_metadata.items() if v}
            simplified_metadatas.append(simplified_metadata)

        self.collection.add(
            documents=[tool.model_dump_json() for tool in self.tools.values()],
            embeddings=[self.save_embeddings(tool.description) for tool in self.tools.values()],
            metadatas=simplified_metadatas,
            ids=list(self.tools.keys()),
        )

    def update_tool(self) -> None:
        # TODO: (Author: Daoji 2025-07-09 18:04:38) 动态更新mcp配置
        pass

    def search(
        self,
        query: str,
        top_k: int = 1,
        similarity_threshold: float | None = None,
    ) -> list[Tool]:
        """
        Search the tool library for tools that are similar to the query.
        """
        if top_k >= len(self.tools) and similarity_threshold is None:
            res = self.tools.values()
        else:
            query_embedding = self.query_embeddings(query)
            res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["distances"],
            )
            cutoff = top_k
            if similarity_threshold:
                for c, distance in enumerate(res["distances"][0]):
                    if distance >= similarity_threshold:
                        cutoff = c
                        break
            res = [self.tools[tool_id] for tool_id in res["ids"][0][:cutoff]]
        return res

    def load_tools_from_class(self, module_class: object) -> None:
        """
        Loading tools from module.

        1. Function to traverse tool classes
        2. Exclude functions starting with an underscore
        """
        for name, func in module_class.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(func, staticmethod) or isinstance(func, classmethod):
                func = func.__func__
            self.tool_library_mcp.tool(func)

    async def reload_tools(self) -> dict[str, Tool]:
        """
        Reload the tools.
        """
        async with self.internal_mcp_client:
            internal_tools = await self.internal_mcp_client.list_tools()
            self.__internal_tools = {tool.name for tool in internal_tools}
        async with self.external_mcp_client:
            external_tools = await self.external_mcp_client.list_tools()
            self.__external_tools = {tool.name for tool in external_tools}

        self.tools = {tool.name: tool for tool in internal_tools + external_tools}

        return self.tools

    async def acall_tool(self, tool_id: str, arguments: dict) -> CallToolResult:
        """
        Call a tool.

        :param tool_id: The ID of the tool to call.
        :param arguments: The arguments to pass to the tool.
        :return: The result of the tool call.
        """
        if tool_id in self.__external_tools:
            return await self.external_mcp_client.call_tool(tool_id, arguments)
        elif tool_id in self.__internal_tools:
            return await self.internal_mcp_client.call_tool(tool_id, arguments)
        else:
            raise ValueError(f"Error: {tool_id} is not a valid tool. Use only the tools available.")
