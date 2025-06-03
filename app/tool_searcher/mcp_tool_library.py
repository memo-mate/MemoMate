import json
import os
import time
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.embeddings import Embeddings
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.log_adapter import logger
from app.rag.embedding.embeeding_model import MemoMateEmbeddings
from app.tool_searcher.mcp_tool import PREFIX, MCPTool
from app.utils.contextlib_tools import temporary_no_proxy


class MCPToolLibrary:
    def __init__(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        chroma_sub_dir: str = "mcp_tools",
        chroma_base_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        + "/data/chroma/",
        description: str = "MCP Tool Library, providing various MCP service tools",
        default_timeout: int = 60,
        default_timeout_message: str = "Error: The tool did not return a response within the specified timeout.",
        update_interval: int = 3600,  # Default: update every hour
    ) -> None:
        self.mcp_servers = mcp_servers
        self.description = description
        self.default_timeout = default_timeout
        self.default_timeout_message = default_timeout_message
        self.update_interval = update_interval
        self.last_update_time = 0  # Timestamp of last update

        self.embedding_model: Embeddings = MemoMateEmbeddings.local_embedding()

        self.tools: dict[str, MCPTool] = {}
        self.mcp_client = None

        chroma_dir = chroma_base_dir + chroma_sub_dir
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="mcp_tools")

        stored_tools = self.collection.get(include=["metadatas"])
        stored_tools_ids = stored_tools["ids"]
        for i, _tool_id in enumerate(stored_tools_ids):
            metadata = stored_tools["metadatas"][i]
            logger.info(f"Loaded MCP tool definition from vector store: {metadata['name']}")

    async def load_tools(self) -> list[MCPTool]:
        try:
            with temporary_no_proxy():
                self.mcp_client = MultiServerMCPClient(self.mcp_servers)

                # Track current tool IDs from server
                current_tool_ids = set()

                lc_tools = await self.mcp_client.get_tools()
                new_tools = []
                updated_tools = []

                for lc_tool in lc_tools:
                    mcp_tool = MCPTool(lc_tool)
                    current_tool_ids.add(mcp_tool.unique_id)

                    # Check if tool already exists and needs updating
                    is_new_tool = True
                    if mcp_tool.unique_id in self.tools:
                        # Tool exists, check if description has changed
                        old_tool = self.tools[mcp_tool.unique_id]
                        if old_tool.description != mcp_tool.description:
                            updated_tools.append(mcp_tool)
                        is_new_tool = False

                    self.tools[mcp_tool.unique_id] = mcp_tool

                    # Check if it's a new tool
                    if is_new_tool:
                        try:
                            search_result = self.collection.get(ids=[mcp_tool.unique_id], include=[])
                            if not search_result or not search_result.get("ids"):
                                new_tools.append(mcp_tool)
                        except Exception:
                            new_tools.append(mcp_tool)

            # Find and remove tools that no longer exist on server
            await self._remove_deleted_tools(current_tool_ids)

            if new_tools:
                self._save_to_vector_store(new_tools)
                logger.info(f"Added {len(new_tools)} new MCP tools to vector database.")

            if updated_tools:
                self._update_in_vector_store(updated_tools)
                logger.info(f"Updated {len(updated_tools)} MCP tools in vector database")

            # Update timestamp
            self.last_update_time = time.time()
            logger.info(f"MCP工具库中共有{len(self.tools)}个工具。")
            return list(self.tools.values())

        except Exception as e:
            logger.exception("Error loading MCP tools", exc_info=e)
            return []

    def _save_to_vector_store(self, tools: list[MCPTool]) -> None:
        tool_lookup = {tool.unique_id: tool for tool in tools}
        logger.info(f"Adding tools to collection {self.collection}: {tool_lookup.keys()}")

        self.collection.add(
            documents=[json.dumps(tool.definition, indent=4) for tool in tool_lookup.values()],
            embeddings=[self.save_embeddings(tool.description) for tool in tool_lookup.values()],
            metadatas=[tool.format_for_chroma() for tool in tool_lookup.values()],
            ids=list(tool_lookup.keys()),
        )

    def _update_in_vector_store(self, tools: list[MCPTool]) -> None:
        if not tools:
            return

        for tool in tools:
            try:
                self.collection.update(
                    ids=[tool.unique_id],
                    documents=json.dumps(tool.definition, indent=4),
                    embeddings=self.save_embeddings(tool.description),
                    metadatas=tool.format_for_chroma(),
                )
            except Exception as e:
                logger.error(f"Error updating tool {tool.name}: {str(e)}")

    async def _remove_deleted_tools(self, current_tool_ids: set) -> None:
        """
        Remove tools that exist in the database but no longer exist on the server
        """
        # Get all tool IDs from vector database
        stored_tools = self.collection.get(include=[])
        stored_tool_ids = set(stored_tools["ids"])

        # Find tool IDs that exist in database but not on server
        deleted_tool_ids = []
        for tool_id in stored_tool_ids:
            # Only consider MCP tools (those with the prefix mcp_tool_)
            if tool_id.startswith(PREFIX) and tool_id not in current_tool_ids:
                deleted_tool_ids.append(tool_id)

        if deleted_tool_ids:
            # Remove from vector database
            self.collection.delete(ids=deleted_tool_ids)

            # Remove from memory cache
            for tool_id in deleted_tool_ids:
                if tool_id in self.tools:
                    del self.tools[tool_id]

            logger.info(f"Removed {len(deleted_tool_ids)} obsolete tools that no longer exist on the MCP server")

    def save_embeddings(self, text: str) -> list[float]:
        embeddings = self.embedding_model.embed_documents([text])
        return embeddings[0]

    def query_embeddings(self, text: str) -> list[float]:
        embeddings = self.embedding_model.embed_query(text)
        return embeddings

    async def check_and_update(self) -> bool:
        """Check if tools need updating and update if necessary"""
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            logger.info("MCP tool information expired, updating...")
            await self.load_tools()
            return True
        return False

    async def search(
        self, problem_description: str, top_k: int = 1, similarity_threshold: float = None
    ) -> list[MCPTool]:
        await self.check_and_update()

        if top_k >= len(self.tools) and similarity_threshold is None:
            return list(self.tools.values())

        query_embedding = self.query_embeddings(problem_description)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["distances"])

        cutoff = top_k
        if similarity_threshold:
            for c, distance in enumerate(results["distances"][0]):
                if distance >= similarity_threshold:
                    cutoff = c
                    break

        found_tools = [self.tools[tool_id] for tool_id in results["ids"][0][:cutoff] if tool_id in self.tools]

        return found_tools

    async def search_langchain_tools(
        self, problem_description: str, top_k: int = 1, similarity_threshold: float = None
    ) -> list:
        mcp_tools = await self.search(problem_description, top_k, similarity_threshold)

        langchain_tools = [tool.langchain_tool for tool in mcp_tools]

        logger.info(f"Found {len(langchain_tools)} relevant tools for query: '{problem_description}'")
        return langchain_tools

    async def execute(self, tool_id: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
        if tool_id not in self.tools:
            return (f"Error: {tool_id} is not a valid tool. Please use only available tools.", True)

        tool = self.tools[tool_id]
        return await tool.execute(**arguments)
