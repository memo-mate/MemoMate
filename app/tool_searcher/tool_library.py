import importlib
import json
import sys
from collections import Counter
from collections.abc import Callable
from inspect import getmembers, isfunction
from os.path import abspath, dirname
from pathlib import Path

import chromadb
from langchain_core.embeddings import Embeddings

from app.core.log_adapter import logger
from app.rag.embedding.embeeding_model import MemoMateEmbeddings
from app.tool_searcher.function_analyzer import FunctionAnalyzer
from app.tool_searcher.tool import Tool


# TODO: (Author: Daoji 2025-05-12 21:38:17) 需要添加 MCP 动态加载工具嵌入功能
class ToolLibrary:
    def __init__(
        self,
        chroma_sub_dir: str = "",
        file_imports: list[tuple[str, list[str] | None]] | None = None,
        instance_imports: list[object] | None = None,
        chroma_base_dir: str = dirname(dirname(dirname(abspath(__file__)))) + "/data/chroma/",
        description: str | None = None,
        default_timeout: int = 60,
        default_timeout_message: str = ("Error: The tool did not return a response within the specified timeout."),
        timeout_settings: dict | None = None,
        verbose_tool_ids: bool = False,
    ) -> None:
        """
        Initialize the tool library: set up the vector store and load the tool information.

        :param chroma_sub_dir: A specific subfolder for the tool library.
        :param file_imports: List of tuples with a module name from which to load tools from and
            an optional list of tools to load. If no tools are specified, all tools are loaded.
        :param instance_imports: List of instances of classes from which to load tools.
        :param chroma_base_dir: Absolute path to the tool library folder.
        :param embedding_model: Name of the embedding model used. Defaults to the one specified in constants.
        :param description: Natural language description of the tool library.
        :param default_timeout: Execution timeout for tools.
        :param default_timeout_message: Default message returned in case of tool execution timeout.
        :param timeout_settings: Tool-specific timeout settings of the form
            {"module_name__tool_name": {"timeout": seconds, "timeout_message": string}}
            NOTE: overriding existing timeout settings is not supported
        :param verbose_tool_ids: Includes module information in tool ID if set to true.
        """
        self.description = description
        self.embedding_model: Embeddings = MemoMateEmbeddings.local_embedding()
        self.function_analyzer = FunctionAnalyzer()
        self.tools: dict[str, Tool] = {}
        self.verbose_tool_ids = verbose_tool_ids

        # timeout settings
        self.default_timeout = default_timeout
        self.default_timeout_message = default_timeout_message
        timeout_settings = timeout_settings if timeout_settings else {}

        # set up directory
        chroma_dir = chroma_base_dir + chroma_sub_dir
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)

        # vector store
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="tulip")
        stored_tools = self.collection.get(include=["metadatas"])
        stored_tools_ids = stored_tools["ids"]

        # load existing tools from vector store and remove unspecified ones from vector store
        class_counts = Counter(type(instance).__name__ for instance in instance_imports) if instance_imports else {}
        duplicates = [t for t, count in class_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate instances detected for classes: {', '.join(duplicates)}")

        instances_by_class = (
            {instance.__class__.__name__: instance for instance in instance_imports} if instance_imports else {}
        )
        functions_by_file = dict(file_imports) if file_imports else {}

        for metadata in stored_tools["metadatas"]:
            instance = None

            if class_name := metadata["class_name"]:
                if class_name not in instances_by_class:
                    self.collection.delete(ids=[metadata["unique_id"]])
                    continue
                else:
                    instance = instances_by_class[class_name]
            else:
                module_name = metadata["module_name"]
                if module_name not in functions_by_file:
                    self.collection.delete(ids=[metadata["unique_id"]])
                    continue
                if functions_by_file[module_name] and metadata["function_name"] not in functions_by_file[module_name]:
                    self.collection.delete(ids=[metadata["unique_id"]])
                    continue

            tool = Tool(
                function_name=metadata["function_name"],
                module_name=metadata["module_name"],
                definition=json.loads(metadata["definition"]),
                instance=instance if instance else None,
                timeout=metadata["timeout"],
                timeout_message=metadata["timeout_message"],
                predecessor=(metadata["predecessor"] if "predecessor" in metadata else None),
                successor=metadata["successor"] if "successor" in metadata else None,
                verbose_id=self.verbose_tool_ids,
            )
            self.tools[tool.unique_id] = tool
        logger.info(
            f"Removed {len(stored_tools['metadatas']) - len(self.tools)} tools from collection {self.collection.name}."
        )
        logger.info(f"Loaded {len(self.tools)} tools from collection {self.collection.name}.")

        # load new tools from files and instances
        if not file_imports and not instance_imports:
            return
        file_imports = file_imports if file_imports else []
        for file_import in file_imports:
            module_name, function_names = file_import
            module = importlib.import_module(module_name)
            functions = [
                f
                for n, f in getmembers(module, isfunction)
                if f.__module__ == module_name and (not function_names or n in function_names)
            ]
            for function in functions:
                function_definition = self.function_analyzer.analyze_function(function)
                tool = Tool(
                    function_name=function.__name__,
                    module_name=module_name,
                    definition=function_definition,
                    timeout=self.default_timeout,
                    timeout_message=self.default_timeout_message,
                    verbose_id=self.verbose_tool_ids,
                )
                if tool.unique_id in timeout_settings:
                    tool.timeout = timeout_settings[tool.unique_id]["timeout"]
                    tool.timeout_message = timeout_settings[tool.unique_id]["timeout_message"]
                self._add_to_tools(tool)
        instance_imports = instance_imports if instance_imports else []
        for instance_import in instance_imports:
            function_definitions = self.function_analyzer.analyze_class(instance_import.__class__)
            for function_definition in function_definitions:
                tool = Tool(
                    function_name=function_definition["function"]["name"],
                    module_name=instance_import.__module__,
                    instance=instance_import,
                    definition=function_definition,
                    timeout=self.default_timeout,
                    timeout_message=self.default_timeout_message,
                    verbose_id=self.verbose_tool_ids,
                )
                if tool.unique_id in timeout_settings:
                    tool.timeout = timeout_settings[tool.unique_id]["timeout"]
                    tool.timeout_message = timeout_settings[tool.unique_id]["timeout_message"]
                self._add_to_tools(tool)
                self.tools[tool.unique_id] = tool

        # store new functions in vector store
        new_tools = [tool for tool_id, tool in self.tools.items() if tool_id not in stored_tools_ids]
        if not new_tools:
            return
        self._save_to_vector_store(new_tools)
        logger.info(f"Added {len(new_tools)} new tools to collection {self.collection.name}.")

    def save_embeddings(self, text: str) -> list[float]:
        embeddings = self.embedding_model.embed_documents([text])
        return embeddings[0]

    def query_embeddings(self, text: str) -> list[float]:
        embeddings = self.embedding_model.embed_query(text)
        return embeddings

    def _add_to_tools(self, tool: Tool) -> None:
        if tool.unique_id in self.tools:
            if tool.module_path != self.tools[tool.unique_id].module_path:
                raise ValueError(
                    f"Name clash for `{tool.unique_id}`. "
                    f"Exists in {tool.module_path} and {self.tools[tool.unique_id].module_path}. "
                    "Consider using `verbose_tool_ids` - requires reloading the library."
                )
        else:
            self.tools[tool.unique_id] = tool

    def _save_to_vector_store(self, tools: list[Tool]) -> None:
        tool_lookup = {tool.unique_id: tool for tool in tools}
        logger.info(f"Adding tools to collection {self.collection}: {tool_lookup.keys()}")
        self.collection.add(
            documents=[json.dumps(tool.definition, indent=4) for tool in tool_lookup.values()],
            embeddings=[self.save_embeddings(tool.description) for tool in tool_lookup.values()],
            metadatas=[tool.format_for_chroma() for tool in tool_lookup.values()],
            ids=list(tool_lookup.keys()),
        )

    def _add_function(
        self,
        function: Callable,
        module_name: str,
        timeout: int | None = None,
        timeout_message: str | None = None,
    ) -> Tool:
        function_definition = self.function_analyzer.analyze_function(function)
        tool = Tool(
            function_name=function.__name__,
            module_name=module_name,
            definition=function_definition,
            timeout=timeout if timeout is not None else self.default_timeout,
            timeout_message=(timeout_message if timeout_message is not None else self.default_timeout_message),
            verbose_id=self.verbose_tool_ids,
        )
        self._add_to_tools(tool)
        self._save_to_vector_store(tools=[tool])
        return tool

    def load_functions_from_file(
        self,
        module_name: str,
        function_names: list[str] | None = None,
        timeout_settings: dict | None = None,
    ) -> list[Tool]:
        timeout_settings = timeout_settings if timeout_settings else {}
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
        functions = [
            f
            for n, f in getmembers(module, isfunction)
            if f.__module__ == module_name and (not function_names or n in function_names)
        ]
        tools = []
        for function in functions:
            tool_id = f"{module_name}__{function.__name__}"
            timeout_settings_ = timeout_settings[tool_id] if tool_id in timeout_settings else {}
            tool = self._add_function(
                function=function,
                module_name=module_name,
                **timeout_settings_,
            )
            tools.append(tool)
        return tools

    def load_functions_from_instance(self, instance: object, timeout_settings: dict | None = None) -> list[Tool]:
        timeout_settings = timeout_settings if timeout_settings else {}
        function_definitions = self.function_analyzer.analyze_class(instance.__class__)
        new_tools = []
        for function_definition in function_definitions:
            if function_definition["function"]["name"].startswith("_"):
                continue
            tool = Tool(
                function_name=function_definition["function"]["name"],
                module_name=instance.__module__,
                instance=instance,
                definition=function_definition,
                timeout=self.default_timeout,
                timeout_message=self.default_timeout_message,
                verbose_id=self.verbose_tool_ids,
            )
            if tool.unique_id in timeout_settings:
                tool.timeout = timeout_settings[tool.unique_id]["timeout"]
                tool.timeout_message = timeout_settings[tool.unique_id]["timeout_message"]
            new_tools.append(tool)
            self._add_to_tools(tool)

        self._save_to_vector_store(tools=new_tools)
        return new_tools

    def remove_tool(
        self,
        tool_id: str,
    ) -> None:
        self.collection.delete(ids=[tool_id])
        self.tools.pop(tool_id)
        logger.info(f"Removed tool {tool_id} from collection {self.collection}.")

    def remove_tools_by_instance(
        self,
        instance: object,
    ) -> None:
        to_be_removed = [tool_id for tool_id, tool in self.tools.items() if tool.instance == instance]
        self.collection.delete(ids=to_be_removed)
        for tool_id in to_be_removed:
            self.tools.pop(tool_id)
        logger.info(f"Removed tools {to_be_removed} from collection {self.collection}.")

    def update_tool(
        self,
        tool_id: str,
        timeout: int | None = None,
        timeout_message: str | None = None,
    ) -> Tool:
        old_tool = self.tools[tool_id]
        module_name = old_tool.module_name
        timeout = timeout or old_tool.timeout
        timeout_message = timeout_message or old_tool.timeout_message

        if old_tool.instance:
            raise ValueError(
                f"The update operation is only supported for modules with exactly one function. "
                f"{tool_id} was loaded from an instance of class {old_tool.instance.__class__.__name__}."
            )
        module_occurrences = len([t for t in self.tools.values() if t.module_name == module_name])
        if module_occurrences != 1:
            raise ValueError(
                f"The update operation is only supported for modules with exactly one function. "
                f"{module_name} includes {module_occurrences}."
            )

        module = importlib.reload(old_tool.module)
        function = getattr(module, old_tool.function_name)

        self.remove_tool(tool_id)
        tool = self._add_function(
            function=function,
            module_name=module.__name__,
            timeout=timeout,
            timeout_message=timeout_message,
        )
        return tool

    def search(
        self,
        problem_description: str,
        top_k: int = 1,
        similarity_threshold: float = None,
    ) -> list[Tool]:
        if top_k >= len(self.tools) and similarity_threshold is None:
            res = self.tools.values()
        else:
            query_embedding = self.query_embeddings(problem_description)
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

    def execute(
        self,
        tool_id: str,
        arguments: dict,
    ) -> tuple:
        if tool_id not in self.tools:
            return (
                f"Error: {tool_id} is not a valid tool. Use only the tools available.",
                True,
            )
        tool = self.tools[tool_id]
        return tool.execute(**arguments)

    async def execute_async(
        self,
        tool_id: str,
        arguments: dict,
    ) -> tuple:
        if tool_id not in self.tools:
            return (
                f"Error: {tool_id} is not a valid tool. Use only the tools available.",
                True,
            )
        tool = self.tools[tool_id]
        return await tool.execute_async(**arguments)
