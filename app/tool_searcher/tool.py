from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import inspect
import json
import logging
import os
import sys
import typing
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import Any, Union

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Tool:
    function_name: str
    module_name: str
    definition: dict
    instance: object | None = None
    class_name: str = ""
    timeout: int | None = None
    timeout_message: str | None = None
    predecessor: str | None = None
    successor: str | None = None
    verbose_id: bool = False
    description: str = field(init=False)
    unique_id: str = field(init=False)
    module_path: str = field(init=False)

    def __post_init__(self) -> None:
        self.module: ModuleType = (
            sys.modules[self.module_name]
            if self.module_name in sys.modules
            else importlib.import_module(self.module_name)
        )
        self.module_path = os.path.abspath(self.module.__file__)
        clean_module_name = self.module_name.replace(".", "__")
        if self.instance:
            if self.verbose_id:
                self.unique_id = f"{clean_module_name}__{self.instance.__class__.__name__}__{self.function_name}"
            else:
                self.unique_id = self.function_name
            self.function: Callable = getattr(self.instance, self.function_name)
            self.class_name = self.instance.__class__.__name__
        else:
            if self.verbose_id:
                self.unique_id = f"{clean_module_name}__{self.function_name}"
            else:
                self.unique_id = self.function_name
            self.function: Callable = getattr(self.module, self.function_name)
        self.description = self.function_name + ":\n" + self.definition["function"]["description"]
        self.definition["function"]["name"] = self.unique_id

        # 添加返回值信息
        self._add_return_type_to_definition()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object {id(self)}: {self.unique_id}>"

    def format_for_chroma(self) -> dict:
        flat_dict = asdict(self)
        flat_dict["definition"] = json.dumps(self.definition, indent=4)
        if self.predecessor is None:
            flat_dict.pop("predecessor")
        if self.successor is None:
            flat_dict.pop("successor")
        flat_dict.pop("instance")
        return flat_dict

    def execute(self, **parameters) -> Any:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                future = executor.submit(self.function, **parameters)
            except Exception as e:
                logger.error(f"{type(e).__name__}: {e}")
                return f"Error: Invalid tool call for {self.unique_id}: {e}", True
            try:
                res = future.result(timeout=self.timeout)
                error = False
            except concurrent.futures.TimeoutError as e:
                logger.error(f"{type(e).__name__}: {self.unique_id} did not return a result before timeout.")
                return self.timeout_message, True
            except Exception as e:
                logger.error(f"{type(e).__name__}: {e}")
                return f"Error: Invalid tool call for {self.unique_id}: {e}", True
        return res, error

    async def execute_async(self, **parameters) -> Any:
        try:
            # 判断函数是否为异步函数
            func = self.function
            is_coroutine = inspect.iscoroutinefunction(func)

            # 设置超时
            try:
                if is_coroutine:
                    # 如果是异步函数，直接使用await调用
                    res = await asyncio.wait_for(func(**parameters), timeout=self.timeout)
                else:
                    # 如果是同步函数，使用run_in_executor执行
                    loop = asyncio.get_event_loop()
                    future = loop.run_in_executor(None, lambda: func(**parameters))
                    res = await asyncio.wait_for(future, timeout=self.timeout)

                error = False
            except TimeoutError:
                logger.error(f"TimeoutError: {self.unique_id} did not return a result before timeout.")
                return self.timeout_message, True

        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            return f"Error: Invalid tool call for {self.unique_id}: {e}", True

        return res, error

    def _add_return_type_to_definition(self) -> None:
        """
        分析函数的返回类型注解并将其添加到definition中。
        支持标准类型、泛型类型和Pydantic模型类型。
        """
        try:
            # 获取函数的返回类型注解
            return_annotation = inspect.signature(self.function).return_annotation

            # 如果返回类型不是inspect._empty，则添加到definition
            if return_annotation != inspect.Signature.empty:
                # 处理返回类型
                return_type_info = self._get_return_type_info(return_annotation)

                # 添加返回类型到definition中
                if "returns" not in self.definition["function"]:
                    self.definition["function"]["returns"] = return_type_info

                # 从docstring中提取返回值描述
                # if self.function.__doc__:
                #     return_desc_match = None
                #     if ":return:" in self.function.__doc__:
                #         return_desc_match = self.function.__doc__.split(":return:")[1].split("\n")[0].strip()
                #     elif ":returns:" in self.function.__doc__:
                #         return_desc_match = self.function.__doc__.split(":returns:")[1].split("\n")[0].strip()

                #     if return_desc_match:
                #         self.definition["function"]["returns"]["description"] = return_desc_match
        except Exception as e:
            # 如果分析过程中发生错误，记录但不中断流程
            logger.warning(f"无法分析函数 {self.unique_id} 的返回类型: {e}")
            pass

    def _get_return_type_info(self, annotation) -> dict:
        """
        获取返回类型的详细信息，支持标准类型、泛型类型和Pydantic模型

        :param annotation: 类型注解
        :return: 包含类型信息的字典
        """
        # 检查是否为Pydantic模型
        if self._is_pydantic_model(annotation):
            # 如果是Pydantic模型，使用其json_schema方法获取完整的模式
            model_schema = annotation.model_json_schema()
            return {"type": "object", "description": f"Pydantic模型: {annotation.__name__}", "schema": model_schema}

        # 处理标准类型
        if not hasattr(annotation, "__origin__"):
            # 处理简单类型如str, int等
            type_name = getattr(annotation, "__name__", str(annotation))
            return {"type": type_name, "description": f"返回值类型: {type_name}"}

        # 处理泛型类型
        origin = annotation.__origin__
        if isinstance(origin, tuple):
            # 特殊处理tuple类型
            args = [self._get_type_name(arg) for arg in annotation.__args__]
            item_schemas = [self._get_return_type_info(arg) for arg in annotation.__args__]
            if len(args) == 2 and args[1] == "...":
                # 对于Tuple[X, ...] 类型
                return {"type": "array", "description": f"返回值类型: tuple[{args[0]}, ...]", "items": item_schemas[0]}
            else:
                # 对于Tuple[X, Y, Z] 类型
                return {
                    "type": "array",
                    "description": f"返回值类型: tuple[{', '.join(args)}]",
                    "prefixItems": item_schemas,
                }
        elif isinstance(origin, list | set | frozenset):
            # 处理list, set, frozenset类型
            if annotation.__args__:
                item_type = self._get_type_name(annotation.__args__[0])
                # 递归处理列表元素类型
                item_schema = self._get_return_type_info(annotation.__args__[0])
                return {
                    "type": "array",
                    "description": f"返回值类型: {origin.__name__}[{item_type}]",
                    "items": item_schema,
                }
            else:
                return {"type": "array", "description": f"返回值类型: {origin.__name__}"}
        elif isinstance(origin, dict):
            # 处理dict类型
            if len(annotation.__args__) >= 2:
                key_type = self._get_type_name(annotation.__args__[0])
                value_type = self._get_type_name(annotation.__args__[1])
                # 递归处理字典值类型
                value_schema = self._get_return_type_info(annotation.__args__[1])
                return {
                    "type": "object",
                    "description": f"返回值类型: dict[{key_type}, {value_type}]",
                    "additionalProperties": value_schema,
                }
            else:
                return {"type": "object", "description": "返回值类型: dict"}
        elif origin == Union or origin == getattr(typing, "_Union", None):
            # 处理Union类型
            types = [self._get_type_name(arg) for arg in annotation.__args__]
            # 递归处理联合类型中的每个类型
            type_schemas = [self._get_return_type_info(arg) for arg in annotation.__args__]
            return {"type": "oneOf", "description": f"返回值类型: Union[{', '.join(types)}]", "oneOf": type_schemas}
        else:
            # 其他类型
            origin_name = getattr(origin, "__name__", str(origin))
            args = [self._get_type_name(arg) for arg in annotation.__args__]
            return {"type": origin_name, "description": f"返回值类型: {origin_name}[{', '.join(args)}]"}

    def _get_type_name(self, type_annotation) -> str:
        """
        获取类型注解的名称

        :param type_annotation: 类型注解
        :return: 类型名称字符串
        """
        if hasattr(type_annotation, "__origin__"):
            origin = getattr(type_annotation.__origin__, "__name__", str(type_annotation.__origin__))
            args = [self._get_type_name(arg) for arg in type_annotation.__args__]
            return f"{origin}[{', '.join(args)}]"
        return getattr(type_annotation, "__name__", str(type_annotation))

    def _is_pydantic_model(self, cls) -> bool:
        """
        检查一个类是否为Pydantic模型

        :param cls: 要检查的类
        :return: 是否为Pydantic模型
        """
        return hasattr(cls, "__mro__") and any(
            base.__module__ == "pydantic.main" and base.__name__ == "BaseModel" for base in cls.__mro__
        )
