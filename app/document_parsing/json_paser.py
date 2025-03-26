import os

import orjson
import pandas as pd
from langchain_core.documents import Document

from app.core import consts, logger
from app.enums.task import DocumentFileTaskType
from app.rag.llm.tokenizers import TokenCounter, TokenizerType

"""
Todo List:
 - [ ] 解析一
 - [x] 解析二
 - [x] 解析三
"""

"""
解析一

json 文件结构示例:
{
    "name": "John",
    "age": 30,
    "city": ["New York", true, 1],
    "address": [
        {
            "street": "123 Main St",
            "city": "Anytown",
            "zip": "12345"
        },
        {
            "street": "456 Oak Ave",
            "city": "Othertown",
            "zip": "67890",
            "country": "USA"
        },
        {
            "street": "789 Pine Rd",
            "city": "Smallville",
            "zip": "98765",
            "house_number": 10
        }
    ]
}
1. 读取json文件
2. 解析 json 文件，获取 json_schema, 保留深度 4 以内的 json_schema
3. 将 json内容 转换为 Document:content, 文件名、json_schema 作为 metadata
4. 将 Document 转换为 list[Document]

PS:
1. 支持 list[Any] 类型
2. 支持 dict[str, Any] 类型
"""

"""
解析二

1. 读取json文件
2. json 扁平化
3. 将 key/columns 作为 Document 的 metadata
4. 将 json内容 转换为 Document:content, key/columns 作为 metadata
5. 将 Document 转换为 list[Document]

PS:
1. 支持 list[Any] 类型
2. 支持 dict[str, Any] 类型
"""

"""
解析三

1. 读取jsonl文件
2. 按 jsonl 文件的每一行，转换为 Document
3. 将 Document 转换为 list[Document]
"""


class JsonParser:
    """
    解析 json 文件, 支持 json 和 jsonl 文件, 按表入库
    """

    def __call__(
        self,
        file_path: str,
        file_name: str,
        file_type: DocumentFileTaskType = DocumentFileTaskType.json,
    ) -> list[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        docs = []
        match file_type:
            case DocumentFileTaskType.json:
                with open(file_path, "rb") as f:
                    df = self.flatten(orjson.loads(f))
                    docs.extend(
                        self.chunk(
                            [df],
                            file_path=file_path,
                            file_name=file_name,
                            file_type=file_type,
                        )
                    )
            case DocumentFileTaskType.jsonl:
                # jsonl 分批加载，内存优化
                df = pd.read_json(file_path, lines=True, chunksize=1000)
                docs.extend(
                    self.chunk(
                        chunk,
                        file_path=file_path,
                        file_name=file_name,
                        file_type=file_type,
                    )
                    for chunk in df
                )
            case _:
                raise ValueError(f"Unsupported file type from JsonParser: {file_type}")
        return docs

    def flatten(self, data: dict | list[dict]) -> pd.DataFrame:
        match type(data):
            case dict():
                return pd.json_normalize(data)
            case list():
                # 如果 data 不是list[dict], 则返回空列表
                if not all(isinstance(item, dict) for item in data):
                    raise ValueError(f"Invalid data type: {type(data)}")
                return pd.json_normalize(data)
            case _:
                raise ValueError(f"Invalid data type: {type(data)}")

    def is_need_split_df(self, df: pd.DataFrame) -> bool:
        """
        如果整表的 token 数超出限制，将 df 按行分割
        """
        max_tokens = consts.BGE_MAX_TOKENS
        counter = TokenCounter(TokenizerType.TRANSFORMERS, consts.BGE_MODEL_PATH)
        table = df.to_json(orient="records", date_format="iso", lines=True)
        tokens = counter.estimate_tokens(table)
        if tokens > max_tokens:
            return True
        else:
            real_tokens = counter.count_tokens(table)
            logger.debug(f"real_tokens: {real_tokens}, max_tokens: {max_tokens}")
            if real_tokens > max_tokens:
                return True
            else:
                return False

    def chunk(
        self,
        data: list[pd.DataFrame],
        is_row: bool = False,
        file_path: str = "",
        file_name: str = "",
        file_type: DocumentFileTaskType = DocumentFileTaskType.json,
    ) -> list[Document]:
        documents = []
        metadata = {
            "source": file_path,
            "file_name": file_name,
            "file_type": file_type,
        }
        is_need_split = is_row or self.is_need_split_df(data)

        for df in data:
            if is_need_split:
                for _, row in df.iterrows():
                    # TODO: 当 row 的 token 数超出限制，将 row 按列分割
                    documents.append(
                        Document(
                            page_content=row.to_json(orient="records", date_format="iso", lines=True),
                            metadata=metadata,
                        )
                    )
            else:
                documents.append(
                    Document(
                        page_content=df.to_json(orient="records", date_format="iso", lines=True),
                        metadata=metadata,
                    )
                )
        return documents
