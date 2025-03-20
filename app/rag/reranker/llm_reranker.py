"""基于LLM的重排序器"""

from collections.abc import Callable
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from app.core.log_adapter import logger
from app.rag.reranker.base import BaseReranker


# 重排序评分模型
class DocumentScore(BaseModel):
    """文档评分模型"""

    document_id: int = Field(description="文档的索引位置")
    relevance_score: float = Field(description="文档与查询的相关性分数，0-10之间")

    @field_validator("relevance_score")
    def check_score_range(cls, v: float) -> float:
        """检查分数范围"""
        if not 0 <= v <= 10:
            raise ValueError("相关性分数必须在0-10之间")
        return v


class RerankedResult(BaseModel):
    """重排序结果模型"""

    scores: list[DocumentScore] = Field(description="每个文档的评分结果")


# 默认的重排序提示模板
DEFAULT_RERANK_PROMPT = """你是一个专业的文档相关性评估专家。你的任务是评估提供的文档与用户查询的相关性。
评分标准是0-10分制，其中10分表示完全相关，0分表示完全不相关。

用户查询: {query}

以下是检索到的文档:
{documents}

请仔细分析每个文档的内容，并根据其与用户查询的相关程度评分。
考虑因素包括但不限于：
1. 内容的直接匹配程度
2. 语义相关性
3. 信息的完整性
4. 文档的权威性和准确性

请以JSON格式输出评分结果，格式如下:
{format_instructions}
"""


class LLMReranker(BaseReranker):
    """基于LLM的重排序器

    使用大型语言模型对检索结果进行重排序
    """

    llm: BaseLLM = Field(description="用于重排序的大型语言模型")
    prompt_template: str = Field(default=DEFAULT_RERANK_PROMPT, description="提示模板")
    document_formatter: Callable[[Document, int], str] = Field(default=None, description="文档格式化函数")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        """初始化LLM重排序器"""
        # 如果没有提供llm，创建默认的ChatOpenAI
        if "llm" not in kwargs:
            kwargs["llm"] = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)

        # 如果没有提供document_formatter，设置默认格式化函数
        if "document_formatter" not in kwargs:
            kwargs["document_formatter"] = self._default_document_formatter

        super().__init__(**kwargs)

    def _default_document_formatter(self, doc: Document, idx: int) -> str:
        """默认的文档格式化函数

        Args:
            doc: 文档
            idx: 文档索引

        Returns:
            格式化后的文档字符串
        """
        # 格式化元数据为字符串
        metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
        return f"文档[{idx}]:\n内容: {doc.page_content}\n元数据: {metadata_str}\n"

    def _format_documents(self, documents: list[Document]) -> str:
        """格式化文档列表

        Args:
            documents: 文档列表

        Returns:
            格式化后的文档字符串
        """
        formatted_docs = []
        # 确保document_formatter不为None
        formatter = cast(Callable[[Document, int], str], self.document_formatter)

        for i, doc in enumerate(documents):
            formatted_docs.append(formatter(doc, i))

        return "\n".join(formatted_docs)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """使用LLM重排序文档

        Args:
            query: 查询文本
            documents: 需要重排序的文档列表

        Returns:
            重排序后的文档列表
        """
        if not documents:
            logger.warning("没有文档需要重排序")
            return []

        # 创建输出解析器
        parser = PydanticOutputParser(pydantic_object=RerankedResult)

        # 创建提示模板
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["query", "documents"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # 格式化文档
        formatted_documents = self._format_documents(documents)

        # 准备LLM输入
        inputs = {"query": query, "documents": formatted_documents}

        try:
            # 获取LLM响应
            logger.debug(f"使用LLM对 {len(documents)} 个文档进行重排序")
            response = self.llm.predict(prompt.format(**inputs))

            # 解析响应
            result = parser.parse(response)

            # 按文档ID和分数进行排序
            scores_by_id = {score.document_id: score.relevance_score for score in result.scores}

            # 将分数添加到文档的元数据中
            for i, doc in enumerate(documents):
                if i in scores_by_id:
                    doc.metadata["rerank_score"] = scores_by_id[i]
                else:
                    doc.metadata["rerank_score"] = 0.0

            # 提取分数并规范化到0-1区间
            scores = [scores_by_id.get(i, 0.0) / 10.0 for i in range(len(documents))]

            # 应用过滤
            reranked_docs = self.filter_documents(documents, scores)
            self.log_rerank_info(query, documents, reranked_docs)

            return reranked_docs

        except Exception as e:
            logger.error(f"LLM重排序过程中发生错误: {e}")
            # 出错时返回原始文档
            return documents[: self.top_k]
