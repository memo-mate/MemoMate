from langchain.chains import create_history_aware_retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangchainBaseRetriever

from app.core import settings
from app.core.log_adapter import logger
from app.rag.llm.completions import LLM, LLMParams, ModelAPIType
from app.rag.retriever.base_retriever import BaseRetriever


class HistoryAwareRetriever:
    """支持历史感知的检索器"""

    def __init__(
        self, base_retriever: BaseRetriever | None = None, llm_name: str = "gpt-3.5-turbo", search_kwargs: dict = None
    ):
        # 如果没有提供基础检索器，则创建一个
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        self.base_retriever = base_retriever or BaseRetriever()
        self.llm_name = llm_name
        self.search_kwargs = search_kwargs
        self._history_aware_retriever = None

    def get_retriever(self) -> LangchainBaseRetriever:
        """获取基础检索器的langchain检索器接口"""
        # 将BaseRetriever转换为langchain检索器接口
        langchain_retriever = self.base_retriever.vector_store.as_retriever(search_kwargs=self.search_kwargs)
        return langchain_retriever

    def create_history_aware_retriever(self):
        """创建支持历史感知的检索器"""

        # 获取基础检索器
        retriever = self.get_retriever()

        # 使用封装好的LLMParams和LLM类
        llm_params = LLMParams(
            api_type=ModelAPIType.OPENAI,
            model_name=self.llm_name,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            temperature=0,
            streaming=False,  # 历史感知检索不需要流式输出
        )

        # 获取原始LLM对象
        llm = LLM().get_llm(llm_params)

        # 创建历史感知检索器
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt_template="""Given the following conversation and a follow up question,
            rephrase the follow up question to be a standalone question,
            taking into account context from the chat history.
            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone Question:""",
        )

        self._history_aware_retriever = history_aware_retriever
        return history_aware_retriever

    def retrieve_with_history(
        self, query: str, chat_history: list[tuple[str, str]] = None, session_id: str = None
    ) -> list[Document]:
        """使用历史记录进行检索"""
        try:
            # 确保历史感知检索器已创建
            if self._history_aware_retriever is None:
                self._history_aware_retriever = self.create_history_aware_retriever()

            # 准备历史记录
            history = chat_history or []

            # 进行检索
            retrieved_docs = self._history_aware_retriever.invoke({"question": query, "chat_history": history})

            logger.info(f"历史感知检索结果数量: {len(retrieved_docs)}")
            return retrieved_docs

        except Exception as e:
            logger.exception("历史感知检索失败", exc_info=e)
            # 失败时回退到基本检索
            return self.base_retriever.retrieve(query)

    def get_context_with_history(self, query: str, chat_history: list[tuple[str, str]] = None, top_k: int = 5) -> str:
        """获取格式化的上下文内容，包含历史感知"""
        # 更新搜索参数
        self.search_kwargs["k"] = top_k

        # 使用历史进行检索
        documents = self.retrieve_with_history(query, chat_history)

        if not documents:
            return ""

        # 将文档内容拼接为上下文字符串
        context_parts = []
        for i, doc in enumerate(documents):
            # 添加文档内容和元数据
            source = doc.metadata.get("filename", "未知来源")
            context_parts.append(f"[文档{i + 1}] 来源: {source}\n{doc.page_content}\n")

        return "\n".join(context_parts)
