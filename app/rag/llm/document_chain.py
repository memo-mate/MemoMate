from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.core import logger, settings
from app.rag.llm.completions import LLMParams, ModelAPIType


class RAGDocumentChain:
    """RAG文档链，用于将检索到的文档整合到回答中"""

    def __init__(
        self,
        llm_params: LLMParams | None = None,
        system_message: str = """你是一个有用的AI助手。使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。""",
    ):
        self.llm_params = llm_params or LLMParams(
            api_type=ModelAPIType.OPENAI,
            model_name=settings.CHAT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            streaming=False,
        )
        self.system_message = system_message

    def get_llm(self):
        """获取LLM模型"""
        if self.llm_params.api_type == ModelAPIType.OPENAI:
            return ChatOpenAI(
                model=self.llm_params.model_name,
                api_key=self.llm_params.api_key,
                base_url=self.llm_params.base_url,
                temperature=self.llm_params.temperature,
                max_tokens=self.llm_params.max_tokens,
                streaming=self.llm_params.streaming,
            )
        else:
            # 根据需要添加其他模型类型
            raise ValueError(f"不支持的API类型: {self.llm_params.api_type}")

    def create_document_chain(self):
        """创建文档链"""
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """请回答以下问题：

问题: {question}

根据以下上下文信息进行回答：
{context}
""",
                ),
            ]
        )

        # 获取LLM
        llm = self.get_llm()

        # 创建文档链
        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt, document_variable_name="context")

        return document_chain

    def run(self, question: str, documents: list[Document], chat_history: list[tuple[str, str]] = None) -> str:
        """运行文档链生成回答"""
        try:
            # 创建文档链
            chain = self.create_document_chain()

            # 准备输入
            inputs = {"question": question, "chat_history": chat_history or [], "context": documents}

            # 执行链
            result = chain.invoke(inputs)
            logger.info("文档链生成回答", question=question, document_count=len(documents))

            return result

        except Exception as e:
            logger.exception("文档链生成回答失败", exc_info=e)
            return f"回答生成失败: {str(e)}"
