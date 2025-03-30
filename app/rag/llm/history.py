from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableWithMessageHistory


class MemoMateMemory:
    """
    记忆助手
    """

    store = {}

    @staticmethod
    def create_prompt_template(template_str: str, system_message: str | None = None):
        """
        创建Prompt模板

        template_str: 模板字符串
        variables: 变量列表
        system_message: 系统消息
        return: ChatPromptTemplate对象
        """
        messages = []

        if system_message:
            messages.append(("system", system_message))

        # 添加历史消息占位符
        messages.append(MessagesPlaceholder(variable_name="chat_history"))

        # 添加用户提问模板
        messages.append(("human", template_str))

        return ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def gen_memory_chain(rag_chain: Runnable) -> RunnableWithMessageHistory:
        """
        生成带有记忆的RAG链

        rag_chain: 基础的RAG链
        return: 带有记忆的RAG链
        """
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            MemoMateMemory.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        return conversational_rag_chain

    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """
        获取会话历史

        session_id: 会话ID
        return: 会话历史
        """
        if session_id not in MemoMateMemory.store:
            # TODO(Daoji): 从数据库中获取会话历史
            MemoMateMemory.store[session_id] = ChatMessageHistory()
        return MemoMateMemory.store[session_id]

    @staticmethod
    def save_session_history(session_id: str, history: BaseChatMessageHistory) -> None:
        """
        保存会话历史
        """
        # TODO(Daoji): 从数据库中保存会话历史
        MemoMateMemory.store[session_id] = history
