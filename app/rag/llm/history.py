from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableWithMessageHistory
from sqlmodel import Session

from app.core.db import engine
from app.crud.history_message import add_history_message, get_history_messages
from app.enums import HistoryMessageType


class MemoMateMemory:
    """
    记忆能力封装类
    """

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
        with Session(engine) as session:
            history_messages = get_history_messages(session=session, session_id=session_id)
            base_messages = []
            for msg in history_messages:
                if msg.message_type == HistoryMessageType.HUMAN:
                    base_messages.append(HumanMessage(content=msg.message))
                elif msg.message_type == HistoryMessageType.AI:
                    base_messages.append(AIMessage(content=msg.message))
            return ChatMessageHistory(messages=base_messages)

    @staticmethod
    def merge_history(session_id: str, client_history: list[tuple[str, str]]) -> None:
        """
        合并客户端历史和数据库历史
        只添加数据库中缺失的对话

        session_id: 会话ID
        client_history: 客户端历史记录，格式为[(用户问题1, AI回答1), ...]
        """
        with Session(engine) as session:
            # 获取数据库中现有历史
            db_messages = get_history_messages(session=session, session_id=session_id)

            # 计算数据库中有多少对完整对话 (每对包含一个人类消息和一个AI消息)
            db_pairs_count = len(db_messages) // 2

            # 比较客户端历史长度
            client_pairs_count = len(client_history)

            # 如果客户端历史更长，只添加新增部分
            if client_pairs_count > db_pairs_count:
                for i in range(db_pairs_count, client_pairs_count):
                    human_msg, ai_msg = client_history[i]
                    add_history_message(
                        session=session, message=human_msg, message_type=HistoryMessageType.HUMAN, session_id=session_id
                    )

                    add_history_message(
                        session=session, message=ai_msg, message_type=HistoryMessageType.AI, session_id=session_id
                    )
