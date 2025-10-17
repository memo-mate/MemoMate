from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentState


def clear_image_history(_state: AgentState) -> list[BaseMessage]:
    """清理图像历史."""
    new_state = []
    for message in _state["messages"]:
        if message.content and isinstance(message.content, list):
            _new_content = []
            for i_content in message.content:
                if i_content["type"] == "image" or i_content["type"] == "image_url":
                    continue
                _new_content.append(i_content)
            message.content = _new_content
        new_state.append(message)
    return new_state
