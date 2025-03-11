import os
from typing import Any

from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_openai import ChatOpenAI
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from app.configs import settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 创建一个全局的控制台对象
console = Console(color_system="auto")


class RichStreamingCallbackHandler(OpenAICallbackHandler):
    """使用Rich实现完整流程可视化的回调处理器"""

    def __init__(self, robot_name: str):
        super().__init__()  # 调用父类初始化方法以确保token计数功能正常
        self.console = Console()
        self.robot_name = robot_name
        self.text = ""
        self.llm_live = None
        self.retrieval_results = []
        self.retrieval_live = None

    def on_retriever_start(self, serialized: dict[str, Any], query: str, *args, **kwargs):
        """检索开始时的回调"""
        self.__retriever_title = f"🔍 知识检索结果 - {query}"
        self.retrieval_results = []
        self.text = ""
        self.retrieval_live = Live(
            Panel(
                Group(Text("正在搜索相关信息...", style="yellow")),
                title=self.__retriever_title,
                border_style="blue",
                box=box.ROUNDED,
                width=200,
                expand=True,
            ),
            refresh_per_second=4,
            console=self.console,
            vertical_overflow="visible",
        )
        self.retrieval_live.start()

    def on_retriever_end(self, documents: list[Document], *args, **kwargs):
        """检索结束时的回调"""
        if self.retrieval_live:
            # 构建最终显示内容
            group_components = []
            if documents:
                for i, doc in enumerate(documents, 1):
                    group_components.append(Text(f"\n文档片段 {i}: {doc.page_content}", style="bold blue"))
            else:
                group_components.append(Text("未找到相关信息", style="yellow"))

            # 使用Group组合所有组件
            self.retrieval_live.update(
                Panel(
                    Group(*group_components),
                    title=self.__retriever_title,
                    border_style="blue",
                    box=box.ROUNDED,
                    width=200,
                    expand=True,
                )
            )
            self.retrieval_live.stop()

    def on_llm_start(self, *args, **kwargs):
        """LLM开始生成时的回调"""
        self.text = ""
        self.llm_live = Live(
            Panel(
                Text("正在思考回答...", style="yellow"),
                title=f"🤖 {self.robot_name}",
                border_style="cyan",
                box=box.ROUNDED,
            ),
            refresh_per_second=4,
            console=self.console,
            vertical_overflow="visible",
        )
        self.llm_live.start()

    def on_llm_new_token(self, token: str, **kwargs):
        """接收到新token时的回调"""
        self.text += token
        if not token:
            return

        self.llm_live.update(
            Panel(
                Markdown(self.text),
                title=f"🤖 {self.robot_name}",
                border_style="cyan",
                box=box.ROUNDED,
                width=200,
                expand=True,
            ),
        )
        # 手动触发滚动 - 使用控制台的print方法但不实际输出内容
        self.console.print("", end="")

    def on_llm_end(self, *args, **kwargs):
        """LLM生成结束时的回调"""
        super().on_llm_end(*args, **kwargs)
        # 创建token信息标题
        token_info = f"输入: {self.prompt_tokens} tokens | 输出: {self.completion_tokens} tokens"

        # 更新面板显示token信息
        if self.llm_live:
            self.llm_live.update(
                Panel(
                    Markdown(self.text),
                    title=f"🤖 {self.robot_name}",
                    subtitle=token_info,
                    subtitle_align="center",
                    border_style="cyan",
                    box=box.ROUNDED,
                    width=200,
                    expand=True,
                )
            )
            self.llm_live.stop()
            console.print(f"{self}\n")


class Assistant:
    def __init__(self, project_name: str, robot_name: str, user: str):
        self.project_name = project_name
        self.robot_name = robot_name
        self.user = user

    def pdf_agent_stream(self, user: str = "user"):
        # 创建回调处理器
        streaming_handler = RichStreamingCallbackHandler(robot_name=self.robot_name)

        # 创建LLM
        llm = ChatOpenAI(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            api_key=settings.openai_api_key,
            base_url="https://api.siliconflow.cn/v1",
            temperature=0,
            max_tokens=3200,
            timeout=None,
            max_retries=3,
            streaming=True,  # 流式输出
            callbacks=[streaming_handler],
        )

        # 交互循环
        console.print(
            f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]你好，我是{self.project_name}的智能助手，你可以叫我{self.robot_name}。"
            "输入[bold yellow] exit[/bold yellow] 或 [bold yellow]bye[/bold yellow] 退出。\n"
        )

        while True:
            message = Prompt.ask(f"[bold] :sunglasses: {self.user}[/bold]")
            if message in ("exit", "bye"):
                break

            # 调用链进行流式输出 - 修改这里
            template = [
                SystemMessage(content="你是一个AI助手，用中文回答用户的问题，以markdown格式输出。"),
                HumanMessage(content=message),
            ]
            response: AIMessage = llm.invoke(template, config={"callbacks": [streaming_handler]})
            useage: UsageMetadata = response.usage_metadata
            streaming_handler.completion_tokens = useage["output_tokens"]
            streaming_handler.prompt_tokens = useage["input_tokens"]
            streaming_handler.total_tokens = useage["total_tokens"]


if __name__ == "__main__":
    assistant = Assistant(project_name="魅魔骂他", robot_name="小魅", user="daoji")
    assistant.pdf_agent_stream()
