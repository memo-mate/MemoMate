import os
import urllib3
import requests
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rich import box, print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from dotenv import load_dotenv
from app.core.vector_search import VectorSearch
from app.core.multi_query import MultiQuerySearch

# 禁用不安全请求的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 加载环境变量
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 禁用代理
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

EMPTY_PROXIES = {
    "http": "",
    "https": "",
}


class RichStreamingCallbackHandler(OpenAICallbackHandler):
    """使用Rich实现流式输出的回调处理器"""

    def __init__(self, robot_name: str):
        self.console = Console()
        self.robot_name = robot_name
        self.text = ""
        self.live = None

    def on_llm_start(self, *args, **kwargs):
        self.text = ""
        self.live = Live(
            Panel(
                Text("思考中...", style="yellow"),
                title=f"🤖 {self.robot_name}",
                border_style="cyan",
                box=box.ROUNDED,
            ),
            refresh_per_second=4,
            console=self.console,
        )
        self.live.start()

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.live.update(
            Panel(
                Markdown(self.text),
                title=f"🤖 {self.robot_name}",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    def on_llm_end(self, *args, **kwargs):
        if self.live:
            self.live.stop()


class Assistant:
    def __init__(self, project_name: str, robot_name: str, user: str):
        self.project_name = project_name
        self.robot_name = robot_name
        self.user = user
        # 初始化本地向量搜索
        self.vector_search = VectorSearch()
        # 初始化多查询检索
        self.multi_query_search = MultiQuerySearch()

        # 设置OpenAI API参数
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv(
            "MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        )
        self.temperature = 0.1

    def _get_context(self, query, use_multi_query=False):
        """从本地向量数据库获取相关上下文"""
        if use_multi_query:
            # 使用多查询增强检索
            results = self.multi_query_search.search(query, k=10, use_multi_query=True)
        else:
            # 使用普通检索
            results = self.vector_search.similarity_search(query, k=10)

        # 格式化上下文
        context_texts = []
        for i, (doc, similarity) in enumerate(results):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            page_info = f"(第{page}页)" if page else ""

            # 添加工作表信息（如果有）
            sheet_name = doc.metadata.get("sheet_name", "")
            sheet_info = f"(工作表: {sheet_name})" if sheet_name else ""

            # 添加段落信息（如果有）
            paragraph = doc.metadata.get("paragraph", "")
            paragraph_info = f"(段落: {paragraph})" if paragraph else ""

            context_text = f"[文档{i+1}] {source}{page_info}{sheet_info}{paragraph_info}\n{doc.page_content}\n"
            context_texts.append(context_text)

        return "\n".join(context_texts)

    def _call_openai_api(self, messages):
        """调用OpenAI API"""
        # 禁用代理
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""

        # 检查API配置
        if not self.api_base:
            raise ValueError("API基础URL未设置，请检查环境变量OPENAI_API_BASE")
        if not self.api_key:
            raise ValueError("API密钥未设置，请检查环境变量OPENAI_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            verify=False,  # 禁用SSL验证，仅用于演示
            proxies=EMPTY_PROXIES,  # 显式设置空代理
        )

        if response.status_code != 200:
            error_msg = f"API调用失败: HTTP {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f" - {error_data['error'].get('message', '')}"
            except Exception:
                error_msg += f" - {response.text[:100]}"
                return error_msg

        return response.json()["choices"][0]["message"]["content"]

    def local_db_agent(self):
        """使用本地数据库的问答代理"""
        print(
            f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]你好，我是{self.project_name}的智能助手，你可以叫我{self.robot_name}。"
            "输入[bold yellow] exit[/bold yellow] 或 [bold yellow]bye[/bold yellow] 退出。\n"
        )

        while True:
            message = Prompt.ask(f"[bold] :sunglasses: {self.user}[/bold]")
            if message in ("exit", "bye"):
                break

            # 获取相关上下文
            context = self._get_context(message)

            # 构建消息
            prompt_template = """
            你是一个专业的助手，请基于以下检索到的信息回答用户的问题。
            如果检索到的信息无法回答用户的问题，请直接说明你不知道，不要编造答案。
            
            检索到的信息:
            {context}
            
            用户问题: {question}
            
            请给出详细、准确的回答，并引用相关的信息来源。
            """

            prompt = prompt_template.format(context=context, question=message)
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的助手，基于检索到的信息回答问题。",
                },
                {"role": "user", "content": prompt},
            ]

            # 调用API
            answer = self._call_openai_api(messages)

            # 打印回答
            print(
                f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]{answer}[/bold green] \n"
            )

    def local_db_agent_stream(self):
        """使用本地数据库的流式问答代理"""
        # 创建回调处理器
        callback_handler = RichStreamingCallbackHandler(robot_name=self.robot_name)

        print(
            f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]你好，我是{self.project_name}的智能助手，你可以叫我{self.robot_name}。"
            "输入[bold yellow] exit[/bold yellow] 或 [bold yellow]bye[/bold yellow] 退出。\n"
        )

        while True:
            message = Prompt.ask(f"[bold] :sunglasses: {self.user}[/bold]")
            if message in ("exit", "bye"):
                break

            # 获取相关上下文
            context = self._get_context(message)

            # 创建LLM
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.api_base,
                temperature=self.temperature,
                max_tokens=5000,
                timeout=None,
                max_retries=3,
                streaming=True,  # 流式输出
                callbacks=[callback_handler],
            )

            # 创建提示模板
            prompt = ChatPromptTemplate.from_template(
                """你是一个专业的助手，请基于以下检索到的信息回答用户的问题。
                如果检索到的信息无法回答用户的问题，请直接说明你不知道，不要编造答案。
                
                检索到的信息:
                {context}
                
                用户问题: {question}
                
                请给出详细、准确的回答，并引用相关的信息来源。"""
            )

            # 调用LLM进行流式输出
            llm.invoke(
                prompt.format(context=context, question=message),
                config={"callbacks": [callback_handler]},
            )

    def local_db_agent_with_multi_query(self):
        """使用多查询增强的本地数据库问答代理"""
        print(
            f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]你好，我是{self.project_name}的智能助手，你可以叫我{self.robot_name}。"
            "输入[bold yellow] exit[/bold yellow] 或 [bold yellow]bye[/bold yellow] 退出。\n"
        )

        while True:
            message = Prompt.ask(f"[bold] :sunglasses: {self.user}[/bold]")
            if message in ("exit", "bye"):
                break

            try:
                # 获取相关上下文（使用多查询增强）
                print("[yellow]正在使用多查询增强检索...[/yellow]")
                context = self._get_context(message, use_multi_query=True)

                # 构建消息
                prompt_template = """
                你是一个专业的助手，请基于以下检索到的信息回答用户的问题。
                如果检索到的信息无法回答用户的问题，请直接说明你不知道，不要编造答案。
                
                检索到的信息:
                {context}
                
                用户问题: {question}
                
                请给出详细、准确的回答，并引用相关的信息来源。
                """

                prompt = prompt_template.format(context=context, question=message)
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个专业的助手，基于检索到的信息回答问题。",
                    },
                    {"role": "user", "content": prompt},
                ]

                # 调用API
                answer = self._call_openai_api(messages)

                # 打印回答
                print(
                    f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]{answer}[/bold green] \n"
                )
            except Exception as e:
                print(f"[bold red]发生错误: {str(e)}[/bold red]")
                print("[bold yellow]请检查网络连接和API配置[/bold yellow]")

    def local_db_agent_with_multi_query_stream(self):
        """使用多查询增强的流式本地数据库问答代理"""
        # 创建回调处理器
        callback_handler = RichStreamingCallbackHandler(robot_name=self.robot_name)

        print(
            f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]你好，我是{self.project_name}的智能助手，你可以叫我{self.robot_name}。"
            "输入[bold yellow] exit[/bold yellow] 或 [bold yellow]bye[/bold yellow] 退出。\n"
        )

        while True:
            message = Prompt.ask(f"[bold] :sunglasses: {self.user}[/bold]")
            if message in ("exit", "bye"):
                break

            try:
                # 获取相关上下文（使用多查询增强）
                print("[yellow]正在使用多查询增强检索...[/yellow]")
                context = self._get_context(message, use_multi_query=True)

                try:
                    # 创建LLM
                    llm = ChatOpenAI(
                        model=self.model_name,
                        api_key=self.api_key,
                        base_url=self.api_base,
                        temperature=self.temperature,
                        max_tokens=3200,
                        timeout=60,  # 增加超时时间
                        max_retries=5,  # 增加重试次数
                        streaming=True,  # 流式输出
                        callbacks=[callback_handler],
                    )

                    # 创建提示模板
                    prompt = ChatPromptTemplate.from_template(
                        """你是一个专业的助手，请基于以下检索到的信息回答用户的问题。
                        如果检索到的信息无法回答用户的问题，请直接说明你不知道，不要编造答案。
                        
                        检索到的信息:
                        {context}
                        
                        用户问题: {question}
                        
                        请给出详细、准确的回答，并引用相关的信息来源。"""
                    )

                    # 调用LLM进行流式输出
                    llm.invoke(
                        prompt.format(context=context, question=message),
                        config={"callbacks": [callback_handler]},
                    )
                except Exception as api_err:
                    print(
                        f"[red]流式API调用失败，尝试使用非流式API: {str(api_err)}[/red]"
                    )

                    # 构建消息
                    prompt_template = """
                    你是一个专业的助手，请基于以下检索到的信息回答用户的问题。
                    如果检索到的信息无法回答用户的问题，请直接说明你不知道，不要编造答案。
                    
                    检索到的信息:
                    {context}
                    
                    用户问题: {question}
                    
                    请给出详细、准确的回答，并引用相关的信息来源。
                    """

                    prompt = prompt_template.format(context=context, question=message)
                    messages = [
                        {
                            "role": "system",
                            "content": "你是一个专业的助手，基于检索到的信息回答问题。",
                        },
                        {"role": "user", "content": prompt},
                    ]

                    # 使用非流式API调用
                    print("[yellow]正在使用非流式API调用...[/yellow]")
                    answer = self._call_openai_api(messages)

                    # 打印回答
                    print(
                        f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]{answer}[/bold green] \n"
                    )
            except Exception as e:
                print(f"[bold red]发生错误: {str(e)}[/bold red]")
                print("[bold yellow]请检查网络连接和API配置[/bold yellow]")

    def demo_run(self):
        print(
            f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]你好，我是{self.project_name}的智能助手，你可以叫我{self.robot_name}。"
            "输入[bold yellow] exit[/bold yellow] 或 [bold yellow]bye[/bold yellow] 退出。\n"
        )

        while True:
            message = Prompt.ask(f"[bold] :sunglasses: {self.user}[/bold]")
            if message in ("exit", "bye"):
                break
            print(
                f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]{message}[/bold green] \n"
            )


if __name__ == "__main__":
    assistant = Assistant(project_name="本地知识库", robot_name="小助手", user="用户")

    # 选择使用的模式
    print("[bold]请选择使用模式:[/bold]")
    print("1. 基本检索模式")
    print("2. 多查询增强检索模式")
    print("3. 多查询增强流式模式")
    print("4. 基本流式模式")
    print("5. 本地模型模式")

    choice = Prompt.ask("请输入选项", choices=["1", "2", "3", "4", "5"], default="3")

    if choice == "1":
        assistant.local_db_agent()
    elif choice == "2":
        assistant.local_db_agent_with_multi_query()
    elif choice == "3":
        assistant.local_db_agent_with_multi_query_stream()
    elif choice == "4":
        assistant.local_db_agent_stream()
    elif choice == "5":
        assistant.local_db_agent_with_local_model()
