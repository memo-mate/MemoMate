import os
import tempfile

import requests
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from rich.prompt import Prompt

from app.configs import settings
from app.utils.stream_handler import RichStreamingCallbackHandler


class Assistant:
    def __init__(self, project_name: str, robot_name: str, user: str):
        self.project_name = project_name
        self.robot_name = robot_name
        self.user = user

    def create_knowledge_base(self, pdf_url: str, collection_name: str, recreate: bool = False):
        """创建知识库"""
        # 下载PDF到本地临时文件
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, "document.pdf")

        # 仅当文件不存在或需要重建时下载
        if not os.path.exists(pdf_path) or recreate:
            response = requests.get(pdf_url)
            with open(pdf_path, "wb") as f:
                f.write(response.content)

        # 加载PDF并分割文档
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)

        # 创建向量存储
        embeddings = HuggingFaceEmbeddings(
            model_name="/Users/datagrand/Code/agent-demo/bge-m3",
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": "mps"},
        )
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=f"./chroma_db_{collection_name}",
        )

        return vectordb

    def pdf_agent_stream(self, user: str = "user"):
        # 创建知识库
        pdf_url = "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
        vectordb = self.create_knowledge_base(pdf_url, "recipes")

        # 创建检索器
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # 创建回调处理器
        streaming_handler = RichStreamingCallbackHandler(robot_name=self.robot_name)

        # 创建LLM
        llm = ChatOpenAI(
            model="Qwen/QwQ-32B",
            api_key=settings.openai_api_key,
            base_url="https://api.siliconflow.cn/v1",
            temperature=0,
            max_tokens=3200,
            timeout=None,
            max_retries=3,
            streaming=True,  # 流式输出
            callbacks=[streaming_handler],
        )

        # 创建提示模板
        prompt = ChatPromptTemplate.from_template(
            """回答以下问题，基于提供的上下文信息。如果无法从上下文中找到答案，请说"我不知道"。

    上下文: {context}
    问题: {input}

    回答:"""
        )

        # 创建文档链和检索链
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

        # 交互循环
        print(
            f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]你好，我是{self.project_name}的智能助手，你可以叫我{self.robot_name}。"
            "输入[bold yellow] exit[/bold yellow] 或 [bold yellow]bye[/bold yellow] 退出。\n"
        )

        while True:
            message = Prompt.ask(f"[bold] :sunglasses: {self.user}[/bold]")
            if message in ("exit", "bye"):
                break

            # 调用链进行流式输出 - 修改这里
            retrieval_chain.invoke({"input": message}, config={"callbacks": [streaming_handler]})
            # 不需要再单独打印结果，因为流式处理器已经打印了


if __name__ == "__main__":
    assistant = Assistant(project_name="魅魔骂他", robot_name="小魅", user="daoji")
    assistant.pdf_agent_stream()
