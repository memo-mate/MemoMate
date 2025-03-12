import os
import tempfile

import requests
import rich
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownTextSplitter
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
            model_name="./bge-m3",
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

    def load_local_md_files(self, collection_name: str = "recipes"):
        """加载本地MD文件"""
        # 创建向量存储
        embeddings = HuggingFaceEmbeddings(
            model_name="./bge-m3",
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": "mps"},
        )

        # 如果存在向量存储，则直接返回
        if os.path.exists(f"./chroma_db_{collection_name}"):
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=f"./chroma_db_{collection_name}",
            )

        dir_path = "./data/Miner2PdfAndWord_Markitdown2Excel"
        loader = DirectoryLoader(dir_path, glob="**/*.md")
        documents = loader.load()
        if not documents:
            raise ValueError("文档目录为空")
        # 使用 MarkdownTextSplitter 分割文档，chunk 大小为 3200，重叠 30
        text_spliter = MarkdownTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

        # 分割文档
        split_docs = text_spliter.split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=f"./chroma_db_{collection_name}",
        )

        return vectordb

    def pdf_agent_stream(self, user: str = "user"):
        # 创建知识库
        # pdf_url = "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
        # vectordb = self.create_knowledge_base(pdf_url, "recipes")
        vectordb = self.load_local_md_files()

        # 创建检索器
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # 创建回调处理器
        streaming_handler = RichStreamingCallbackHandler(robot_name=self.robot_name)

        # 创建LLM
        llm = ChatOpenAI(
            # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
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
        #     prompt = ChatPromptTemplate.from_template(
        #         """回答以下问题，基于提供的上下文信息。如果无法从上下文中找到答案，请说"我不知道"。

        # 上下文: {context}
        # 问题: {input}

        # 回答:"""
        #     )

        # 创建文档链和检索链
        # question_answer_chain = create_stuff_documents_chain(llm, prompt)
        # retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Output parser will split the LLM result into a list of queries
        class LineListOutputParser(BaseOutputParser[list[str]]):
            """Output parser for a list of lines."""

            def parse(self, text: str) -> list[str]:
                lines = text.strip().split("\n")
                return list(filter(None, lines))  # Remove empty lines

        output_parser = LineListOutputParser()

        QUERY_PROMPT = PromptTemplate.from_template(
            template="""你是一个ai语言模型助手。你的任务是生成五个不同的版本的用户问题，以从向量数据库中检索相关文档。通过生成用户问题的多个视角，您的目标是帮助用户克服基于距离的相似度搜索的一些局限性。
以下为替代问题， 以换行符分隔，不要输出推理过程。
原始问题：{question}""",
        )

        llm_chain = QUERY_PROMPT | llm | output_parser
        # 创建多查询检索器
        query_retriever = MultiQueryRetriever(retriever=retriever, llm_chain=llm_chain, parser_key="lines")

        # 交互循环
        rich.print(
            f"\n[bold cyan] 🤖 AI:[/bold cyan] [bold green]你好，我是{self.project_name}的智能助手，你可以叫我{self.robot_name}。"
            "输入[bold yellow] exit[/bold yellow] 或 [bold yellow]bye[/bold yellow] 退出。\n"
        )
        # 打印模型信息
        rich.print(f"[bold cyan] 🤖 AI:[/bold cyan] [bold green]模型名称: {llm.model_name}[/bold green] \n")
        rich.print(f"[bold cyan] 🤖 AI:[/bold cyan] [bold green]最大上下文: {llm.max_tokens}[/bold green] \n")

        while True:
            message = Prompt.ask(f"[bold] :sunglasses: {self.user}[/bold]")
            if message in ("exit", "bye"):
                break

            # 调用链进行流式输出 - 修改这里
            # retrieval_chain.invoke({"input": message}, config={"callbacks": [streaming_handler]})
            # 使用多查询检索器
            print(f"prompt: {QUERY_PROMPT.format(question=message)}")
            docs = query_retriever.invoke(message)
            rich.inspect(docs)
            rich.print("🎬 退出")
            break


if __name__ == "__main__":
    assistant = Assistant(project_name="魅魔骂他", robot_name="小魅", user="daoji")
    assistant.pdf_agent_stream()
