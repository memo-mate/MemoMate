from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI


# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


MULTI_QUERY_PROMPT = """你是一个ai语言模型助手。你的任务是生成五个不同的版本的用户问题，以从向量数据库中检索相关文档。通过生成用户问题的多个视角，您的目标是帮助用户克服基于距离的相似度搜索的一些局限性。
以下为替代问题， 以换行符分隔，不要输出推理过程。
原始问题：{question}"""


def get_multi_query_retriever(llm: ChatOpenAI, vectordb: Chroma, k: int = 3) -> MultiQueryRetriever:
    # 创建检索器
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate.from_template(template=MULTI_QUERY_PROMPT)

    llm_chain = QUERY_PROMPT | llm | output_parser
    # 创建多查询检索器
    query_retriever = MultiQueryRetriever(retriever=retriever, llm_chain=llm_chain, parser_key="lines")

    # 创建一个转换函数，将检索到的文档转换为正确的字典格式
    def format_docs(docs):
        return {"context": docs, "question": RunnablePassthrough()}

    # 将检索器和格式转换函数组合
    retriever_chain = query_retriever | format_docs

    return retriever_chain


def get_custom_retriever(llm: ChatOpenAI, vectordb: Chroma, k: int = 3):
    # 创建检索器
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate.from_template(template=MULTI_QUERY_PROMPT)

    # 创建生成多查询的链
    generate_queries = QUERY_PROMPT | llm | output_parser

    # 创建一个转换函数，将检索到的文档转换为正确的字典格式
    def format_docs(input_dict):
        question = input_dict["question"]
        queries = generate_queries.invoke({"question": question})
        all_docs = []
        for query in queries:
            docs = retriever.get_relevant_documents(query)
            all_docs.extend(docs)
        return all_docs

    # 创建并行运行的链
    retriever_chain = RunnableParallel(context=format_docs, question=RunnablePassthrough())

    return retriever_chain
