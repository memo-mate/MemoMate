from langchain_core.messages import AnyMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.configs import settings
from app.rag.embedding import vector_store
from app.rag.web_search import web_search


def process_rag(
    query: str,
    history: list[AnyMessage],
    is_web_search: bool = False,
) -> str:
    prompt = PromptTemplate.from_template(
        """
        ## 任务
        你是问题排查的小助手，根据用户问题{note_time}，结合网络搜索结果和数据库查询结果、历史记录，给出回答。

        ## 用户问题
        {query}

        ## 网络搜索结果
        {web_search_result}

        ## 数据库查询结果
        {db_search_result}

        ## 历史记录
        {history}
        """
    )
    # 时间敏感检测
    time_sensitive = any(word in query for word in ["今天", "最新", "今年", "当前", "最近", "刚刚", "现在", "如今"])

    note_time = "，注意这是时间敏感的问题，请优先使用最新信息" if time_sensitive and is_web_search else ""
    web_search_result = "没有进行网络搜索，请根据历史记录和知识库回答问题。"
    if is_web_search:
        web_result = web_search.search(query)
        print(web_result)

        # artifact 类型:
        # {'query': str,
        #  'follow_up_questions': NoneType,
        #  'answer': str,
        #  'images': list,
        #  'results': list,
        #  'response_time': float}
        # 取前 5 个置信度最高的网络搜索结果
        web_search_result = ""
        web_search_result += f"查询语句: {web_result.artifact['query']}\n"
        web_search_result += f"检索答案: {web_result.artifact['answer']}\n"
        web_search_result += f"后续问题: {web_result.artifact['follow_up_questions']}\n"

        web_search_result += "检索引用如下: \n"
        for result in web_result.artifact["results"]:
            web_search_result += f"url: {result['url']}\n"
            web_search_result += f"title: {result['title']}\n"
            web_search_result += f"content: {result['content']}\n"

    documents = vector_store.similarity_search(query, k=5, include=["documents", "metadatas"])
    db_search_result = ""
    for doc in documents:
        db_search_result += f"文件名: {doc.metadata['source']}\n"
        db_search_result += f"内容: {doc.page_content}\n"

    prompt = prompt.format(
        query=query,
        note_time=note_time,
        web_search_result=web_search_result,
        db_search_result=db_search_result,
        history=history,
    )

    print(prompt)
    # 使用 OpenAI 的 API 进行推理
    llm = ChatOpenAI(
        model="Qwen/QwQ-32B",
        api_key=settings.openai_api_key,
        base_url="https://api.siliconflow.cn/v1",
        temperature=0,
        max_tokens=3200,
        timeout=None,
        max_retries=3,
    )

    response = llm.invoke(prompt)
    return response.content
