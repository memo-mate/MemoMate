"""RAG Agent."""

import base64
import datetime
import json
import os
import re
from typing import Annotated, Literal, NotRequired, TypedDict

import httpx
import jieba3  # type: ignore
import tenacity
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.chat_agent_executor import _get_state_value
from langgraph.types import Checkpointer

from app import prompts
from app.core import logger, settings
from app.plugins.rerank.service import rerank_documents
from app.rag.embedding.embedding_db.custom_qdrant import QdrantVectorStore
from app.rag.embedding.embeeding_model import EmbeddingFactory
from app.rag.llm.completions import LLM, LLMParams

trimmer: Runnable = trim_messages(
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=settings.MAX_RAG_SESSION_TOKENS,
    include_system=True,
)


def clean_text(text: str) -> str:
    """Clean text."""
    return re.sub(r"\(cid:\d+\)", "", text).replace("  ", " ").strip()


def hybrid_search(query: str, k: int = 5) -> list[Document]:
    """Perform a Hybrid Search (similarity_search + BM25Retriever) in the collection."""
    # Create vector store instance
    vector_store = QdrantVectorStore(
        collection_name=settings.RAG_COLLECTION_NAME,
        embeddings=EmbeddingFactory.get(),
        path=settings.QDRANT_PATH,
    )

    # Get all documents from the Qdrant collection
    documents = vector_store.get_all()
    if not documents:
        logger.error("No documents found in the collection")
        return []

    # Create BM25Retriever from the documents
    tokenizer = jieba3.jieba3()
    bm25_retriever = BM25Retriever.from_documents(documents=documents, k=k, preprocess_func=tokenizer.cut_query)

    # Create vector search retriever from Qdrant instance
    similarity_search_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Ensemble the retrievers using Langchain's EnsembleRetriever Object
    ensemble_retriever = EnsembleRetriever(retrievers=[similarity_search_retriever, bm25_retriever], weights=[0.6, 0.4])

    # Retrieve k relevant documents for the query
    return ensemble_retriever.invoke(query)


class RagState(TypedDict):
    """RAG State."""

    messages: Annotated[list[BaseMessage], add_messages]
    context: NotRequired[list[Document]]
    retrieved_docs: list[Document]  # 新增：存储检索后的原始文档
    answer: NotRequired[str]
    time_context: NotRequired[str]
    search_type: Literal["knowledge_base", "real_time", "hybrid", "direct"]
    needs_retrieval: NotRequired[bool]


def get_latest_user_message(messages: list[BaseMessage]) -> str:
    """Get the latest user message from messages."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            file = None
            content = None
            file_list = []
            if isinstance(message.content, str):
                return message.content
            for file in message.content:
                if file["type"] in ["file", "image"]:
                    file_name = file["metadata"]["filename"]
                    data = file["data"]
                    file_list.append(file_name)
                    if not os.path.exists(settings.UPLOAD_DIR):
                        os.makedirs(settings.UPLOAD_DIR)
                    with open(os.path.join(settings.UPLOAD_DIR, file_name), "wb") as f:
                        binary_data = base64.b64decode(data)
                        f.write(binary_data)
                elif file["type"] == "text":
                    content = file["text"]
            return content
    # 如果没有HumanMessage，返回最后一个消息的内容
    if messages:
        last_message = messages[-1]
        return last_message.content[0]["text"]
    return ""


def classify_query(state: RagState) -> RagState:
    """Classify query type using LLM."""
    question = get_latest_user_message(state["messages"])
    logger.info(
        "Classify query",
        question=question,
        collection_name=settings.RAG_COLLECTION_NAME,
        model=settings.CHAT_MODEL,
    )

    # 创建分类提示词
    classification_prompt = f"""Analyze the following user question and determine its query type.

User Question: {question}

Please respond in the following format (must strictly follow the format):
需要检索：是/否
查询类型：直接回答/知识库检索/混合检索
原因：[brief explanation]

Classification criteria:
1. 直接回答 (Direct Answer): Questions about conversation history, simple greetings, feature inquiries, or questions that don't require document lookup
2. 知识库检索 (Knowledge Base Retrieval): Questions requiring document search but not involving latest information
3. 混合检索 (Hybrid Retrieval): Questions requiring both documents and External knowledge base

Examples:
- "What was my previous question?" → 直接回答
- "What is machine learning?" → 知识库检索
- "Latest AI news today" → 混合检索"""

    # 调用模型进行分类
    llm = LLM().get_llm(LLMParams(model_name=settings.CHAT_MODEL))
    try:
        response = llm.invoke(classification_prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        # 解析模型响应
        lines = response_text.strip().split("\n")
        needs_retrieval = True
        search_type = "knowledge_base"

        for line in lines:
            line = line.strip()
            if line.startswith("需要检索："):
                needs_retrieval = "是" in line
            elif line.startswith("查询类型："):
                if "直接回答" in line:
                    search_type = "direct"
                    needs_retrieval = False
                elif "混合检索" in line:
                    search_type = "hybrid"
                elif "知识库检索" in line:
                    search_type = "knowledge_base"

        logger.info(
            "LLM classification result",
            needs_retrieval=needs_retrieval,
            search_type=search_type,
        )

    except Exception as e:
        logger.error(
            "Failed to classify query with LLM, falling back to rule-based",
            error=str(e),
        )
        # 回退到规则方式
        needs_retrieval = True
        search_type = "knowledge_base"

    # 生成时间上下文
    current_time = datetime.datetime.now()
    time_context = f"当前时间：{current_time.strftime('%Y年%m月%d日 %H:%M')}"

    return {
        **state,
        "time_context": time_context,
        "search_type": search_type,
        "needs_retrieval": needs_retrieval,
    }


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
def fetch_rerank_api(
    query: str,
    documents: list[str],
    top_n: int = 10,
    timeout: int = 300,
) -> dict:
    """Fetch rerank api."""
    url = settings.RERANK_BASE_URL
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.RERANKER_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }
    with httpx.Client(verify=False) as client:
        response: httpx.Response = client.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code == 200:
        content = response.json()["results"]
        return content
    else:
        raise Exception(f"API Error: {response.text}")


def deduplicate_documents(
    docs: list[Document],
    mode: Literal["content", "content+metadata"] = "content",
) -> list[Document]:
    """去重."""
    seen = set()
    unique_docs = []
    for doc in docs:
        if mode == "content":
            key = doc.page_content.strip()
        elif mode == "content+metadata":
            key = (doc.page_content.strip(), tuple(sorted(doc.metadata.items())))
        else:
            raise ValueError("mode 必须是 'content' 或 'content+metadata'")

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    logger.info("Deduplicate docs", org_docs=len(docs), unique_docs=len(unique_docs))
    return unique_docs


def retrieve_documents(state: RagState) -> RagState:
    """Unified retrieve function for both knowledge base and hybrid retrieval."""
    question = get_latest_user_message(state["messages"])
    search_type = state.get("search_type", "knowledge_base")

    if search_type == "hybrid":
        # 先执行知识库检索
        kb_docs = hybrid_search(question, k=settings.SEARCH_MAX_DOCS)
        logger.info("Retrieved KB docs", docs=len(kb_docs))

        # 再执行实时检索
        real_time_docs = retrieve_real_time_data(question)
        logger.info("Retrieved real-time docs", docs=len(real_time_docs))

        # 合并检索结果
        retrieved_docs = kb_docs + real_time_docs
    else:
        # 仅知识库检索
        retrieved_docs = hybrid_search(question, k=settings.SEARCH_MAX_DOCS)
        logger.info("Retrieved docs", docs=len(retrieved_docs))

    # 删除重复的文档
    retrieved_docs = deduplicate_documents(retrieved_docs)

    # 将检索到的文档存储到 retrieved_docs 字段
    return {**state, "retrieved_docs": retrieved_docs}


def retrieve_real_time_data(question: str) -> list[Document]:
    """Retrieve real-time information (can be integrated with search APIs)."""
    # 这里可以接入实时搜索API，比如：
    # - Google Search API
    # - Bing Search API
    # - 企业内部实时数据接口
    # - Tavily Search API

    # 模拟实时检索结果
    real_time_docs = [
        Document(
            page_content=f"关于'{question}'的实时信息：这是模拟的实时数据，时间：{datetime.datetime.now()}",
            metadata={
                "source": "real_time_api",
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "real_time",
            },
        )
    ]

    return real_time_docs


def rerank_documents_unified(state: RagState) -> RagState:
    """Unified rerank function for both knowledge base and hybrid retrieval."""
    question = get_latest_user_message(state["messages"])
    retrieved_docs = state.get("retrieved_docs", [])
    search_type = state.get("search_type", "knowledge_base")

    if not retrieved_docs:
        logger.warning("No retrieved documents to rerank")
        return {**state, "context": []}

    # 使用统一的插件系统进行重排序
    try:
        reranked_docs = rerank_documents(
            query=question,
            documents=retrieved_docs,
            top_n=settings.RERANK_MAX_DOCS,
            metadata={
                "source": search_type,
                "search_type": search_type,
            },
        )
        logger.info(f"Plugin rerank completed: {len(reranked_docs)} documents for {search_type}")
    except Exception as e:
        logger.error(f"Plugin rerank failed, falling back to direct API method: {e}")
        # 回退到直接API调用方法
        try:
            documents_str = [doc.page_content for doc in retrieved_docs]
            scores_results = fetch_rerank_api(question, documents_str, top_n=settings.RERANK_MAX_DOCS)

            reranked_docs = []
            for item in scores_results:
                idx: int = item["index"]
                score: float = item["relevance_score"]

                doc = Document(
                    page_content=retrieved_docs[idx].page_content,
                    metadata=retrieved_docs[idx].metadata,
                )
                doc.metadata.update({"relevance_score": round(score, 4)})
                reranked_docs.append(doc)

            # 应用阈值过滤
            reranked_docs = [
                doc for doc in reranked_docs if doc.metadata.get("relevance_score", 0) >= settings.RERANK_THRESHOLD
            ]
        except Exception as fallback_error:
            logger.error(f"Fallback rerank also failed: {fallback_error}")
            # 最终回退：返回原始文档的前20个
            reranked_docs = retrieved_docs[: settings.RERANK_MAX_DOCS]

    state["context"] = reranked_docs
    logger.info("rerank state", state_context=len(state["context"]))
    return state


def _build_source_info(doc: Document) -> str:
    """构建文档来源信息."""
    if not doc.metadata:
        return ""
    source = doc.metadata.get("source", "")
    timestamp = doc.metadata.get("timestamp", "")
    if source:
        source_info = f"（来源：{source}"
        if timestamp:
            source_info += f"，时间：{timestamp}"
        return source_info + "）"
    return ""


def _build_metadata(doc: Document) -> str:
    """构建文档元数据信息."""
    if not doc.metadata:
        return ""
    # 移除多个keys
    for key in [
        # "relevance_score",
        "type",
        "join_id",
        "embedded_at",
        "kid",
        "original_index",
    ]:
        doc.metadata.pop(key, None)
    metadata = json.dumps(doc.metadata, ensure_ascii=False)
    return f"\n元数据: {metadata}"


def ensure_human_message(messages: list[BaseMessage]) -> list[BaseMessage]:
    """检查 messages 列表中是否至少有一个 HumanMessage."""
    if not any(isinstance(m, HumanMessage) for m in messages):
        raise ValueError("❌ 对话信息过多，请重新开始对话!")
    return messages


def generate_answer(state: RagState) -> RagState:
    """Generate answer."""
    question = get_latest_user_message(state["messages"])

    # Base Chat LCEL Chain
    check_human = RunnableLambda(ensure_human_message)
    base_chain = trimmer | check_human | LLM().get_llm(LLMParams(model_name=settings.CHAT_MODEL)) | StrOutputParser()
    remaining_steps = _get_state_value(state, "remaining_steps", None)
    logger.info("Generating answer...", remaining_steps=remaining_steps)
    try:
        if state["search_type"] == "direct":
            # 直接回答：使用消息历史
            answer = base_chain.invoke(state["messages"])
        else:
            # 构建上下文
            context_parts = [
                f"文档{i}{_build_source_info(doc)}：\n{doc.page_content}{_build_metadata(doc)}"
                for i, doc in enumerate(state["context"], 1)
            ]
            context_text = "\n\n".join(context_parts)

            # 选择提示词模板并构建输入
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompts.RAG_PROMPT),  # 系统指令部分
                    MessagesPlaceholder(variable_name="history"),  # 历史消息
                    (
                        "human",
                        "## Knowledge Base:\n{context}\n\n## User Question:\n{question}",
                    ),  # 当前问题
                ]
            )
            prompt_input = {
                "context": context_text,
                "question": question,
                "system_time": state["time_context"],
                "history": state.get("messages", []),
            }

            # 使用 RAG 链生成答案
            answer = (prompt | base_chain).invoke(prompt_input)
    except Exception as e:
        answer = f"抱歉，生成回答时出现错误：{str(e)}"

    # 创建AI消息并添加到messages中
    ai_message = AIMessage(content=answer.strip())

    return {
        **state,
        "messages": state["messages"] + [ai_message],
        "answer": answer.strip(),
    }


def route_retrieval(
    state: RagState,
) -> Literal["retrieve", "generate"]:
    """Route retrieval strategy."""
    if state["search_type"] == "direct":
        return "generate"  # 直接生成答案，跳过检索
    else:
        return "retrieve"  # 统一的检索节点


def finish_answer(state: RagState) -> RagState:
    """Finish answer."""
    state["context"] = []
    state["retrieved_docs"] = []
    return state


# 创建状态图
def create_time_sensitive_rag_graph(checkpointer: Checkpointer = None, name: str = "rag_agent"):
    """Create time-sensitive RAG graph."""
    # 创建图
    workflow = StateGraph(RagState)

    # 添加节点
    workflow.add_node("classify", classify_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rerank", rerank_documents_unified)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("finish", finish_answer)

    # 定义流程
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_retrieval,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    # 检索后进行重排序
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "finish")
    workflow.add_edge("finish", END)
    compile_params = {"checkpointer": checkpointer, "name": name}
    logger.info("Compile RAG graph params", compile_params=compile_params)
    # 过滤不用的参数
    compile_params = {k: v for k, v in compile_params.items() if v is not None}
    workflow = workflow.compile(**compile_params)
    return workflow


# 使用示例
def create_smart_qa_system():
    """Create intelligent Q&A system."""
    # short-term memory
    memory_checkpointer = InMemorySaver()

    app = create_time_sensitive_rag_graph(checkpointer=memory_checkpointer)

    def ask(question: str) -> str:
        """Ask function."""
        result = app.invoke(
            RagState(messages=[HumanMessage(content=[{"type": "text", "text": question}])]),
            config={"thread_id": "rag_agent"},
        )

        return result["answer"]

    return ask


# 批量问答处理
def batch_qa(questions: list[str]) -> list[dict]:
    """Batch process Q&A."""
    qa_system = create_smart_qa_system()
    results = []

    for question in questions:
        start_time = datetime.datetime.now()
        answer = qa_system(question)
        end_time = datetime.datetime.now()

        results.append(
            {
                "question": question,
                "answer": answer,
                "response_time": (end_time - start_time).total_seconds(),
                "timestamp": start_time.isoformat(),
            }
        )

    return results


def get_rag_graph() -> StateGraph:
    """Lazy-initialize and cache the compiled RAG graph."""
    return create_time_sensitive_rag_graph()


rag_graph = get_rag_graph()

# 使用示例
if __name__ == "__main__":
    qa_system = create_smart_qa_system()

    # 测试不同类型的问题
    test_questions = [
        # "什么是机器学习？",  # 一般问题
        # "最新的AI发展趋势是什么？",  # 时间敏感问题
        # "2024年的技术突破有哪些？",  # 明确时间问题
        # "当前市场状况如何？",  # 时间敏感问题
        "研究水下爆破多场耦合效应有哪些文章？"
    ]

    for question in test_questions:
        logger.info(f"问题：{question}")
        answer = qa_system(question)
        logger.info(f"回答：{answer}")
        logger.info("-" * 50)
