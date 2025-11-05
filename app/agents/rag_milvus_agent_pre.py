"""Enhanced Milvus RAG Agent with hybrid search, reranking, and deduplication."""

import json
import re
from typing import Literal, NotRequired

import httpx
import tenacity
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field
from structlog.stdlib import get_logger

from app import prompts
from app.core.config import settings
from app.plugins.rerank import rerank_documents
from app.rag.embedding import get_embeddings
from app.rag.llm import LLM

logger = get_logger(__name__)


class EnhancedRagState(MessagesState):
    """Enhanced RAG state with document tracking."""

    context: NotRequired[list[Document]]
    retrieved_docs: NotRequired[list[Document]]


logger.info(
    "Initializing Milvus vector store",
    uri="http://localhost:19530",
    collection=settings.RAG_COLLECTION_NAME,
    drop_old=False,
)

vectorstore = Milvus(
    embedding_function=get_embeddings(),
    connection_args={"uri": "http://localhost:19530"},
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    consistency_level="Bounded",
    collection_name=settings.RAG_COLLECTION_NAME,
    drop_old=False,  # Drop the old Milvus collection if it exists
    # partition_key_field="namespace",  # Use the "namespace" field as the partition key
    auto_id=True,
    enable_dynamic_field=True,  # 动态字段
)
llm: ChatOpenAI = LLM().get_llm()

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": settings.SEARCH_MAX_DOCS,
        "ranker_type": "weighted",
        "ranker_params": {"weights": [0.6, 0.4]},  # 60% vector, 40% BM25
    }
)

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_documents",
    "Search and retrieve relevant documents from the knowledge base. "
    "Use this tool when you need to answer questions based on stored information.",
    response_format="content_and_artifact",
)

logger.info("Milvus RAG agent initialized successfully")


def clean_text(text: str) -> str:
    """Clean text by removing OCR artifacts and extra spaces."""
    return re.sub(r"\(cid:\d+\)", "", text).replace("  ", " ").strip()


def get_latest_user_message(messages: list[BaseMessage]) -> str:
    """Extract the latest user message from the messages list."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                return message.content
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg.content, str):
            return last_msg.content
    return ""


def deduplicate_documents(
    docs: list[Document],
    mode: Literal["content", "content+metadata"] = "content",
) -> list[Document]:
    """Remove duplicate documents."""
    seen = set()
    unique_docs = []

    for doc in docs:
        if mode == "content":
            key = doc.page_content.strip()
        elif mode == "content+metadata":
            key = (doc.page_content.strip(), tuple(sorted(doc.metadata.items())))
        else:
            raise ValueError("mode must be 'content' or 'content+metadata'")

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    logger.info(
        "Document deduplication completed",
        original_count=len(docs),
        unique_count=len(unique_docs),
        duplicates_removed=len(docs) - len(unique_docs),
    )
    return unique_docs


def build_context_with_metadata(docs: list[Document]) -> str:
    """Build context text with metadata and sources."""
    context_parts = []

    for i, doc in enumerate(docs, 1):
        source_info = ""
        if doc.metadata:
            source = doc.metadata.get("source", "")
            timestamp = doc.metadata.get("timestamp", "")
            if source:
                source_info = f"（来源：{source}"
                if timestamp:
                    source_info += f"，时间：{timestamp}"
                source_info += "）"

        metadata_info = ""
        if doc.metadata:
            filtered_metadata = {
                k: v
                for k, v in doc.metadata.items()
                if k
                not in [
                    "type",
                    "join_id",
                    "embedded_at",
                    "kid",
                    "original_index",
                ]
            }
            if filtered_metadata:
                metadata_info = f"\n元数据: {json.dumps(filtered_metadata, ensure_ascii=False)}"

        content = clean_text(doc.page_content)
        context_parts.append(f"文档{i}{source_info}：\n{content}{metadata_info}")

    return "\n\n".join(context_parts)


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def fetch_rerank_api(
    query: str,
    documents: list[str],
    top_n: int = 10,
    timeout: int = 300,
) -> dict:
    """Fetch rerank API with retry."""
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
        return response.json()["results"]
    else:
        raise Exception(f"Rerank API Error: {response.text}")


class GradeDocuments(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


def generate_query_or_respond(state: EnhancedRagState) -> dict:
    """Decide whether to retrieve or respond directly."""
    logger.info("Evaluating query - deciding whether to retrieve or respond directly")
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


def extract_artifacts_to_state(state: EnhancedRagState) -> dict:
    """Extract Document artifacts from ToolMessage and store in state."""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and hasattr(message, "artifact") and message.artifact:
            docs = message.artifact
            logger.info(
                "Extracted documents from current retrieval (replaces previous)",
                doc_count=len(docs) if isinstance(docs, list) else 0,
                tool_name=message.name,
            )
            return {"retrieved_docs": docs if isinstance(docs, list) else []}

    logger.warning("No ToolMessage with artifacts found, clearing retrieved_docs")
    return {"retrieved_docs": []}


def rerank_documents_node(state: EnhancedRagState) -> dict:
    """Rerank retrieved documents with fallback strategies."""
    question = get_latest_user_message(state["messages"])
    retrieved_docs = state.get("retrieved_docs", [])

    logger.info("Starting document reranking", doc_count=len(retrieved_docs), question=question[:100])

    if not retrieved_docs:
        logger.warning("No documents to rerank")
        return {"context": [], "retrieved_docs": []}

    retrieved_docs = deduplicate_documents(retrieved_docs)

    try:
        reranked_docs = rerank_documents(
            query=question,
            documents=retrieved_docs,
            top_n=settings.RERANK_MAX_DOCS,
            metadata={"source": "milvus_hybrid"},
        )

        reranked_docs = [
            doc for doc in reranked_docs if doc.metadata.get("relevance_score", 0) >= settings.RERANK_THRESHOLD
        ]

        logger.info("Plugin rerank completed", reranked_count=len(reranked_docs))
        return {"context": reranked_docs, "retrieved_docs": retrieved_docs}

    except Exception as e:
        logger.error("Plugin rerank failed, trying direct API", error=str(e))

    try:
        documents_str = [doc.page_content for doc in retrieved_docs]
        scores_results = fetch_rerank_api(question, documents_str, top_n=settings.RERANK_MAX_DOCS)

        reranked_docs = []
        for item in scores_results:
            idx: int = item["index"]
            score: float = item["relevance_score"]

            if score >= settings.RERANK_THRESHOLD:
                doc = Document(
                    page_content=retrieved_docs[idx].page_content,
                    metadata={**retrieved_docs[idx].metadata, "relevance_score": round(score, 4)},
                )
                reranked_docs.append(doc)

        logger.info("Direct API rerank completed", reranked_count=len(reranked_docs))
        return {"context": reranked_docs, "retrieved_docs": retrieved_docs}

    except Exception as fallback_error:
        logger.error("Direct API rerank also failed, using original documents", error=str(fallback_error))

    fallback_docs = retrieved_docs[: settings.RERANK_MAX_DOCS]
    logger.warning("Using fallback strategy", doc_count=len(fallback_docs))
    return {"context": fallback_docs, "retrieved_docs": retrieved_docs}


def grade_documents(state: EnhancedRagState) -> Literal["generate", "rewrite"]:
    """Determine document relevance."""
    logger.info("Grading document relevance")

    question = get_latest_user_message(state["messages"])
    docs = state.get("context", [])

    if not docs:
        logger.warning("No documents in context, will rewrite query")
        return "rewrite"

    docs_text = "\n\n".join([f"Document {i + 1}: {doc.page_content[:200]}..." for i, doc in enumerate(docs[:3])])

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    grade_prompt = f"""You are a grader assessing relevance of retrieved documents to a user question.

    Retrieved documents:
    {docs_text}

    User question:
    {question}

    If the documents contain keyword(s) or semantic meaning related to the user question, grade them as relevant.
    Give a binary score 'yes' or 'no' to indicate whether the documents are relevant to the question."""

    try:
        score = structured_llm_grader.invoke([{"role": "user", "content": grade_prompt}]).binary_score

        if score == "yes":
            logger.info("Documents graded as relevant - proceeding to generation")
            return "generate"
        else:
            logger.info("Documents graded as not relevant - will rewrite query")
            return "rewrite"
    except Exception as e:
        logger.error("Document grading failed, defaulting to generate", error=str(e))
        return "generate"


def rewrite_question(state: EnhancedRagState) -> dict:
    """Transform query for better retrieval."""
    logger.info("Rewriting query for better retrieval")

    question = get_latest_user_message(state["messages"])

    rewrite_prompt = f"""You are an expert at query expansion and transformation.

    Look at the input question and try to reason about the underlying semantic intent / meaning.

    Here is the initial question:
    {question}

    Formulate an improved question that will retrieve better documents from a vector database:"""

    response = llm.invoke([{"role": "user", "content": rewrite_prompt}])

    logger.info("Query rewritten", original=question[:100], rewritten=response.content[:100])
    return {"messages": [{"role": "user", "content": response.content}]}


def generate(state: EnhancedRagState) -> dict:
    """Generate answer from context."""
    logger.info("Generating answer from context")

    question = get_latest_user_message(state["messages"])
    docs = state.get("context", [])

    if not docs:
        logger.warning("No context documents for generation")
        response = llm.invoke([{"role": "user", "content": question}])
        return {"messages": [response]}

    context_text = build_context_with_metadata(docs)

    logger.info(
        "Generating with context",
        question_length=len(question),
        context_docs=len(docs),
        context_length=len(context_text),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.RAG_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "## Knowledge Base:\n{context}\n\n## User Question:\n{question}"),
        ]
    )

    prompt_input = {
        "context": context_text,
        "question": question,
        "system_time": "",
        "history": state.get("messages", [])[:-1],
    }
    try:
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(prompt_input)
        logger.info("Answer generated successfully", answer_length=len(answer))
    except Exception as e:
        logger.error("Answer generation failed", error=str(e))
        answer = f"抱歉，生成回答时出现错误：{str(e)}"

    return {"messages": [{"role": "assistant", "content": answer.strip()}]}


def create_enhanced_rag_graph(
    checkpointer: Checkpointer | None = None,
    name: str = "milvus_rag_agent_pre",
):
    """Create enhanced RAG graph."""
    logger.info("Building enhanced RAG graph", graph_name=name)

    workflow = StateGraph(EnhancedRagState)

    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("extract_artifacts", extract_artifacts_to_state)
    workflow.add_node("rerank", rerank_documents_node)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "generate_query_or_respond")

    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )

    workflow.add_edge("retrieve", "extract_artifacts")
    workflow.add_edge("extract_artifacts", "rerank")

    workflow.add_conditional_edges(
        "rerank",
        grade_documents,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    workflow.add_edge("rewrite", "generate_query_or_respond")
    workflow.add_edge("generate", END)
    compile_params = {"checkpointer": checkpointer, "name": name}
    compile_params = {k: v for k, v in compile_params.items() if v is not None}

    compiled_graph = workflow.compile(**compile_params)
    logger.info("Enhanced RAG graph compiled successfully")

    return compiled_graph


milvus_rag_agent_graph_pre = create_enhanced_rag_graph()

logger.info("Enhanced Milvus RAG Agent module loaded successfully")
