import re
from typing import Literal

import orjson
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import PlaywrightURLLoader, WebBaseLoader
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool as langchain_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.embedding.embeeding_model import EmbeddingFactory
from app.rag.llm.tokenizers import TokenCounter

web_loader_type: Literal["playwright", "webbase"] = "webbase"


def _get_web_loader(web_loader_type: Literal["playwright", "webbase"], urls: list[str]) -> WebBaseLoader:
    match web_loader_type:
        case "playwright":
            return PlaywrightURLLoader(urls, headless=True)
        case "webbase":
            return WebBaseLoader(urls)
        case _:
            raise ValueError(f"Invalid web loader type: {web_loader_type}")


def _get_num_tokens(docs: list[Document]) -> int:
    return sum(TokenCounter.estimate_tokens(doc.page_content) for doc in docs)


def _clean_text(text: str) -> str:
    # 替换形如 "\n \n \n" 的混合空行为单个 \n
    RE_MIXED_NEWLINES = re.compile(r"(?:\s*\n\s*){2,}")
    RE_SPACES = re.compile(r"[ \t]+")

    text = text.strip()
    text = RE_MIXED_NEWLINES.sub("\n", text)  # 替换混合换行
    text = RE_SPACES.sub(" ", text)  # 合并空格和 tab
    return text


@langchain_tool
async def duck_search(query: str) -> list[Document]:
    """Search the web for information."""
    search_results = await DuckDuckGoSearchResults(output_format="json", max_results=10).arun(query)
    if not search_results:
        return "No search results found"
    search_results = orjson.loads(search_results)
    web_loader = _get_web_loader(web_loader_type, [x["link"] for x in search_results])
    docs: list[Document] = []
    async for doc in web_loader.alazy_load():
        docs.append(doc)
    docs = [Document(page_content=_clean_text(doc.page_content), metadata=doc.metadata) for doc in docs]
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embedding = EmbeddingFactory.get()
    vectorstore = Chroma.from_documents(
        docs, embedding=embedding, client_settings=ChromaSettings(anonymized_telemetry=False)
    )
    retrieved_docs = vectorstore.similarity_search(query, k=10)
    return retrieved_docs
