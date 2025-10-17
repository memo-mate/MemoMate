"""Simple rerank plugins with only core functionality."""

from typing import Any

import httpx
import tenacity
from langchain_core.documents import Document

from app.core import logger, settings
from app.plugins.rerank.base import RerankContext, RerankPlugin


class DuplicateRemovalPlugin(RerankPlugin):
    """Plugin that removes duplicate documents."""

    weight: float = 10.0

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize duplicate removal plugin."""
        super().__init__(config)

    def preprocess(self, context: RerankContext) -> RerankContext:
        """Add original indices to documents."""
        for i, doc in enumerate(context.documents):
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["original_index"] = i
        return context

    def rerank(self, context: RerankContext) -> RerankContext:
        """Remove duplicate documents."""
        if not context.documents:
            return context

        seen_content = set()
        unique_docs = []

        for doc in context.documents:
            # Simple content-based deduplication
            content_hash = hash(doc.page_content.strip())

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        original_count = len(context.documents)
        context.documents = unique_docs

        logger.info(f"Removed {original_count - len(unique_docs)} duplicate documents")
        return context


class APIRerankPlugin(RerankPlugin):
    """Simple API-based rerank plugin."""

    weight: float = 5.0

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize API rerank plugin."""
        super().__init__(config)

        # Configuration
        self.base_url = self.config.get("base_url", settings.RERANK_BASE_URL)
        self.api_key = self.config.get("api_key", settings.OPENAI_API_KEY)
        self.model = self.config.get("model", settings.RERANKER_MODEL)
        self.timeout = self.config.get("timeout", 300)
        self.threshold = self.config.get("threshold", settings.RERANK_THRESHOLD)

    def rerank(self, context: RerankContext) -> RerankContext:
        """Rerank documents using API."""
        if not self.base_url or not self.api_key:
            logger.warning("API rerank not configured, skipping")
            return context

        if not context.documents:
            return context

        try:
            # Extract document contents
            documents_str = [doc.page_content for doc in context.documents]

            # Call rerank API
            rerank_results = self._fetch_rerank_api(context.query, documents_str, context.top_n)

            # Apply rerank results
            reranked_docs = self._apply_rerank_scores(context.documents, rerank_results)

            # Filter by threshold
            filtered_docs = [doc for doc in reranked_docs if doc.metadata.get("relevance_score", 0) >= self.threshold]

            context.documents = filtered_docs
            logger.info(f"Reranked {len(context.documents)} documents")

        except Exception as e:
            logger.error(f"API rerank failed: {e}")
            # Fall back to original documents on error

        return context

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    )
    def _fetch_rerank_api(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        """Fetch rerank API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }

        with httpx.Client(verify=False) as client:
            response = client.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)

        if response.status_code == 200:
            return response.json()["results"]
        else:
            raise Exception(f"API Error: {response.text}")

    def _apply_rerank_scores(self, documents: list[Document], rerank_results: list[dict[str, Any]]) -> list[Document]:
        """Apply rerank scores to documents."""
        reranked_docs = []

        for item in rerank_results:
            idx = item["index"]
            score = item["relevance_score"]

            if 0 <= idx < len(documents):
                _doc: Document = documents[idx]
                doc = _doc.model_copy()
                doc.metadata["relevance_score"] = round(score, 4)
                reranked_docs.append(doc)

        return reranked_docs


class ThresholdFilterPlugin(RerankPlugin):
    """Plugin that filters documents by relevance threshold."""

    weight: float = 1.0

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize threshold filter plugin."""
        super().__init__(config)
        self.threshold = self.config.get("threshold", 0.5)

    def rerank(self, context: RerankContext) -> RerankContext:
        """Filter documents by threshold."""
        if not context.documents:
            return context

        # Only filter if documents have relevance scores
        scored_docs = [doc for doc in context.documents if "relevance_score" in doc.metadata]

        if scored_docs:
            filtered_docs = [doc for doc in scored_docs if doc.metadata["relevance_score"] >= self.threshold]

            context.documents = filtered_docs
            logger.info(f"Filtered by threshold {self.threshold}: {len(context.documents)} remaining")

        return context
