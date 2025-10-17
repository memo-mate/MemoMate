"""Simple rerank service with dynamic loading support."""

from typing import Any

from langchain_core.documents import Document

from app.core import logger

from .manager import RerankManager, get_rerank_manager

# Global rerank manager
_rerank_manager: RerankManager | None = None


def get_configured_rerank_manager() -> RerankManager:
    """Get or create a configured rerank manager."""
    global _rerank_manager
    if _rerank_manager is None:
        _rerank_manager = get_rerank_manager()
        # Initialize with default plugins
        _rerank_manager._load_all_plugins()

    return _rerank_manager


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = 10,
    metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Rerank documents using the simple plugin system.

    Args:
        query: Search query
        documents: Documents to rerank
        top_n: Number of top documents to return
        metadata: Additional metadata

    Returns:
        Reranked documents
    """
    if not documents:
        return documents

    try:
        manager = get_configured_rerank_manager()
        return manager.rerank(query, documents, top_n, metadata)
    except Exception as e:
        logger.error(f"Simple rerank failed: {e}")
        # Fallback to original documents
        return documents[:top_n]


def reload_rerank_plugins() -> dict[str, Any]:
    """Dynamically reload all rerank plugins."""
    global _rerank_manager
    try:
        if _rerank_manager is None:
            _rerank_manager = get_configured_rerank_manager()

        _rerank_manager.reload_plugins()

        logger.info("Successfully reloaded rerank plugins")
        return {
            "status": "success",
            "message": "Plugins reloaded successfully",
            **_rerank_manager.get_status(),
        }
    except Exception as e:
        logger.error(f"Failed to reload plugins: {e}")
        return {"status": "error", "error": str(e)}


def get_rerank_status() -> dict[str, Any]:
    """Get status of the rerank system.

    Returns:
        Status information
    """
    try:
        manager = get_configured_rerank_manager()
        return manager.get_status()
    except Exception as e:
        return {"error": str(e)}
