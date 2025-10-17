"""Simple rerank plugin system for dynamic document reranking."""

from .base import RerankContext, RerankPlugin
from .loader import PluginLoader, get_plugin_loader
from .manager import RerankManager, get_rerank_manager
from .service import (
    get_rerank_status,
    reload_rerank_plugins,
    rerank_documents,
)

__all__ = [
    "RerankPlugin",
    "RerankContext",
    "RerankManager",
    "PluginLoader",
    "get_rerank_manager",
    "get_plugin_loader",
    "rerank_documents",
    "get_rerank_status",
    "reload_rerank_plugins",
]
