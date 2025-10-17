"""Simple rerank manager for plugin system."""

from typing import Any

from langchain_core.documents import Document

from app.core import logger, settings

from .base import RerankContext, RerankPlugin
from .loader import get_plugin_loader


class RerankManager:
    """Simple manager for rerank plugins."""

    def __init__(self):
        """Initialize the rerank manager."""
        self.plugins: list[RerankPlugin] = []
        self.loader = get_plugin_loader()

    def add_plugin(self, plugin: RerankPlugin) -> None:
        """Add a plugin to the manager.

        Args:
            plugin: Plugin instance to add
        """
        self.plugins.append(plugin)
        self._sort_plugins()
        logger.info(f"Added plugin: {plugin}")

    def load_plugin(
        self,
        plugin_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Load and add a plugin by name.

        Args:
            plugin_name: Name of the plugin to load
            weight: Plugin weight
            config: Plugin configuration
        """
        plugin = self.loader.create_plugin(plugin_name, config)
        self.add_plugin(plugin)

    def _sort_plugins(self) -> None:
        """Sort plugins by weight (descending)."""
        self.plugins.sort(key=lambda p: p.weight, reverse=True)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Execute the rerank pipeline.

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

        # Create initial context
        context = RerankContext(
            query=query,
            documents=documents.copy(),
            top_n=top_n,
            metadata=metadata or {},
        )

        logger.info(f"Starting rerank with {len(documents)} documents, {len(self.plugins)} plugins")

        # Execute plugins in order
        for plugin in self.plugins:
            try:
                # Execute preprocess hook
                context = plugin.preprocess(context)

                # Execute main rerank logic
                context = plugin.rerank(context)

                # Execute postprocess hook
                context = plugin.postprocess(context)

                logger.debug(f"Plugin {plugin.name} processed {len(context.documents)} documents")

            except Exception as e:
                logger.error(f"Error in plugin {plugin.name}: {e}")
                # Continue with other plugins on error
                continue

        # Apply top_n limit
        if len(context.documents) > top_n:
            context.documents = context.documents[:top_n]

        logger.info(f"Rerank completed, returning {len(context.documents)} documents")
        return context.documents

    def clear(self) -> None:
        """Clear all plugins."""
        self.plugins.clear()
        logger.info("Cleared all plugins")

    def reload_plugins(self) -> None:
        """Reload all plugins and reinitialize with defaults."""
        logger.info("Reloading plugin system...")

        # Clear current plugins
        self.clear()

        # Reload plugin definitions
        self.loader.reload_plugins()

        # Reinitialize with default plugins
        self._load_all_plugins()

    def _load_all_plugins(self) -> None:
        """Reload plugins."""
        self._initialize_default_plugins()
        self._load_custom_plugins()

    def _load_custom_plugins(self) -> None:
        """Load custom plugins."""
        logger.info("Loading custom plugins...")

        # Reload custom plugins
        _plugins = self.loader.list_plugins()
        for plugin in _plugins:
            if plugin in [
                "DuplicateRemovalPlugin",
                "APIRerankPlugin",
                "ThresholdFilterPlugin",
            ]:
                continue
            self.load_plugin(plugin)

    def _initialize_default_plugins(self) -> None:
        """Initialize default plugins."""
        try:
            # Duplicate removal (高优先级)
            self.load_plugin("DuplicateRemovalPlugin")

            # API rerank (中等优先级)
            self.load_plugin("APIRerankPlugin")

            # Threshold filter (最低优先级)
            self.load_plugin("ThresholdFilterPlugin", config={"threshold": settings.RERANK_THRESHOLD})

        except Exception as e:
            logger.error(f"Failed to load default plugins: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current status of the manager."""
        return {
            "plugins": [str(p) for p in self.plugins],
            "available_plugins": self.loader.list_plugins(),
        }


# Global rerank manager instance
_rerank_manager: RerankManager | None = None


def get_rerank_manager() -> RerankManager:
    """Get the global rerank manager instance."""
    global _rerank_manager
    if _rerank_manager is None:
        _rerank_manager = RerankManager()
    return _rerank_manager
