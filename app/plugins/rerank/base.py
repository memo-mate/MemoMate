"""Simplified rerank plugin base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class RerankContext:
    """Context passed through the rerank pipeline."""

    query: str
    documents: list[Document]
    top_n: int
    metadata: dict[str, Any]

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class RerankPlugin(ABC):
    """Base class for rerank plugins."""

    weight: float = 1.0  # Plugin weight for ordering (higher weight = higher priority)

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize plugin.

        Args:
            weight: Plugin weight for ordering (higher weight = higher priority)
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self.name = self.__class__.__name__

    def preprocess(self, context: RerankContext) -> RerankContext:
        """Preprocess hook called before reranking.

        Args:
            context: The rerank context

        Returns:
            Modified context
        """
        return context

    @abstractmethod
    def rerank(self, context: RerankContext) -> RerankContext:
        """Rerank processing core logic.

        Args:
            context: The rerank context

        Returns:
            Context with reranked documents
        """
        pass

    def postprocess(self, context: RerankContext) -> RerankContext:
        """Postprocess hook called after reranking.

        Args:
            context: The rerank context

        Returns:
            Modified context
        """
        return context

    def __lt__(self, other: "RerankPlugin") -> bool:
        """Compare plugins by weight for sorting."""
        return self.weight < other.weight

    def __str__(self):
        """Return string representation."""
        return f"{self.name}(weight={self.weight})"
