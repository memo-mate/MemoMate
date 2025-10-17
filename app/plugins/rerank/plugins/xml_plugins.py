"""Xml document aggregation plugin.

Aggregates document chunks by doc_id and reconstructs full documents.
Runs in preprocessing stage with weight 100.0 for highest priority.
"""

from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

from app.core import logger, settings
from app.enums.embedding import EmbeddingDriverEnum
from app.plugins.rerank.base import RerankContext, RerankPlugin
from app.rag.embedding.embedding_db.custom_qdrant import QdrantVectorStore
from app.rag.embedding.embeeding_model import MemoMateEmbeddings


def get_full_document(vector_store: QdrantVectorStore, doc_id: str) -> list[Document]:
    """Get all chunks for a document ID from vector store.

    Args:
        vector_store: Qdrant vector store instance
        doc_id: Document unique identifier

    Returns:
        List of documents
    """
    results = vector_store.search_by_metadata(metadatas={"doc_id": doc_id})
    return results


class XMLAggregationPlugin(RerankPlugin):
    """Document aggregation plugin for chunked documents.

    Groups document chunks by doc_id and reconstructs full documents.
    Runs with weight 100.0 to ensure preprocessing priority.
    """

    weight: float = 100.0  # Plugin execution weight (higher = earlier execution)

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize plugin with weight 100.0 for highest priority.

        Args:
            config: Plugin configuration with optional collection_name
        """
        super().__init__(config)
        self.collection_name = (
            config.get("collection_name", settings.RAG_COLLECTION_NAME) if config else settings.RAG_COLLECTION_NAME
        )

        logger.info(
            f"Initialized {self.name} plugin",
            weight=self.weight,
            collection_name=self.collection_name,
        )

    def preprocess(self, context: RerankContext) -> RerankContext:
        """Aggregate document chunks by doc_id before reranking.

        Groups chunks, retrieves full documents from vector store,
        sorts by chunk_id and concatenates into complete documents.

        Args:
            context: Rerank context with documents to process

        Returns:
            Context with aggregated full documents
        """
        original_count = len(context.documents)
        logger.info(
            f"{self.name}: Starting chunk aggregation",
            original_docs_count=original_count,
            query=context.query,
        )

        if not context.documents:
            logger.warning(f"{self.name}: No documents to process")
            return context

        try:
            # Group documents by doc_id
            docs_by_id: defaultdict[str, list[Document]] = defaultdict(list)
            for doc in context.documents:
                doc_id = doc.metadata.get("doc_id")
                if doc_id:
                    docs_by_id[doc_id].append(doc)
                else:
                    logger.warning(
                        f"{self.name}: Document missing doc_id",
                        doc_content=doc.page_content[:100],
                    )

            logger.info(
                f"{self.name}: Grouping by doc_id completed",
                unique_doc_ids=len(docs_by_id),
                doc_ids=list(docs_by_id.keys()),
            )

            # Get vector store
            vector_store = QdrantVectorStore(
                collection_name=self.collection_name,
                embeddings=MemoMateEmbeddings.local_embedding(driver=EmbeddingDriverEnum.MAC),
                path=settings.QDRANT_PATH,
            )
            logger.debug(
                f"{self.name}: Vector store retrieved",
                collection_name=self.collection_name,
            )

            # Retrieve and aggregate full documents
            full_docs = []
            for doc_id, chunk_docs in docs_by_id.items():
                if not chunk_docs[0].metadata.get("is_merge", True):
                    full_docs.extend(chunk_docs)
                    continue
                logger.debug(
                    f"{self.name}: Processing document {doc_id}",
                    chunk_count=len(chunk_docs),
                )

                try:
                    # Get all chunks for this document
                    chunks = get_full_document(vector_store, doc_id)
                    if not chunks:
                        logger.warning(f"{self.name}: No chunks found for document {doc_id}")
                        continue

                    # Sort by chunk_id
                    chunks = sorted(chunks, key=lambda d: d.metadata.get("chunk_id", 0))
                    logger.debug(
                        f"{self.name}: Document {doc_id} sorted chunks",
                        sorted_chunk_count=len(chunks),
                    )

                    # Concatenate full text
                    full_text = "\n".join([d.page_content for d in chunks])

                    # Keep base metadata (exclude chunk_id)
                    base_meta = {k: v for k, v in chunks[0].metadata.items() if k not in ["chunk_id"]}

                    full_doc = Document(page_content=full_text, metadata=base_meta)
                    full_docs.append(full_doc)

                    logger.debug(
                        f"{self.name}: Document {doc_id} aggregation completed",
                        full_text_length=len(full_text),
                        metadata_keys=list(base_meta.keys()),
                    )

                except Exception as e:
                    logger.error(
                        f"{self.name}: Error processing document {doc_id}",
                        error=str(e),
                        doc_id=doc_id,
                    )
                    # Continue processing other documents on error
                    continue

            # Update documents in context
            context.documents = full_docs

            final_count = len(full_docs)
            logger.info(
                f"{self.name}: Chunk aggregation completed",
                original_count=original_count,
                final_count=final_count,
                reduction_ratio=f"{((original_count - final_count) / original_count * 100):.1f}%"
                if original_count > 0
                else "0%",
            )

            # Record processing info in metadata
            context.metadata["xml_aggregation"] = {
                "processed": True,
                "original_count": original_count,
                "final_count": final_count,
                "unique_doc_ids": len(docs_by_id),
            }

        except Exception as e:
            logger.error(
                f"{self.name}: Aggregation processing failed",
                error=str(e),
                original_count=original_count,
            )
            # Keep original documents on failure
            context.metadata["xml_aggregation"] = {
                "processed": False,
                "error": str(e),
            }

        return context

    def rerank(self, context: RerankContext) -> RerankContext:
        """Skip rerank stage - this plugin only does preprocessing.

        Args:
            context: Rerank context

        Returns:
            Unmodified context
        """
        # This plugin only works in preprocessing stage
        logger.debug(f"{self.name}: Skipping rerank stage (preprocessing only)")
        return context

    def postprocess(self, context: RerankContext) -> RerankContext:
        """Log final aggregation statistics.

        Args:
            context: Rerank context after all processing

        Returns:
            Context with complete processing stats
        """
        aggregation_info = context.metadata.get("xml_aggregation", {})
        if aggregation_info.get("processed"):
            logger.info(
                f"{self.name}: Postprocess statistics",
                final_docs_count=len(context.documents),
                aggregation_stats=aggregation_info,
            )

        return context
