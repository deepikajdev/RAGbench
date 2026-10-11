"""Abstract base class interface for RAG retrieval engines."""

from abc import ABC, abstractmethod
from typing import List, Sequence

from ragbench.core.models import Document, ScoredDocument


class BaseRetriever(ABC):
    """Abstract base class defining the retrieval interface for RAGBench."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[ScoredDocument]:
        """Search the indexed corpus and return ranked ScoredDocument objects.

        Args:
            query: The search query string.
            top_k: Maximum number of ranked results to return.

        Returns:
            List of ScoredDocument objects ordered by relevance rank (rank 1 first).
        """
        pass

    def retrieve_ids(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve ranked document IDs directly for a given search query.

        Args:
            query: The search query string.
            top_k: Maximum number of ranked results to return.

        Returns:
            List of string document IDs ordered from highest to lowest relevance.
        """
        return [doc.doc_id for doc in self.search(query, top_k=top_k)]
