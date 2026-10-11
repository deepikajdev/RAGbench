"""Lightweight, dependency-free BM25Okapi keyword retriever implementation."""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set

from ragbench.core.models import Document, ScoredDocument
from ragbench.retrievers.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25 (Best Matching 25) keyword search retriever.

    Implements BM25Okapi scoring using only Python's standard library:
      - Term frequency saturation governed by k1.
      - Document length normalization governed by b.
      - Smooth non-negative Inverse Document Frequency (IDF).
    """

    def __init__(
        self,
        documents: Optional[Sequence[Document]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize and optionally index documents.

        Args:
            documents: Optional sequence of Document instances to index immediately.
            k1: Term frequency saturation tuning parameter (default 1.5). Must be > 0.
            b: Document length normalization tuning parameter (default 0.75). Must be in [0, 1].

        Raises:
            ValueError: If k1 <= 0 or b < 0 or b > 1.
        """
        if k1 <= 0:
            raise ValueError(f"k1 must be positive, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be between 0.0 and 1.0, got {b}")

        self.k1 = k1
        self.b = b

        self.documents: List[Document] = []
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.term_frequencies: List[Counter[str]] = []
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        self.idf: Dict[str, float] = {}

        if documents:
            self.index(documents)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words."""
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def index(self, documents: Sequence[Document]) -> None:
        """Index a collection of documents for BM25 retrieval.

        Args:
            documents: Sequence of Document instances.
        """
        self.documents = list(documents)
        self.doc_ids = [doc.doc_id for doc in self.documents]
        num_docs = len(self.documents)

        self.doc_lengths = []
        self.term_frequencies = []
        self.inverted_index = defaultdict(list)
        self.idf = {}

        if num_docs == 0:
            self.avg_doc_length = 0.0
            return

        total_length = 0
        doc_freqs: Counter[str] = Counter()

        for idx, doc in enumerate(self.documents):
            # Combine title (if present) and body content
            full_text = f"{doc.title or ''} {doc.content}".strip()
            tokens = self.tokenize(full_text)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            tf = Counter(tokens)
            self.term_frequencies.append(tf)

            for term in tf.keys():
                self.inverted_index[term].append(idx)
                doc_freqs[term] += 1

        self.avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0

        # Calculate Robertson/Lucene non-negative smooth IDF for all indexed terms
        for term, df in doc_freqs.items():
            self.idf[term] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[ScoredDocument]:
        """Search the indexed documents using BM25 scoring.

        Args:
            query: The search query string.
            top_k: Maximum number of ranked results to return (default 5).

        Returns:
            List of ScoredDocument objects ordered by BM25 score descending.
            Returns an empty list if the query is empty or produces no matching terms.

        Raises:
            ValueError: If top_k <= 0.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        if not self.documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Find candidate documents that contain at least one query token
        candidate_indices: Set[int] = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_indices.update(self.inverted_index[token])

        if not candidate_indices:
            return []

        scored_candidates: List[tuple[int, float]] = []
        avgdl = self.avg_doc_length if self.avg_doc_length > 0 else 1.0

        for doc_idx in candidate_indices:
            doc_len = self.doc_lengths[doc_idx]
            len_norm = 1.0 - self.b + self.b * (doc_len / avgdl)
            score = 0.0
            tf_map = self.term_frequencies[doc_idx]

            for token in query_tokens:
                if token in tf_map:
                    freq = tf_map[token]
                    idf_val = self.idf.get(token, 0.0)
                    tf_component = (freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm)
                    score += idf_val * tf_component

            if score > 0.0:
                scored_candidates.append((doc_idx, score))

        # Sort by score descending; break ties deterministically by doc_id
        scored_candidates.sort(
            key=lambda item: (-item[1], self.doc_ids[item[0]])
        )

        top_candidates = scored_candidates[:top_k]

        return [
            ScoredDocument(
                doc_id=self.doc_ids[doc_idx],
                score=round(score, 4),
                rank=rank,
            )
            for rank, (doc_idx, score) in enumerate(top_candidates, start=1)
        ]

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Retriever":
        """Factory method to instantiate and index a BM25Retriever from documents."""
        return cls(documents=documents, k1=k1, b=b)
