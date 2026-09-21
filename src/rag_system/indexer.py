from pathlib import Path

import bm25s

from rag_system.models import Chunk


class Indexer:
    """Build and search a BM25 index over code chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        corpus = [chunk.content for chunk in chunks]

        self.bm25 = bm25s.BM25()
        tokens = bm25s.tokenize(corpus)
        self.bm25.index(tokens)

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Return the k most relevant chunks."""
        query_tokens = bm25s.tokenize([query])

        results, scores = self.bm25.retrieve(
            query_tokens,
            k=k,
        )

        return [
            (self.chunks[int(index)], float(score))
            for index, score in zip(
                results[0],
                scores[0],
            )
        ]

    def save(self, path: Path) -> None:
        """Save the BM25 index."""
        path.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(path))

    @classmethod
    def load(
        cls,
        path: Path,
        chunks: list[Chunk],
    ) -> "Indexer":
        """Load a BM25 index."""
        indexer = cls.__new__(cls)
        indexer.chunks = chunks
        indexer.bm25 = bm25s.BM25.load(str(path))
        return indexer