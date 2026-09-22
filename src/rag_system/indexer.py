from pathlib import Path
import json

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
        chunks_data = [
            chunk.model_dump()
            for chunk in self.chunks
        ]
        with open(path / "chunks.json", "w") as file:
            json.dump(chunks_data, file)

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "Indexer":
        """Load a BM25 index."""
        indexer = cls.__new__(cls)
        indexer.bm25 = bm25s.BM25.load(str(path))
        with open(path / "chunks.json", "r") as file:
            chunks_data = json.load(file)
        indexer.chunks = [
            Chunk.model_validate(data)
            for data in chunks_data
        ]
        return indexer