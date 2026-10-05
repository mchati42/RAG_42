import json
from pathlib import Path

import bm25s

from .models import Chunk


class Retriever:
    """Search the BM25 index and return matching chunks."""
    def __init__(self, index_dir: Path) -> None:
        self.bm25 = bm25s.BM25.load(str(index_dir))
        with open(index_dir / "chunks.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        self.chunks = [
            Chunk.model_validate(item)
            for item in data
        ]
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Search the index."""
        query_tokens = bm25s.tokenize([query])
        results = self.bm25.retrieve(query_tokens, k=k)
        document_indexes = results.documents[0]
        scores = results.scores[0]
        search_results = []
        for chunk_id, score in zip(document_indexes, scores):
            chunk = self.chunks[chunk_id]
            search_results.append((chunk, score))
        return search_results
