import io
import tokenize

from rank_bm25 import BM25Okapi

from rag_system.models import Chunk


def tokenize_code(code: str) -> list[str]:
    """Convert Python code into tokens."""
    tokens = tokenize.generate_tokens(
        io.StringIO(code).readline
    )

    return [
        token.string
        for token in tokens
        if token.type != tokenize.ENCODING
    ]


class Indexer:
    """Build and search a BM25 index over code chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        tokenized_chunks = [
            tokenize_code(chunk.content)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Return the k most relevant chunks."""
        query_tokens = tokenize_code(query)

        scores = self.bm25.get_scores(query_tokens)
        print(f"scores: {scores}")

        top_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        print(f"top_indexes: {top_indexes}")

        return [
            (self.chunks[i], float(scores[i]))
            for i in top_indexes
        ]
