"""Command-line interface for the RAG system."""

from pathlib import Path
import fire

from src.rag_system.chunker import (
    chunk_markdown,
    chunk_python,
    chunk_text,
)
from src.rag_system.corpus_loader import load_corpus
from src.rag_system.indexer import build_index
from src.rag_system.models import Chunk


class CLI:
    """Command-line interface for the RAG system."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build the BM25 index."""
        root = Path("data/raw/vllm-0.10.1/vllm-0.10.1")
        index_dir = Path("data/processed/index")

        documents = load_corpus(root)
        chunks: list[Chunk] = []

        for doc in documents:
            if doc.file_path.endswith(".py"):
                raw_chunks = chunk_python(
                    text=doc.content,
                    chunk_size=max_chunk_size,
                )
            elif doc.file_path.endswith(".md"):
                raw_chunks = chunk_markdown(
                    text=doc.content,
                    chunk_size=max_chunk_size,
                )
            elif doc.file_path.endswith(".txt"):
                raw_chunks = chunk_text(
                    text=doc.content,
                    chunk_size=max_chunk_size,
                )
            else:
                continue

            for content, start, end in raw_chunks:
                chunk = Chunk(
                    file_path=doc.file_path,
                    content=content,
                    first_character_index=start,
                    last_character_index=end,
                )
                chunks.append(chunk)

        build_index(
            chunks=chunks,
            index_dir=index_dir,
        )

if __name__ == "__main__":
    fire.Fire(CLI)
