from pathlib import Path

import fire

from src.rag_system.chunker import chunk_markdown, chunk_python
from src.rag_system.corpus_loader import load_corpus
from src.rag_system.indexer import Indexer
from src.rag_system.models import Chunk
from src.rag_system.retriever import Retriever


class CLI:
    """Command-line interface for the RAG system."""

    def index(self) -> None:
        """Build and save the BM25 index."""
        root = Path("data/raw/vllm-0.10.1/vllm-0.10.1")

        documents = load_corpus(root)

        chunks: list[Chunk] = []

        for document in documents:
            if document.file_path.endswith(".py"):
                raw_chunks = chunk_python(document.content)
            elif document.file_path.endswith(".md"):
                raw_chunks = chunk_markdown(document.content)
            else:
                continue

            for content, start, end in raw_chunks:
                chunks.append(
                    Chunk(
                        file_path=document.file_path,
                        content=content,
                        first_character_index=start,
                        last_character_index=end,
                    )
                )
        print(f"Created {len(chunks)} chunks")

        indexer = Indexer(chunks)

        index_path = Path("data/processed/index")
        indexer.save(index_path)

        print(f"Index saved to {index_path}")
    def search(self, query: str, k: int = 5) -> None:
        retriever = Retriever(
            Path("data/processed/index")
        )
        result = retriever.search(
            "cli",
            query,
            k,
        )
        print(result)

if __name__ == "__main__":
    fire.Fire(CLI)
