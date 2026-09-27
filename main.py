from pathlib import Path

import fire

from src.rag_system.chunker import chunk_markdown, chunk_python
from src.rag_system.corpus_loader import load_corpus
from src.rag_system.evaluate import evaluate_recall
from src.rag_system.indexer import Indexer
from src.rag_system.models import Chunk
from src.rag_system.retriever import Retriever
from src.rag_system.search_dataset import search_dataset


class CLI:
    """Command-line interface for the RAG system."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build and save the BM25 index."""
        root = Path("data/raw/vllm-0.10.1/vllm-0.10.1")

        documents = load_corpus(root)

        chunks: list[Chunk] = []

        for document in documents:
            if document.file_path.endswith(".py"):
                raw_chunks = chunk_python(
                    document.content,
                    max_chunk_size,
                )
            elif document.file_path.endswith(".md"):
                raw_chunks = chunk_markdown(
                    document.content,
                    max_chunk_size,
                )
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
        """Search the index."""
        retriever = Retriever(
            Path("data/processed/index")
        )

        result = retriever.search(
            "cli",
            query,
            k,
        )

        print(result)

    def search_dataset(
        self,
        dataset_path: str,
        index_path: str,
        k: int = 5,
        save_directory: str = "data/processed/search_results",
    ) -> None:
        """Search all questions in a dataset."""
        search_dataset(
            Path(dataset_path),
            Path(index_path),
            k,
            Path(save_directory),
        )

    def evaluate(
        self,
        dataset_path: str,
        results_path: str,
        k: int = 5,
    ) -> None:
        """Evaluate retrieval Recall@k."""
        recall = evaluate_recall(
            Path(dataset_path),
            Path(results_path),
            k,
        )

        print(f"Recall@{k}: {recall:.2%}")


if __name__ == "__main__":
    fire.Fire(CLI)
