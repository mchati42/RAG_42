"""Command-line interface for the RAG system."""

import json
from pathlib import Path

import fire

from src.rag_system.chunker import (
    chunk_markdown,
    chunk_python,
    chunk_text,
)
from src.rag_system.corpus_loader import load_corpus
from src.rag_system.indexer import build_index
from src.rag_system.models import (
    Chunk,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)
from src.rag_system.retriever import Retriever


class CLI:
    """Command-line interface for the RAG system."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build the BM25 index."""
        root = Path("data/raw/vllm-0.10.1")
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

    def search(self, query: str, k: int = 5) -> None:
        """Search the indexed corpus."""
        retriever = Retriever(Path("data/processed/index"))

        results = retriever.search(query, k)

        for chunk, score in results:
            print(f"\nScore: {score}")
            print(f"File: {chunk.file_path}")
            print(f"Start: {chunk.first_character_index}")
            print(f"End: {chunk.last_character_index}")
            print(chunk.content)

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 10,
    ) -> None:
        """Search all questions in a RAG dataset."""
        with open(dataset_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        retriever = Retriever(Path("data/processed/index"))

        search_results: list[MinimalSearchResults] = []

        for question in data["rag_questions"]:
            results = retriever.search(
                question["question"],
                k,
            )

            retrieved_sources: list[MinimalSource] = []

            for chunk, score in results:
                source = MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )

                retrieved_sources.append(source)

            search_result = MinimalSearchResults(
                question_id=question["question_id"],
                question=question["question"],
                retrieved_sources=retrieved_sources,
            )

            search_results.append(search_result)

        student_results = StudentSearchResults(
            search_results=search_results,
            k=k,
        )

        output_file = Path(save_directory) / Path(dataset_path).name
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(
                student_results.model_dump(),
                file,
                indent=2,
            )


if __name__ == "__main__":
    fire.Fire(CLI)