"""Command-line interface for the RAG system."""

import json
from tqdm import  tqdm
from pathlib import Path
from src.rag_system.generator import Generator
from src.rag_system.evaluate import evaluation_data

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
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
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

        for question in tqdm(data["rag_questions"]):
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
    def  answer(self, query: str) -> None:
        retriever = Retriever(Path("data/processed/index"))
        results = retriever.search(query, k=5)
        context = "\n\n".join(
            chunk.content for chunk, score in results
        )

        generator = Generator()
        answer = generator.generate(
            question=query,
            context=context,
        )
        print(answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate answers for existing search results."""
        with open(student_search_results_path, encoding="utf-8") as file:
            data = StudentSearchResults.model_validate_json(file.read())

        generator = Generator()
        answers: list[MinimalAnswer] = []

        for result in tqdm(data.search_results):
            context_parts: list[str] = []

            for source in result.retrieved_sources:
                file_path = Path(source.file_path)
                if not file_path.exists():
                    file_path = (
                        Path("data/raw/vllm-0.10.1")
                        / "vllm-0.10.1"
                        / file_path.relative_to("data/raw/vllm-0.10.1")
                    )

                with open(file_path, encoding="utf-8") as file:
                    content = file.read()

                context = content[
                    source.first_character_index:source.last_character_index
                ]
                context_parts.append(context)

            context = "\n\n".join(context_parts)

            answer = generator.generate(
                question=result.question,
                context=context,
            )

            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=data.k,
        )

        output_file = Path(save_directory) / Path(
            student_search_results_path
        ).name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def evaluate(self, dataset_path, search_results_path):
        """Evaluate search results."""
        evaluation_data(dataset_path, search_results_path)
        


if __name__ == "__main__":
    fire.Fire(CLI)
