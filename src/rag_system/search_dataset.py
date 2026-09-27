"""Search a complete RAG dataset."""

import json
from pathlib import Path

from tqdm import tqdm

from .models import RagDataset, StudentSearchResults
from .retriever import Retriever


def search_dataset(
    dataset_path: Path,
    index_path: Path,
    k: int,
    save_directory: Path,
) -> None:
    """Search all questions in a dataset and save the results."""

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    dataset = RagDataset.model_validate(data)

    retriever = Retriever(index_path)

    results = []

    for question in tqdm(
        dataset.rag_questions,
        desc="Searching dataset",
    ):
        result = retriever.search(
            question.question_id,
            question.question,
            k,
        )

        results.append(result)

    student_results = StudentSearchResults(
        search_results=results,
        k=k,
    )

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = save_directory / "search_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            student_results.model_dump(),
            file,
            indent=2,
        )

    print(f"Results saved to {output_path}")