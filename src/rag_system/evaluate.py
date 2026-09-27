"""Evaluate retrieval results using Recall@k."""

import json
from pathlib import Path

from .models import RagDataset, StudentSearchResults


def ranges_overlap(
    first_a: int,
    last_a: int,
    first_b: int,
    last_b: int,
) -> bool:
    """Return True when two character ranges overlap."""
    return first_a < last_b and first_b < last_a


def evaluate_recall(
    dataset_path: Path,
    results_path: Path,
    k: int = 5,
) -> float:
    """Calculate Recall@k."""
    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset_data = json.load(file)

    with open(results_path, "r", encoding="utf-8") as file:
        results_data = json.load(file)

    dataset = RagDataset.model_validate(dataset_data)
    results = StudentSearchResults.model_validate(results_data)

    correct = 0
    total = 0

    for question, search_result in zip(
        dataset.rag_questions,
        results.search_results,
    ):
        if not hasattr(question, "sources"):
            continue

        total += 1

        expected_sources = question.sources
        retrieved_sources = search_result.retrieved_sources[:k]

        found = False

        for expected in expected_sources:
            for retrieved in retrieved_sources:
                same_file = (
                    expected.file_path == retrieved.file_path
                )

                overlapping = ranges_overlap(
                    expected.first_character_index,
                    expected.last_character_index,
                    retrieved.first_character_index,
                    retrieved.last_character_index,
                )

                if same_file and overlapping:
                    found = True
                    break

            if found:
                break

        if found:
            correct += 1

    if total == 0:
        return 0.0

    return correct / total