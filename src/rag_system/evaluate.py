"""Evaluate search results."""

import json
from pathlib import Path

from .models import (
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def load_json(file_path: Path) -> dict:
    """Load data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_iou(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> float:
    """Calculate the intersection over union of two ranges."""
    if start_a >= end_a or start_b >= end_b:
        return 0.0

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)

    intersection_length = max(
        0, intersection_end - intersection_start
    )

    union_length = (
        (end_a - start_a)
        + (end_b - start_b)
        - intersection_length
    )

    if union_length == 0:
        return 0.0

    return intersection_length / union_length


def is_source_correct(
    expected_source: MinimalSource,
    retrieved_source: MinimalSource,
) -> bool:
    """Check if a retrieved source matches the expected source."""
    if expected_source.file_path != retrieved_source.file_path:
        return False

    iou = calculate_iou(
        expected_source.first_character_index,
        expected_source.last_character_index,
        retrieved_source.first_character_index,
        retrieved_source.last_character_index,
    )

    return iou >= 0.05


def calculate_recall_at_k(
    expected_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    k: int,
) -> float:
    """Calculate Recall@K for one question."""
    if not expected_sources or k <= 0:
        return 0.0

    count = 0

    for expected_source in expected_sources:
        for retrieved_source in retrieved_sources[:k]:
            if is_source_correct(expected_source, retrieved_source):
                count += 1
                break

    return count / len(expected_sources)


def recallk(
    dataset_obj: RagDataset,
    student: dict,
    k: int,
) -> float:
    """Calculate average Recall@K for the dataset."""
    questions = dataset_obj.rag_questions

    if not questions:
        return 0.0

    total_recall = 0.0

    for dataset_question in questions:
        student_question = student.get(dataset_question.question_id)

        if student_question is None:
            continue

        question_recall = calculate_recall_at_k(
            dataset_question.sources,
            student_question.retrieved_sources,
            k,
        )

        total_recall += question_recall

    return total_recall / len(questions)


def evaluation_data(
    dataset_path: Path,
    results_path: Path,
) -> None:
    """Load data and print evaluation results."""
    data_dataset = load_json(dataset_path)
    data_results = load_json(results_path)

    dataset_obj = RagDataset.model_validate(data_dataset)
    results_obj = StudentSearchResults.model_validate(data_results)
    print(f"📊 Questions evaluated: {len(dataset_obj.rag_questions)}")
    student = {
        result.question_id: result
        for result in results_obj.search_results
    }

    for k in (1, 3, 5, 10):
        recall = recallk(dataset_obj, student, k)
        print(f"📈Recall@{k}: {recall:.3f}")


if __name__ == "__main__":
    evaluation_data(
        Path(
            "data/datasets/AnsweredQuestions/"
            "dataset_docs_public.json"
        ),
        Path(
            "data/output/search_results/"
            "dataset_docs_public.json"
        ),
    )