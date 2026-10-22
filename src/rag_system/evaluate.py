"""Evaluate search results."""

from models import(
    RagDataset, StudentSearchResults,
    MinimalSource
)
import json 
from pathlib import Path


def load_json(file_path: Path) -> dict:
    """load data from a json file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        raise
    return data

def load_evaluation_data(
        dataset_path: Path,
        results_path: Path,
) -> tuple[RagDataset, StudentSearchResults]:
    """Load the dataset and search results."""
    data_dataset = load_json(dataset_path)
    data_results = load_json(results_path)
    dataset_obj = RagDataset.model_validate(data_dataset)
    results_obj = StudentSearchResults.model_validate(data_results)

    return (dataset_obj, results_obj)


def calculate_iou(
        start_a: int,
        end_a: int,
        start_b: int,
        end_b: int
) -> float:
    """Calculate the intersection over union of two ranges."""

    if start_a >= end_a or start_b >= end_b:
        return 0.0
    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)

    intersection_length = max(
        0, intersection_end - intersection_start
    )

    union_length = (end_a - start_a) + (end_b - start_b) - intersection_length
    if union_length == 0:
        return 0.0
    return intersection_length / union_length

def is_source_correct(
        expected_source: MinimalSource,
        retrieved_source: MinimalSource,
) -> bool:
    """check if a retrieved source matches the expected source."""
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
) -> float:
    """Calculate Recall@K"""
    for source in retrieved_sources[:5]:
        
    



print(calculate_iou(100, 200, 300, 400))