from pathlib import Path

from rag_system.indexer import Indexer


def test_search_returns_k_results() -> None:
    """Test that search returns the requested number of results."""
    index_path = Path("data/processed/index")

    indexer = Indexer.load(index_path)

    results = indexer.search(
        "What activation formats does the fused batched MoE layer return in vLLM?",
        k=5,
    )

    assert len(results) == 5


def test_search_returns_chunks_and_scores() -> None:
    """Test that search returns chunks with scores."""
    index_path = Path("data/processed/index")

    indexer = Indexer.load(index_path)

    results = indexer.search(
        "What is vLLM?",
        k=5,
    )

    for chunk, score in results:
        assert chunk.file_path != ""
        assert chunk.content != ""
        assert isinstance(score, float)