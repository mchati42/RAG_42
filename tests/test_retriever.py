from pathlib import Path

from rag_system.retriever import Retriever


def test_retriever() -> None:
    """Test that the retriever returns search results."""
    index_path = Path("data/processed/index")

    retriever = Retriever(index_path)

    result = retriever.search(
        "Q1",
        "What activation formats does the fused batched MoE layer return in vLLM?",
        5,
    )

    assert result.question_id == "Q1"
    assert result.question != ""
    assert len(result.retrieved_sources) == 5