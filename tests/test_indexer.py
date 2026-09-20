from rag_system.indexer import Indexer
from rag_system.models import Chunk


def create_chunks() -> list[Chunk]:
    return [
        Chunk(
            file_path="a.py",
            content="vllm uses moe models",
            first_character_index=0,
            last_character_index=20,
        ),
        Chunk(
            file_path="b.py",
            content="python classes and objects",
            first_character_index=0,
            last_character_index=25,
        ),
        Chunk(
            file_path="c.py",
            content="database connection",
            first_character_index=0,
            last_character_index=18,
        ),
    ]


def test_indexer_search() -> None:
    chunks = create_chunks()
    indexer = Indexer(chunks)

    results = indexer.search("vllm moe", k=2)

    assert len(results) == 2
    assert results[0][0].file_path == "a.py"
    assert results[0][1] > results[1][1]


def test_indexer_respects_k() -> None:
    chunks = create_chunks()
    indexer = Indexer(chunks)

    results = indexer.search("vllm", k=1)

    assert len(results) == 1


def test_indexer_returns_correct_chunk() -> None:
    chunks = create_chunks()
    indexer = Indexer(chunks)

    results = indexer.search("database", k=1)

    assert results[0][0].file_path == "c.py"


def test_indexer_returns_score() -> None:
    chunks = create_chunks()
    indexer = Indexer(chunks)

    results = indexer.search("vllm", k=1)

    chunk, score = results[0]

    assert isinstance(chunk, Chunk)
    assert isinstance(score, float)
