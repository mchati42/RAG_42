"""Split source documents into chunks."""

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def chunk_python(
    text: str,
    chunk_size: int = CHUNK_SIZE,
) -> list[tuple[str, int, int]]:
    """Split Python code into chunks."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_text(text)

    return _add_offsets(text, chunks)


def chunk_markdown(
    text: str,
    chunk_size: int = CHUNK_SIZE,
) -> list[tuple[str, int, int]]:
    """Split Markdown into chunks."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_text(text)

    return _add_offsets(text, chunks)


def _add_offsets(
    text: str,
    chunks: list[str],
) -> list[tuple[str, int, int]]:
    """Add character offsets to chunks."""
    results = []
    search_from = 0

    for chunk in chunks:
        start = text.find(chunk, search_from)

        if start == -1:
            continue

        end = start + len(chunk)

        results.append((chunk, start, end))

        search_from = start + 1

    return results