from langchain_text_splitters import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)


def _split(
    text: str,
    splitter: TextSplitter,
) -> list[tuple[str, int, int]]:
    """Split text and return content with character positions."""
    chunks = splitter.split_text(text)

    result: list[tuple[str, int, int]] = []
    search_from = 0

    for chunk in chunks:
        start = text.find(chunk, search_from)
        end = start + len(chunk)

        result.append((chunk, start, end))

        search_from = end

    return result


def chunk_python(
    text: str,
    chunk_size: int = 2000,
) -> list[tuple[str, int, int]]:
    """Split Python code."""
    splitter = PythonCodeTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    return _split(text, splitter)


def chunk_markdown(
    text: str,
    chunk_size: int = 2000,
) -> list[tuple[str, int, int]]:
    """Split Markdown."""
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    return _split(text, splitter)


def chunk_text(
    text: str,
    chunk_size: int = 2000,
) -> list[tuple[str, int, int]]:
    """Split plain text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    return _split(text, splitter)

