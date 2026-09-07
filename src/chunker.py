"""Utilities for splitting Python and text source files into chunks."""

import ast

from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 2000


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[str]:
    """Split generic text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    return splitter.split_text(text)


def get_chunk_offsets(
    text: str,
    chunks: list[str],
) -> list[tuple[str, int, int]]:
    """Return chunks with their offsets inside the given text."""
    results: list[tuple[str, int, int]] = []
    search_start = 0

    for chunk in chunks:
        start = text.find(chunk, search_start)

        if start == -1:
            raise ValueError(
                "Chunk not found in original text"
            )

        end = start + len(chunk)

        results.append(
            (chunk, start, end)
        )

        search_start = end

    return results


def get_line_offset(
    lines: list[str],
    line_number: int,
) -> int:
    """Convert a 1-based line number into a character offset."""
    count = 0

    for line in lines[:line_number - 1]:
        count += len(line)

    return count


def get_node_offsets(
    lines: list[str],
    node: ast.AST,
) -> tuple[int, int] | None:
    """Return absolute character offsets for an AST node."""
    if not hasattr(node, "lineno"):
        return None

    end_lineno = getattr(node, "end_lineno", None)

    if end_lineno is None:
        return None

    start_line = node.lineno

    decorators = getattr(node, "decorator_list", [])

    if decorators:
        decorator_lines = [
            decorator.lineno
            for decorator in decorators
            if hasattr(decorator, "lineno")
        ]

        if decorator_lines:
            start_line = min(
                start_line,
                min(decorator_lines),
            )

    start_offset = get_line_offset(
        lines,
        start_line,
    )

    end_offset = (
        get_line_offset(
            lines,
            end_lineno,
        )
        + len(lines[end_lineno - 1])
    )

    return start_offset, end_offset


def split_source_chunk(
    text: str,
    start_offset: int,
    chunk_size: int,
) -> list[tuple[str, int, int]]:
    """Split source text and convert offsets to absolute positions."""
    if len(text) <= chunk_size:
        return [
            (
                text,
                start_offset,
                start_offset + len(text),
            )
        ]

    small_chunks = chunk_text(
        text,
        chunk_size,
    )

    chunks_with_offsets = get_chunk_offsets(
        text,
        small_chunks,
    )

    results: list[tuple[str, int, int]] = []

    for (
        chunk,
        relative_start,
        relative_end,
    ) in chunks_with_offsets:
        absolute_start = (
            start_offset + relative_start
        )

        absolute_end = (
            start_offset + relative_end
        )

        results.append(
            (
                chunk,
                absolute_start,
                absolute_end,
            )
        )

    return results


def get_class_chunks(
    text: str,
    lines: list[str],
    class_node: ast.ClassDef,
    chunk_size: int,
) -> list[tuple[str, int, int]]:
    """Extract all top-level statements from a large class."""
    chunks: list[tuple[str, int, int]] = []

    for item in class_node.body:
        offsets = get_node_offsets(
            lines,
            item,
        )

        if offsets is None:
            continue

        start_offset, end_offset = offsets

        item_text = text[
            start_offset:end_offset
        ]

        item_chunks = split_source_chunk(
            item_text,
            start_offset,
            chunk_size,
        )

        chunks.extend(item_chunks)

    return chunks


def chunk_python(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[tuple[str, int, int]]:
    """Split Python source code into chunks with absolute offsets."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)

    chunks: list[tuple[str, int, int]] = []

    for node in tree.body:
        offsets = get_node_offsets(
            lines,
            node,
        )

        if offsets is None:
            continue

        start_offset, end_offset = offsets

        node_text = text[
            start_offset:end_offset
        ]

        if len(node_text) <= chunk_size:
            chunks.append(
                (
                    node_text,
                    start_offset,
                    end_offset,
                )
            )
            continue

        if isinstance(node, ast.ClassDef):
            class_chunks = get_class_chunks(
                text,
                lines,
                node,
                chunk_size,
            )

            chunks.extend(class_chunks)
            continue

        node_chunks = split_source_chunk(
            node_text,
            start_offset,
            chunk_size,
        )

        chunks.extend(node_chunks)

    return chunks


if __name__ == "__main__":
    text = """
from pydantic import BaseModel


def hello():
    print("hello")


async def fetch_data():
    print("fetching")


class User:
    name = "Mohamed"
    age = 25

    def __init__(self, name):
        self.name = name

    def greet(self):
        print(self.name)

    async def fetch(self):
        print("fetching user")


def another_function():
    print("another function")
"""

    chunks = chunk_python(
        text,
        chunk_size=200,
    )

    for chunk, start, end in chunks:
        print("=" * 60)
        print(f"START: {start}")
        print(f"END: {end}")
        print(f"LENGTH: {end - start}")
        print(chunk)