import ast

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    return splitter.split_text(text)


def get_chunk_offsets(
    text: str,
    chunks: list[str],
) -> list[tuple[str, int, int]]:
    results: list[tuple[str, int, int]] = []
    search_start = 0

    for chunk in chunks:
        start = text.find(chunk, search_start)

        if start == -1:
            raise ValueError("Chunk not found in original text")

        end = start + len(chunk)
        results.append((chunk, start, end))
        search_start = end

    return results


def get_line_offset(lines: list[str], line_number: int) -> int:
    count = 0

    for line in lines[:line_number - 1]:
        count += len(line)

    return count

def chunk_python(
    text: str,
    chunk_size: int = 2000,
) -> list[tuple[str, int, int]]:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    chunks: list[tuple[str, int, int]] = []

    for node in tree.body:
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue

        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if not isinstance(
                    method,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    continue

                if method.end_lineno is None:
                    continue

                start_offset = get_line_offset(
                    lines,
                    method.lineno,
                )

                end_offset = (
                    get_line_offset(lines, method.end_lineno)
                    + len(lines[method.end_lineno - 1])
                )

                chunk = text[start_offset:end_offset]

                if len(chunk) <= chunk_size:
                    chunks.append(
                        (chunk, start_offset, end_offset)
                    )
                else:
                    small_chunks = chunk_text(
                        chunk,
                        chunk_size,
                    )

                    small_chunks_with_offsets = get_chunk_offsets(
                        chunk,
                        small_chunks,
                    )

                    for (
                        small_chunk,
                        relative_start,
                        relative_end,
                    ) in small_chunks_with_offsets:
                        absolute_start = (
                            start_offset + relative_start
                        )
                        absolute_end = (
                            start_offset + relative_end
                        )

                        chunks.append(
                            (
                                small_chunk,
                                absolute_start,
                                absolute_end,
                            )
                        )

            continue

        if node.end_lineno is None:
            continue

        start_offset = get_line_offset(
            lines,
            node.lineno,
        )

        end_offset = (
            get_line_offset(lines, node.end_lineno)
            + len(lines[node.end_lineno - 1])
        )

        chunk = text[start_offset:end_offset]

        if len(chunk) <= chunk_size:
            chunks.append(
                (chunk, start_offset, end_offset)
            )
        else:
            small_chunks = chunk_text(
                chunk,
                chunk_size,
            )

            small_chunks_with_offsets = get_chunk_offsets(
                chunk,
                small_chunks,
            )

            for (
                small_chunk,
                relative_start,
                relative_end,
            ) in small_chunks_with_offsets:
                absolute_start = (
                    start_offset + relative_start
                )
                absolute_end = (
                    start_offset + relative_end
                )

                chunks.append(
                    (
                        small_chunk,
                        absolute_start,
                        absolute_end,
                    )
                )

    return chunks

if __name__ == "__main__":
    text = """
from pydantic import BaseModel

def hello():
    print("hello")


async def fetch_data():
    print("fetching")


class User:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(self.name)
"""

    chunks = chunk_python(text)

    for chunk, start, end in chunks:
        print("=" * 40)
        print(f"START: {start}")
        print(f"END: {end}")
        print(chunk)