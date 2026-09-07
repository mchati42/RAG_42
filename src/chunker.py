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


def chunk_python(text: str) -> list[tuple[str, int, int]]:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    chunks: list[tuple[str, int, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.end_lineno is None:
                continue

            start_offset = get_line_offset(lines, node.lineno)

            end_offset = (
                get_line_offset(lines, node.end_lineno)
                + len(lines[node.end_lineno - 1])
            )

            chunk = text[start_offset:end_offset]
            chunks.append((chunk, start_offset, end_offset))

    return chunks


if __name__ == "__main__":
    text = """
class mohamed:
    def hello():
        print(mchati)
"""

    chunks = chunk_python(text)

    for chunk in chunks:
        print(repr(chunk))
