import ast
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    return splitter.split_text(text)


def get_chunk_offsets(text: str, chunks: list[str]) -> list[tuple[str, int, int]]:
    results = []
    search_start = 0

    for chunk in chunks:
        start = text.find(chunk, search_start)
        if start == -1:
            raise ValueError("Chunk not found in original text")
        end = start + len(chunk)
        results.append((chunk, start, end))
        search_start = end
    return results

code = """ 
    def hello(name):
        print(name)
    """
    tree = ast.parse(code)
    print(tree)
if __name__ == "__main__":
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_text(text, chunk_size=5)

    print(chunks)