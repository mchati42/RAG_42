from rag_system.chunker import chunk_python


def test_large_class_is_split():
    text = """class Example:
    def first(self):
        print("first")

    def second(self):
        print("second")

    def third(self):
        print("third")
"""

    chunks = chunk_python(
        text,
        chunk_size=50,
    )

    assert len(chunks) > 1

    for chunk, start, end in chunks:
        assert len(chunk) <= 50
        assert text[start:end] == chunk
