from rag_system.chunker import chunk_python


def test_python_chunk_offsets():
    text = """def hello():
    print("hello")
"""

    chunks = chunk_python(text)

    for chunk, start, end in chunks:
        assert text[start:end] == chunk


def test_python_chunk_size():
    text = """def hello():
    print("hello")
    print("this is a longer line")
    print("another line")
"""

    chunks = chunk_python(
        text,
        chunk_size=50,
    )

    for chunk, start, end in chunks:
        assert len(chunk) <= 50
        assert end - start == len(chunk)


def test_large_function_is_split():
    text = """def hello():
    print("line 1")
    print("line 2")
    print("line 3")
    print("line 4")
    print("line 5")
"""

    chunks = chunk_python(
        text,
        chunk_size=40,
    )

    assert len(chunks) > 1

    for chunk, start, end in chunks:
        assert len(chunk) <= 40
        assert text[start:end] == chunk


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
