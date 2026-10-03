from pathlib import Path

import bm25s

from .models import Chunk

import json



def build_index(
    chunks: list[Chunk],
    index_dir: Path
) -> None:
    "Build and save the BM25 index."
    corpus = [chunk.content for chunk in chunks]
    corpus_tokens = bm25s.tokenize(
        corpus,
        stopwords="en"
    )
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    index_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    retriever.save(index_dir)
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as file:
        json.dump(
            [chunk.model_dump() for chunk in chunks],
            file,
            indent=2,
        )