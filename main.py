import fire
from pathlib import Path
from src.rag_system.corpus_loader import load_corpus
from src.rag_system.chunker import chunk_python, chunk_markdown
from src.rag_system.models import Chunk
from src.rag_system.indexer import Indexer


class CLI:
    def index(self) -> None:
        root = Path("data/raw/vllm-0.10.1/vllm-0.10.1")

        documents = load_corpus(root)
        chunks: list[Chunk] =[]
        for doc in documents:
            if doc.file_path.endswith(".py"):
                raw_chunks = chunk_python(doc.content)
            elif doc.file_path.endswith(".md"):
                raw_chunks = chunk_markdown(doc.content)
            else: 
                continue
            for content, start, end in raw_chunks:
                chunks.append(
                    Chunk(
                        file_path=doc.file_path,
                        content=content,
                        first_character_index=start,
                        last_character_index=end
                    )
                )
        indexer = Indexer(chunks)
        index_path = Path("data/processed/index")
        indexer.save(index_path)
        print(f"Index saved to {index_path}")
if __name__ == "__main__":
    fire.Fire(CLI)