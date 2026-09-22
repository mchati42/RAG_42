from pathlib import Path

from .indexer import Indexer

from .models import MinimalSearchResults, MinimalSource

class Retriever:
    def __init__(
        self,
        index_path: Path,
    ) -> None:
        self.indexer = Indexer.load(index_path)


    def search(
        self,
        question_id: str,
        query: str,
        k: int = 5,
    ) -> MinimalSearchResults:
        results = self.indexer.search(query, k)

        sources = [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            for chunk, score in results
        ]

        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=sources,
        )