"""Load source documents from the vLLM corpus."""

from pathlib import Path

from .models import Document


ROOT = Path("data/raw/vllm-0.10.1/vllm-0.10.1")
SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}


def load_corpus(root: Path) -> list[Document]:
    """Load supported files from the corpus directory."""
    documents: list[Document] = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            content = path.read_text(encoding="utf-8")

            document = Document(
                file_path=str(path),
                content=content,
            )

            documents.append(document)

    return documents


def main() -> None:
    """Load the corpus and print the number of documents."""
    documents = load_corpus(ROOT)
    print(f"Total documents: {len(documents)}")


if __name__ == "__main__":
    main()
