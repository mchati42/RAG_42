from pathlib import Path
from typing import List
from tqdm import tqdm
from .models import Document

PY_MD_TXT_FILES = {".py", ".md", ".txt"}

def load_corpus(root: Path) -> List[Document]:
    documents: List[Document] = []
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
        
    paths = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in PY_MD_TXT_FILES
    ]
    
    for path in tqdm(paths, desc="Loading corpus", ncols=80):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            document = Document(
                file_path=str(path).replace("\\", "/"),
                content=content,
            )
            documents.append(document)
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            
    return documents