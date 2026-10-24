Installation and project structure
[ ] uv sync works from the repository root
[ ] pyproject.toml and uv.lock are present
[ ] src/ is runnable with python -m src
[ ] Makefile has install, run, debug, clean and lint targets
[ ] README.md and .gitignore are present

Indexing and retrieval
[ ] Python code has its own chunking strategy
[ ] Markdown/text has its own chunking strategy
[ ] TF-IDF or BM25 is implemented
[ ] Maximum chunk size is configurable and defaults to 2000
[ ] Index is saved under data/processed/
[ ] Indexing completes within 5 minutes
[ ] Single-question search works
[ ] Dataset search works and uses tqdm for long operations

Answer generation and evaluation
[ ] Default model is Qwen/Qwen3-0.6B
[ ] Single-question answer works
[ ] answer_dataset creates valid answer JSON
[ ] evaluate accepts configurable result and dataset paths
[ ] Recall@5 reaches 80% for docs and 50% for code

Data and reliability
[ ] Pydantic models match the required data structures
[ ] Search output includes question_id, question and retrieved_sources
[ ] Source paths and character indexes are correct
[ ] Missing files and malformed JSON are handled without unhandled tracebacks
[ ] Empty queries and k=0 are handled safely
[ ] flake8 and the required mypy command pass

Submission
[ ] README contains all required sections in English
[ ] Generated outputs, caches and model weights are not committed
[ ] All tests pass
[ ] Changes are committed and pushed to GitHub