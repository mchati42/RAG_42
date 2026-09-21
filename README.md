# RAG Against the Machine

A Retrieval-Augmented Generation (RAG) system built as part of the 42 curriculum.

The goal of this project is to build a RAG pipeline that can search the provided **vLLM 0.10.1** source code and use the retrieved information to answer questions.

## Project Goal

The system follows this pipeline:

```text
Source Code
    ↓
Corpus Loading
    ↓
Chunking
    ↓
BM25 Indexing
    ↓
Retrieval
    ↓
Context Augmentation
    ↓
Qwen/Qwen3-0.6B
    ↓
Answer
    ↓
Evaluation
```

The project also includes evaluation using the provided question datasets.

---

## Project Structure

```text
RAG_42/
│
├── data/
│   ├── raw/
│   │   └── vllm-0.10.1/
│   ├── datasets/
│   │   └── AnsweredQuestions/
│   └── processed/
│
├── src/
│   └── rag_system/
│       ├── __init__.py
│       ├── chunker.py
│       ├── corpus_loader.py
│       ├── dataset_loader.py
│       ├── indexer.py
│       └── models.py
│
├── tests/
│   ├── test_chunker.py
│   └── test_indexer.py
│
├── main.py
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Requirements

* Python 3.10+
* `uv`
* Git

The project uses:

* Pydantic
* LangChain Text Splitters
* bm25s
* tqdm
* Python Fire
* pytest
* mypy
* flake8

---

## Installation

Clone the repository:

```bash
git clone git@github.com:mchati42/RAG_42.git
cd RAG_42
```

Install the project dependencies:

```bash
uv sync
```

---

## Dataset

The project uses the provided RAG question datasets.

The answered questions contain:

* question
* answer
* source file
* source character range
* difficulty
* validity

Example source metadata:

```text
file_path
first_character_index
last_character_index
```

These values are used to evaluate whether retrieved chunks match the expected source.

---

# Indexing

The indexing pipeline currently performs these steps:

```text
vLLM source files
        ↓
Corpus Loader
        ↓
LangChain Chunker
        ↓
Chunk objects
        ↓
BM25
        ↓
Persistent index
```

## Corpus Loading

The corpus loader scans the vLLM source directory and loads supported source files.

Currently supported extensions include:

```text
.py
.md
.txt
```

Long-running corpus loading uses `tqdm` to display progress.

---

## Chunking Strategy

The project uses LangChain's:

```python
RecursiveCharacterTextSplitter
```

Python files use:

```python
RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
)
```

Markdown files use:

```python
RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
)
```

The default chunk size is:

```text
2000 characters
```

The current overlap is:

```text
200 characters
```

Each chunk keeps its original source metadata:

```text
file_path
first_character_index
last_character_index
```

This allows the retrieval system to identify where the chunk came from in the original source file.

---

# BM25 Indexing

The project uses **BM25** for lexical retrieval.

The implementation uses the `bm25s` Python library.

The index is built from the generated source chunks.

The generated index is stored under:

```text
data/processed/index/
```

The index is generated locally and should not be committed to Git.

To build the index:

```bash
uv run python main.py index
```

The current indexing run processes the vLLM corpus and creates thousands of chunks before saving the BM25 index.

---

# Retrieval

Retrieval is the next major part of the project.

The planned retrieval pipeline is:

```text
User question
      ↓
BM25 search
      ↓
Top-k chunks
      ↓
MinimalSearchResults
```

The retriever will return the most relevant source chunks together with:

```text
file_path
first_character_index
last_character_index
```

The value of `k` will be configurable.

---

# Answer Generation

After retrieval, the retrieved chunks will be used as context for the language model.

The required model is:

```text
Qwen/Qwen3-0.6B
```

The planned pipeline is:

```text
Question
    ↓
BM25 Retrieval
    ↓
Top-k Sources
    ↓
Context
    ↓
Qwen/Qwen3-0.6B
    ↓
Answer
```

The generated answer should be based on the retrieved source context.

---

# Evaluation

The project evaluates retrieval using:

```text
Recall@k
```

The evaluation checks whether the retrieved source matches the expected source from the dataset.

A retrieval is considered correct when the retrieved source:

1. points to the expected file
2. has a character range that overlaps the expected source range

The project will evaluate several values of `k`, including:

```text
Recall@1
Recall@3
Recall@5
Recall@10
```

The target requirements from the subject are:

```text
Documentation: ≥ 80% Recall@5
Code:          ≥ 50% Recall@5
```

These values will be measured after the retrieval system is implemented.

---

# Performance

The indexing process must complete within the time limit defined by the subject.

The project will measure:

* total indexing time
* number of documents
* number of chunks
* retrieval time

`tqdm` is used for long-running operations so that progress is visible to the user.

---

# Data Models

The project uses Pydantic models to validate structured data.

Important models include:

```python
MinimalSource
AnsweredQuestion
UnansweredQuestion
RagDataset
MinimalSearchResults
MinimalAnswer
StudentSearchResults
StudentSearchResultsAndAnswer
Document
Chunk
```

A `Chunk` contains:

```text
file_path
content
first_character_index
last_character_index
```

---

# Command Line Interface

Python Fire is used to expose project functionality through the command line.

Current command:

```bash
uv run python main.py index
```

The final CLI will provide the commands required by the subject:

```text
index
search
search_dataset
answer
answer_dataset
evaluate
```

The CLI will support configurable parameters such as:

```text
k
chunk size
input paths
output paths
```

---

# Testing

Run the test suite with:

```bash
uv run pytest
```

The project currently contains tests for:

* chunking
* BM25 indexing

The test suite will be expanded as retrieval, generation, and evaluation are implemented.

---

# Code Quality

The project follows:

* PEP 8
* type hints
* docstrings
* flake8
* mypy

Run mypy with:

```bash
uv run mypy src
```

Run flake8 with:

```bash
uv run flake8
```

Run tests with:

```bash
uv run pytest
```

---

# AI Usage

AI tools were used during development to help understand concepts, debug errors, review implementation ideas, and improve the project structure.

The final implementation is tested and understood by the student.

---

# Design Decisions

## BM25

BM25 was selected because the project requires either BM25 or TF-IDF and the corpus contains source code where exact words and identifiers are important.

## LangChain Text Splitters

LangChain's `RecursiveCharacterTextSplitter` is used instead of maintaining a large custom text-splitting implementation.

Language-specific separators are used for Python and Markdown.

## Character Offsets

Character offsets are preserved because the evaluation dataset identifies expected sources using file paths and character ranges.

## Persistent Index

The BM25 index is persisted under:

```text
data/processed/
```

This avoids rebuilding the index every time retrieval is performed.

---

# Challenges

Some of the main challenges during development include:

* handling a large source-code corpus
* preserving exact source character offsets
* choosing a suitable chunking strategy
* building and persisting the BM25 index
* handling Python source files that cannot always be parsed by Python's AST
* keeping the implementation compatible with the project requirements
* measuring retrieval quality with Recall@k

---

# Current Progress

```text
[x] Project setup
[x] Dependencies
[x] Pydantic models
[x] Dataset loading
[x] Corpus loading
[x] Python chunking
[x] Markdown chunking
[x] BM25 indexing
[x] Persistent index

[ ] Retrieval
[ ] Dataset search
[ ] Answer generation
[ ] Evaluation
[ ] Recall@k measurement
[ ] Performance measurement
[ ] Final CLI
[ ] Final documentation
```

---

# Example

Build the index:

```bash
uv run python main.py index
```

The command:

1. loads the vLLM source corpus
2. splits the source files into chunks
3. creates `Chunk` objects
4. builds the BM25 index
5. saves the index under `data/processed/index/`

Retrieval and answer-generation commands will be added as the project progresses.

---

# License

This project was created as part of the 42 curriculum.
