*This project has been created as part of the 42 curriculum by mchati.*

# RAG Against the Machine

A Retrieval-Augmented Generation system that indexes a large codebase, retrieves relevant source code and documentation, and uses a small local language model to generate grounded answers.

The project focuses on understanding and implementing the complete RAG pipeline:

**Indexing → Retrieval → Augmentation → Generation → Evaluation**

The system is designed to answer questions about the provided **vLLM** codebase while measuring retrieval quality using **Recall@k**.

---

# Project Goal

Large codebases contain thousands of files and hundreds of thousands of lines of code.

Finding the correct information manually can be difficult, especially when a question does not use the exact same words as the source code.

The goal of this project is to build a system that can:

* Ingest the provided vLLM repository
* Split source files into searchable chunks
* Build a persistent search index
* Retrieve the most relevant source locations
* Return the top-k relevant snippets
* Provide retrieved context to a language model
* Generate grounded answers
* Process individual questions
* Process complete question datasets
* Measure retrieval quality using Recall@k
* Handle invalid and edge-case inputs gracefully

The final system will combine **information retrieval** with **local language model generation**.

---

# What is RAG?

Retrieval-Augmented Generation, or RAG, is a technique that combines information retrieval with language generation.

Instead of asking a language model to answer only from its internal knowledge, the system first searches an external knowledge source and gives the relevant information to the model.

The pipeline used in this project is:

```text
Question
   │
   ▼
Retrieval
   │
   ▼
Relevant Sources
   │
   ▼
Context
   │
   ▼
Qwen/Qwen3-0.6B
   │
   ▼
Grounded Answer
```

The project subject describes four main stages:

```text
Indexing
   ↓
Retrieving
   ↓
Augmenting
   ↓
Generating
```

---

# Core Architecture

The project will use a modular architecture:

```text
┌─────────────────────────────────────────────┐
│                 vLLM Corpus                 │
│                                             │
│  Python files        Markdown / Text files │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                  Indexer                    │
│                                             │
│  File discovery                             │
│  Python chunking                            │
│  Markdown/Text chunking                     │
│  Metadata extraction                        │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│               Search Index                  │
│                                             │
│              TF-IDF / BM25                 │
└──────────────────────┬──────────────────────┘
                       │
                       │ Question
                       ▼
┌─────────────────────────────────────────────┐
│                 Retriever                   │
│                                             │
│  Query processing                           │
│  Relevance scoring                          │
│  Ranking                                    │
│  Top-k selection                            │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│               Retrieved Context             │
│                                             │
│  file_path                                  │
│  first_character_index                      │
│  last_character_index                      │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              Qwen/Qwen3-0.6B               │
│                                             │
│       Question + Retrieved Context          │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                    Answer                   │
└─────────────────────────────────────────────┘
```

The architecture will be implemented incrementally.

The goal is to first build a correct and measurable retrieval system before adding improvements.

---

# Technology Stack

## Language

* Python 3.10+
* Type hints
* Pydantic

## Package Management

* uv
* `pyproject.toml`
* `uv.lock`

The project must use **uv**, because the evaluator runs `uv sync`.

## Retrieval

The project will implement at least one classic lexical retrieval method:

* TF-IDF
* BM25

The final choice will be documented after implementation and testing.

## Language Model

Required default model:

```text
Qwen/Qwen3-0.6B
```

The system must remain compatible with this model.

## CLI

The command-line interface will use:

* Python Fire
* tqdm

Long-running operations will display progress information.

## Code Quality

* flake8
* mypy
* PEP 257 docstrings
* Type hints
* Exception handling

---

# Project Structure

The repository will progressively evolve toward the following structure:

```text
RAG_42/
│
├── src/
│   ├── __main__.py
│   ├── cli.py
│   ├── indexer.py
│   ├── chunker.py
│   ├── retriever.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── models.py
│   └── ...
│
├── data/
│   ├── raw/
│   │   └── vllm-0.10.1/
│   │
│   ├── processed/
│   │
│   ├── datasets/
│   │   ├── AnsweredQuestions/
│   │   └── UnansweredQuestions/
│   │
│   └── output/
│       ├── search_results/
│       │   ├── AnsweredQuestions/
│       │   └── UnansweredQuestions/
│       │
│       └── search_results_and_answer/
│           ├── AnsweredQuestions/
│           └── UnansweredQuestions/
│
├── tests/
│
├── Makefile
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

The evaluation expects `src/`, `pyproject.toml`, `uv.lock`, `README.md`, the raw corpus, processed index, datasets, and output directories to follow the required layout.

Large raw datasets, generated outputs, and model weights should not be committed to the repository.

---

# Data Flow

The complete system follows this flow:

```text
vLLM Repository
       │
       ▼
File Discovery
       │
       ▼
File Type Detection
       │
       ├───────────────┐
       ▼               ▼
Python Chunking   Markdown/Text Chunking
       │               │
       └───────┬───────┘
               ▼
          Chunk Metadata
               │
               ▼
          Search Index
               │
               ▼
            Question
               │
               ▼
          Query Search
               │
               ▼
          Ranking
               │
               ▼
             Top-k
               │
               ▼
       Retrieved Context
               │
               ▼
        Qwen/Qwen3-0.6B
               │
               ▼
             Answer
```

---

# Indexing

Indexing is the first major stage of the project.

The indexer will:

1. Read files from `data/raw/`
2. Identify useful files
3. Detect the file type
4. Apply the correct chunking strategy
5. Create metadata for every chunk
6. Build the search index
7. Persist the index under `data/processed/`

The complete corpus must be indexed in **at most 5 minutes**.

---

# Chunking Strategy

Python code and Markdown/Text documents must use different chunking strategies.

## Python Chunking

Python source code should be divided while trying to preserve meaningful code units.

The strategy will consider elements such as:

* Functions
* Classes
* Methods
* Logical code blocks
* Source boundaries

The implementation will be evaluated based on how well the resulting chunks support retrieval.

## Markdown / Text Chunking

Markdown and text files will use a text-oriented chunking strategy.

The strategy will try to preserve:

* Headings
* Paragraphs
* Sections
* Related text

## Maximum Chunk Size

The chunk size is configurable:

```bash
--max_chunk_size <int>
```

Default:

```text
2000 characters
```

A retrieved source must never exceed 2000 characters.

The effect of different chunk sizes on retrieval quality will be measured and documented.

---

# Retrieval

The retrieval system receives a question and returns the most relevant source locations.

Each result contains:

```text
file_path
first_character_index
last_character_index
```

The system must return the **top-k** relevant sources.

The source path must match the original corpus path exactly because the evaluator compares paths verbatim.

---

# Retrieval Algorithm

The project must implement at least one lexical retrieval method.

Possible approaches:

```text
Option 1:
TF-IDF

Option 2:
BM25
```

The selected algorithm will:

1. Process the user query
2. Compare it with indexed chunks
3. Calculate relevance scores
4. Rank the chunks
5. Return the top-k results

The final implementation will document why the chosen method was selected.

---

# Answer Generation

After retrieval, the system passes the relevant context to:

```text
Qwen/Qwen3-0.6B
```

The model receives:

```text
Question
+
Retrieved Context
```

and generates a natural-language answer.

The generated answer should be:

* Coherent
* Relevant
* Grounded in retrieved sources
* Free from major hallucinations

The project prioritizes retrieval quality and grounding over perfect language generation because the required model has limited reasoning capabilities.

---

# Data Models

The project will use **Pydantic** for validation of data exchanged between pipeline stages.

## MinimalSource

```python
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
```

## UnansweredQuestion

```python
class UnansweredQuestion(BaseModel):
    question_id: str
    question: str
```

## AnsweredQuestion

```python
class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
```

## RagDataset

```python
class RagDataset(BaseModel):
    rag_questions: List[
        AnsweredQuestion | UnansweredQuestion
    ]
```

## MinimalSearchResults

```python
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]
```

## MinimalAnswer

```python
class MinimalAnswer(MinimalSearchResults):
    answer: str
```

## StudentSearchResults

```python
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int
```

## StudentSearchResultsAndAnswer

```python
class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
```

These models define the main data contracts between the different pipeline stages.

---

# CLI

The entire project will be accessible through a Python Fire CLI.

Commands will follow:

```bash
uv run python -m src <command>
```

## Index

Build the searchable index:

```bash
uv run python -m src index --max_chunk_size 2000
```

## Search

Search a single question:

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

## Search Dataset

Search an entire dataset:

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

## Answer

Answer a single question:

```bash
uv run python -m src answer \
    "How to configure OpenAI server?" \
    --k 10
```

## Answer Dataset

Generate answers for a dataset:

```bash
uv run python -m src answer_dataset \
    --student_search_results_path <path> \
    --save_directory <directory>
```

## Evaluate

Evaluate retrieval quality:

```bash
uv run python -m src evaluate \
    --student_search_results_path <path> \
    --dataset_path <path>
```

These commands are required by the project subject. Input and output paths must remain configurable and must not be hard-coded.

---

# Output Format

Search operations produce a JSON file following:

```text
StudentSearchResults
```

Answer generation produces:

```text
StudentSearchResultsAndAnswer
```

Each source contains:

```text
file_path
first_character_index
last_character_index
```

The file path is relative to the project root and must match the ingested corpus exactly.

---

# Makefile

The project will provide a Makefile to automate common development tasks.

Required commands:

```bash
make install
make run
make debug
make clean
make lint
```

Optional:

```bash
make lint-strict
```

The required lint command uses:

```text
flake8 .
```

and:

```text
mypy . --warn-return-any
      --warn-unused-ignores
      --ignore-missing-imports
      --disallow-untyped-defs
      --check-untyped-defs
```

The project subject requires these Makefile rules.

---

# Testing Strategy

Testing will be part of development instead of being added only at the end.

Tests will cover:

## Indexing

* File discovery
* File filtering
* Python chunking
* Markdown/Text chunking
* Chunk size limits
* Index persistence

## Retrieval

* Exact identifiers
* Natural-language questions
* Empty queries
* Unknown queries
* `k=0`
* Large values of `k`

## Data Validation

* Valid Pydantic models
* Invalid source ranges
* Invalid JSON
* Missing fields

## CLI

* Valid commands
* Missing files
* Invalid arguments
* Empty input
* Malformed datasets

The CLI must handle degenerate inputs without producing an unhandled traceback.

---

# Evaluation

Retrieval quality is measured using **Recall@k**.

A retrieved source is considered correct when:

* It belongs to the correct file
* Its character range overlaps the expected source range

The exact span does not need to be identical because the evaluation allows a small overlap.

## Target Performance

The project must reach:

| Metric                      |  Requirement |
| --------------------------- | -----------: |
| Documentation Recall@5      |        ≥ 80% |
| Code Recall@5               |        ≥ 50% |
| Full indexing time          |  ≤ 5 minutes |
| Retrieval for 200 questions | ≤ 90 seconds |

---

# Performance Analysis

Performance will be measured during development.

## Indexing

| Metric         | Result |
| -------------- | -----: |
| Files indexed  |    TBD |
| Chunks created |    TBD |
| Index size     |    TBD |
| Indexing time  |    TBD |

## Retrieval

| Dataset       | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| ------------- | -------: | -------: | -------: | --------: |
| Documentation |      TBD |      TBD |      TBD |       TBD |
| Code          |      TBD |      TBD |      TBD |       TBD |

The results will be updated as the implementation improves.

---

# Design Decisions

Important technical decisions will be documented throughout development.

## Chunking

The project uses different strategies for Python and Markdown/Text because source code and natural-language documents have different structures.

## Retrieval

A lexical retrieval method is required by the subject.

The final choice between TF-IDF and BM25 will be based on:

* Retrieval quality
* Query behavior
* Performance
* Implementation complexity

## Local Model

The required model is:

```text
Qwen/Qwen3-0.6B
```

Using a local model keeps the answer-generation stage compatible with the project requirements.

## Pydantic

Pydantic is used to create clear and validated data contracts between pipeline stages.

## CLI

Python Fire provides a simple interface for running each pipeline stage independently and makes the complete system easy to test.

---

# Challenges Faced

This section will document the real problems encountered during development.

Expected areas include:

* Processing a large codebase
* Designing useful chunks
* Preserving exact source locations
* Improving lexical retrieval
* Handling different file types
* Keeping chunks below the 2000-character limit
* Improving Recall@5
* Running the model efficiently
* Handling malformed input
* Keeping the CLI stable
* Meeting indexing and retrieval performance requirements

Each challenge will be documented with:

```text
Problem
   ↓
Investigation
   ↓
Possible Solutions
   ↓
Technical Decision
   ↓
Implementation
   ↓
Testing
   ↓
Result
```

---

# Development Roadmap

The project will be developed incrementally.

## Phase 1 — Project Foundation

* [x] Create GitHub repository
* [x] Configure Python project
* [x] Configure uv
* [x] Create `.gitignore`
* [x] Add question datasets
* [x] Create `src/`
* [x] Configure project dependencies
* [x] Create Makefile
* [x] Configure flake8
* [x] Configure mypy

## Phase 2 — Data Models

* [x] Create `MinimalSource`
* [x] Create `UnansweredQuestion`
* [x] Create `AnsweredQuestion`
* [x] Create `RagDataset`
* [x] Create `MinimalSearchResults`
* [x] Create `MinimalAnswer`
* [x] Create `StudentSearchResults`
* [x] Create `StudentSearchResultsAndAnswer`
* [x] Validate JSON datasets

## Phase 3 — Corpus Indexing

* [x] Load vLLM corpus
* [x] Discover relevant files
* [ ] Implement Python chunking
* [ ] Implement Markdown/Text chunking
* [ ] Add character offsets
* [ ] Add configurable chunk size
* [ ] Build persistent index
* [ ] Store index under `data/processed/`
* [ ] Add tqdm progress bars
* [ ] Measure indexing time

## Phase 4 — Retrieval

* [ ] Select TF-IDF or BM25
* [ ] Build lexical index
* [ ] Implement query processing
* [ ] Implement relevance scoring
* [ ] Implement ranking
* [ ] Implement top-k retrieval
* [ ] Implement exact source locations
* [ ] Test single-query search
* [ ] Test dataset search
* [ ] Measure Recall@k

## Phase 5 — RAG Generation

* [ ] Integrate Qwen/Qwen3-0.6B
* [ ] Build retrieval context
* [ ] Design generation prompt
* [ ] Generate grounded answers
* [ ] Validate generated output
* [ ] Implement single-question answering
* [ ] Implement dataset answering

## Phase 6 — CLI & Evaluation

* [ ] Implement `index`
* [ ] Implement `search`
* [ ] Implement `search_dataset`
* [ ] Implement `answer`
* [ ] Implement `answer_dataset`
* [ ] Implement `evaluate`
* [ ] Handle invalid inputs
* [ ] Handle missing files
* [ ] Handle malformed JSON
* [ ] Handle empty queries
* [ ] Handle `k=0`

## Phase 7 — Quality & Performance

* [ ] Run flake8
* [ ] Run mypy
* [ ] Add docstrings
* [ ] Add unit tests
* [ ] Optimize indexing
* [ ] Optimize retrieval
* [ ] Measure retrieval throughput
* [ ] Improve documentation Recall@5
* [ ] Improve code Recall@5
* [ ] Analyze chunk size impact

## Phase 8 — Documentation

* [ ] Complete README
* [ ] Document architecture
* [ ] Document chunking strategy
* [ ] Document retrieval algorithm
* [ ] Document design decisions
* [ ] Document challenges
* [ ] Document performance results
* [ ] Add usage examples
* [ ] Document AI usage
* [ ] Document known limitations

---

# Bonus Features

The mandatory part will be completed before working on bonus features.

Possible bonus improvements include:

### Semantic Embeddings

Add a lightweight CPU-based semantic embedding index.

### Hybrid Retrieval

Combine lexical and semantic retrieval.

### Incremental Indexing

Only re-index files that have changed.

### Caching

Cache the index and repeated query results.

### Local HTTP API

Expose search and answer generation through a local HTTP API.

These are the five bonus directions defined by the project subject.

---

# Git Workflow

The main branch should remain stable.

Feature branches can be used for important changes.

Example:

```bash
git checkout -b feature/indexer
```

## Commit Convention

Commits should clearly describe the change:

```text
feat: add project foundation
feat: implement python chunking
feat: add bm25 retrieval
feat: add qwen answer generation

test: add chunking tests
test: add retrieval tests

fix: handle empty search query

refactor: separate indexing and retrieval

docs: update architecture documentation
```

Before committing:

```text
Code
  ↓
Test
  ↓
Run lint
  ↓
Review diff
  ↓
Commit
  ↓
Push
```

---

# Problem-Solving Approach

For each important part of the project:

```text
Understand the requirement
        ↓
Understand the problem
        ↓
Design possible solutions
        ↓
Choose an approach
        ↓
Implement
        ↓
Test
        ↓
Measure
        ↓
Improve
        ↓
Document
```

The goal is not only to make the system work, but to understand why each component exists and how the complete RAG pipeline works.

---

# Project Principles

### 1. Understand Before Implementing

Every component should be understood before being added to the project.

### 2. Build the Mandatory Part First

Bonus features will only be considered after the mandatory requirements are working correctly.

### 3. Measure Retrieval Quality

A RAG system should not be judged only by how good an answer sounds.

Retrieval quality must be measured using Recall@k.

### 4. Preserve Source Information

Every retrieved result must maintain the correct file path and character range.

### 5. Keep Components Separate

Indexing, retrieval, generation, evaluation, and CLI logic should have clear responsibilities.

### 6. Handle Errors Gracefully

The system should not crash because of an empty query, invalid JSON, missing files, or invalid CLI arguments.

### 7. Understand AI-Generated Code

AI can be used as a learning and productivity tool, but generated code must be reviewed, tested, and understood.

---

# Skills Demonstrated

| Skill                       | Project Evidence                     |
| --------------------------- | ------------------------------------ |
| Python                      | Complete RAG pipeline                |
| Python typing               | Typed functions and data structures  |
| Pydantic                    | Data validation                      |
| Information Retrieval       | TF-IDF / BM25                        |
| Natural Language Processing | Query and document processing        |
| RAG                         | Retrieval + generation pipeline      |
| LLMs                        | Qwen/Qwen3-0.6B                      |
| CLI Development             | Python Fire                          |
| Data Processing             | Codebase ingestion and chunking      |
| Performance Optimization    | Indexing and retrieval optimization  |
| Testing                     | Unit and integration tests           |
| Git                         | Version control                      |
| Linux / Bash                | Development environment              |
| Problem Solving             | Retrieval and architecture decisions |

---

# Project Status

**Status: 🚧 In Development**

The project is currently in the foundation stage.

Current repository state:

```text
Project Repository
       ↓
uv configuration
       ↓
Dataset preparation
       ↓
Project foundation
       ↓
Indexing
       ↓
Retrieval
       ↓
RAG Generation
       ↓
Evaluation
       ↓
Optimization
       ↓
Final Documentation
```

Features will only be marked as complete after they have been implemented and tested.

---

# Current Development Direction

The current priority is to build the project from the bottom up:

```text
Project Foundation
        ↓
Dependencies
        ↓
Pydantic Models
        ↓
File Loading
        ↓
Chunking
        ↓
Index
        ↓
Retrieval
        ↓
CLI
        ↓
Dataset Search
        ↓
Recall@k
        ↓
Qwen Generation
        ↓
End-to-End Pipeline
        ↓
Performance Optimization
```

---

# Resources

## Project Subject

The official project subject is the primary reference for the implementation and evaluation requirements.

## Retrieval-Augmented Generation

Resources about:

* Retrieval-Augmented Generation
* Information Retrieval
* TF-IDF
* BM25
* Text chunking
* Language models

will be documented here during development.

## Python

Relevant Python documentation and references will be added here.

## Pydantic

Documentation related to data validation and typed models will be added here.

## Qwen

Documentation and references related to:

```text
Qwen/Qwen3-0.6B
```

will be added here.

## AI Usage

AI tools are used as development and learning assistants.

AI may be used for:

* Understanding technical concepts
* Explaining project requirements
* Discussing architecture
* Debugging errors
* Reviewing implementation ideas
* Improving documentation
* Exploring possible solutions

AI-generated suggestions are reviewed, tested, and understood before being used in the project.

---

# Known Limitations

This section will contain limitations discovered during development.

Current limitations will be documented as the project evolves.

Examples may include:

* Retrieval limitations
* Model reasoning limitations
* Performance limitations
* Chunking limitations
* Dataset limitations

---

# Future Improvements

After the mandatory implementation is complete, possible improvements include:

* Semantic embeddings
* Hybrid retrieval
* Incremental indexing
* Query caching
* Local HTTP API
* Better chunking strategies
* Retrieval re-ranking
* Additional evaluation metrics

---

# Final Goal

The final system should allow a developer to:

1. Clone the repository
2. Run `uv sync`
3. Prepare the corpus
4. Build the index
5. Search questions
6. Retrieve relevant source locations
7. Generate grounded answers
8. Search complete datasets
9. Evaluate Recall@k
10. Analyze and improve retrieval performance

The complete pipeline should work reproducibly through the required CLI commands.

---

# Author

**mchati**

42 / 1337
