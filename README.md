# RAG Against the Machine

This project has been created as part of the **42 curriculum by mchati**.

**RAG Against the Machine** is a Retrieval-Augmented Generation (RAG) system designed to answer questions about a large codebase.

The project uses the **vLLM repository** as its knowledge base. The system will:

1. Load the source code and documentation.
2. Split the files into meaningful chunks.
3. Build a persistent search index.
4. Retrieve the most relevant source chunks for a question.
5. Provide the retrieved context to a local language model.
6. Generate a grounded answer.
7. Measure retrieval quality using Recall@k.

The main goal is to understand and implement a complete RAG pipeline while keeping retrieval measurable and reproducible.

---

# Table of Contents

* [Project Goal](#project-goal)
* [What is RAG?](#what-is-rag)
* [Architecture](#architecture)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Data Flow](#data-flow)
* [Development Progress](#development-progress)
* [Phase 1 — Project Foundation](#phase-1--project-foundation)
* [Phase 2 — Data Models](#phase-2--data-models)
* [Phase 3 — Corpus Indexing](#phase-3--corpus-indexing)
* [Phase 4 — Retrieval](#phase-4--retrieval)
* [Phase 5 — RAG Generation](#phase-5--rag-generation)
* [Phase 6 — CLI & Evaluation](#phase-6--cli--evaluation)
* [Phase 7 — Quality & Performance](#phase-7--quality--performance)
* [Chunking Strategy](#chunking-strategy)
* [Retrieval](#retrieval)
* [Answer Generation](#answer-generation)
* [Data Models](#data-models)
* [Testing Strategy](#testing-strategy)
* [Evaluation](#evaluation)
* [Performance Requirements](#performance-requirements)
* [Design Decisions](#design-decisions)
* [Challenges](#challenges)
* [Git Workflow](#git-workflow)
* [Resources](#resources)
* [AI Usage](#ai-usage)
* [Known Limitations](#known-limitations)
* [Future Improvements](#future-improvements)
* [Final Goal](#final-goal)
* [Author](#author)

---

# Project Goal

Large repositories contain thousands of files and a very large amount of source code.

Finding the correct information manually can be difficult, especially when a question uses different words from the source code.

The goal of this project is to build a system that can:

* Ingest the provided vLLM repository.
* Discover relevant source files.
* Split source files into searchable chunks.
* Preserve the original source locations.
* Build a persistent lexical search index.
* Retrieve the most relevant chunks.
* Return the top-k relevant sources.
* Provide retrieved context to a local language model.
* Generate grounded answers.
* Process individual questions.
* Process complete question datasets.
* Measure retrieval quality using Recall@k.
* Handle invalid and edge-case inputs gracefully.

The project is being implemented incrementally.

---

# What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**.

Instead of asking a language model to answer a question only from its internal knowledge, a RAG system first searches an external knowledge source.

The retrieved information is then given to the language model as context.

The basic idea is:

```text
User Question
      │
      ▼
   Retrieval
      │
      ▼
Relevant Source Chunks
      │
      ▼
   Context
      │
      ▼
Language Model
      │
      ▼
Grounded Answer
```

For this project, the external knowledge source is the **vLLM codebase**.

---

# Architecture

The planned architecture is:

```text
┌──────────────────────────────────────────────┐
│                 vLLM Corpus                  │
│                                              │
│   Python files       Markdown / Text files  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                   Indexer                    │
│                                              │
│   File discovery                             │
│   File type detection                        │
│   Python chunking                            │
│   Markdown/Text chunking                     │
│   Metadata extraction                         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                Search Index                  │
│                                              │
│                 BM25 / TF-IDF               │
└──────────────────────┬───────────────────────┘
                       │
                       │ Question
                       ▼
┌──────────────────────────────────────────────┐
│                  Retriever                   │
│                                              │
│   Query processing                            │
│   Relevance scoring                           │
│   Ranking                                     │
│   Top-k selection                             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             Retrieved Context               │
│                                              │
│   file_path                                   │
│   first_character_index                       │
│   last_character_index                        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              Qwen/Qwen3-0.6B                │
│                                              │
│        Question + Retrieved Context          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
                  Grounded Answer
```

The architecture is implemented incrementally.

The current priority is to build a correct chunking and retrieval foundation before adding generation.

---

# Technology Stack

## Language

* Python 3.10+
* Type hints
* Pydantic

## Package Management

The project uses:

* `uv`
* `pyproject.toml`
* `uv.lock`

The evaluator uses:

```bash
uv sync
```

Therefore, `uv` is the required package manager.

## Retrieval

The project will use a lexical retrieval algorithm:

* BM25
* or TF-IDF

The final choice will be based on retrieval quality and performance.

## Language Model

Required default model:

```text
Qwen/Qwen3-0.6B
```

## CLI

The final CLI will use:

* Python Fire
* tqdm

## Code Quality

The project is intended to use:

* flake8
* mypy
* PEP 257 docstrings
* type hints
* graceful exception handling

---

# Project Structure

The repository currently contains the project foundation and is progressively evolving toward:

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
│   │
│   └── output/
│
├── tests/
│
├── Makefile
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

Generated indexes, large datasets, model weights, and other large generated files should not be committed unnecessarily.

---

# Data Flow

The complete pipeline will follow:

```text
vLLM Repository
      │
      ▼
File Discovery
      │
      ▼
File Type Detection
      │
      ├─────────────────────┐
      ▼                     ▼
Python Chunking      Markdown/Text Chunking
      │                     │
      └──────────┬──────────┘
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
                 │
                 ▼
             Evaluation
```

---

# Development Progress

The project is developed incrementally.

A feature is marked complete only after it has been implemented and tested.

Current development direction:

```text
Project Foundation       ✅
        │
        ▼
Dependencies              ✅
        │
        ▼
Pydantic Models           ✅
        │
        ▼
Corpus Loading            ✅
        │
        ▼
Python Chunking           🚧
        │
        ▼
Markdown/Text Chunking    ⏳
        │
        ▼
Persistent Index          ⏳
        │
        ▼
Retrieval                 ⏳
        │
        ▼
Evaluation                ⏳
        │
        ▼
Qwen Generation           ⏳
        │
        ▼
End-to-End Pipeline       ⏳
```

Legend:

```text
✅ Complete
🚧 In progress
⏳ Not started
```

---

# Phase 1 — Project Foundation

* [x] Create GitHub repository
* [x] Configure Python project
* [x] Configure uv
* [x] Create `.gitignore`
* [x] Add project dependencies
* [x] Create `src/`
* [x] Create Makefile foundation
* [x] Prepare project documentation

---

# Phase 2 — Data Models

* [x] Create `MinimalSource`
* [x] Create `AnsweredQuestion`
* [x] Create `UnansweredQuestion`
* [x] Create `RagDataset`
* [x] Create `MinimalSearchResults`
* [x] Create `MinimalAnswer`
* [x] Create `StudentSearchResults`
* [x] Create `StudentSearchResultsAndAnswer`
* [x] Create `Document`
* [x] Validate dataset structure with Pydantic

---

# Phase 3 — Corpus Indexing

## Corpus

* [x] Load vLLM corpus
* [x] Discover relevant files
* [x] Filter supported files

## Python Chunking

* [x] Parse Python files with `ast`
* [x] Extract top-level functions
* [x] Extract async functions
* [x] Extract class methods
* [x] Calculate character offsets
* [x] Split large functions and methods
* [ ] Decide final class chunking strategy
* [ ] Handle large classes
* [ ] Validate the 2000-character limit for all Python chunks

## Markdown / Text Chunking

* [ ] Implement Markdown chunking
* [ ] Implement text chunking
* [ ] Preserve headings and paragraphs where possible
* [ ] Add character offsets
* [ ] Validate chunk size

## Chunk Model

* [ ] Create final `Chunk` model
* [ ] Store chunk text
* [ ] Store file path
* [ ] Store character offsets
* [ ] Store useful metadata

## Indexing

* [ ] Build persistent index
* [ ] Select BM25 or TF-IDF
* [ ] Store index under `data/processed/`
* [ ] Add tqdm progress bars
* [ ] Measure indexing time
* [ ] Validate indexing under 5 minutes

---

# Phase 4 — Retrieval

* [ ] Select final lexical retrieval algorithm
* [ ] Load persistent index
* [ ] Implement query processing
* [ ] Calculate relevance scores
* [ ] Rank results
* [ ] Implement top-k retrieval
* [ ] Return exact source locations
* [ ] Validate file paths
* [ ] Validate character offsets
* [ ] Test single-question search
* [ ] Test dataset search
* [ ] Measure Recall@1
* [ ] Measure Recall@3
* [ ] Measure Recall@5
* [ ] Measure Recall@10

---

# Phase 5 — RAG Generation

* [ ] Integrate `Qwen/Qwen3-0.6B`
* [ ] Build retrieval context
* [ ] Design generation prompt
* [ ] Generate grounded answers
* [ ] Validate generated output
* [ ] Implement single-question answering
* [ ] Implement dataset answering

---

# Phase 6 — CLI & Evaluation

The final project should expose the main functionality through a Python Fire CLI.

Planned commands:

```bash
uv run python -m src index
```

```bash
uv run python -m src search "How does the scheduler work?" --k 5
```

```bash
uv run python -m src search_dataset \
    --dataset_path <path> \
    --k 5 \
    --save_directory <path>
```

```bash
uv run python -m src answer \
    "How does the scheduler work?" \
    --k 5
```

```bash
uv run python -m src answer_dataset \
    --student_search_results_path <path> \
    --save_directory <path>
```

```bash
uv run python -m src evaluate \
    --student_search_results_path <path> \
    --dataset_path <path>
```

The commands will be implemented progressively.

The final CLI should:

* Accept configurable paths.
* Accept configurable `k`.
* Accept configurable chunk size.
* Handle invalid arguments.
* Handle missing files.
* Handle malformed JSON.
* Handle empty queries.
* Handle `k=0`.
* Avoid unhandled tracebacks for expected user errors.

---

# Chunking Strategy

Python and Markdown/Text require different strategies because they have different structures.

---

## Python Chunking

Python source code contains meaningful structural units.

The chunker should try to preserve:

* Functions
* Async functions
* Classes
* Methods
* Logical code blocks
* Source boundaries

The current implementation uses Python's `ast` module to understand the structure of Python source code.

The current approach is:

```text
Python file
     │
     ├── Function
     │      │
     │      └── Chunk
     │
     ├── Async Function
     │      │
     │      └── Chunk
     │
     └── Class
            │
            ├── Method
            │     └── Chunk
            │
            └── Method
                  └── Chunk
```

Large Python units are split further when they exceed the configured maximum chunk size.

The final class strategy will be chosen after testing retrieval quality.

---

# Markdown / Text Chunking

Markdown and plain-text files do not have Python's AST structure.

The strategy should therefore preserve natural text structure where possible.

Important structures include:

* Headings
* Paragraphs
* Sections
* Related text

A recursive text splitter can be used as part of this strategy.

The implementation must also preserve exact character offsets.

---

# Maximum Chunk Size

The default maximum chunk size is:

```text
2000 characters
```

The chunk size must be configurable.

Planned CLI option:

```bash
--max_chunk_size 2000
```

No retrieved source should exceed the required maximum.

The effect of chunk size on retrieval quality will be measured later.

---

# Character Offsets

Every retrieved source must contain:

```text
file_path
first_character_index
last_character_index
```

Example:

```json
{
    "file_path": "data/raw/vllm-0.10.1/vllm-0.10.1/example.py",
    "first_character_index": 120,
    "last_character_index": 450
}
```

The offsets refer to the original source file.

The source path must remain exactly consistent with the ingested corpus because the evaluator compares source paths.

---

# Retrieval

The retrieval system receives a question and returns the most relevant source chunks.

The retrieval process is:

```text
Question
   │
   ▼
Tokenization / Query Processing
   │
   ▼
Search Index
   │
   ▼
Relevance Scores
   │
   ▼
Ranking
   │
   ▼
Top-k Results
```

Each result must contain the source location:

```text
file_path
first_character_index
last_character_index
```

---

# Retrieval Algorithm

The project must implement at least one lexical retrieval method.

Possible choices:

### BM25

BM25 is designed for ranking documents based on the terms contained in a query.

### TF-IDF

TF-IDF represents the importance of terms in documents and queries.

The final choice will be based on:

* Retrieval quality
* Recall@k
* Query behavior
* Speed
* Implementation complexity

The selected method will be documented after implementation and evaluation.

---

# Answer Generation

After retrieval, the system will provide the retrieved context to:

```text
Qwen/Qwen3-0.6B
```

The model receives:

```text
Question
+
Retrieved Context
```

and produces:

```text
Grounded Answer
```

The generated answer should be:

* Relevant
* Coherent
* Grounded in retrieved sources
* Resistant to hallucination

The project prioritizes retrieval quality and source grounding.

---

# Data Models

Pydantic is used to validate data exchanged between pipeline stages.

## MinimalSource

```python
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
```

## AnsweredQuestion

```python
class AnsweredQuestion(BaseModel):
    question_id: str
    question: str
    answer: str
    sources: list[MinimalSource]
    difficulty: str
    is_valid: bool
```

## UnansweredQuestion

```python
class UnansweredQuestion(BaseModel):
    question_id: str
    question: str
    difficulty: str
    is_valid: bool
```

## RagDataset

```python
class RagDataset(BaseModel):
    rag_questions: list[
        AnsweredQuestion | UnansweredQuestion
    ]
```

## MinimalSearchResults

```python
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]
```

## MinimalAnswer

```python
class MinimalAnswer(MinimalSearchResults):
    answer: str
```

## StudentSearchResults

```python
class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int
```

## StudentSearchResultsAndAnswer

```python
class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int
```

A final `Chunk` model will be added during the chunking stage.

---

# Testing Strategy

Testing is part of development rather than something added only at the end.

## Corpus Tests

* File discovery
* File filtering
* Supported extensions
* Missing corpus
* Empty corpus

## Chunking Tests

* Python function extraction
* Python method extraction
* Class handling
* Async functions
* Character offsets
* Large functions
* Large classes
* Markdown chunking
* Text chunking
* Chunk size limits
* Empty files
* Invalid Python files

## Indexing Tests

* Index creation
* Index persistence
* Index loading
* Metadata persistence
* Large corpus indexing

## Retrieval Tests

* Exact identifiers
* Function names
* Class names
* Natural-language questions
* Empty queries
* Unknown queries
* `k=0`
* Large values of `k`

## Data Validation Tests

* Valid Pydantic models
* Invalid source ranges
* Missing fields
* Invalid JSON
* Invalid datasets

## CLI Tests

* Valid commands
* Missing files
* Invalid arguments
* Empty input
* Malformed datasets
* Invalid `k`

The CLI should handle expected errors without producing an unhandled traceback.

---

# Evaluation

Retrieval quality is measured using **Recall@k**.

A retrieved source is considered correct when:

1. It belongs to the correct file.
2. Its character range overlaps the expected source range.

The exact character span does not have to be identical.

---

# Performance Requirements

The project has the following target requirements:

| Metric                      |  Requirement |
| --------------------------- | -----------: |
| Documentation Recall@5      |        ≥ 80% |
| Code Recall@5               |        ≥ 50% |
| Full indexing time          |  ≤ 5 minutes |
| Retrieval for 200 questions | ≤ 90 seconds |

These measurements will be collected after the complete retrieval pipeline is implemented.

---

# Performance Analysis

Performance results will be recorded during development.

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

These values will be updated after implementation and testing.

---

# Design Decisions

## 1. Different Strategies for Python and Text

Python code has a formal syntax tree.

Markdown and plain text do not.

Therefore:

```text
Python
  → AST-aware chunking

Markdown/Text
  → structure-aware text chunking
```

This should produce more meaningful retrieval units.

---

## 2. Character Offsets

Character offsets are preserved because the evaluator does not only care about the text returned.

It also checks whether the retrieved source corresponds to the correct location in the original file.

Therefore every chunk must maintain:

```text
file_path
start_offset
end_offset
```

---

## 3. Maximum Chunk Size

The default maximum chunk size is:

```text
2000 characters
```

A large source unit must therefore be split into smaller pieces.

The important goal is not simply to split text mechanically, but to preserve useful context whenever possible.

---

## 4. Pydantic

Pydantic is used to define clear data contracts between different stages.

This helps prevent malformed data from silently moving through the pipeline.

---

## 5. Lexical Retrieval

A lexical retrieval algorithm is required by the project.

The final choice between BM25 and TF-IDF will be based on actual experiments rather than assumptions.

---

## 6. Local Language Model

The required model is:

```text
Qwen/Qwen3-0.6B
```

Using a local model keeps the generation stage compatible with the project requirements.

---

# Challenges

Important challenges will be documented during implementation.

Expected challenges include:

* Processing a large repository.
* Designing useful chunks.
* Preserving exact source locations.
* Handling Python syntax correctly.
* Handling large functions and classes.
* Handling Markdown and text structure.
* Keeping chunks below the maximum size.
* Building a persistent index efficiently.
* Improving Recall@5.
* Handling malformed input.
* Meeting indexing performance requirements.
* Meeting retrieval performance requirements.

For important problems, the development process will follow:

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
Measurement
   ↓
Result
```

---

# Git Workflow

The `main` branch should remain stable.

For larger features, feature branches can be used.

Example:

```bash
git checkout -b feature/indexer
```

## Commit Convention

Commits should clearly describe the change.

Examples:

```text
feat: add project foundation
feat: implement python chunking
feat: implement markdown chunking
feat: add persistent bm25 index
feat: add retrieval
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
Lint
  ↓
Review diff
  ↓
Commit
  ↓
Push
```

---

# Problem-Solving Approach

For each important component:

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

The goal is not only to make the system work.

The goal is also to understand:

* Why each component exists.
* How the components communicate.
* How retrieval works.
* How RAG works.
* How retrieval quality is measured.
* Why particular design decisions were made.

---

# Project Principles

## 1. Understand Before Implementing

Every component should be understood before being added to the project.

## 2. Build the Mandatory Part First

Bonus features will only be considered after the mandatory requirements are complete and tested.

## 3. Measure Retrieval Quality

A RAG system should not be judged only by how good the generated answer sounds.

Retrieval quality must be measured using Recall@k.

## 4. Preserve Source Information

Every retrieved result must maintain:

```text
file_path
first_character_index
last_character_index
```

## 5. Keep Components Separate

The following components should have clear responsibilities:

```text
Indexing
Retrieval
Generation
Evaluation
CLI
```

## 6. Handle Errors Gracefully

The system should handle expected problems such as:

* Empty queries
* Invalid JSON
* Missing files
* Invalid arguments
* Invalid datasets

without crashing unexpectedly.

## 7. Understand AI-Generated Code

AI tools may be used as learning and productivity assistants.

However, generated code must be:

* Reviewed
* Tested
* Understood
* Modified when necessary

---

# Development Roadmap

## Phase 1 — Project Foundation

* [x] Create GitHub repository
* [x] Configure Python project
* [x] Configure uv
* [x] Create `.gitignore`
* [x] Configure dependencies
* [x] Create `src/`
* [x] Create project Makefile

## Phase 2 — Data Models

* [x] Create Pydantic models
* [x] Load and validate dataset structure

## Phase 3 — Corpus Indexing

### Corpus

* [x] Load vLLM corpus
* [x] Discover relevant files

### Python Chunking

* [x] AST parsing
* [x] Function extraction
* [x] Method extraction
* [x] Character offsets
* [x] Split large functions/methods
* [x] Decide class strategy
* [x] Handle large classes

### Markdown/Text

* [x] Markdown chunking
* [x] Text chunking
* [x] Character offsets
* [x] Chunk size validation

### Chunk Model

* [x] Final `Chunk` model
* [x] Chunk metadata
* [x] Chunk validation

### Indexing

* [ ] Build BM25/TF-IDF index
* [ ] Persist index
* [ ] Store index under `data/processed/`
* [ ] Add tqdm
* [ ] Measure indexing time

## Phase 4 — Retrieval

* [ ] Load persistent index
* [ ] Query processing
* [ ] Relevance scoring
* [ ] Ranking
* [ ] Top-k
* [ ] Source locations
* [ ] Single-question retrieval
* [ ] Dataset retrieval
* [ ] Recall@k

## Phase 5 — RAG Generation

* [ ] Integrate Qwen/Qwen3-0.6B
* [ ] Build context
* [ ] Prompt design
* [ ] Generate grounded answers
* [ ] Single-question answering
* [ ] Dataset answering

## Phase 6 — CLI & Evaluation

* [ ] `index`
* [ ] `search`
* [ ] `search_dataset`
* [ ] `answer`
* [ ] `answer_dataset`
* [ ] `evaluate`
* [ ] Error handling

## Phase 7 — Quality & Performance

* [ ] flake8
* [ ] mypy
* [ ] Unit tests
* [ ] Integration tests
* [ ] Indexing optimization
* [ ] Retrieval optimization
* [ ] Measure throughput
* [ ] Improve Recall@5
* [ ] Analyze chunk-size impact

## Phase 8 — Documentation

* [ ] Final architecture documentation
* [ ] Final chunking documentation
* [ ] Retrieval documentation
* [ ] Design decisions
* [ ] Challenges
* [ ] Performance results
* [ ] Usage examples
* [ ] AI usage
* [ ] Known limitations

---

# Bonus Features

Bonus work will only begin after the mandatory project is complete.

Possible improvements include:

## Semantic Embeddings

Add a lightweight CPU-based semantic search index.

## Hybrid Retrieval

Combine lexical retrieval with semantic retrieval.

## Incremental Indexing

Only re-index files that have changed.

## Caching

Cache:

* Search results
* Repeated queries
* Index-related operations

## Local HTTP API

Expose the search and generation pipeline through a local HTTP API.

## Retrieval Re-ranking

Add a second ranking stage to improve retrieval quality.

---

# Resources

The main resources used during development include:

* Python documentation
* Python `ast` documentation
* Pydantic documentation
* BM25 / TF-IDF references
* Text chunking documentation
* Qwen model documentation
* RAG and information retrieval resources
* The official project subject

The project subject remains the primary source for mandatory requirements.

---

# AI Usage

AI tools are used as development and learning assistants.

AI may be used for:

* Understanding technical concepts.
* Explaining project requirements.
* Discussing architecture.
* Debugging errors.
* Reviewing implementation ideas.
* Exploring possible solutions.
* Improving documentation.
* Learning Python libraries.
* Comparing implementation approaches.

AI-generated suggestions are not treated as automatically correct.

They are reviewed, tested, and understood before being used in the project.

The objective is to use AI as a learning tool while maintaining understanding of the complete implementation.

---

# Known Limitations

The following limitations will be documented as development continues:

* Retrieval quality is not yet measured.
* The final lexical retrieval algorithm has not yet been selected.
* Markdown/Text chunking is still under development.
* Python class chunking strategy is still under development.
* The persistent search index is not yet implemented.
* Qwen generation is not yet integrated.
* Final Recall@k results are not yet available.
* Performance measurements are not yet available.

---

# Future Improvements

After the mandatory implementation is complete, possible improvements include:

* Semantic embeddings
* Hybrid retrieval
* Incremental indexing
* Query caching
* Retrieval re-ranking
* Better chunking strategies
* Additional evaluation metrics
* Local HTTP API
* Improved generation prompts
* Better error reporting

---

# Final Goal

The final system should allow a developer to:

1. Clone the repository.
2. Run `uv sync`.
3. Prepare the vLLM corpus.
4. Build the search index.
5. Search individual questions.
6. Retrieve relevant source locations.
7. Generate grounded answers.
8. Search complete datasets.
9. Evaluate Recall@k.
10. Analyze retrieval performance.
11. Improve the system based on measured results.

The complete pipeline should be reproducible and accessible through the project CLI.

---

# Author

**mchati**

42 / 1337

GitHub:

`https://github.com/mchati42/RAG_42`

---

# Current Status

🚧 **In Development**

Current focus:

```text
Corpus Loading
      ↓
Python Chunking        ✅
      ↓
Class Strategy         🚧
      ↓
Markdown/Text          ⏳
      ↓
Final Chunk Model      ⏳
      ↓
Chunking Tests         ⏳
      ↓
Persistent Index       ⏳
      ↓
Retrieval              ⏳
      ↓
Evaluation             ⏳
      ↓
Qwen Generation        ⏳
      ↓
Complete RAG Pipeline  ⏳
```

The next implementation milestone is to finish the **chunking stage** before starting the persistent search index.
