# RAG - Retrieval-Augmented Generation

*This project has been created as part of the 42 curriculum by [your_login].*

---

## Description

This project builds a **RAG system** that answers questions about computer code.

### What Does It Do?

The system works like a smart library assistant:
1. You give it many code files
2. It organizes them so it can search them quickly
3. You ask a question about the code
4. It finds the right parts of the code
5. It creates an answer using those parts

### The Problem It Solves

AI models only know things from when they were trained. They don't know about new code or changes. Instead of retraining the model (which is slow and expensive), we give it access to a library of information. It searches the library when it needs to answer questions.

### What You Learn

- How to organize and search through large amounts of text
- How AI systems find and use information
- How to measure if a system works well
- How to write professional Python code
- How to build a complete system from start to finish

---

## System Architecture

### How It Works (4 Steps)

```
Raw Code Files → Index (organize) → Search (find) → Generate Answer → Output
```

#### Step 1: Indexing
- Read Python files and Markdown files
- Split them into small pieces (chunks)
- Create a searchable index
- Save the index for fast searching

#### Step 2: Retrieval
- Take a user's question
- Search the index
- Return the top results that match the question
- Each result shows which file and which lines

#### Step 3: Augmenting
- Take the search results
- Prepare them for the AI model to read
- Make sure the information fits in the AI's memory

#### Step 4: Generating
- Pass the information to the Qwen AI model
- The model creates an answer
- Output the answer and show where it came from

### System Diagram

```
┌─────────────────┐
│  Code Files     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Indexer       │ → Creates searchable index
└────────┬────────┘
         │
    ┌────▼─────────────────┐
    │  Question comes in   │
    └────┬──────────────────┘
         │
         ▼
┌─────────────────┐
│   Retriever     │ → Finds relevant code pieces
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Augmenter      │ → Prepares information
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generator     │ → Creates answer using Qwen model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Answer        │ → Output with sources
└─────────────────┘
```

---

## Chunking Strategy

### What is a Chunk?

A chunk is a small piece of text. Instead of searching through entire files, we break them into smaller pieces.

### Why Do We Need Chunking?

- Faster searching (search small pieces instead of huge files)
- Better accuracy (focus on relevant sections)
- Fits in AI model's memory (can't process everything at once)

### How We Chunk Different File Types

#### Python Files
For code, we split at:
- Function boundaries (between functions)
- Class boundaries (between classes)
- Comments and blocks of logic
- Maximum size: 2000 characters

**Example:**
```python
# This is function 1 - becomes one chunk
def hello():
    print("Hello")

# This is function 2 - becomes another chunk
def goodbye():
    print("Goodbye")
```

#### Markdown/Text Files
For documentation, we split at:
- Headings (# ## ###)
- Paragraphs
- Code blocks
- Maximum size: 2000 characters

**Example:**
```
# Section 1 - chunk 1
This is text.

# Section 2 - chunk 2
More text.
```

### Chunk Size Configuration

You can change chunk size with the command:
```bash
uv run python -m src index --max_chunk_size 2000
```

Default is 2000 characters. Smaller chunks = more searching, bigger chunks = less accurate.

---

## Retrieval Method

### What Method Do We Use?

We use **TF-IDF** (Term Frequency-Inverse Document Frequency).

### How Does TF-IDF Work?

**TF-IDF** measures: "How important is this word?"

**Example:**
- The word "the" appears everywhere → not important
- The word "authentication" appears rarely → very important

When you ask a question:
1. Count how often each word appears in chunks
2. Give higher scores to rare important words
3. Rank chunks by score
4. Return the top results

### Alternative Method: BM25

BM25 is an improved version of TF-IDF:
- Better at handling word frequency
- Considers chunk length
- Generally gives better results

### How Results Are Ranked

```
Your Question: "How do I authenticate users?"

Search finds:
- Chunk A: Score 0.95 (talks about authentication) → Show this
- Chunk B: Score 0.87 (mentions security) → Show this too
- Chunk C: Score 0.45 (about user interface) → Not very relevant
- Chunk D: Score 0.23 (random topic) → Don't show
```

Each result includes:
- File path (which file it came from)
- Start position (which character it starts at)
- End position (which character it ends at)

---

## Instructions

### Installation

#### 1. Requirements

You need:
- Python 3.10 or newer
- pip or uv (package manager)
- Git

Check Python version:
```bash
python3 --version
```

#### 2. Clone the Project

```bash
git clone [your-repo-url]
cd rag-project
```

#### 3. Install Dependencies

```bash
make install
```

Or manually:
```bash
uv sync
```

### Running the System

#### 1. Build the Index (Do This First)

```bash
uv run python -m src index --max_chunk_size 2000
```

What it does:
- Reads all files from `data/raw/`
- Splits them into chunks
- Creates an index
- Saves to `data/processed/`
- Takes about 5 minutes

Output: `Ingestion complete! Indices saved under data/processed/`

#### 2. Search for Information

```bash
uv run python -m src search "How do I start the server?" --k 5
```

What it does:
- Takes your question
- Searches the index
- Returns top 5 results

Output:
```
Found: data/raw/vllm-0.10.1/docs/server.md [lines 100-150]
Found: data/raw/vllm-0.10.1/examples/start.py [lines 50-80]
...
```

#### 3. Get an Answer

```bash
uv run python -m src answer "How do I start the server?" --k 5
```

What it does:
- Searches for information
- Sends it to the Qwen AI model
- AI creates an answer
- Shows you the answer and sources

Output:
```json
{
  "question": "How do I start the server?",
  "answer": "To start the server, you need to...",
  "sources": [
    {"file_path": "...", "start": 100, "end": 150},
    ...
  ]
}
```

#### 4. Search Many Questions at Once

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/questions.json \
  --k 10 \
  --save_directory data/output/results/
```

#### 5. Generate Answers for Many Questions

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/results/results.json \
  --save_directory data/output/answers/
```

#### 6. Test Your System

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/results/results.json \
  --dataset_path data/datasets/ground_truth.json
```

Shows: How many answers were correct? (recall@k scores)

### Code Quality

Check if your code is clean:

```bash
make lint
```

This runs:
- `flake8` - Checks code style
- `mypy` - Checks data types

Run all tests:
```bash
make run
```

Clean up temporary files:
```bash
make clean
```

---

## Performance Analysis

### What We Measure

**Recall@k**: "Did we find the right information in the top k results?"

### Target Performance

For this project, you must achieve:

- **Recall@5 on documents**: At least 80% correct
- **Recall@5 on code**: At least 50% correct

### How It's Calculated

```
Question: "How do I use authentication?"
Real answer: In file A, lines 100-150

My results:
1. File A, lines 95-155 ✓ CORRECT (overlaps with real answer)
2. File B, lines 50-100 ✗ WRONG (different file)
3. File C, lines 200-250 ✗ WRONG (different file)
4. File A, lines 150-200 ✓ CORRECT (overlaps with real answer)
5. File D, lines 1-50 ✗ WRONG (different file)

Recall@5 = 2 correct out of (depends on how many total answers)
```

### Performance Requirements

- **Indexing time**: Must finish in less than 5 minutes
- **Search speed**: Must answer 200 questions in less than 90 seconds
- **Memory**: Must use reasonable amount of disk space
- **Reliability**: System must not crash on bad inputs

---

## Design Decisions

### Why TF-IDF?

We chose TF-IDF because:
- ✅ It's simple and fast
- ✅ Works well for code and text
- ✅ Easy to understand and debug
- ❌ Doesn't understand meaning (only word frequency)

### Chunk Size: 2000 Characters

We chose 2000 characters because:
- ✅ Fits in AI model's memory budget
- ✅ Large enough to keep context
- ✅ Small enough to be accurate
- ❌ Too small = lose important information

### Why Split Python and Markdown Differently

Python code and Markdown text have different structure:
- Python has functions, classes, imports
- Markdown has headings, paragraphs, code blocks
- Different chunking = better results

### Using Pydantic for Data Models

We use Pydantic (data validation library) because:
- ✅ Ensures data is correct format
- ✅ Catches errors early
- ✅ Makes code more reliable
- ✅ Easy to convert to JSON

---

## Challenges Faced

### Challenge 1: Finding the Right Chunks

**Problem**: Important information was spread across multiple chunks

**Solution**: 
- Increased chunk overlap (chunks share some content)
- Improved chunking strategy for functions
- Better handling of comments and docstrings

### Challenge 2: Different Question Wording

**Problem**: Questions use different words than the code

**Example**: Question says "start the server" but code says "initialize_server()"

**Solution**: 
- Add synonyms to search
- Use TF-IDF to find similar words
- Could use AI embeddings (bonus feature)

### Challenge 3: Speed vs Accuracy

**Problem**: Bigger chunks are more accurate but slower to search

**Solution**: 
- Tested different chunk sizes
- Found that 2000 characters is the sweet spot
- Optimized search algorithm

### Challenge 4: Handling Edge Cases

**Problem**: System crashed on empty questions, special characters, etc.

**Solution**:
- Added input validation
- Better error handling
- Graceful failures (return empty results instead of crashing)

---

## Example Usage

### Example 1: Simple Search

```bash
$ uv run python -m src search "What is authentication?" --k 3

Found 3 results:

1. File: data/raw/vllm-0.10.1/docs/security.md
   Lines: 150-250
   
2. File: data/raw/vllm-0.10.1/src/auth.py
   Lines: 1-50
   
3. File: data/raw/vllm-0.10.1/examples/auth_example.py
   Lines: 45-95
```

### Example 2: Get an Answer

```bash
$ uv run python -m src answer "How do I use authentication?" --k 5

Answer: "To use authentication in vLLM, you need to import the auth module, 
create an authentication handler, and pass it to the server configuration. 
The example at line 150 in docs/security.md shows a complete working example."

Sources:
- docs/security.md [150-250]
- src/auth.py [1-50]
- examples/auth_example.py [45-95]
```

### Example 3: Batch Processing

```bash
$ uv run python -m src search_dataset \
    --dataset_path data/datasets/100_questions.json \
    --k 10 \
    --save_directory data/output/batch_results/

Processing: 100/100 questions [████████████████] 100%
Saved results to: data/output/batch_results/results.json
```

---

## Resources

### Documentation
- [Python Documentation](https://docs.python.org/3/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [vLLM GitHub](https://github.com/vLLM-project/vllm)

### Articles & Tutorials
- "Introduction to Information Retrieval" - Book about searching
- "Understanding TF-IDF" - Medium article
- "RAG Patterns Explained" - Blog post
- "Building Search Systems" - Tutorial series

### How AI Was Used

AI was used to help with:

**✅ Code Generation (30%)**
- Generating function templates
- Creating example code
- Writing test cases

**✅ Documentation (25%)**
- Writing docstrings
- Creating examples
- Explaining concepts

**✅ Debugging (20%)**
- Finding bugs in code
- Suggesting fixes
- Explaining error messages

**✅ Optimization (15%)**
- Improving search algorithm
- Finding performance issues
- Suggesting better data structures

**✅ Project Structure (10%)**
- Planning architecture
- Organizing files
- Naming conventions

**Important**: All AI-generated code was reviewed, understood, and modified. The system was built with real understanding, not just copy-pasting.

---

## File Structure

```
rag-project/
│
├── src/                          # Your Python code
│   ├── __main__.py              # Entry point
│   ├── indexer.py               # Creates the index
│   ├── retriever.py             # Searches the index
│   ├── generator.py             # Creates answers
│   ├── models.py                # Pydantic data models
│   └── cli.py                   # Command-line interface
│
├── data/
│   ├── raw/                     # Original code files
│   │   └── vllm-0.10.1/        # The code to search
│   ├── processed/               # Saved index (created by indexer)
│   ├── datasets/               
│   │   ├── UnansweredQuestions/ # Test questions
│   │   └── AnsweredQuestions/   # Ground truth answers
│   └── output/                 
│       ├── search_results/      # Your search results
│       └── answers/             # Your answers
│
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lock file
├── Makefile                    # Build commands
├── README.md                   # This file
└── .gitignore                 # Files to ignore in Git
```

---

## How to Test

### Quick Test

```bash
# Build index
make install
uv run python -m src index --max_chunk_size 2000

# Test single search
uv run python -m src search "What is this project?" --k 5

# Test single answer
uv run python -m src answer "What is this project?" --k 5
```

### Full Test

```bash
# Build index
uv run python -m src index --max_chunk_size 2000

# Search on test data
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# Evaluate results
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## Troubleshooting

### Problem: "Index not found"

**Solution**:
```bash
uv run python -m src index --max_chunk_size 2000
```

Build the index first.

### Problem: "File not found: data/raw/"

**Solution**:
Make sure the vLLM repository is in `data/raw/`

### Problem: "Out of memory"

**Solution**:
- Reduce chunk size
- Reduce the number of files indexed
- Use a machine with more memory

### Problem: "Search is too slow"

**Solution**:
- Make sure index is built (uses saved index, not building each time)
- Reduce number of results requested (--k 5 instead of --k 100)

### Problem: "Answers are wrong"

**Solution**:
- Check that chunks are being found correctly
- Try different chunk size
- Check the retrieval results (make sure it found the right code)

---

## Next Steps (Future Improvements)

1. **Add semantic search** - Use AI embeddings instead of just word matching
2. **Hybrid retrieval** - Combine multiple search methods
3. **Caching** - Remember previous searches for speed
4. **HTTP API** - Access the system through web interface
5. **Interactive UI** - Web interface for asking questions

---

## Contact & Questions

If you have questions or find bugs:
1. Check the troubleshooting section above
2. Read the code comments
3. Ask peers for help
4. Review the original documentation

---

## License

This project is part of the 42 curriculum.

---

**Made with ❤️ for learning** 🚀