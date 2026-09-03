# Chunking

## 1. What is Chunking?

**Chunking** means splitting a large document or file into smaller pieces called **chunks**.

For example, if we have a file with 10,000 characters, we do not always want to use the whole file as one piece.

We can split it:

```text
10,000 characters
       ↓
 ┌─────┬─────┬─────┬─────┬─────┐
 │chunk│chunk│chunk│chunk│chunk│
 │  1  │  2  │  3  │  4  │  5  │
 └─────┴─────┴─────┴─────┴─────┘
```

Each chunk contains a smaller part of the original file.

---

## 2. Why Do We Need Chunking?

The main reason is **retrieval**.

Imagine a very large Python file.

A user asks:

> How does the `login()` function work?

We do not want to give the retrieval system the entire file.

We want to find the small part that contains the `login()` function.

Without chunking:

```text
Large file
     ↓
Search
     ↓
Entire file
```

With chunking:

```text
Large file
     ↓
Small chunks
     ↓
Search
     ↓
Relevant chunk
```

This makes retrieval more **precise**.

The LLM also receives only the useful information instead of a huge amount of unrelated text.

---

# 3. What Makes a Good Chunk?

A good chunk should contain **related information**.

For example, this is a good chunk:

```python
def login(user):
    check_user(user)
    create_session(user)
```

The complete function has one meaning.

A bad chunk could cut the function in the middle:

```python
def login(user):
    check_user(user)
```

and another chunk:

```python
    create_session(user)
```

The information is now separated.

So we want chunks to preserve the **meaning and structure** of the original file.

---

# 4. Chunk Size

Our project has a maximum chunk size.

The default maximum is:

```text
2000 characters
```

This means:

```text
chunk length <= 2000 characters
```

A chunk should normally not be bigger than 2000 characters.

The chunk size should be configurable so we can change it later.

For example:

```text
chunk_size = 2000
```

or:

```text
chunk_size = 1000
```

---

# 5. The Problem With Simple Fixed-Size Chunking

A simple solution is to cut a file every 2000 characters:

```text
0 ---------------- 2000
2000 ------------- 4000
4000 ------------- 6000
```

This is called **fixed-size chunking**.

It is simple, but it can have a problem.

We might cut something important in the middle.

For example:

```python
def calculate_price(product):
    price = product.price
    discount = get_discount(product)
    ...
```

If we cut the file in the middle of this function, the function may be separated into two chunks.

This can make retrieval less useful.

---

# 6. Structure-Aware Chunking

Instead of only looking at character count, we can also look at the **structure** of the file.

The goal is:

> Keep related information together when possible.

Different file types have different structures.

```text
.py
 ↓
Python structure

.md / .txt
 ↓
Document structure
```

---

# 7. Chunking Python Files

Python has a clear structure.

For example:

```python
class User:

    def login(self):
        ...

    def logout(self):
        ...


def calculate_price():
    ...
```

There are:

* classes
* functions
* methods
* other Python structures

We can use the **AST (Abstract Syntax Tree)** to understand this structure.

---

# 8. What is AST?

AST means:

**Abstract Syntax Tree**

It is a tree that represents the structure of Python code.

For example:

```python
def hello():
    print("Hello")
```

The AST understands that this code contains a function.

Conceptually:

```text
Module
 └── FunctionDef
      ├── name: hello
      └── body
           └── print(...)
```

Python provides the `ast` module to work with this structure.

---

# 9. Why Use AST for Python Chunking?

We use AST because Python code has meaningful structures.

For example:

```text
Python file
│
├── class User
│
├── function login()
│
├── function logout()
│
└── function calculate()
```

Instead of randomly cutting the file, we can try to create chunks around these structures.

This makes the chunks more meaningful for retrieval.

Important:

> AST does not create our chunks automatically.

AST helps us **understand the Python structure**.

Our chunker decides how to use that structure to create chunks.

---

# 10. What If a Function Is Bigger Than 2000 Characters?

Sometimes one function can be very large.

For example:

```text
Function = 3500 characters
Maximum chunk size = 2000
```

We cannot simply return the whole function because:

```text
3500 > 2000
```

So our chunker needs a strategy for large structures.

A possible strategy is:

```text
Large function
     ↓
Try to split at meaningful boundaries
     ↓
Chunk 1 <= 2000
Chunk 2 <= 2000
```

The important idea is:

> Respect the maximum size while trying to preserve meaningful structure.

We will design the exact algorithm when we implement the chunker.

---

# 11. Chunking Markdown Files

Markdown also has structure.

For example:

```markdown
# Installation

Install Python.

Install uv.

# Configuration

Set the database URL.

# Usage

Run the application.
```

The headings give meaning to the document.

We can think about the document like this:

```text
Installation
 ├── Install Python
 └── Install uv

Configuration
 └── Set the database URL

Usage
 └── Run the application
```

---

# 12. Why Use Headings for Markdown?

Suppose the user asks:

> How do I configure the application?

We want to retrieve:

```markdown
# Configuration

Set the database URL.
```

We do not want to randomly retrieve:

```markdown
# Installation

Install Python.

# Configuration

Set the database URL.
```

The heading helps us keep related information together.

So:

> **Markdown headings help us understand the document structure.**

---

# 13. Markdown/Text Chunking

For Markdown and text files, we can try to preserve:

* headings
* paragraphs
* sections
* related text

The goal is to avoid breaking related information unnecessarily.

---

# 14. Character Offsets

Every chunk should remember **where it came from**.

For example:

```text
file.py
```

contains:

```text
01234567890123456789...
```

Suppose a chunk starts at character `100` and ends at character `500`.

We store:

```text
first_character_index = 100
last_character_index = 500
```

These are called **character offsets**.

---

# 15. Why Do We Need Offsets?

Offsets tell us the exact location of the chunk in the original file.

Without offsets, we might know:

```text
file.py
```

but we do not know exactly where the retrieved text came from.

With offsets:

```text
file.py
start = 100
end = 500
```

we know the exact source location.

This is important for our project because the evaluator checks whether our retrieved source overlaps the correct source range.

---

# 16. The Source Information of a Chunk

A chunk should keep information such as:

```text
file_path
first_character_index
last_character_index
```

For example:

```text
file_path = "vllm/model.py"

first_character_index = 1200

last_character_index = 1850
```

This tells us:

> This chunk comes from `vllm/model.py`, from character 1200 to character 1850.

---

# 17. Chunking Pipeline

Our chunking process will look approximately like this:

```text
Original file
     ↓
Read file
     ↓
Identify file type
     ↓
 ┌───────────────┐
 │               │
.py            .md/.txt
 │               │
 ↓               ↓
AST           Headings /
structure     paragraphs
 │               │
 └───────┬───────┘
         ↓
Create chunks
         ↓
Check size
         ↓
Calculate offsets
         ↓
Store chunks
```

---

# 18. Different Files Need Different Strategies

We should not use exactly the same chunking method for every file.

```text
Python
  ↓
AST
  ↓
Functions / classes / methods


Markdown
  ↓
Headings
  ↓
Sections


Text
  ↓
Paragraphs / logical blocks
```

This is called **structure-aware chunking**.

---

# 19. Chunking in Our RAG System

Our RAG system will use chunking before retrieval.

The complete idea is:

```text
vLLM source code
       ↓
     Chunking
       ↓
Small meaningful chunks
       ↓
Character offsets
       ↓
Search index
       ↓
Retriever
       ↓
Top-k relevant chunks
       ↓
LLM
       ↓
Answer
```

Chunking is therefore one of the first important parts of our RAG pipeline.

---

# 20. Important Concepts to Remember

### Chunk

A small piece of a larger document.

### Chunking

The process of splitting a document into chunks.

### Chunk size

The maximum amount of text allowed in a chunk.

Our default:

```text
2000 characters
```

### Character offset

The position of a chunk inside the original file.

```text
start → first_character_index
end   → last_character_index
```

### AST

A tree that represents the structure of Python code.

It helps us identify things like:

```text
functions
classes
methods
```

### Structure-aware chunking

Creating chunks based on the meaning and structure of the document instead of only cutting by character count.

---

# 21. The Main Idea

Remember these four sentences:

> **Chunking splits large files into smaller pieces.**

> **Good chunks keep related information together.**

> **Offsets tell us exactly where a chunk came from.**

> **AST and Markdown structure help us create meaningful chunks instead of random pieces.**

---

# 22. What We Will Implement

For our project, we will eventually build:

```text
src/
├── models.py
├── dataset_loader.py
├── chunker.py
└── ...
```

Our `chunker.py` will need to:

1. Read a document.
2. Detect its type.
3. Use AST for Python files.
4. Use headings/paragraphs for Markdown/Text.
5. Respect the maximum chunk size.
6. Calculate character offsets.
7. Return structured chunks.

We will **not implement all of this at once**.

We will first build a very small chunking function, test it, understand it, and then improve it.

---

# 23. Learning Rule

For this project, follow this cycle:

```text
Learn
  ↓
Explain the concept in your own words
  ↓
Small exercise
  ↓
Implement
  ↓
Test
  ↓
Commit
```

The goal is not only to make the RAG system work.

The goal is to **understand why it works**.
