
# Task 2: RAG Document Retrieval Assistant

## Overview
A local document retrieval assistant built using Python, LangChain, Sentence Transformers, and ChromaDB. It retrieves relevant information from TXT and PDF documents based on user questions.

## Features
- Load TXT and PDF documents
- Split documents into smaller text chunks
- Generate embeddings locally using Sentence Transformers
- Store and search document embeddings using ChromaDB
- Retrieve relevant content with source filenames
- Run without a paid OpenAI API key

## Technologies
- Python
- LangChain
- Sentence Transformers
- ChromaDB
- PyPDF
- Pytest

## Project Structure
```text
task2_rag_assistant/
├── documents/
│   └── internship_notes.txt
├── rag_Pipeline.py
├── test_rag_pipeline.py
├── requirements.txt
└── README.md
```

## Setup
From the repository root, activate your virtual environment and install the dependencies:

```bash
pip install -r task2_rag_assistant/requirements.txt
```

## Run the Assistant
```bash
python task2_rag_assistant/rag_Pipeline.py
```

Ask a question about the documents. Type `exit` to quit.

## Run Tests
```bash
python -m pytest task2_rag_assistant/test_rag_pipeline.py -v
```

## Current Scope
This project performs retrieval-augmented document search: it retrieves relevant document chunks and displays their source and content. It does not yet generate new answers using a large language model.
