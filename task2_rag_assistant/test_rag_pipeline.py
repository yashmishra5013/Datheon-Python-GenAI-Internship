
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_Pipeline import ask_question


def test_ask_question_returns_relevant_content():
    """Check that retrieved document content is returned."""

    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [
        MagicMock(
            page_content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"source": "internship_notes.txt"},
        )
    ]

    result = ask_question(vector_store, "What is RAG?")

    assert "RAG stands for Retrieval-Augmented Generation." in result
    assert "internship_notes.txt" in result
    vector_store.similarity_search.assert_called_once_with(
        "What is RAG?", k=3
    )


def test_ask_question_returns_message_when_no_results():
    """Check behavior when retrieval finds no documents."""

    vector_store = MagicMock()
    vector_store.similarity_search.return_value = []

    result = ask_question(vector_store, "Unknown topic")

    assert result == "No relevant information found."


def test_ask_question_handles_multiple_documents():
    """Check that content from multiple retrieved documents is included."""

    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [
        MagicMock(
            page_content="Python is a programming language.",
            metadata={"source": "python.txt"},
        ),
        MagicMock(
            page_content="ChromaDB stores vectors.",
            metadata={"source": "rag.txt"},
        ),
    ]

    result = ask_question(vector_store, "Tell me about Python and ChromaDB")

    assert "Python is a programming language." in result
    assert "ChromaDB stores vectors." in result
    assert "python.txt" in result
    assert "rag.txt" in result
