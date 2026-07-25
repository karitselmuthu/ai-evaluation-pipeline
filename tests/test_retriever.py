import pytest
import os
import shutil
from pathlib import Path
from src.core.retriever import BM25Retriever, RAGRetriever

def test_bm25_retriever():
    corpus = [
        {"id": 0, "text": "The quick brown fox jumps over the lazy dog.", "source": "test.txt"},
        {"id": 1, "text": "Python is a popular programming language for AI development.", "source": "test.txt"},
        {"id": 2, "text": "Mitochondria are known as the powerhouse of the cell.", "source": "test.txt"}
    ]
    
    retriever = BM25Retriever(corpus)
    
    # Query matching python
    results = retriever.search("python language", top_n=2)
    assert len(results) >= 1
    assert results[0][0] == 1  # Matches python doc
    
    # Query matching mitochondria
    results = retriever.search("powerhouse cell", top_n=1)
    assert len(results) == 1
    assert results[0][0] == 2  # Matches powerhouse doc

def test_rag_retriever_indexing(tmp_path):
    # Create temp docs
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    doc1 = docs_dir / "doc1.txt"
    doc1.write_text("Hello world. This is a test document.")
    
    doc2 = docs_dir / "doc2.txt"
    doc2.write_text("RAG systems combine retrieval and generation.\n\nHere is a second paragraph.")
    
    retriever = RAGRetriever(docs_dir=str(docs_dir))
    
    assert len(retriever.chunks) >= 3
    # Check that chunks contain the text
    texts = [c["text"] for c in retriever.chunks]
    assert any("Hello world" in t for t in texts)
    assert any("RAG systems" in t for t in texts)
    assert any("second paragraph" in t for t in texts)
