# Basic tests for the loading/chunking logic. These don't need Groq or
# any API key set up, they just test the ingestion pipeline on its own.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion import chunk_documents, discover_files, load_documents
from langchain_core.documents import Document

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_discover_files_finds_sample_docs():
    files = discover_files(SAMPLE_DIR)
    assert len(files) >= 3
    assert all(f.suffix.lower() in {".md", ".pdf", ".txt"} for f in files)


def test_load_documents_reads_markdown():
    files = [f for f in discover_files(SAMPLE_DIR) if f.suffix.lower() == ".md"]
    docs = load_documents(files)
    assert len(docs) >= 3
    assert all(doc.metadata.get("source") for doc in docs)


def test_chunk_documents_respects_overlap_and_ids():
    # repeating a sentence a bunch of times so it's long enough to split
    raw_text = "Sentence one. " * 200
    doc = Document(page_content=raw_text, metadata={"source": "synthetic.txt"})
    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    # each chunk should get an id like synthetic.txt::chunk-1, chunk-2, etc
    for i, chunk in enumerate(chunks, start=1):
        assert chunk.metadata["chunk_id"] == f"synthetic.txt::chunk-{i}"


def test_chunk_documents_empty_input_returns_empty():
    assert chunk_documents([]) == []
