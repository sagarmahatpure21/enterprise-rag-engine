# This file handles loading documents and splitting them into chunks.
# Basically: read the file -> break it into smaller pieces -> tag each
# piece with where it came from (so we can cite it later).

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

logger = logging.getLogger(__name__)

# only supporting pdf/txt/md for now to keep things simple
# markdown just uses the plain text loader, no need for a fancy md parser
_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


@dataclass
class IngestionResult:
    source_files: List[str] = field(default_factory=list)
    num_documents: int = 0
    num_chunks: int = 0
    errors: List[str] = field(default_factory=list)


def _load_single_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    loader_cls = _LOADERS.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")

    try:
        loader = loader_cls(str(path))
    except TypeError:
        # TextLoader sometimes wants an encoding, just fall back to utf-8
        loader = loader_cls(str(path), encoding="utf-8")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata.setdefault("page", doc.metadata.get("page", 0))
    return docs


def load_documents(paths: List[Path]) -> List[Document]:
    # loads a bunch of files and returns them as LangChain Document objects
    all_docs: List[Document] = []
    for path in paths:
        try:
            all_docs.extend(_load_single_file(path))
        except Exception as exc:
            logger.exception("Failed to load %s", path)
            raise RuntimeError(f"Failed to load {path.name}: {exc}") from exc
    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    # splits documents into smaller overlapping chunks so we can embed them
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # give each chunk an id like "filename::chunk-1" so we can cite it later
    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        per_source_counter[src] = per_source_counter.get(src, 0) + 1
        chunk.metadata["chunk_id"] = f"{src}::chunk-{per_source_counter[src]}"
    return chunks


def discover_files(directory: Path) -> List[Path]:
    # grabs every file in a folder that we know how to read
    return [
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]


def ingest_paths(paths: List[Path]) -> tuple[List[Document], IngestionResult]:
    # runs the whole load -> chunk pipeline and returns the chunks + a summary
    result = IngestionResult(source_files=[p.name for p in paths])
    documents = load_documents(paths)
    result.num_documents = len(documents)
    chunks = chunk_documents(documents)
    result.num_chunks = len(chunks)
    return chunks, result
