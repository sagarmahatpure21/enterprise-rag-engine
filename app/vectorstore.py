# This handles turning text into vectors (embeddings) and storing/searching
# them with FAISS. Using sentence-transformers here so everything runs
# locally on CPU, no API key needed for this part.

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app import config

logger = logging.getLogger(__name__)

_FAISS_INDEX_NAME = "index"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


class VectorStoreManager:
    # basically a wrapper around FAISS so I don't repeat load/save logic everywhere
    def __init__(self):
        self.embeddings = get_embeddings()
        self._store: Optional[FAISS] = None

    def load(self) -> bool:
        index_file = config.VECTOR_STORE_DIR / f"{_FAISS_INDEX_NAME}.faiss"
        if not index_file.exists():
            return False
        self._store = FAISS.load_local(
            str(config.VECTOR_STORE_DIR),
            self.embeddings,
            index_name=_FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True,
            normalize_L2=True,
        )
        return True

    def build(self, chunks: List[Document]) -> None:
        if not chunks:
            raise ValueError("No chunks provided to build the vector store.")

        if self._store is None:
            self._store = FAISS.from_documents(
                chunks,
                self.embeddings,
                normalize_L2=True,
            )
        else:
            self._store.add_documents(chunks)

        self._store.save_local(str(config.VECTOR_STORE_DIR), index_name=_FAISS_INDEX_NAME)
        logger.info("Indexed %d chunks into FAISS.", len(chunks))

    def similarity_search(self, query: str, k: int = config.TOP_K):
        if self._store is None:
            raise RuntimeError("Vector store not initialized. Call load() or build() first.")
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def as_retriever(self, k: int = config.TOP_K):
        if self._store is None:
            raise RuntimeError("Vector store not initialized. Call load() or build() first.")
        return self._store.as_retriever(search_kwargs={"k": k})

    def document_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

# keeping one shared instance so we don't reload the model on every request
_manager: Optional[VectorStoreManager] = None


def get_manager() -> VectorStoreManager:
    global _manager
    if _manager is None:
        _manager = VectorStoreManager()
        _manager.load()
    return _manager