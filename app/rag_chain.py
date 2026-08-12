# This is where the actual "RAG" happens:
# 1. search FAISS for chunks related to the question
# 2. stuff those chunks into a prompt
# 3. send it to Groq and get an answer back
# 4. build a citations list from the chunks we used

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app import config
from app.vectorstore import VectorStoreManager

# telling the model to only use the given context and not make stuff up
SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer the \
user's question using ONLY the context provided below, which was retrieved \
from the organization's internal documents.

Rules:
- If the context does not contain enough information to answer, say so \
  plainly instead of guessing.
- Be concise and factual. Do not invent information that isn't in the context.
- When you use a fact from a specific source, reference it by its source \
  name (e.g. "(Source: policy.pdf)").

Context:
{context}
"""

USER_PROMPT = "Question: {question}"


@dataclass
class Citation:
    source: str
    chunk_id: str
    page: int
    snippet: str
    relevance_score: float


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)


def _format_context(scored_docs) -> str:
    # turns the retrieved chunks into one text block to feed the LLM
    blocks = []
    for doc, score in scored_docs:
        blocks.append(
            f"[Source: {doc.metadata.get('source', 'unknown')} | "
            f"chunk: {doc.metadata.get('chunk_id', 'n/a')}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, store_manager: VectorStoreManager):
        self.store_manager = store_manager

        # fail early if there's no API key instead of a confusing error later
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com and add it to your .env file."
            )

        self.llm = ChatGroq(
            model=config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.LLM_TEMPERATURE,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def answer(self, question: str, k: int = config.TOP_K) -> RAGAnswer:
        scored_docs = self.store_manager.similarity_search(question, k=k)

        # if nothing relevant was found, don't even bother calling the LLM
        if not scored_docs:
            return RAGAnswer(
                question=question,
                answer=(
                    "I couldn't find any relevant information in the knowledge "
                    "base to answer that question. Try ingesting more documents "
                    "or rephrasing your question."
                ),
                citations=[],
            )

        context = _format_context(scored_docs)
        generated = self.chain.invoke({"context": context, "question": question})

        # build citations from the retrieved chunks, not from the LLM's text
        # (this way citations are always accurate even if the model messes up)
        citations = [
            Citation(
                source=doc.metadata.get("source", "unknown"),
                chunk_id=doc.metadata.get("chunk_id", "n/a"),
                page=doc.metadata.get("page", 0),
                snippet=doc.page_content[:300].strip() + ("..." if len(doc.page_content) > 300 else ""),
                relevance_score=round(float(score), 4),
            )
            for doc, score in scored_docs
        ]

        return RAGAnswer(question=question, answer=generated, citations=citations)
