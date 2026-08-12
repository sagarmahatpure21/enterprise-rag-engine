# The chat UI. This just talks to the FastAPI backend over HTTP,
# it doesn't know anything about embeddings or FAISS directly.

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

# keeps the chat history around between reruns (streamlit reruns the
# whole script on every interaction otherwise)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def call_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.ok
    except requests.RequestException:
        return False


def call_stats():
    try:
        r = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Could not fetch stats: {exc}")
        return None


def call_ingest(files):
    multipart = [("files", (f.name, f.getvalue())) for f in files]
    r = requests.post(f"{API_BASE_URL}/ingest", files=multipart, timeout=600)
    r.raise_for_status()
    return r.json()


def call_query(question: str, top_k: int):
    r = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


# sidebar - upload docs + show index stats
with st.sidebar:
    st.title("📚 Knowledge Base")

    healthy = call_health()
    st.markdown(f"**API status:** {'🟢 online' if healthy else '🔴 offline'}")

    st.divider()
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Ingest documents", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Chunking, embedding, and indexing..."):
            try:
                result = call_ingest(uploaded_files)
                st.success(
                    f"Indexed {result['num_chunks']} chunks from "
                    f"{len(result['ingested_files'])} file(s). "
                    f"Total vectors: {result['total_vectors']}"
                )
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("Index stats")
    stats = call_stats()
    if stats:
        st.metric("Vectors indexed", stats["total_vectors"])
        st.caption(f"Embedding model: {stats['embedding_model']}")
        st.caption(f"LLM: {stats['llm_model']}")
        st.caption(f"Chunk size / overlap: {stats['chunk_size']} / {stats['chunk_overlap']}")

    st.divider()
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=4)

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# main chat area
st.title("Enterprise RAG Knowledge Management Assistant")
st.caption(
    "Ask questions about your organization's documents. Answers are grounded "
    "in retrieved context and cite their sources."
)

for question, answer, citations in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for c in citations:
                    st.markdown(
                        f"**{c['source']}** — chunk `{c['chunk_id']}` "
                        f"(relevance: {c['relevance_score']:.2f})"
                    )
                    st.caption(c["snippet"])

question = st.chat_input("Ask a question about your documents...")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = call_query(question, top_k)
                st.write(result["answer"])
                citations = result["citations"]
                if citations:
                    with st.expander(f"Sources ({len(citations)})"):
                        for c in citations:
                            st.markdown(
                                f"**{c['source']}** — chunk `{c['chunk_id']}` "
                                f"(relevance: {c['relevance_score']:.2f})"
                            )
                            st.caption(c["snippet"])
                st.session_state.chat_history.append((question, result["answer"], citations))
            except requests.RequestException as exc:
                st.error(f"Query failed: {exc}")
