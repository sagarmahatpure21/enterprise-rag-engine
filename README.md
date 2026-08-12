# RAG-Based Enterprise Knowledge Assistant

An AI-powered RAG (Retrieval-Augmented Generation) app for asking questions
over a set of documents and getting back answers with real source citations.

Built with **FastAPI**, **Streamlit**, **LangChain**, and **FAISS**. Uses
**Sentence-Transformers** for local embeddings and the **Groq API** for
fast LLM responses. Fully containerized with **Docker**.

## What it does

You upload documents (PDF, TXT, or MD), the app splits them into chunks,
turns each chunk into a vector (embedding), and stores them in a FAISS
index. When you ask a question, it finds the most relevant chunks and
sends them to an LLM (via Groq) to generate an answer — along with the
exact source and chunk it came from.

## Tech stack

- **FastAPI** – backend REST API
- **Streamlit** – chat interface
- **LangChain** – document loading, chunking, prompt handling
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) – local embeddings, no API key needed
- **FAISS** – vector similarity search
- **Groq API** (`llama-3.1-8b-instant`) – answer generation
- **Docker / Docker Compose** – containerized deployment

## Project structure

```
enterprise-rag-engine/
├── app/
│   ├── config.py         # settings (models, paths, chunk size, API key)
│   ├── ingestion.py       # loads + chunks documents
│   ├── vectorstore.py     # embeddings + FAISS
│   ├── rag_chain.py       # retrieval + Groq generation
│   └── main.py            # FastAPI app / endpoints
├── streamlit_ui/
│   └── app.py             # chat UI
├── scripts/
│   └── ingest_cli.py       # bulk-ingest from the command line
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.streamlit
├── data/sample_docs/       # sample HR/IT/onboarding docs to test with
├── tests/
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

### 1. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — no credit card needed.

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Open `.env` and paste in your `GROQ_API_KEY`.

### 3. Ingest the sample documents
```bash
python -m scripts.ingest_cli --path data/sample_docs
```

### 4. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```
Swagger docs: http://localhost:8000/docs

### 5. Run the UI (separate terminal)
```bash
streamlit run streamlit_ui/app.py
```
Open: http://localhost:8501

## Running with Docker

```bash
docker compose up --build
```
Make sure `.env` has a valid `GROQ_API_KEY` before running — the API
container reads it from there.

## API endpoints

| Method | Endpoint  | What it does |
|--------|-----------|----------------|
| GET    | `/health` | Check if the server is up |
| GET    | `/stats`  | Index size + current settings |
| POST   | `/ingest` | Upload files to embed and index |
| POST   | `/query`  | Ask a question, get an answer + citations |
| DELETE | `/index`  | Clear the vector index |

Example:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do employees get?"}'
```

## Running tests
```bash
pytest tests/ -v
```

## Notes

- Only PDF, TXT, and MD files are supported right now.
- Embeddings happen 100% locally — only the final prompt (retrieved text +
  your question) is sent to Groq, never the raw documents.
- The `vectorstore_index/` folder is generated automatically and excluded
  from git — it can always be rebuilt by re-running ingestion.
