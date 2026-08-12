# All the settings for the project live here.
# I'm reading most of them from a .env file so I don't hardcode
# stuff like my Groq API key directly in the code.

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# folder paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = Path(os.getenv("RAG_VECTOR_STORE_DIR", BASE_DIR / "vectorstore_index"))
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"

# make sure these folders exist before anything tries to use them
for p in (DATA_DIR, UPLOAD_DIR, VECTOR_STORE_DIR):
    p.mkdir(parents=True, exist_ok=True)

# embedding model - this runs locally on CPU, no API key needed
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM settings - using Groq's free API instead of running a model locally
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))

# chunking settings
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))

# how many chunks to pull back when answering a question
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# API server settings
API_HOST = os.getenv("RAG_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RAG_API_PORT", "8000"))

# only handling these file types for now
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
