# Quick script to bulk-ingest a folder of documents without needing the
# API running. Usage: python -m scripts.ingest_cli --path data/sample_docs

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import config
from app.ingestion import discover_files, ingest_paths
from app.vectorstore import VectorStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest-cli")


def main():
    parser = argparse.ArgumentParser(description="Bulk-ingest documents into the RAG vector store.")
    parser.add_argument("--path", required=True, help="Directory containing documents to ingest.")
    args = parser.parse_args()

    directory = Path(args.path)
    if not directory.exists():
        logger.error("Path does not exist: %s", directory)
        sys.exit(1)

    files = discover_files(directory)
    if not files:
        logger.warning("No supported files found under %s", directory)
        sys.exit(0)

    logger.info("Discovered %d file(s): %s", len(files), [f.name for f in files])

    chunks, result = ingest_paths(files)
    logger.info("Loaded %d document(s) -> %d chunk(s)", result.num_documents, result.num_chunks)

    # load whatever index already exists (if any) and add the new chunks to it
    manager = VectorStoreManager()
    manager.load()
    manager.build(chunks)

    logger.info(
        "Done. Vector store now contains %d vectors (saved to %s).",
        manager.document_count(),
        config.VECTOR_STORE_DIR,
    )


if __name__ == "__main__":
    main()
