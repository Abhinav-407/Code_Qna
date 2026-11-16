from pathlib import Path

from app.rag.code_chunker import chunk_project
from app.rag.embeddings import embed_chunks
from app.rag.vector_store import store_chunks


def ingest_project(project_path):
    project_path = Path(project_path)

    chunks = chunk_project(project_path)

    vectors = embed_chunks(chunks)

    ids = store_chunks(chunks, vectors)

    return ids

if __name__ == "__main__":
    ids = ingest_project(".")

    print(f"Number of stored chunks: {len(ids)}")