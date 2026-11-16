import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="codebase",
    metadata={"hnsw:space": "cosine"}
)

def store_chunks(chunks, vector):
    texts= [chunk.page_content for chunk in chunks]
    metadata = [
        {
            key: value
            for key, value in chunk.metadata.items()
            if value is not None
        }
        for chunk in chunks
    ]

    ids = [ f"{chunk.metadata['source']}-{chunk.metadata['type']}-{chunk.metadata['name']}-{i}"
           for i, chunk in enumerate(chunks)]
    collection.upsert(
        ids = ids,
        documents = texts,
        embeddings = vector,
        metadatas = metadata
    )
    return ids

if __name__ == "__main__":
    from app.rag.code_chunker import chunk_project
    from app.rag.embeddings import embed_chunks

    chunks = chunk_project(".")

    print("Number of chunks:", len(chunks))

    vectors = embed_chunks(chunks)

    print("Number of vectors:", len(vectors))

    ids = store_chunks(chunks, vectors)

    print("Stored chunks:", len(ids))

    results = collection.get()

    print("Total records in Chroma:", len(results["ids"]))

    print("\nFirst ID:")
    print(results["ids"][0])

    print("\nFirst document:")
    print(results["documents"][0])

    print("\nFirst metadata:")
    print(results["metadatas"][0])