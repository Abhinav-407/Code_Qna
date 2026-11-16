import chromadb

from app.rag.embeddings import embeddings


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="codebase",
    metadata={"hnsw:space": "cosine"}
)


def retrieve(query, top_k=5):
    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    retrieved_chunks = []

    for i in range(len(results["ids"][0])):
        retrieved_chunks.append(
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
        )

    return retrieved_chunks


if __name__ == "__main__":
    results = retrieve(
        "Where is the code that extracts functions and classes from Python source?",
        top_k=3
    )

    for result in results:
        print("=" * 60)

        metadata = result["metadata"]

        print("File:", metadata["source"])
        print("Type:", metadata["type"])
        print("Name:", metadata["name"])
        print(
            "Lines:",
            metadata["start_line"],
            "-",
            metadata["end_line"]
        )
        print("Distance:", result["distance"])

        print("\nCode:")
        print(result["content"])