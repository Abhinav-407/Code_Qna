from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def embed_chunks(chunks):
    texts= [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    return vectors

if __name__ == "__main__":
    test_chunks = [
        "Function: login\n\ndef login(username, password):\n    return True",
        "Function: logout\n\ndef logout(user):\n    print('logged out')",
    ]

    vectors = embeddings.embed_documents(test_chunks)

    print("Number of vectors:", len(vectors))
    print("Vector dimension:", len(vectors[0]))
    print("First 5 values:", vectors[0][:5])