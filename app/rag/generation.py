from openai import OpenAI
from dotenv import load_dotenv
from app.rag.retrieval import retrieve

load_dotenv()

client = OpenAI()

def build_prompt(query, retrieved_chunks):
    context = ""

    for chunk in retrieved_chunks:
        context += (
            f"File: {chunk['metadata']['source']}\n"
            f"Type: {chunk['metadata']['type']}\n"
            f"Name: {chunk['metadata']['name']}\n"
            f"Lines: {chunk['metadata']['start_line']}-"
            f"{chunk['metadata']['end_line']}\n\n"
            f"{chunk['content']}\n"
            f"{'-' * 60}\n"
        )

    prompt = f"""
You are a code assistant.

Answer the user's question using the provided code context.

If the answer cannot be determined from the provided context,
say that the relevant code was not found.

Code context:
{context}

User question:
{query}
"""

    return prompt

def generate_answer(prompt):
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text

if __name__ == "__main__":
    query = "Where are methods inside a Python class extracted?"

    retrieved_chunks = retrieve(
        query,
        top_k=3
    )

    prompt = build_prompt(
        query,
        retrieved_chunks
    )

    answer = generate_answer(prompt)

    print(answer)