
import json
import math
from pathlib import Path
from ollama import embed


BASE_DIR = Path(__file__).parent
MODEL = "nomic-embed-text"


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def semantic_search(error_message):
    with open(BASE_DIR / "knowledge_base.json", "r") as file:
        documents = json.load(file)

    query_embedding = embed(
        model=MODEL,
        input="search_query: " + error_message
    ).embeddings[0]

    document_texts = [
        "search_document: " + doc["title"] + " " + doc["content"]
        for doc in documents
    ]

    document_embeddings = embed(
        model=MODEL,
        input=document_texts
    ).embeddings

    results = []

    for doc, doc_embedding in zip(documents, document_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)

        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "similarity": score
        })

    results.sort(key=lambda item: item["similarity"], reverse=True)

    return results


if __name__ == "__main__":
    error = "The application cannot reach its SQL server"

    results = semantic_search(error)

    print("Error:", error)
    print("\nMost relevant documents:")

    for result in results:
        print(f"\nDocument: {result['title']}")
        print(f"Similarity: {result['similarity']:.3f}")
