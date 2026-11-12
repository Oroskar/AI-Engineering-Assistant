
from retriever import retrieve_docs
from semantic_retriever import semantic_search


test_cases = [
    (
        "Connection refused by PostgreSQL",
        "PostgreSQL Connection Troubleshooting"
    ),
    (
        "The application cannot reach its SQL server",
        "PostgreSQL Connection Troubleshooting"
    ),
    (
        "HTTP request timed out after 30 seconds",
        "Request Timeout Troubleshooting"
    ),
    (
        "The API response is taking too long",
        "Request Timeout Troubleshooting"
    ),
    (
        "Memory allocation failed",
        "Memory Troubleshooting"
    ),
    (
        "CUDA ran out of GPU memory",
        "Memory Troubleshooting"
    )
]


keyword_correct = 0
semantic_correct = 0

for message, expected in test_cases:

    keyword_results = retrieve_docs(message)
    semantic_results = semantic_search(message)

    keyword_top = (
        keyword_results[0]["title"]
        if keyword_results else None
    )

    semantic_top = semantic_results[0]["title"]

    if keyword_top == expected:
        keyword_correct += 1

    if semantic_top == expected:
        semantic_correct += 1

    print(f"\nError: {message}")
    print(f"Expected: {expected}")
    print(f"Keyword Search: {keyword_top}")
    print(f"Semantic Search: {semantic_top}")
    print(f"Similarity: {semantic_results[0]['similarity']:.3f}")


total = len(test_cases)

print("\n========== RETRIEVAL EVALUATION ==========")

print(f"Keyword Search Accuracy: {keyword_correct / total:.2%}")
print(f"Semantic Search Accuracy: {semantic_correct / total:.2%}")
