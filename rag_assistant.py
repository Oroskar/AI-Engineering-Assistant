
from ollama import chat
from semantic_retriever import semantic_search


def suggest_solution(error_message):

    # Step 1: Retrieve documents using semantic search
    documents = semantic_search(error_message)

    if not documents:
        return "No relevant documentation found."

    # Step 2: Select the highest-ranked document
    best_document = documents[0]

    context = f"""
Document: {best_document['title']}
Content: {best_document['content']}
"""

    # Step 3: Generate an answer using the document
    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": """
You are an AI Engineering Assistant.

Your job is to help software engineers
understand and troubleshoot software errors.

Use the provided technical documentation
as your source of troubleshooting guidance.

Rules:

1. Explain the possible causes of the error.

2. Suggest practical troubleshooting steps.

3. Do not assume which database engine,
programming language, or technology is used
unless explicitly mentioned in the error.

4. Do not present possible causes as
confirmed facts.

5. Do not invent technical information.

6. Mention the supporting document.

7. If information is missing, explain
what additional information is needed.

8. If the documentation is not relevant
or sufficient, say so clearly.

Provide your answer with these sections:

- Possible Causes
- Troubleshooting Steps
- Supporting Documentation
- Missing Information
"""
            },
            {
                "role": "user",
                "content": f"""
Error:
{error_message}

Technical Documentation:
{context}
"""
            }
        ],
        options={"temperature": 0}
    )

    return response.message.content


if __name__ == "__main__":

    error = "The application cannot reach its SQL server"

    solution = suggest_solution(error)

    print("AI Suggested Solution:")
    print(solution)
