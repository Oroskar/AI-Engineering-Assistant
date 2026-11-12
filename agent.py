
import json
import re
from pathlib import Path

from ollama import chat
from semantic_retriever import semantic_search


BASE_DIR = Path(__file__).parent
MODEL = "llama3.2"

TOPIC_PATTERNS = {
    "DATABASE": r"\b(database|postgresql|postgres|mysql|sql|mongodb)\b",
    "TIMEOUT": r"\b(timeout|timed out|times out|deadline|time limit|too long)\b",
    "MEMORY": r"\b(memory|ram|vram|cuda|memory allocation)\b"
}


def identify_topics(text):
    return [
        topic
        for topic, pattern in TOPIC_PATTERNS.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]


def search_technical_docs(error_message: str) -> str:
    """
    Search local technical documentation.

    Args:
        error_message: Software error to investigate.
    """
    topics = identify_topics(error_message)

    if not topics:
        return json.dumps({
            "status": "NO_DOCUMENTATION"
        })

    documents = semantic_search(error_message)

    for doc in documents:
        title = doc["title"].lower()

        if "timeout" in title:
            doc_topic = "TIMEOUT"
        elif "memory" in title:
            doc_topic = "MEMORY"
        elif "database" in title or "postgresql" in title:
            doc_topic = "DATABASE"
        else:
            continue

        if doc_topic in topics:
            return json.dumps({
                "status": "FOUND",
                "title": doc["title"],
                "content": doc["content"],
                "similarity": round(doc["similarity"], 3)
            })

    return json.dumps({
        "status": "NO_DOCUMENTATION"
    })


def read_error_summary() -> str:
    """Read saved error counts from the local report."""

    path = BASE_DIR / "analysis_results.json"

    if not path.exists():
        return json.dumps({"status": "MISSING_REPORT"})

    with open(path, "r", encoding="utf-8") as file:
        results = json.load(file)

    categories = {}

    for item in results:
        category = item["category"]
        categories[category] = categories.get(category, 0) + 1

    return json.dumps({
        "status": "FOUND",
        "total_errors": len(results),
        "categories": categories,
        "total_requests": None
    })


def run_agent(question: str):

    # Guardrail: Do not calculate rates without request counts
    asks_for_rate = (
        re.search(r"\b(percentage|percent|rate)\b|%", question, re.I)
        and re.search(r"\b(failed|failure|requests|errors)\b", question, re.I)
    )

    if asks_for_rate:
        print("Guardrail activated: Missing denominator")
        print("Tool selected: read_error_summary")

        return (
            "Cannot calculate the percentage of failed requests. "
            "The saved report does not contain total request counts "
            "for the requested period."
        )

    tools = [
        search_technical_docs,
        read_error_summary
    ]

    messages = [
        {
            "role": "system",
            "content": """
You are an AI Engineering Assistant.

Choose the appropriate tools for the user's request.

Use search_technical_docs for troubleshooting questions.
Use read_error_summary for saved report statistics.

You may select multiple tools when needed.
Do not invent information.
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = chat(
        model=MODEL,
        messages=messages,
        tools=tools,
        options={"temperature": 0}
    )

    tool_calls = response.message.tool_calls or []

    if not tool_calls:
        return (
            "No appropriate tool was selected. "
            "I cannot provide a verified answer."
        )

    answers = []

    for tool_call in tool_calls:
        name = tool_call.function.name
        arguments = tool_call.function.arguments or {}

        print("Tool selected:", name)

        if name == "read_error_summary":

            data = json.loads(read_error_summary())

            if data["status"] != "FOUND":
                answers.append("No saved report is available.")
                continue

            lines = [
                "SAVED ERROR REPORT",
                f"Total error events: {data['total_errors']}",
                "Errors by category:"
            ]

            for category, count in data["categories"].items():
                lines.append(f"- {category}: {count}")

            lines.append(
                "Note: Total request counts are not available."
            )

            answers.append("\n".join(lines))

        elif name == "search_technical_docs":

            error = arguments.get("error_message", "")

            if not isinstance(error, str) or not error.strip():
                answers.append("Invalid error message.")
                continue

            data = json.loads(search_technical_docs(error))

            if data["status"] != "FOUND":
                print("Guardrail activated: No relevant documentation")
                answers.append(
                    "No relevant local documentation is available."
                )
                continue

            answers.append(
                "TECHNICAL DOCUMENTATION\n"
                f"Source: {data['title']}\n"
                f"Suggested checks from source:\n{data['content']}\n"
                "These checks are general guidance, "
                "not a confirmed diagnosis."
            )

        else:
            answers.append("Unknown tool requested.")

    return "\n\n".join(answers)


if __name__ == "__main__":
    question = (
        "How many errors are in my saved report? "
        "Also, explain how to troubleshoot "
        "a PostgreSQL connection failure."
    )

    print(run_agent(question))
