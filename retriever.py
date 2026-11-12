
import json
from pathlib import Path


def retrieve_docs(error_message):
    knowledge_path = Path(__file__).parent / "knowledge_base.json"

    with open(knowledge_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    matches = []

    for doc in documents:
        for keyword in doc["keywords"]:
            if keyword.lower() in error_message.lower():
                matches.append(doc)
                break

    return matches
