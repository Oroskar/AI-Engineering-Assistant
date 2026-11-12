
import json
from ollama import chat


def classify_with_ai(error_message):
    schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "DATABASE",
                    "TIMEOUT",
                    "MEMORY",
                    "NETWORK",
                    "UNKNOWN"
                ]
            },
            "explanation": {
                "type": "string"
            }
        },
        "required": ["category", "explanation"],
        "additionalProperties": False
    }

    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": """
You are an expert software error classifier.

Your task is to classify software errors
based on the affected component or resource.

Available categories:

DATABASE:
Errors related to database systems,
including PostgreSQL, MySQL, database
connections, and authentication.

TIMEOUT:
Operations that exceed their allowed
execution or response time.

MEMORY:
Errors involving RAM, GPU VRAM,
memory allocation, or out-of-memory
conditions during program execution.

NETWORK:
General network connection failures,
DNS problems, or network communication
errors not specifically related to databases.

UNKNOWN:
Errors that do not fit any of the
available categories.

IMPORTANT CLASSIFICATION RULES:

1. Disk space and storage are NOT RAM.

2. Disk full, storage capacity, and file
system errors must be classified as UNKNOWN,
because STORAGE is not an available category.

3. PostgreSQL and MySQL connection problems
must be classified as DATABASE, even when
the error mentions a network connection.

4. Classify based on the affected resource,
not just individual keywords.

5. If the error does not clearly fit an
available category, return UNKNOWN.

Return a JSON object containing:
- category
- explanation

Keep the explanation short and factual.
Do not invent information.
"""
            },
            {
                "role": "user",
                "content": error_message
            }
        ],
        format=schema,
        options={
            "temperature": 0
        }
    )

    result = json.loads(response.message.content)

    return result
