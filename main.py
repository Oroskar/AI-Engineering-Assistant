
import json
from pathlib import Path

from log_parser import parse_multiline_log
from ai_classifier import classify_with_ai
from rag_assistant import suggest_solution


BASE_DIR = Path(__file__).parent


def classify_error(message):
    message = message.lower()

    if "database" in message:
        return "DATABASE"
    elif "timeout" in message:
        return "TIMEOUT"
    elif "memory" in message:
        return "MEMORY"
    else:
        return "UNKNOWN"


def analyze_logs(file_path):
    records = parse_multiline_log(file_path)

    results = []
    categories = {}

    for log in records:

        if log["level"] != "ERROR":
            continue

        message = log["message"]
        details = log["details"]

        # Combine error message and stack trace
        full_error = "\n".join([message] + details)

        print("\n" + "=" * 50)
        print("DETECTED ERROR:", message)

        if details:
            print("Stack Trace detected!")

        # Step 1: Classify the error
        category = classify_error(message)
        source = "RULE_BASED"
        explanation = "Classified using predefined rules"

        if category == "UNKNOWN":
            print("Classifying with AI...")

            ai_result = classify_with_ai(full_error)

            category = ai_result["category"]
            explanation = ai_result["explanation"]
            source = "AI"

        print("Category:", category)

        # Step 2: Retrieve documentation and generate advice
        print("Searching technical documentation...")

        solution = suggest_solution(full_error)

        print("\nAI Suggested Solution:")
        print(solution)

        # Step 3: Save the result
        results.append({
            "date": log["date"],
            "time": log["time"],
            "message": message,
            "stack_trace": details,
            "category": category,
            "explanation": explanation,
            "classification_source": source,
            "suggested_solution": solution
        })

        categories[category] = categories.get(category, 0) + 1

    # Save complex-log results separately
    output_path = BASE_DIR / "analysis_results_realistic.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print("Total log events:", len(records))
    print("Total errors:", len(results))
    print("Errors by category:", categories)
    print("Results saved to:", output_path)


if __name__ == "__main__":
    log_path = BASE_DIR / "sample" / "realistic.log"
    analyze_logs(log_path)
