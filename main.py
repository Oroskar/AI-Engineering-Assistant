
import json
from pathlib import Path

from ai_classifier import classify_with_ai
from rag_assistant import suggest_solution


BASE_DIR = Path(__file__).parent


def parse_log_line(line):
    parts = line.strip().split(maxsplit=3)

    if len(parts) != 4:
        return None

    date, time, level, message = parts

    return {
        "date": date,
        "time": time,
        "level": level,
        "message": message
    }


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
    results = []
    categories = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            log = parse_log_line(line)

            if log is None or log["level"] != "ERROR":
                continue

            message = log["message"]
            category = classify_error(message)
            source = "RULE_BASED"
            explanation = "Classified using predefined rules"

            # Classify unknown errors using AI
            if category == "UNKNOWN":
                print("\nClassifying with AI...")

                ai_result = classify_with_ai(message)

                category = ai_result["category"]
                explanation = ai_result["explanation"]
                source = "AI"

            # Retrieve documentation and suggest solutions
            print(f"\n[{category}] {message}")
            print("Searching technical documentation...")

            solution = suggest_solution(message)

            print("\nSuggested Solution:")
            print(solution)

            # Store the analysis
            results.append({
                "date": log["date"],
                "time": log["time"],
                "message": message,
                "category": category,
                "explanation": explanation,
                "classification_source": source,
                "suggested_solution": solution
            })

            categories[category] = categories.get(category, 0) + 1

    # Save results to JSON
    output_path = BASE_DIR / "analysis_results.json"

    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=4, ensure_ascii=False)

    print("\n" + "=" * 40)
    print("ANALYSIS SUMMARY")
    print("=" * 40)

    print("Total errors:", len(results))
    print("Errors by category:", categories)
    print("Results saved to:", output_path)


if __name__ == "__main__":
    analyze_logs(BASE_DIR / "sample" / "sample.log")
