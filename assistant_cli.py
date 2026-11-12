
import json
from pathlib import Path

from rag_assistant import suggest_solution
from grounded_rag import generate_grounded_answer


BASE_DIR = Path(__file__).parent
REPORT_PATH = BASE_DIR / "analysis_results_realistic.json"


def load_errors():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            "Run main.py first to generate the realistic error report."
        )

    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("The error report must contain a list.")

    return [
        record
        for record in records
        if isinstance(record, dict)
        and "message" in record
        and "category" in record
    ]


def select_error(records):
    print("\n" + "=" * 55)
    print("AI ENGINEERING ASSISTANT")
    print("=" * 55)

    print("\nAvailable errors:")

    for index, record in enumerate(records, start=1):
        print(
            f"{index}. [{record['category']}] "
            f"{record['message']}"
        )

    while True:
        choice = input(
            "\nChoose an error number (or q to quit): "
        ).strip().lower()

        if choice == "q":
            return None

        if choice.isdigit():
            index = int(choice) - 1

            if 0 <= index < len(records):
                return records[index]

        print("Invalid selection. Try again.")


def select_mode():
    print("\nChoose an answer mode:")
    print("1. Generative RAG - Detailed AI answer")
    print("2. Grounded RAG - Source-only recommendations")
    print("3. Compare both approaches")

    while True:
        choice = input("\nMode (1/2/3): ").strip()

        if choice in {"1", "2", "3"}:
            return choice

        print("Please choose 1, 2, or 3.")


def build_full_error(record):
    parts = [str(record["message"])]

    details = record.get("stack_trace", [])

    if isinstance(details, list):
        parts.extend(str(line) for line in details)

    return "\n".join(parts)


def run_generative(record):
    print("\n" + "=" * 55)
    print("GENERATIVE RAG")
    print("=" * 55)

    full_error = build_full_error(record)

    print("\nGenerating AI troubleshooting advice...\n")

    answer = suggest_solution(full_error)

    print(answer)

    print(
        "\nNOTE: AI-generated commands and recommendations "
        "are not independently verified."
    )


def run_grounded(record):
    print("\n" + "=" * 55)
    print("GROUNDED RAG")
    print("=" * 55)

    result = generate_grounded_answer(record)

    if result["status"] != "FOUND":
        print("\nStatus:", result["status"])
        print("No source-backed recommendation was returned.")
        return

    print("\nSource:", result["source"])
    print("Selection method:", result["selection_method"])

    print("\nRecommendations from documentation:")

    count = 0

    for recommendation in result["recommendations"]:
        if not recommendation.get("exact_source_match", False):
            continue

        count += 1

        print(f"\n{count}. {recommendation['text']}")
        print("   Exact source match: True")

    if count == 0:
        print("No exact source quotations available.")

    print(
        "\nNOTE: Exact quotation does not establish "
        "technical correctness or relevance."
    )


def main():
    try:
        records = load_errors()
    except (OSError, ValueError) as error:
        print("Could not load error report:", error)
        return

    if not records:
        print("No errors available for analysis.")
        return

    while True:
        record = select_error(records)

        if record is None:
            print("\nGoodbye!")
            break

        print("\nSelected error:", record["message"])

        mode = select_mode()

        try:
            if mode in {"1", "3"}:
                run_generative(record)

            if mode in {"2", "3"}:
                run_grounded(record)

        except Exception as error:
            print(
                "\nProcessing failed:",
                type(error).__name__,
                str(error)
            )

        again = input(
            "\nAnalyze another error? (y/n): "
        ).strip().lower()

        if again != "y":
            print("\nSession completed.")
            break


if __name__ == "__main__":
    main()
