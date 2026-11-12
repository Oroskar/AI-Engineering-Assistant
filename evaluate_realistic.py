
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "analysis_results_realistic.json"
OUTPUT_FILE = BASE_DIR / "evaluation_results_realistic.json"

TEST_CASES = {
    "failed to connect to postgresql": {
        "expected_category": "DATABASE",
        "required_concepts": [
            r"\b(postgresql|database)\b",
            r"\b(hostname|host|port|connection settings)\b",
            r"\b(server|service|running|status)\b"
        ]
    },
    "cuda out of memory while running inference": {
        "expected_category": "MEMORY",
        "required_concepts": [
            r"\b(cuda|gpu|vram)\b",
            r"\b(memory|allocation)\b",
            r"\b(batch|input size|quantization|model)\b"
        ]
    },
    "request timeout after 30 seconds": {
        "expected_category": "TIMEOUT",
        "required_concepts": [
            r"\b(timeout|timed out)\b",
            r"\b(server|service|response|network)\b",
            r"\b(check|monitor|measure|investigate|verify)\b"
        ]
    }
}

UNCERTAINTY_PATTERN = (
    r"\b(possible|possibly|may|might|could|"
    r"if|additional information|missing information|"
    r"cannot determine|not enough information)\b"
)


def detect_command_issues(answer):
    issues = []

    if re.search(r"\bpg_log\b", answer, re.IGNORECASE):
        issues.append(
            "Non-standard PostgreSQL command: pg_log"
        )

    if re.search(r"\bpg_ctl\s+status\b", answer, re.IGNORECASE):
        if not re.search(r"\bPGDATA\b|pg_ctl\s+(?:-D|--pgdata)", answer):
            issues.append(
                "pg_ctl status may require -D or PGDATA"
            )

    if re.search(r"\bufw\b", answer, re.IGNORECASE):
        issues.append(
            "ufw is Linux-specific; verify the server OS"
        )

    if re.search(r"\bnvidia-smi\b", answer, re.IGNORECASE):
        issues.append(
            "nvidia-smi requires a compatible NVIDIA environment"
        )

    return issues


def evaluate_answer(record):
    message = record["message"].strip().lower()
    answer = str(record["suggested_solution"])
    category = record["category"]

    case = TEST_CASES.get(message)

    if case is None:
        return None

    category_correct = (
        category == case["expected_category"]
    )

    relevance_checks = [
        bool(re.search(pattern, answer, re.IGNORECASE))
        for pattern in case["required_concepts"]
    ]

    relevant = all(relevance_checks)

    expresses_uncertainty = bool(
        re.search(UNCERTAINTY_PATTERN, answer, re.IGNORECASE)
    )

    command_issues = detect_command_issues(answer)

    has_known_invalid_command = any(
        "Non-standard" in issue
        for issue in command_issues
    )

    checks = {
        "category_correct": category_correct,
        "relevant_concepts_present": relevant,
        "expresses_uncertainty": expresses_uncertainty,
        "no_known_invalid_command": not has_known_invalid_command
    }

    score = sum(checks.values())

    return {
        "error": record["message"],
        "category": category,
        "score": score,
        "max_score": 4,
        "checks": checks,
        "warnings": command_issues,
        "groundedness": "NOT_EVALUATED"
    }


def run_evaluation():
    if not INPUT_FILE.exists():
        print("Missing analysis file:", INPUT_FILE)
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    results = []

    print("\n" + "=" * 55)
    print("AI ENGINEERING ASSISTANT - EVALUATION")
    print("=" * 55)

    for record in records:
        result = evaluate_answer(record)

        if result is None:
            print("\nSKIPPED:", record.get("message"))
            continue

        results.append(result)

        print("\nERROR:", result["error"])
        print("CATEGORY:", result["category"])
        print("HEURISTIC SCORE:", result["score"], "/ 4")

        for check, passed in result["checks"].items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check}")

        for warning in result["warnings"]:
            print("  [WARNING]", warning)

        print("  Groundedness: NOT EVALUATED")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    print("Evaluated answers:", len(results))

    if results:
        average = sum(
            result["score"] for result in results
        ) / len(results)

        print("Average heuristic score:", round(average, 2), "/ 4")

    print("Groundedness verification: NOT IMPLEMENTED")
    print("Results saved to:", OUTPUT_FILE.name)


if __name__ == "__main__":
    run_evaluation()
