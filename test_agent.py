
import io
from contextlib import redirect_stdout

from agent import run_agent


test_cases = [
    {
        "question": "How can I troubleshoot a PostgreSQL connection failure?",
        "expected_tool": "search_technical_docs"
    },
    {
        "question": "What should I check when a request times out?",
        "expected_tool": "search_technical_docs"
    },
    {
        "question": "How many errors were detected in my saved report?",
        "expected_tool": "read_error_summary"
    },
    {
        "question": "Show me the number of errors in each category.",
        "expected_tool": "read_error_summary"
    }
]


def evaluate_agent():
    correct = 0

    for index, test in enumerate(test_cases, start=1):
        question = test["question"]
        expected = test["expected_tool"]

        # Capture the tool selected by the agent
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            answer = run_agent(question)

        output = captured_output.getvalue()

        selected_tool = "NONE"

        for line in output.splitlines():
            if line.startswith("Tool selected:"):
                selected_tool = line.split(":", 1)[1].strip()
                break

        is_correct = selected_tool == expected

        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"

        print(f"\nTEST {index}: {status}")
        print("Question:", question)
        print("Expected tool:", expected)
        print("Selected tool:", selected_tool)
        print("Agent answer:", answer)

    total = len(test_cases)
    accuracy = correct / total if total else 0

    print("\n========== AGENT EVALUATION ==========")
    print(f"Tool Selection Accuracy: {accuracy:.2%}")
    print(f"Correct selections: {correct}/{total}")


if __name__ == "__main__":
    evaluate_agent()
