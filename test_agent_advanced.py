
from agent import run_agent


test_cases = [
    {
        "name": "MULTI-TOOL REQUEST",
        "question": (
            "How many errors are in my saved report? "
            "Also, explain how to troubleshoot "
            "a PostgreSQL connection failure."
        )
    },
    {
        "name": "MISSING DATA",
        "question": (
            "What percentage of all requests failed "
            "yesterday? Use my saved report and "
            "do not invent missing information."
        )
    },
    {
        "name": "UNKNOWN DOCUMENTATION",
        "question": (
            "According to our local technical "
            "documentation, how do I fix a "
            "Kubernetes certificate expiration error?"
        )
    }
]


def test_agent():
    for index, test in enumerate(test_cases, start=1):

        print("\n" + "=" * 50)
        print(f"TEST {index}: {test['name']}")
        print("=" * 50)

        print("\nQUESTION:")
        print(test["question"])

        print("\nAGENT EXECUTION:")

        try:
            answer = run_agent(test["question"])

            print("\nAGENT ANSWER:")
            print(answer)

        except Exception as error:
            print("Agent failed:", error)


if __name__ == "__main__":
    test_agent()
