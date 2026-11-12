
from ai_classifier import classify_with_ai


TEST_CASES = [
    (
        "DNS NXDOMAIN",
        "DNS lookup returned NXDOMAIN for payments.internal",
        "NETWORK"
    ),
    (
        "Socket failure",
        "socket.gaierror: nodename nor servname provided, or not known",
        "NETWORK"
    ),
    (
        "Redis authentication",
        "Redis authentication failed: invalid credentials",
        "DATABASE"
    ),
    (
        "PostgreSQL permissions",
        "psycopg2.errors.InsufficientPrivilege: permission denied for table orders",
        "DATABASE"
    ),
    (
        "GPU allocation",
        "RuntimeError: Failed to allocate 2048 MiB on CUDA device",
        "MEMORY"
    ),
    (
        "Python memory",
        "MemoryError: Unable to allocate array",
        "MEMORY"
    ),
    (
        "Storage quota",
        "Storage quota exceeded while saving upload",
        "STORAGE"
    ),
    (
        "Disk capacity",
        "Write failed: filesystem has no free blocks",
        "STORAGE"
    ),
    (
        "Upstream deadline",
        "Upstream request exceeded its 45 second deadline",
        "TIMEOUT"
    ),
    (
        "Generic file permissions",
        "PermissionError: Cannot open config.yaml",
        "UNKNOWN"
    ),
    (
        "Invalid value",
        "ValueError: invalid literal for int() with base 10",
        "UNKNOWN"
    ),
    (
        "Generic application failure",
        "Unexpected error occurred while processing task",
        "UNKNOWN"
    )
]


def main():
    passed = 0
    failures = []

    print("=" * 55)
    print("UNSEEN CLASSIFICATION EVALUATION")
    print("=" * 55)

    for index, (name, message, expected) in enumerate(
        TEST_CASES, start=1
    ):
        try:
            result = classify_with_ai(message)
            predicted = result["category"]

            success = predicted == expected

            if success:
                passed += 1
            else:
                failures.append({
                    "test": name,
                    "expected": expected,
                    "predicted": predicted
                })

            status = "PASS" if success else "FAIL"

            print(f"\nTEST {index}: {name}")
            print("STATUS:", status)
            print("EXPECTED:", expected)
            print("PREDICTED:", predicted)

        except Exception as error:
            failures.append({
                "test": name,
                "expected": expected,
                "error": str(error)
            })

            print(f"\nTEST {index}: {name}")
            print("STATUS: ERROR")
            print("DETAILS:", error)

    total = len(TEST_CASES)

    print("\n" + "=" * 55)
    print("UNSEEN EVALUATION SUMMARY")
    print("=" * 55)

    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", len(failures))
    print(
        "Accuracy on this test set:",
        f"{passed / total * 100:.1f}%"
    )

    if failures:
        print("\nFAILED CASES:")

        for failure in failures:
            print(failure)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
