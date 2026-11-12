
import sys

from ai_classifier import classify_with_ai


TEST_CASES = [
    {
        "name": "PostgreSQL connection",
        "error": (
            "psycopg2.OperationalError: "
            "could not connect to PostgreSQL server"
        ),
        "expected": "DATABASE"
    },
    {
        "name": "SQL query error",
        "error": (
            "sqlite3.OperationalError: "
            "database is locked"
        ),
        "expected": "DATABASE"
    },
    {
        "name": "HTTP timeout",
        "error": (
            "HTTP 504 Gateway Timeout "
            "while calling upstream API"
        ),
        "expected": "TIMEOUT"
    },
    {
        "name": "GPU out of memory",
        "error": (
            "torch.cuda.OutOfMemoryError: "
            "CUDA out of memory"
        ),
        "expected": "MEMORY"
    },
    {
        "name": "DNS lookup failure",
        "error": (
            "DNS lookup failed for "
            "api.production.internal"
        ),
        "expected": "NETWORK"
    },
    {
        "name": "Network unreachable",
        "error": (
            "OSError: [Errno 101] "
            "Network is unreachable"
        ),
        "expected": "NETWORK"
    },
    {
        "name": "Disk full",
        "error": (
            "OSError: [Errno 28] "
            "No space left on device"
        ),
        "expected": "STORAGE"
    },
    {
        "name": "Storage ENOSPC",
        "error": (
            "ENOSPC while writing "
            "data to storage volume"
        ),
        "expected": "STORAGE"
    },
    {
        "name": "Missing file",
        "error": (
            "FileNotFoundError: "
            "model_weights.bin not found"
        ),
        "expected": "UNKNOWN"
    },
    {
        "name": "Permission denied",
        "error": (
            "PermissionError: [Errno 13] "
            "Permission denied"
        ),
        "expected": "UNKNOWN"
    }
]


def run_regression_tests():
    passed = 0
    failed = 0

    print("\n" + "=" * 55)
    print("AI CLASSIFICATION REGRESSION TESTS")
    print("=" * 55)

    for index, test in enumerate(TEST_CASES, start=1):
        print(f"\nTEST {index}: {test['name']}")

        try:
            result = classify_with_ai(test["error"])
            predicted = result.get("category", "UNKNOWN")

            if predicted == test["expected"]:
                passed += 1
                print("RESULT: PASS")
            else:
                failed += 1
                print("RESULT: FAIL")

            print("EXPECTED:", test["expected"])
            print("PREDICTED:", predicted)
            print(
                "EXPLANATION:",
                result.get("explanation", "")
            )

        except Exception as error:
            failed += 1
            print("RESULT: ERROR")
            print("DETAILS:", str(error))

    total = passed + failed
    accuracy = passed / total * 100 if total else 0

    print("\n" + "=" * 55)
    print("REGRESSION TEST SUMMARY")
    print("=" * 55)

    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print(f"Accuracy on this test set: {accuracy:.1f}%")

    if failed == 0:
        print("STATUS: ALL TESTS PASSED")
    else:
        print("STATUS: SOME TESTS FAILED")

    return failed


if __name__ == "__main__":
    failures = run_regression_tests()
    sys.exit(1 if failures else 0)
