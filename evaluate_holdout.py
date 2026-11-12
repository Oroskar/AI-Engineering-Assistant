
from main import classify_error
from ai_classifier import classify_with_ai


test_cases = [
    ("MongoDB server selection failed", "DATABASE"),
    ("PostgreSQL authentication failed", "DATABASE"),
    ("Database schema migration failed", "DATABASE"),

    ("Gateway request exceeded deadline", "TIMEOUT"),
    ("Worker exceeded execution time limit", "TIMEOUT"),

    ("GPU failed to allocate VRAM", "MEMORY"),
    ("Process exceeded RAM limit", "MEMORY"),
    ("Memory usage exceeded threshold", "MEMORY"),

    ("DNS lookup failed for service.local", "NETWORK"),
    ("TCP handshake failed with remote server", "NETWORK"),

    ("No space left on device", "UNKNOWN"),
    ("Permission denied when creating file", "UNKNOWN")
]


correct = 0
rule_correct = 0
rule_total = 0
ai_correct = 0
ai_total = 0

for message, expected in test_cases:

    predicted = classify_error(message)
    source = "RULE_BASED"

    if predicted == "UNKNOWN":
        result = classify_with_ai(message)
        predicted = result["category"]
        source = "AI"

    is_correct = predicted == expected

    if is_correct:
        correct += 1

    if source == "RULE_BASED":
        rule_total += 1
        rule_correct += int(is_correct)
    else:
        ai_total += 1
        ai_correct += int(is_correct)

    status = "PASS" if is_correct else "FAIL"

    print(f"\n{status}: {message}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"Source: {source}")


total = len(test_cases)

print("\n========== HOLDOUT EVALUATION ==========")
print(f"Overall accuracy: {correct / total:.2%}")
print(f"Correct predictions: {correct}/{total}")

if rule_total:
    print(f"Rule-based accuracy: {rule_correct / rule_total:.2%}")

if ai_total:
    print(f"AI accuracy: {ai_correct / ai_total:.2%}")
