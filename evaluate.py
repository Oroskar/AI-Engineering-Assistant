
from main import classify_error
from ai_classifier import classify_with_ai


# New test cases - not used during development
test_cases = [
    ("MySQL authentication rejected for user app", "DATABASE"),
    ("Database connection pool exhausted", "DATABASE"),
    ("Request timeout after 30 seconds", "TIMEOUT"),
    ("HTTP call exceeded its time limit", "TIMEOUT"),
    ("Memory allocation failed while loading model", "MEMORY"),
    ("CUDA out of memory during inference", "MEMORY"),
    ("DNS resolution failed for api.example.com", "NETWORK"),
    ("Connection lost to remote API gateway", "NETWORK"),
    ("Disk is full: cannot write output file", "UNKNOWN"),
    ("Invalid JSON syntax in config file", "UNKNOWN")
]


total = len(test_cases)
correct = 0

rule_total = 0
rule_correct = 0

ai_total = 0
ai_correct = 0


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


print("\n========== EVALUATION RESULTS ==========")

print(f"Overall accuracy: {correct / total:.2%}")
print(f"Correct predictions: {correct}/{total}")

if rule_total:
    print(f"Rule-based accuracy: {rule_correct / rule_total:.2%}")

if ai_total:
    print(f"AI accuracy: {ai_correct / ai_total:.2%}")
