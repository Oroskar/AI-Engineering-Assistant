
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

LOG_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<time>\d{2}:\d{2}:\d{2}) "
    r"(?P<level>INFO|WARNING|ERROR|DEBUG|CRITICAL) "
    r"(?P<message>.+)$"
)


def parse_multiline_log(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\r\n")

            if not line.strip():
                continue

            match = LOG_PATTERN.match(line)

            if match:
                record = match.groupdict()
                record["details"] = []
                records.append(record)

            elif records:
                records[-1]["details"].append(line)

    return records


def analyze_complex_logs(file_path):
    records = parse_multiline_log(file_path)

    errors = [
        record for record in records
        if record["level"] == "ERROR"
    ]

    for error in errors:
        print("\n" + "=" * 50)
        print("ERROR:", error["message"])
        print("Time:", error["time"])

        if error["details"]:
            print("\nStack Trace / Additional Details:")

            for detail in error["details"]:
                print(detail)

    print("\n" + "=" * 50)
    print("Total log events:", len(records))
    print("Total errors:", len(errors))


if __name__ == "__main__":
    log_path = BASE_DIR / "sample" / "realistic.log"
    analyze_complex_logs(log_path)
