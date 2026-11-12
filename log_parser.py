
import re
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).parent

ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>"
    r"(?P<date>\d{4}-\d{2}-\d{2})[ T]"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:[.,]\d{1,6})?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
    r")"
    r"\s+\[?"
    r"(?P<level>TRACE|DEBUG|INFO|WARNING|WARN|"
    r"ERROR|CRITICAL|FATAL)"
    r"\]?"
    r"(?:\s*[-:]\s*|\s+)"
    r"(?P<message>\S.*)$",
    re.IGNORECASE
)

# Detect lines that appear to begin with a timestamp,
# even when the timestamp or log level is malformed.
HEADER_CANDIDATE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]"
)


def is_valid_timestamp(timestamp):
    normalized = timestamp.replace(" ", "T", 1)
    normalized = normalized.replace(",", ".")

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        datetime.fromisoformat(normalized)
        return True

    except ValueError:
        return False


def _parse_text(text):
    if not isinstance(text, str):
        raise TypeError("Log content must be text")

    records = []
    ignored_prefix_lines = 0
    malformed_header_lines = 0

    current_record = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")

        if not line.strip():
            continue

        match = LOG_PATTERN.match(line)

        if match:
            timestamp = match.group("timestamp")

            if not is_valid_timestamp(timestamp):
                malformed_header_lines += 1
                current_record = None

                if not records:
                    ignored_prefix_lines += 1

                continue

            level = match.group("level").upper()

            if level == "WARN":
                level = "WARNING"

            record = {
                "date": match.group("date"),
                "time": match.group("time"),
                "level": level,
                "message": match.group("message"),
                "details": []
            }

            records.append(record)
            current_record = record

        elif HEADER_CANDIDATE.match(line):
            # A malformed log header must not become
            # part of the preceding stack trace.
            malformed_header_lines += 1
            current_record = None

            if not records:
                ignored_prefix_lines += 1

        elif current_record is not None:
            # Normal continuation line, such as traceback.
            current_record["details"].append(line)

        elif not records:
            ignored_prefix_lines += 1

    stats = {
        "records": records,
        "ignored_prefix_lines": ignored_prefix_lines,
        "malformed_header_lines": malformed_header_lines
    }

    return stats


def parse_log_text(text):
    result = _parse_text(text)
    return result["records"]


def parse_log_text_with_stats(text):
    return _parse_text(text)


def parse_multiline_log(file_path):
    path = Path(file_path)

    text = path.read_text(encoding="utf-8-sig")

    return parse_log_text(text)


def analyze_complex_logs(file_path):
    records = parse_multiline_log(file_path)

    errors = [
        record
        for record in records
        if record["level"] in ERROR_LEVELS
    ]

    for error in errors:
        print("\n" + "=" * 50)

        print("LEVEL:", error["level"])
        print("ERROR:", error["message"])
        print("DATE:", error["date"])
        print("TIME:", error["time"])

        if error["details"]:
            print("\nStack Trace / Additional Details:")

            for detail in error["details"]:
                print(detail)

    print("\n" + "=" * 50)
    print("LOG PARSER SUMMARY")
    print("=" * 50)

    print("Total log events:", len(records))
    print("Total errors:", len(errors))

    return records


if __name__ == "__main__":
    log_path = BASE_DIR / "sample" / "realistic.log"
    analyze_complex_logs(log_path)
