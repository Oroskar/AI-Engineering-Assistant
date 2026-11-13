
import re
from pathlib import Path

from log_parser import (
    ERROR_LEVELS,
    parse_multiline_log
)


BASE_DIR = Path(__file__).parent

TOPIC_PATTERNS = [
    (
        r"\bpostgres(?:ql)?\b",
        r"\bpostgres(?:ql)?\b"
    ),
    (
        r"\bcuda\b",
        r"\bcuda\b"
    ),
    (
        r"\bdatabase\b",
        r"\b(database|postgresql|postgres|sql)\b"
    ),
    (
        r"\b(memory|gpu)\b",
        r"\b(memory|gpu|cuda)\b"
    ),
    (
        r"\b(timeout|timed out)\b",
        r"\b(timeout|timed out)\b"
    ),
    (
        r"\b(disk|storage)\b",
        r"\b(disk|storage|space left)\b"
    ),
    (
        r"\b(dns|hostname|network)\b",
        r"\b(dns|hostname|network|resolve)\b"
    )
]


def is_error(event):
    return (
        str(event.get("level", "")).upper()
        in ERROR_LEVELS
    )


def format_event(event):
    date = event.get("date", "")
    time = event.get("time", "")
    level = event.get("level", "UNKNOWN")
    message = event.get("message", "")

    return (
        f"[{date} {time}] "
        f"{level}: {message}"
    )


def has_stack_trace(event):
    details = event.get("details", [])

    if not isinstance(details, list):
        return False

    trace_pattern = re.compile(
        r"traceback \(most recent call last\)"
        r'|^\s*File ".*", line \d+'
        r"|^\s*at\s+\S+",
        re.IGNORECASE
    )

    return any(
        trace_pattern.search(str(line))
        for line in details
    )


def first_error(events):
    for event in events:
        if is_error(event):
            return (
                "First error in log file order:\n"
                + format_event(event)
                + "\n\nNote: File order is not "
                "necessarily chronological order."
            )

    return "No error events found."


def last_error(events):
    for event in reversed(events):
        if is_error(event):
            return (
                "Last error in log file order:\n"
                + format_event(event)
            )

    return "No error events found."


def errors_with_stack_traces(events):
    matches = [
        event
        for event in events
        if is_error(event)
        and has_stack_trace(event)
    ]

    if not matches:
        return "No errors with detected stack traces."

    lines = [
        f"Errors with stack traces: {len(matches)}"
    ]

    for index, event in enumerate(
        matches[:20], start=1
    ):
        lines.append(
            f"{index}. {format_event(event)}"
        )

    if len(matches) > 20:
        lines.append(
            f"...and {len(matches) - 20} more."
        )

    return "\n".join(lines)


def find_target_pattern(question):
    for question_pattern, event_pattern in TOPIC_PATTERNS:
        if re.search(
            question_pattern,
            question,
            re.IGNORECASE
        ):
            return event_pattern

    return None


def event_before_target(question, events):
    target_pattern = find_target_pattern(question)

    if target_pattern is None:
        return (
            "Please specify which error to inspect, "
            "for example: 'What happened before "
            "the PostgreSQL failure?'"
        )

    matches = []

    for index, event in enumerate(events):
        if not is_error(event):
            continue

        searchable_text = "\n".join([
            str(event.get("message", "")),
            *[
                str(detail)
                for detail in event.get("details", [])
            ]
        ])

        if re.search(
            target_pattern,
            searchable_text,
            re.IGNORECASE
        ):
            matches.append(index)

    if not matches:
        return (
            "No matching error was found "
            "in the current log."
        )

    if len(matches) > 1:
        return (
            f"Found {len(matches)} matching errors. "
            "Please specify a more precise target."
        )

    target_index = matches[0]
    target_event = events[target_index]

    if target_index == 0:
        return (
            "The matching error is the first "
            "event in the file. "
            "No earlier log event is available."
        )

    previous_event = events[target_index - 1]

    return (
        "Target error:\n"
        + format_event(target_event)
        + "\n\nImmediately preceding log event:\n"
        + format_event(previous_event)
        + "\n\nNote: This is the previous event "
        "in file order, not a proven cause."
    )


def answer_event_question(question, events):
    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    if not isinstance(events, list):
        raise TypeError("Events must be a list.")

    normalized = question.lower().strip()

    if not normalized:
        return "Please enter a question."

    if not events:
        return "No log events are available."

    if re.search(
        r"\b(stack traces?|tracebacks?|stacktrace)\b",
        normalized
    ):
        return errors_with_stack_traces(events)

    if re.search(r"\bbefore\b", normalized):
        return event_before_target(question, events)

    if (
        re.search(r"\b(first|earliest)\b", normalized)
        and re.search(r"\berrors?\b", normalized)
    ):
        return first_error(events)

    if (
        re.search(r"\b(last|latest)\b", normalized)
        and re.search(r"\berrors?\b", normalized)
    ):
        return last_error(events)

    return (
        "This event question is not supported yet."
    )


def main():
    log_path = BASE_DIR / "sample" / "realistic.log"

    events = parse_multiline_log(log_path)

    questions = [
        "Which error happened first?",
        "Which errors have stack traces?",
        "What happened before the PostgreSQL failure?"
    ]

    print("\n" + "=" * 55)
    print("EVENT INVESTIGATION TEST")
    print("=" * 55)

    for question in questions:
        print("\nQUESTION:", question)
        print(
            answer_event_question(question, events)
        )
        print("-" * 55)


if __name__ == "__main__":
    main()
