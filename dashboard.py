
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import streamlit as st

from ai_classifier import classify_with_ai
from evidence_aware import analyze_error


BASE_DIR = Path(__file__).parent

DEMO_REPORT = BASE_DIR / "analysis_results_realistic.json"
SAVED_ANALYSIS = BASE_DIR / "evidence_aware_results.json"

MAX_UPLOAD_BYTES = 2 * 1024 * 1024

ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}

VALID_CATEGORIES = {
    "DATABASE",
    "MEMORY",
    "TIMEOUT",
    "NETWORK",
    "STORAGE",
    "UNKNOWN"
}

LOG_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T]"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:[.,]\d{1,6})?"
    r"(?:Z|[+-]\d{2}:?\d{2})?\s+"
    r"\[?(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\]?"
    r"\s*(?:[-:]\s*)?(?P<message>.*)$",
    re.IGNORECASE
)


st.set_page_config(
    page_title="AI Engineering Assistant",
    page_icon="🛠️",
    layout="wide"
)


def load_json(path):
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list")

    return data


def classify_with_rules(text):
    text = text.lower()

    storage_terms = (
        "no space left on device",
        "no space left",
        "disk full",
        "disk is full",
        "insufficient disk space",
        "file system is full",
        "filesystem full",
        "storage exhausted",
        "enospc"
    )

    if any(term in text for term in storage_terms):
        return "STORAGE"

    memory_terms = (
        "out of memory",
        "memoryerror",
        "cuda",
        "memory allocation failed",
        "memory allocation error"
    )

    if any(term in text for term in memory_terms):
        return "MEMORY"

    timeout_terms = (
        "timeout",
        "timed out",
        "deadline exceeded"
    )

    if any(term in text for term in timeout_terms):
        return "TIMEOUT"

    if re.search(
        r"\b(database|postgresql|postgres|"
        r"mysql|mongodb|sqlite|sql)\b",
        text
    ):
        return "DATABASE"

    network_terms = (
        "dns",
        "unable to resolve hostname",
        "name or service not known",
        "hostname not found",
        "host not found",
        "network unreachable",
        "connection refused",
        "connection reset"
    )

    if any(term in text for term in network_terms):
        return "NETWORK"

    return "UNKNOWN"


def parse_uploaded_text(text):
    events = []
    ignored_prefix_lines = 0

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            continue

        match = LOG_PATTERN.match(line)

        if match:
            record = match.groupdict()

            record["level"] = record["level"].upper()
            record["stack_trace"] = []

            events.append(record)

        elif events:
            events[-1]["stack_trace"].append(line)

        else:
            ignored_prefix_lines += 1

    errors = []

    for event in events:
        if event["level"] not in ERROR_LEVELS:
            continue

        full_error = "\n".join(
            [event["message"]]
            + event["stack_trace"]
        )

        category = classify_with_rules(full_error)

        errors.append({
            "date": event["date"],
            "time": event["time"],
            "level": event["level"],
            "message": event["message"],
            "stack_trace": event["stack_trace"],
            "category": category,
            "classification_source": (
                "RULE_BASED"
                if category != "UNKNOWN"
                else "NOT_YET_CLASSIFIED"
            )
        })

    return {
        "events": events,
        "errors": errors,
        "ignored_prefix_lines": ignored_prefix_lines
    }


def build_full_error(record):
    details = record.get("stack_trace", [])

    if not isinstance(details, list):
        details = []

    return "\n".join(
        [str(record["message"])]
        + [str(line) for line in details]
    )


def analyze_selected_record(record):
    updated_record = dict(record)

    category = updated_record.get(
        "category", "UNKNOWN"
    )

    classification_source = updated_record.get(
        "classification_source", "UNKNOWN"
    )

    if category == "UNKNOWN":
        prediction = classify_with_ai(
            build_full_error(updated_record)
        )

        predicted_category = str(
            prediction.get("category", "UNKNOWN")
        ).upper().strip()

        if predicted_category in VALID_CATEGORIES:
            category = predicted_category

        updated_record["category"] = category
        classification_source = "AI"

    updated_record[
        "classification_source"
    ] = classification_source

    result = analyze_error(updated_record)

    result["category"] = category
    result["classification_source"] = classification_source

    return result


def show_log_explorer(records, analysis_cache):
    with st.expander(
        "Log Explorer - Detected Errors"
    ):
        rows = []

        for index, record in enumerate(records):
            cached = analysis_cache.get(index, {})

            rows.append({
                "ID": index + 1,
                "Time": record.get("time", ""),
                "Level": record.get("level", "ERROR"),
                "Category": cached.get(
                    "category",
                    record.get("category", "UNKNOWN")
                ),
                "Message": record.get("message", "")
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )


def show_analysis(result):
    left, right = st.columns(2, gap="large")

    documentation = result.get(
        "documentation", {}
    )

    ai_analysis = result.get(
        "ai_analysis", {}
    )

    with left:
        st.subheader("Documented Recommendations")

        source = documentation.get("source")

        st.caption(
            f"Source: {source or 'No matching documentation'}"
        )

        recommendations = documentation.get(
            "exact_source_quotes", []
        )

        if recommendations:
            for index, recommendation in enumerate(
                recommendations, start=1
            ):
                st.markdown(
                    f"**{index}.** {recommendation}"
                )
        else:
            st.info(
                "No exact source quotations available."
            )

        st.caption(
            "These quotes come from local sample "
            "documentation. They are not independently "
            "verified for technical accuracy."
        )

    with right:
        st.subheader("AI Hypotheses")

        st.warning(
            "Possible causes - not confirmed diagnoses."
        )

        hypotheses = ai_analysis.get(
            "hypotheses", []
        )

        if hypotheses:
            for hypothesis in hypotheses:
                st.markdown(f"- {hypothesis}")
        else:
            st.info("No hypotheses available.")

        st.divider()

        st.subheader("Missing Information")

        missing = ai_analysis.get(
            "missing_information", []
        )

        if missing:
            for item in missing:
                st.markdown(f"- {item}")
        else:
            st.info("No missing information identified.")


def main():
    st.sidebar.title("AI Engineering Assistant")

    data_source = st.sidebar.radio(
        "Data Source",
        ["Demo Data", "Upload Log File"]
    )

    st.sidebar.divider()

    saved_results = []

    if data_source == "Demo Data":
        records = load_json(DEMO_REPORT)
        saved_results = load_json(SAVED_ANALYSIS)

        source_key = "demo"
        total_events = 6

        st.info("Using synthetic demo logs.")

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a log file",
            type=["log", "txt"]
        )

        st.sidebar.caption(
            "Maximum file size: 2 MB."
        )

        if uploaded_file is None:
            st.title("Upload Logs")

            st.info(
                "Choose a .log or .txt file in the sidebar."
            )
            st.stop()

        file_bytes = uploaded_file.getvalue()

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            st.error("File exceeds the 2 MB limit.")
            st.stop()

        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("The file must be UTF-8 encoded.")
            st.stop()

        parsed = parse_uploaded_text(text)

        records = parsed["errors"]
        total_events = len(parsed["events"])

        source_key = (
            "upload:"
            + hashlib.sha256(file_bytes).hexdigest()
        )

        st.success(
            f"Loaded file: {uploaded_file.name}"
        )

        if parsed["ignored_prefix_lines"]:
            st.warning(
                "Some lines before the first recognized "
                "log event were ignored."
            )

        if total_events == 0:
            st.error(
                "No recognized log events were found."
            )
            st.stop()

    if st.session_state.get("active_source") != source_key:
        st.session_state["active_source"] = source_key
        st.session_state["analysis_cache"] = {}

    analysis_cache = st.session_state["analysis_cache"]

    def effective_category(index, record):
        return analysis_cache.get(index, {}).get(
            "category",
            record.get("category", "UNKNOWN")
        )

    category_counts = Counter(
        effective_category(index, record)
        for index, record in enumerate(records)
    )

    st.title("🛠️ AI Engineering Assistant")

    st.caption(
        "Local Log Analysis | Llama 3.2 | "
        "RAG | Evidence-Aware AI"
    )

    st.divider()

    overview = st.columns(2)

    overview[0].metric(
        "Total Log Events", total_events
    )

    overview[1].metric(
        "Detected Errors", len(records)
    )

    st.markdown("#### Error Categories")

    categories = [
        "DATABASE",
        "NETWORK",
        "STORAGE",
        "MEMORY",
        "TIMEOUT"
    ]

    metric_columns = st.columns(5)

    for column, category in zip(
        metric_columns, categories
    ):
        column.metric(
            category.title(),
            category_counts.get(category, 0)
        )

    if category_counts.get("UNKNOWN", 0):
        st.caption(
            f"Unknown errors: "
            f"{category_counts['UNKNOWN']}"
        )

    st.divider()

    if not records:
        st.info(
            "No ERROR, CRITICAL or FATAL events detected."
        )
        return

    show_log_explorer(
        records,
        analysis_cache
    )

    st.subheader("Select an Error")

    selected_index = st.selectbox(
        "Error Event",
        options=list(range(len(records))),
        format_func=lambda index: (
            f"[{effective_category(index, records[index])}] "
            f"{records[index].get('message', '')[:110]}"
        )
    )

    record = records[selected_index]

    fresh_result = analysis_cache.get(
        selected_index
    )

    displayed_category = effective_category(
        selected_index,
        record
    )

    displayed_source = (
        fresh_result.get("classification_source", "AI")
        if fresh_result is not None
        else record.get(
            "classification_source", "SAVED"
        )
    )

    st.markdown(
        f"**Selected Error:** {record['message']}"
    )

    st.caption(
        f"Category: {displayed_category} | "
        f"Classification: {displayed_source}"
    )

    with st.expander("Raw Error / Stack Trace"):
        st.code(
            build_full_error(record),
            language="text"
        )

    st.divider()

    run_analysis = st.button(
        "🤖 Analyze Selected Error",
        type="primary"
    )

    if run_analysis:
        success = False

        with st.spinner("Running local AI analysis..."):
            try:
                result = analyze_selected_record(record)

                analysis_cache[selected_index] = result
                success = True

            except Exception as error:
                st.error(
                    f"Analysis failed: "
                    f"{type(error).__name__}: {error}"
                )

        if success:
            st.rerun()

    result = analysis_cache.get(
        selected_index
    )

    if result is not None:
        st.caption("Fresh AI analysis")

    elif data_source == "Demo Data":
        result = next(
            (
                item for item in saved_results
                if item.get("error") == record["message"]
            ),
            None
        )

        if result is not None:
            st.caption("Previously saved analysis")

    if result is None:
        st.info(
            "Click Analyze Selected Error "
            "to generate recommendations."
        )
        return

    st.divider()

    show_analysis(result)

    st.divider()

    st.download_button(
        "Download Analysis JSON",
        data=json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        ),
        file_name="ai_analysis.json",
        mime="application/json"
    )

    st.caption(
        "Local prototype with synthetic test data. "
        "Uploaded logs are processed in memory. "
        "AI results require verification."
    )


if __name__ == "__main__":
    main()
