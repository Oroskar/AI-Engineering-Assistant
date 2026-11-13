
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import streamlit as st

from ai_classifier import classify_with_ai
from evidence_aware import analyze_error
from event_queries import answer_event_question
from log_parser import ERROR_LEVELS, parse_log_text_with_stats
from session_agent import run_session_agent


BASE_DIR = Path(__file__).parent

DEMO_LOG = BASE_DIR / "sample" / "realistic.log"
SAVED_ANALYSIS = BASE_DIR / "evidence_aware_results.json"

MAX_UPLOAD_BYTES = 2 * 1024 * 1024

VALID_CATEGORIES = {
    "DATABASE",
    "MEMORY",
    "TIMEOUT",
    "NETWORK",
    "STORAGE",
    "UNKNOWN"
}

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

    if any(term in text for term in (
        "no space left",
        "disk full",
        "insufficient disk space",
        "filesystem full",
        "file system is full",
        "enospc"
    )):
        return "STORAGE"

    if any(term in text for term in (
        "out of memory",
        "memoryerror",
        "cuda",
        "memory allocation failed"
    )):
        return "MEMORY"

    if any(term in text for term in (
        "timeout",
        "timed out",
        "deadline exceeded"
    )):
        return "TIMEOUT"

    if re.search(
        r"\b(database|postgresql|postgres|mysql|"
        r"mongodb|sqlite|sql|redis)\b",
        text
    ):
        return "DATABASE"

    if any(term in text for term in (
        "dns",
        "unable to resolve hostname",
        "name or service not known",
        "hostname not found",
        "network unreachable",
        "connection refused",
        "connection reset"
    )):
        return "NETWORK"

    return "UNKNOWN"


def build_error_records(events):
    errors = []

    for event in events:
        if event["level"] not in ERROR_LEVELS:
            continue

        full_error = "\n".join(
            [event["message"]] + event["details"]
        )

        category = classify_with_rules(full_error)

        errors.append({
            "date": event["date"],
            "time": event["time"],
            "level": event["level"],
            "message": event["message"],
            "stack_trace": list(event["details"]),
            "category": category,
            "classification_source": (
                "RULE_BASED"
                if category != "UNKNOWN"
                else "NOT_YET_CLASSIFIED"
            )
        })

    return errors


def full_error_text(record):
    return "\n".join(
        [str(record["message"])]
        + [
            str(line)
            for line in record.get("stack_trace", [])
        ]
    )


def run_analysis(record):
    updated = dict(record)

    category = updated["category"]
    source = updated["classification_source"]

    if category == "UNKNOWN":
        prediction = classify_with_ai(
            full_error_text(updated)
        )

        predicted = str(
            prediction.get("category", "UNKNOWN")
        ).upper().strip()

        if predicted in VALID_CATEGORIES:
            category = predicted

        source = "AI"

    updated["category"] = category
    updated["classification_source"] = source

    result = analyze_error(updated)

    result["category"] = category
    result["classification_source"] = source

    return result


def display_parser_health(parsed):
    st.subheader("Parsing Quality")

    malformed = parsed.get(
        "malformed_header_lines", 0
    )

    ignored = parsed.get(
        "ignored_prefix_lines", 0
    )

    columns = st.columns(3)

    columns[0].metric(
        "Parsed Events",
        len(parsed["records"])
    )

    columns[1].metric(
        "Malformed Headers",
        malformed
    )

    columns[2].metric(
        "Ignored Prefix Lines",
        ignored
    )

    if malformed:
        st.warning(
            f"{malformed} malformed header(s) were "
            "skipped. Analysis may be incomplete."
        )

    if ignored:
        st.warning(
            f"{ignored} prefix line(s) were ignored. "
            "This count may overlap with malformed headers."
        )

    if not malformed and not ignored:
        st.success(
            "No malformed timestamp-like headers or "
            "ignored prefix lines detected."
        )

    with st.expander("About Parsing Quality"):
        st.caption(
            "The parser supports defined timestamp and "
            "log-level formats. Unrecognized continuation "
            "lines may still be attached to an earlier event."
        )


def display_analysis(result):
    documentation = result.get(
        "documentation", {}
    )

    ai = result.get(
        "ai_analysis", {}
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Documented Recommendations")

        source = documentation.get("source")

        st.caption(
            "Source: "
            + str(source or "No matching documentation")
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
            "Quotes come from local sample documentation. "
            "Technical correctness is not independently verified."
        )

    with right:
        st.subheader("AI Hypotheses")

        st.warning(
            "Possible causes, not confirmed diagnoses."
        )

        hypotheses = ai.get("hypotheses", [])

        if hypotheses:
            for hypothesis in hypotheses:
                st.markdown(f"- {hypothesis}")
        else:
            st.info("No hypotheses generated.")

        st.divider()

        st.subheader("Missing Information")

        missing = ai.get(
            "missing_information", []
        )

        if missing:
            for item in missing:
                st.markdown(f"- {item}")
        else:
            st.info(
                "No missing information identified."
            )


def prepare_session_records(records, analysis_cache):
    session_records = []

    for index, record in enumerate(records):
        updated = dict(record)

        analysis = analysis_cache.get(index)

        if analysis is not None:
            updated["category"] = analysis.get(
                "category",
                record["category"]
            )

            updated["classification_source"] = (
                analysis.get(
                    "classification_source",
                    record["classification_source"]
                )
            )

        session_records.append(updated)

    return session_records


def is_event_question(question):
    """
    Route supported event questions to deterministic
    Python logic instead of the LLM session agent.
    """
    text = question.lower()

    if re.search(
        r"\b(stack\s*traces?|stacktraces?|tracebacks?)\b",
        text
    ):
        return True

    if re.search(r"\bbefore\b", text):
        return True

    has_position = bool(
        re.search(
            r"\b(first|earliest|last|latest)\b",
            text
        )
    )

    mentions_error = bool(
        re.search(r"\berrors?\b", text)
    )

    return has_position and mentions_error


def route_chat_question(question, records, events):
    """
    Event questions: deterministic Python.
    Summary and troubleshooting: session AI agent.
    """
    if is_event_question(question):
        return answer_event_question(
            question,
            events
        )

    return run_session_agent(
        question,
        records
    )


def display_log_chat(records, events, analysis_cache):
    st.subheader("💬 Ask Your Logs")

    st.caption(
        "Ask questions about the currently loaded log file. "
        "Event investigation uses original log events. "
        "Summary questions use the session agent."
    )

    with st.expander("Example Questions"):
        st.markdown(
            "- How many errors are in this file?\n"
            "- What is the most common error category?\n"
            "- How should I investigate the most common errors?\n"
            "- Which error happened first?\n"
            "- Which errors have stack traces?\n"
            "- What happened before the PostgreSQL failure?"
        )

    history = st.session_state["chat_history"]

    for message in history:
        with st.chat_message(message["role"]):
            st.text(message["content"])

    question = st.chat_input(
        "Ask a question about your logs...",
        key="log_chat_input"
    )

    if question:
        history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        session_records = prepare_session_records(
            records,
            analysis_cache
        )

        with st.chat_message("assistant"):
            with st.spinner("Investigating logs..."):
                try:
                    answer = route_chat_question(
                        question,
                        session_records,
                        events
                    )

                except Exception as error:
                    answer = (
                        "Unable to answer this question. "
                        f"{type(error).__name__}: {error}"
                    )

            st.text(answer)

        history.append({
            "role": "assistant",
            "content": answer
        })

    if history and st.button(
        "Clear Chat History",
        key="clear_log_chat"
    ):
        st.session_state["chat_history"] = []
        st.rerun()


def main():
    st.sidebar.title("AI Engineering Assistant")

    data_source = st.sidebar.radio(
        "Data Source",
        ["Demo Data", "Upload Log File"]
    )

    st.sidebar.divider()

    if data_source == "Demo Data":
        if not DEMO_LOG.exists():
            st.error("Demo log file is missing.")
            st.stop()

        file_bytes = DEMO_LOG.read_bytes()
        filename = DEMO_LOG.name

        saved_results = load_json(
            SAVED_ANALYSIS
        )

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a log file",
            type=["log", "txt"]
        )

        st.sidebar.caption(
            "Maximum file size: 2 MB"
        )

        if uploaded_file is None:
            st.title("Upload Logs")
            st.info(
                "Choose a .log or .txt file in the sidebar."
            )
            st.stop()

        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        saved_results = []

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("File exceeds the 2 MB limit.")
        st.stop()

    try:
        text = file_bytes.decode("utf-8-sig")

    except UnicodeDecodeError:
        st.error(
            "Only UTF-8 encoded logs are supported."
        )
        st.stop()

    parsed = parse_log_text_with_stats(text)

    events = parsed["records"]
    records = build_error_records(events)

    source_key = (
        data_source + ":"
        + hashlib.sha256(file_bytes).hexdigest()
    )

    # New file = new session.
    if st.session_state.get("active_source") != source_key:
        st.session_state["active_source"] = source_key
        st.session_state["analysis_cache"] = {}
        st.session_state["chat_history"] = []

    cache = st.session_state["analysis_cache"]

    def effective_category(index):
        return cache.get(index, {}).get(
            "category",
            records[index]["category"]
        )

    counts = Counter(
        effective_category(index)
        for index in range(len(records))
    )

    st.title("🛠️ AI Engineering Assistant")

    st.caption(
        "Log Analysis | Llama 3.2 | RAG | "
        "Evidence Tracking | Event Investigation"
    )

    if data_source == "Demo Data":
        st.info("Using synthetic demo logs.")
    else:
        st.success(f"Loaded file: {filename}")

    st.divider()

    overview = st.columns(2)

    overview[0].metric(
        "Total Log Events",
        len(events)
    )

    overview[1].metric(
        "Detected Errors",
        len(records)
    )

    st.divider()

    display_parser_health(parsed)

    st.divider()

    if not events:
        st.error("No supported log events found.")
        st.stop()

    st.subheader("Error Categories")

    categories = [
        "DATABASE",
        "NETWORK",
        "STORAGE",
        "MEMORY",
        "TIMEOUT",
        "UNKNOWN"
    ]

    columns = st.columns(6)

    for column, category in zip(
        columns, categories
    ):
        column.metric(
            category.title(),
            counts.get(category, 0)
        )

    st.divider()

    if not records:
        st.info(
            "No ERROR, CRITICAL or FATAL events found."
        )
        return

    display_log_chat(
        records,
        events,
        cache
    )

    st.divider()

    with st.expander("Log Explorer"):
        rows = []

        for index, record in enumerate(records):
            rows.append({
                "ID": index + 1,
                "Time": record["time"],
                "Level": record["level"],
                "Category": effective_category(index),
                "Message": record["message"]
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Select an Error")

    selected_index = st.selectbox(
        "Error Event",
        options=list(range(len(records))),
        format_func=lambda index: (
            f"[{effective_category(index)}] "
            f"{records[index]['message'][:110]}"
        )
    )

    record = records[selected_index]
    result = cache.get(selected_index)

    category = effective_category(
        selected_index
    )

    classification_source = (
        result.get("classification_source", "AI")
        if result is not None
        else record["classification_source"]
    )

    st.markdown(
        f"**Selected Error:** {record['message']}"
    )

    st.caption(
        f"Category: {category} | "
        f"Classification: {classification_source}"
    )

    with st.expander("Raw Error / Stack Trace"):
        st.code(
            full_error_text(record),
            language="text"
        )

    st.divider()

    if st.button(
        "🤖 Analyze Selected Error",
        type="primary"
    ):
        with st.spinner("Running local AI analysis..."):
            try:
                new_result = run_analysis(record)

                cache[selected_index] = new_result
                st.rerun()

            except Exception as error:
                st.error(
                    f"Analysis failed: "
                    f"{type(error).__name__}: {error}"
                )

    result = cache.get(selected_index)

    if result is not None:
        st.caption("Fresh AI analysis")

    elif data_source == "Demo Data":
        result = next(
            (
                saved
                for saved in saved_results
                if saved.get("error") == record["message"]
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

    else:
        st.divider()

        display_analysis(result)

        st.divider()

        st.download_button(
            "Download Analysis JSON",
            data=json.dumps(
                result,
                ensure_ascii=False,
                indent=4
            ),
            file_name="ai_analysis.json",
            mime="application/json"
        )

    st.caption(
        "Local prototype using synthetic test data "
        "and sample documentation. AI conclusions "
        "require verification."
    )


if __name__ == "__main__":
    main()
