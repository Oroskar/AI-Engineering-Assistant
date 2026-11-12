
import unittest

from log_parser import (
    ERROR_LEVELS,
    parse_log_text,
    parse_log_text_with_stats
)


class TestLogParserEdgeCases(unittest.TestCase):

    def test_empty_file(self):
        records = parse_log_text("")

        self.assertEqual(len(records), 0)

    def test_blank_lines(self):
        text = "\n\n   \n"

        records = parse_log_text(text)

        self.assertEqual(len(records), 0)

    def test_orphan_lines(self):
        text = (
            "Unexpected text before logs\n"
            "Another invalid line\n"
            "2026-11-12 10:00:00 INFO Started\n"
        )

        result = parse_log_text_with_stats(text)

        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(
            result["ignored_prefix_lines"], 2
        )

    def test_windows_line_endings(self):
        text = (
            "2026-11-12 10:00:00 INFO Started\r\n"
            "2026-11-12 10:01:00 ERROR Failed\r\n"
        )

        records = parse_log_text(text)

        self.assertEqual(len(records), 2)

    def test_warning_normalization(self):
        text = (
            "2026-11-12 10:00:00 WARN Retry scheduled\n"
        )

        records = parse_log_text(text)

        self.assertEqual(
            records[0]["level"], "WARNING"
        )

    def test_critical_and_fatal(self):
        text = (
            "2026-11-12 10:00:00 CRITICAL Service crashed\n"
            "2026-11-12 10:01:00 FATAL Process terminated\n"
        )

        records = parse_log_text(text)

        errors = [
            record for record in records
            if record["level"] in ERROR_LEVELS
        ]

        self.assertEqual(len(errors), 2)

    def test_invalid_clock_time(self):
        text = (
            "2026-11-12 25:99:99 ERROR Invalid time\n"
        )

        records = parse_log_text(text)

        self.assertEqual(len(records), 0)

    def test_invalid_calendar_date(self):
        text = (
            "2026-02-30 10:00:00 ERROR Invalid date\n"
        )

        records = parse_log_text(text)

        self.assertEqual(len(records), 0)

    def test_malformed_header_not_added_to_trace(self):
        text = (
            "2026-11-12 10:00:00 ERROR First error\n"
            "2026-11-12 10:XX:00 ERROR Malformed event\n"
        )

        records = parse_log_text(text)

        self.assertEqual(len(records), 1)

        # A malformed log header should not silently
        # become part of the previous stack trace.
        self.assertEqual(records[0]["details"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
