
import tempfile
import unittest
from pathlib import Path

from log_parser import parse_multiline_log


class TestLogParser(unittest.TestCase):

    def parse_text(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.log"
            path.write_text(text, encoding="utf-8")
            return parse_multiline_log(path)

    def test_standard_log_events(self):
        text = (
            "2026-11-12 10:00:00 INFO Application started\n"
            "2026-11-12 10:01:00 ERROR Database unavailable\n"
        )

        records = self.parse_text(text)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["level"], "ERROR")
        self.assertEqual(
            records[1]["message"],
            "Database unavailable"
        )

    def test_multiline_stack_trace(self):
        text = (
            "2026-11-12 10:01:00 ERROR Connection failed\n"
            "Traceback (most recent call last):\n"
            '  File "db.py", line 42, in connect\n'
            "ConnectionRefusedError: Connection refused\n"
        )

        records = self.parse_text(text)

        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["details"]), 3)
        self.assertIn(
            "ConnectionRefusedError",
            records[0]["details"][2]
        )

    def test_critical_log_level(self):
        text = (
            "2026-11-12 10:01:00 CRITICAL Service crashed\n"
        )

        records = self.parse_text(text)

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["level"],
            "CRITICAL"
        )

    def test_milliseconds_and_brackets(self):
        text = (
            "2026-11-12 10:01:00.123 [ERROR] Disk full\n"
        )

        records = self.parse_text(text)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["level"], "ERROR")

    def test_iso_timestamp(self):
        text = (
            "2026-11-12T10:01:00Z ERROR DNS lookup failed\n"
        )

        records = self.parse_text(text)

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["message"],
            "DNS lookup failed"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
