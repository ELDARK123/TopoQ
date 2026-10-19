import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from topoq.main import (
    _build_run_context,
    _default_output_path,
    _preflight_output_path,
    _resolve_output_path,
    _write_json_result,
)


class ResultOutputSafetyTests(unittest.TestCase):
    def test_default_output_uses_timestamped_run_directory(self):
        now = datetime(
            2026, 7, 19, 14, 30, 25, 123456,
            tzinfo=timezone(timedelta(hours=8)),
        )
        context = _build_run_context(now)
        with tempfile.TemporaryDirectory() as root:
            path = _default_output_path(
                "all_results.json", context["run_id"], results_dir=root
            )
            expected = os.path.abspath(os.path.join(
                root,
                "runs",
                "20260719-143025-123456+0800",
                "all_results.json",
            ))
            self.assertEqual(path, expected)
            self.assertEqual(context["started_at"], now.isoformat())

    def test_existing_output_is_rejected_without_content_change(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "result.json")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("original")

            with self.assertRaises(FileExistsError):
                _preflight_output_path(path, overwrite=False)
            with self.assertRaises(FileExistsError):
                _write_json_result(path, {"new": True}, overwrite=False)

            with open(path, "r", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "original")

    def test_explicit_overwrite_replaces_existing_json(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "result.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({"old": True}, stream)

            _preflight_output_path(path, overwrite=True)
            _write_json_result(path, {"new": True}, overwrite=True)

            with open(path, "r", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), {"new": True})

    def test_writer_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "nested", "run", "result.json")
            _write_json_result(path, {"ok": True})
            with open(path, "r", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), {"ok": True})

    def test_explicit_and_default_filenames_remain_distinct(self):
        with tempfile.TemporaryDirectory() as root:
            run_id = "run-id"
            single = _resolve_output_path(
                None, "VQE-50_Small.json", run_id, results_dir=root
            )
            suite = _resolve_output_path(
                None, "all_results.json", run_id, results_dir=root
            )
            explicit = _resolve_output_path(
                os.path.join(root, "chosen.json"),
                "ignored.json",
                run_id,
                results_dir=root,
            )
            self.assertNotEqual(single, suite)
            self.assertTrue(single.endswith("VQE-50_Small.json"))
            self.assertTrue(suite.endswith("all_results.json"))
            self.assertTrue(explicit.endswith("chosen.json"))


if __name__ == "__main__":
    unittest.main()
