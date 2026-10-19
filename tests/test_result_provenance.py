import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from topoq.config import get_experiment
from topoq.main import main
from topoq.provenance import (
    EXPERIMENT_PROTOCOL_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    build_file_manifest,
    build_result_provenance,
    canonical_sha256,
    report_compatibility_key,
    sha256_file,
    validate_report_provenance,
)
from topoq.runner import run_experiment_spec


class ResultProvenanceTests(unittest.TestCase):
    def test_file_digest_matches_known_sha256(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "value.txt"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb"
                "410ff61f20015ad",
            )

    def test_manifest_is_order_independent_and_uses_relative_paths(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            first = root_path / "data" / "a.json"
            second = root_path / "code" / "b.py"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text('{"a": 1}', encoding="utf-8")
            second.write_text("VALUE = 2\n", encoding="utf-8")

            forward = build_file_manifest(
                [first, second], project_root=root_path
            )
            reverse = build_file_manifest(
                [second, first], project_root=root_path
            )

            self.assertEqual(
                forward["aggregate_sha256"], reverse["aggregate_sha256"]
            )
            self.assertEqual(
                list(forward["files"]), ["code/b.py", "data/a.json"]
            )
            self.assertNotIn(str(root_path), str(forward["files"]))

    def test_manifest_changes_when_content_changes(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            path = root_path / "input.json"
            path.write_text('{"value": 1}', encoding="utf-8")
            before = build_file_manifest([path], project_root=root_path)
            path.write_text('{"value": 2}', encoding="utf-8")
            after = build_file_manifest([path], project_root=root_path)
            self.assertNotEqual(
                before["aggregate_sha256"], after["aggregate_sha256"]
            )

    def test_protocol_digest_uses_canonical_json(self):
        self.assertEqual(
            canonical_sha256({"b": 2, "a": 1}),
            canonical_sha256({"a": 1, "b": 2}),
        )

    def test_result_provenance_contains_protocol_code_inputs_environment(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            code = root_path / "topoq" / "module.py"
            data = root_path / "dataset" / "input.json"
            code.parent.mkdir()
            data.parent.mkdir()
            code.write_text("VALUE = 1\n", encoding="utf-8")
            data.write_text('{"value": 1}', encoding="utf-8")

            result = build_result_provenance(
                {"kind": "test", "repetitions": 1},
                [data],
                project_root=root_path,
                code_paths=[code],
            )

            self.assertEqual(
                result["schema_version"], PROVENANCE_SCHEMA_VERSION
            )
            self.assertEqual(
                result["protocol"]["version"],
                EXPERIMENT_PROTOCOL_VERSION,
            )
            self.assertEqual(result["code"]["file_count"], 1)
            self.assertEqual(result["inputs"]["file_count"], 1)
            self.assertEqual(
                result["inputs"]["scope"], "exact_experiment_inputs"
            )
            self.assertIn("python_version", result["environment"])
            self.assertIn("sha256", result["environment"])

    def test_report_compatibility_allows_distinct_exact_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            code = root_path / "topoq" / "module.py"
            first = root_path / "dataset" / "first.json"
            second = root_path / "dataset" / "second.json"
            code.parent.mkdir()
            first.parent.mkdir()
            code.write_text("VALUE = 1\n", encoding="utf-8")
            first.write_text('{"value": 1}', encoding="utf-8")
            second.write_text('{"value": 2}', encoding="utf-8")
            first_provenance = build_result_provenance(
                {"kind": "single", "experiment_id": "first"},
                [first], project_root=root_path, code_paths=[code],
            )
            second_provenance = build_result_provenance(
                {"kind": "single", "experiment_id": "second"},
                [second], project_root=root_path, code_paths=[code],
            )
            results = {
                "first": {"metadata": {"provenance": first_provenance}},
                "second": {"metadata": {"provenance": second_provenance}},
            }

            key = validate_report_provenance(results)

            self.assertEqual(
                key, report_compatibility_key(results["first"])
            )
            self.assertNotEqual(
                first_provenance["inputs"]["aggregate_sha256"],
                second_provenance["inputs"]["aggregate_sha256"],
            )

    def test_report_compatibility_rejects_changed_dataset_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            code = root_path / "topoq" / "module.py"
            data = root_path / "dataset" / "input.json"
            code.parent.mkdir()
            data.parent.mkdir()
            code.write_text("VALUE = 1\n", encoding="utf-8")
            data.write_text('{"value": 1}', encoding="utf-8")
            before = build_result_provenance(
                {"kind": "single"}, [data],
                project_root=root_path, code_paths=[code],
            )
            data.write_text('{"value": 2}', encoding="utf-8")
            after = build_result_provenance(
                {"kind": "single"}, [data],
                project_root=root_path, code_paths=[code],
            )

            with self.assertRaisesRegex(ValueError, "dataset_sha256"):
                validate_report_provenance([
                    {"metadata": {"provenance": before}},
                    {"metadata": {"provenance": after}},
                ])

    def test_experiment_result_records_exact_inputs_and_seed_sequence(self):
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0, "runtime_seconds": 0.1},
        ):
            result = run_experiment_spec(
                get_experiment("VQE-50_Small"),
                profile="quick",
                repetitions=2,
                algorithms=("Greedy", "TopoQ"),
            )

        provenance = result["metadata"]["provenance"]
        protocol = provenance["protocol"]
        self.assertEqual(protocol["random_seeds"], [42, 43])
        self.assertEqual(protocol["algorithms"], ["Greedy", "TopoQ"])
        self.assertEqual(provenance["inputs"]["file_count"], 4)
        self.assertEqual(
            set(provenance["inputs"]["files"]),
            {
                "dataset/network_topologies/fattree_k4.json",
                "dataset/sim_params/sim_params.json",
                "dataset/system_configs/small.json",
                "dataset/workload_dags/vqe_dag_50q_10iter.json",
            },
        )

    def test_full_suite_records_effective_repetitions_and_dataset_snapshot(self):
        main_module = importlib.import_module("topoq.main")
        with tempfile.TemporaryDirectory() as root, patch.object(
            main_module, "experiment_makespan_comparison", return_value={}
        ), patch.object(
            main_module, "experiment_communication_overhead", return_value={}
        ), patch.object(
            main_module, "experiment_fidelity", return_value={}
        ), patch.object(
            main_module, "experiment_scalability", return_value=({}, {})
        ), patch.object(main_module, "print_summary"):
            output = Path(root) / "suite.json"
            result = main([
                "--profile", "quick",
                "--algorithm", "Greedy",
                "--output", str(output),
            ])

        provenance = result["metadata"]["provenance"]
        protocol = provenance["protocol"]
        self.assertEqual(protocol["repetitions"], 1)
        self.assertEqual(protocol["random_seeds"], [42])
        self.assertEqual(protocol["algorithms"], ["Greedy"])
        self.assertEqual(
            provenance["inputs"]["scope"],
            "complete_dataset_json_snapshot",
        )
        self.assertGreater(provenance["inputs"]["file_count"], 4)


if __name__ == "__main__":
    unittest.main()
