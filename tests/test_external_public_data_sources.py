import json
import tempfile
import unittest
from pathlib import Path

from topoq.config import (
    DATASET_DIR,
    dataset_source_context,
    load_circuit,
    load_dag,
    load_network_topology,
    load_sim_params,
    load_system_config,
    resolve_circuit_path,
    resolve_dag_path,
)
from topoq.provenance import build_result_provenance


class ExternalPublicDataSourceTests(unittest.TestCase):
    def test_context_loads_external_dag_and_nested_circuit_then_restores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dag_dir = root / "dags"
            circuit_dir = root / "circuits"
            nested = circuit_dir / "mqt_bench"
            dag_dir.mkdir()
            nested.mkdir(parents=True)
            system_root = root / "system_data"
            (system_root / "system_configs").mkdir(parents=True)
            (system_root / "network_topologies").mkdir()
            (system_root / "sim_params").mkdir()
            (dag_dir / "public_test.json").write_text(
                json.dumps({"id": "public_test", "value": "external"}),
                encoding="utf-8",
            )
            (nested / "public_circuit.json").write_text(
                json.dumps({"id": "public_circuit", "n_qubits": 4}),
                encoding="utf-8",
            )
            (system_root / "system_configs" / "small.json").write_text(
                json.dumps({"source": "external-system"}),
                encoding="utf-8",
            )
            (system_root / "network_topologies" / "network.json").write_text(
                json.dumps({"source": "external-network"}),
                encoding="utf-8",
            )
            (system_root / "sim_params" / "sim_params.json").write_text(
                json.dumps({"source": "external-sim"}),
                encoding="utf-8",
            )

            with dataset_source_context(
                dag_dir=dag_dir,
                circuit_dirs=[circuit_dir],
                system_data_root=system_root,
            ):
                self.assertEqual(load_dag("public_test")["value"], "external")
                self.assertEqual(load_circuit("public_circuit")["n_qubits"], 4)
                self.assertEqual(
                    load_system_config("small")["source"],
                    "external-system",
                )
                self.assertEqual(
                    load_network_topology("network")["source"],
                    "external-network",
                )
                self.assertEqual(
                    load_sim_params()["source"],
                    "external-sim",
                )
                self.assertEqual(
                    Path(resolve_dag_path("public_test")).resolve(),
                    (dag_dir / "public_test.json").resolve(),
                )
                self.assertEqual(
                    Path(resolve_circuit_path("public_circuit")).resolve(),
                    (nested / "public_circuit.json").resolve(),
                )

            self.assertEqual(
                Path(resolve_dag_path("public_test")),
                Path(DATASET_DIR) / "workload_dags" / "public_test.json",
            )

    def test_duplicate_external_circuit_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "circuits" / "first"
            second = root / "circuits" / "second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            payload = json.dumps({"id": "duplicate"})
            (first / "duplicate.json").write_text(payload, encoding="utf-8")
            (second / "duplicate.json").write_text(payload, encoding="utf-8")

            with dataset_source_context(circuit_dirs=[root / "circuits"]):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    resolve_circuit_path("duplicate")

    def test_provenance_accepts_explicit_external_dataset_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "code.py"
            experiment_input = root / "input.json"
            dataset_input = root / "dataset.json"
            code.write_text("VALUE = 1\n", encoding="utf-8")
            experiment_input.write_text('{"input": 1}\n', encoding="utf-8")
            dataset_input.write_text('{"dataset": 2}\n', encoding="utf-8")

            provenance = build_result_provenance(
                protocol={"kind": "public_test"},
                input_paths=[experiment_input],
                project_root=root,
                code_paths=[code],
                dataset_paths=[dataset_input],
            )

            self.assertEqual(
                provenance["dataset_snapshot"]["file_count"],
                1,
            )
            self.assertIn(
                "dataset.json",
                provenance["dataset_snapshot"]["files"],
            )


if __name__ == "__main__":
    unittest.main()
