import unittest

from topoq.diagnostics import compute_critical_path
from topoq.models import Workload


class DiagnosticsRegressionTests(unittest.TestCase):
    def test_critical_path_components_sum_to_path(self):
        dag = {
            "id": "diagnostics_probe",
            "type": "VQE",
            "n_qubits": 2,
            "n_iterations": 1,
            "tasks": [
                {"id": "a", "type": "quantum", "n_qubits": 2},
                {"id": "b", "type": "classical"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }
        workload = Workload(dag)
        sched = {
            "a": {"start": 0.0, "end": 10.0},
            "b": {
                "start": 15.0,
                "end": 25.0,
                "dependency_transfers": [
                    {"predecessor": "a", "start": 10.0, "end": 12.0}
                ],
            },
        }
        critical = compute_critical_path(workload, sched)
        self.assertEqual(critical["tasks"], ["a", "b"])
        self.assertAlmostEqual(critical["total_path_ms"], 25.0)
        self.assertAlmostEqual(
            critical["exec_ms"]
            + critical["transfer_ms"]
            + critical["wait_ms"],
            critical["total_path_ms"],
        )
        self.assertAlmostEqual(critical["transfer_ms"], 2.0)
        self.assertAlmostEqual(critical["wait_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
