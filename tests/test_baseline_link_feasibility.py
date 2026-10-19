import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import get_experiment
from topoq.mapping import (
    BASELINE_FEASIBILITY_BEAM_WIDTH,
    BASELINE_FEASIBILITY_MAX_BEAM_WIDTH,
    count_direct_link_partition_violations,
    count_quantum_dependency_link_violations,
    project_partition_to_direct_links,
    project_quantum_task_assignments,
)
from topoq.runner import run_experiment_spec, run_single_experiment


class _SparseQuantumSystem:
    def __init__(self, capacity=2):
        self.qpus = {
            qid: {"n_qubits": capacity}
            for qid in ("Q0", "Q1", "Q2")
        }
        self.links = {
            frozenset(("Q0", "Q1")),
            frozenset(("Q1", "Q2")),
        }

    def get_qpu_ids(self):
        return sorted(self.qpus)

    def get_quantum_link(self, left, right):
        if frozenset((left, right)) in self.links:
            return {"rate_hz": 1000.0}
        return None


class BaselineLinkFeasibilityTests(unittest.TestCase):
    def test_already_feasible_partition_is_preserved_exactly(self):
        system = _SparseQuantumSystem()
        graph = {
            "nodes": [0, 1, 2, 3],
            "edges": {
                (0, 1): {"two_qubit_gates": 3},
                (1, 2): {"two_qubit_gates": 2},
                (2, 3): {"two_qubit_gates": 1},
            },
        }
        original = {0: "Q0", 1: "Q0", 2: "Q1", 3: "Q1"}

        projected, diagnostics = project_partition_to_direct_links(
            graph, original, system
        )

        self.assertEqual(projected, original)
        self.assertFalse(diagnostics["projection_applied"])
        self.assertEqual(diagnostics["moved_logical_qubits"], 0)
        self.assertEqual(diagnostics["after"]["violating_gates"], 0)

    def test_partition_projection_removes_violations_with_minimum_moves(self):
        system = _SparseQuantumSystem()
        graph = {
            "nodes": [0, 1, 2, 3],
            "edges": {
                (0, 1): {"two_qubit_gates": 3},
                (1, 2): {"two_qubit_gates": 2},
                (2, 3): {"two_qubit_gates": 1},
            },
        }
        original = {0: "Q0", 1: "Q2", 2: "Q0", 3: "Q2"}

        projected, diagnostics = project_partition_to_direct_links(
            graph, original, system
        )

        self.assertEqual(diagnostics["before"]["violating_edges"], 3)
        self.assertEqual(diagnostics["before"]["violating_gates"], 6)
        self.assertEqual(diagnostics["moved_logical_qubits"], 2)
        self.assertEqual(
            count_direct_link_partition_violations(
                graph, projected, system
            ),
            {"violating_edges": 0, "violating_gates": 0},
        )
        for qpu_id, qpu in system.qpus.items():
            assigned = sum(value == qpu_id for value in projected.values())
            self.assertLessEqual(assigned, qpu["n_qubits"])

    def test_quantum_dependency_projection_keeps_classical_assignments(self):
        system = _SparseQuantumSystem()
        tasks = {
            "q0": {"type": "quantum"},
            "q1": {"type": "quantum"},
            "q2": {"type": "quantum"},
            "c0": {"type": "classical"},
        }
        workload = SimpleNamespace(
            tasks=tasks,
            quantum_tasks=["q0", "q1", "q2"],
            edges=[
                {"source": "q0", "target": "q1"},
                {"source": "q1", "target": "q2"},
                {"source": "q2", "target": "c0"},
            ],
        )
        original = {
            "q0": "Q0", "q1": "Q2", "q2": "Q0", "c0": "P7"
        }

        projected, diagnostics = project_quantum_task_assignments(
            workload, original, system
        )

        self.assertEqual(diagnostics["before_violations"], 2)
        self.assertEqual(diagnostics["after_violations"], 0)
        self.assertEqual(diagnostics["moved_quantum_tasks"], 1)
        self.assertEqual(projected["c0"], "P7")
        self.assertEqual(
            count_quantum_dependency_link_violations(
                workload, projected, system
            ),
            0,
        )

    def test_runner_metadata_declares_common_physical_domain(self):
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0},
        ):
            result = run_experiment_spec(
                get_experiment("VQE-50_Small"),
                profile="quick",
                repetitions=1,
                algorithms=("Random", "Greedy", "TopoQ"),
            )

        policy = result["metadata"]["baseline_physical_feasibility"]
        self.assertEqual(
            policy["method"], "minimum_hamming_direct_link_projection"
        )
        self.assertEqual(
            policy["bounded_beam_widths"],
            [
                BASELINE_FEASIBILITY_BEAM_WIDTH,
                BASELINE_FEASIBILITY_MAX_BEAM_WIDTH,
            ],
        )
        self.assertTrue(policy["direct_link_required"])
        self.assertFalse(policy["topoq_constraints_or_objective_reused"])
        self.assertEqual(policy["applies_to"], ["Random", "Greedy"])

    def test_real_baseline_exposes_zero_post_projection_violations(self):
        metrics = run_single_experiment(
            "vqe_dag_50q_10iter",
            "small",
            "fattree_k4",
            "Greedy",
            seed=42,
        )

        self.assertEqual(
            metrics["baseline_dependency_violations_after"], 0
        )
        self.assertEqual(
            metrics["baseline_partition_violating_edges_after"], 0
        )
        self.assertEqual(
            metrics["baseline_partition_violating_gates_after"], 0
        )


if __name__ == "__main__":
    unittest.main()
