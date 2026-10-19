import unittest
from unittest.mock import patch

from topoq.config import get_experiment
from topoq.mapping import (
    PARTITION_FILL_PENALTY,
    PARTITION_MAX_REFINEMENT_PASSES,
    PARTITION_METHOD_NAME,
    PARTITION_NODE_ORDER,
    partition_interaction_graph,
)
from topoq.runner import run_experiment_spec


class _PartitionSystem:
    def __init__(self):
        self.qpus = {
            "Q0": {"n_qubits": 2},
            "Q1": {"n_qubits": 2},
        }
        self._link = {
            "rate_hz": 1000.0,
            "distance_km": 10.0,
            "max_concurrent_epr": 2,
            "decoherence_rate_per_ms": 0.01,
        }

    def get_quantum_link(self, source, target):
        return self._link if source != target else None


class Phase2PartitioningPolicyTests(unittest.TestCase):
    def test_partition_is_deterministic_and_capacity_feasible(self):
        graph = {
            "nodes": [0, 1, 2, 3],
            "edges": {
                (0, 1): {"weight": 4.0},
                (1, 2): {"weight": 1.0},
                (2, 3): {"weight": 4.0},
            },
        }
        system = _PartitionSystem()

        first = partition_interaction_graph(
            graph, ("Q0", "Q1"), system
        )
        second = partition_interaction_graph(
            graph, ("Q0", "Q1"), system
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(graph["nodes"]))
        for qpu_id, qpu in system.qpus.items():
            assigned = sum(value == qpu_id for value in first.values())
            self.assertLessEqual(assigned, qpu["n_qubits"])

    def test_public_method_constants_match_implemented_defaults(self):
        self.assertEqual(
            PARTITION_METHOD_NAME,
            "capacity_topology_greedy_single_vertex_refinement",
        )
        self.assertEqual(
            PARTITION_NODE_ORDER,
            "descending_weighted_degree_then_node_id",
        )
        self.assertEqual(PARTITION_FILL_PENALTY, 1e-6)
        self.assertEqual(PARTITION_MAX_REFINEMENT_PASSES, 5)

    def test_runner_metadata_describes_non_multilevel_refinement(self):
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0},
        ):
            result = run_experiment_spec(
                get_experiment("VQE-50_Small"),
                profile="quick",
                repetitions=1,
                algorithms=("Greedy", "TopoQ"),
            )

        policy = result["metadata"]["phase2_partitioning"]
        self.assertEqual(policy["method"], PARTITION_METHOD_NAME)
        self.assertFalse(policy["multilevel"])
        self.assertFalse(policy["coarsening"])
        self.assertFalse(policy["uncoarsening"])
        self.assertFalse(policy["pair_swaps"])
        self.assertEqual(policy["refinement_move"], "single_vertex")
        self.assertEqual(
            policy["refinement_acceptance"],
            "strict_objective_decrease",
        )
        self.assertEqual(
            policy["maximum_refinement_passes"],
            PARTITION_MAX_REFINEMENT_PASSES,
        )
        self.assertEqual(policy["applies_to"], ["TopoQ"])


if __name__ == "__main__":
    unittest.main()
