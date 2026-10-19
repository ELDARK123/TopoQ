"""Regression tests for the 2026-10-18 baseline/runner bug fixes."""

import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from topoq.baselines import (
    RandomScheduler,
    GreedyScheduler,
    QOnlyScheduler,
    DQTetrisScheduler,
)
from topoq.config import (
    load_dag,
    load_network_topology,
    load_sim_params,
    load_system_config,
    validate_objective_mode,
)
from topoq.mapping import (
    dqtetris_partition,
    interaction_only_partition,
)
from topoq.models import HybridSystem, Workload
from topoq.validators import validate_schedule
import topoq.runner as runner


class BaselinePriorityTests(unittest.TestCase):
    def test_priority_marks_longest_path_to_exit(self):
        # Two branches: t -> a -> b (exit) and t -> s2 -> s3 (exit).
        predecessors = {
            "t": [], "a": ["t"], "b": ["a"],
            "s2": ["t"], "s3": ["s2"],
        }
        successors = {
            "t": ["a", "s2"], "a": ["b"], "b": [],
            "s2": ["s3"], "s3": [],
        }
        durations = {"t": 2.0, "a": 1.0, "b": 0.0, "s2": 1.0, "s3": 0.0}

        scheduler = RandomScheduler.__new__(RandomScheduler)
        scheduler.workload = SimpleNamespace(
            tasks={tid: {} for tid in predecessors},
            predecessors=predecessors,
            successors=successors,
        )
        scheduler._reference_duration = lambda tid: durations[tid]

        priorities = scheduler._compute_priorities()

        self.assertAlmostEqual(priorities["b"], 0.0)
        self.assertAlmostEqual(priorities["a"], 1.0)
        self.assertAlmostEqual(priorities["s3"], 0.0)
        self.assertAlmostEqual(priorities["s2"], 1.0)
        self.assertAlmostEqual(priorities["t"], 3.0)
        # Upward rank must dominate every single successor.
        for tid, succs in successors.items():
            for succ in succs:
                self.assertGreaterEqual(
                    priorities[tid] + 1e-9,
                    durations[tid] + priorities[succ],
                )


class RepetitionAggregationTests(unittest.TestCase):
    def test_string_fields_do_not_crash_averaging(self):
        sequence = [
            {"makespan": 10.0, "mode": "proactive"},
            {"makespan": 20.0, "mode": "proactive"},
        ]
        with patch.object(
            runner, "run_single_experiment", side_effect=sequence
        ):
            result = runner.run_with_repetitions(
                "dag", "small", "fattree_k4", "TopoQ", n_rep=2
            )

        self.assertAlmostEqual(result["makespan"], 15.0)
        self.assertAlmostEqual(result["makespan_std"], 5.0)
        self.assertEqual(result["mode"], "proactive")
        self.assertIn("runtime_seconds", result)

    def test_inconsistent_string_fields_are_preserved_as_variants(self):
        sequence = [
            {"makespan": 10.0, "mode": "proactive"},
            {"makespan": 20.0, "mode": "on_demand"},
        ]
        with patch.object(
            runner, "run_single_experiment", side_effect=sequence
        ):
            result = runner.run_with_repetitions(
                "dag", "small", "fattree_k4", "TopoQ", n_rep=2
            )

        self.assertEqual(result["mode"], ["proactive", "on_demand"])


class BaselineConstraintRegressionTests(unittest.TestCase):
    def test_baseline_schedules_have_no_c3_or_c6_violations(self):
        dag = load_dag("vqe_dag_50q_10iter")
        system = HybridSystem(
            load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )
        workload = Workload(dag)
        for scheduler_class in (
            RandomScheduler, GreedyScheduler, QOnlyScheduler, DQTetrisScheduler
        ):
            with self.subTest(scheduler=scheduler_class.__name__):
                sched, assignment, mapping = scheduler_class(
                    system, workload, seed=42
                ).schedule()
                counts = Counter(
                    violation["constraint"]
                    for violation in validate_schedule(
                        system, workload, sched, assignment, mapping, params={}
                    )
                )
                self.assertEqual(counts.get("C3_DAG_PRECEDENCE", 0), 0)
                self.assertEqual(counts.get("C6_MAPPING_CONSISTENCY", 0), 0)


class ObjectiveModeTests(unittest.TestCase):
    def test_objective_mode_validation(self):
        self.assertEqual(validate_objective_mode("joint"), "joint")
        self.assertEqual(validate_objective_mode("makespan"), "makespan")
        with self.assertRaises(ValueError):
            validate_objective_mode("unknown")


class DQTetrisPartitionTests(unittest.TestCase):
    def test_cut_weight_not_worse_than_greedy(self):
        class _System:
            qpus = {
                "Q0": {"n_qubits": 4},
                "Q1": {"n_qubits": 4},
                "Q2": {"n_qubits": 4},
            }

            def get_quantum_link(self, left, right):
                return {"rate_hz": 1000.0}

        system = _System()
        graph = {
            "nodes": list(range(12)),
            "edges": {
                (0, 1): {"weight": 5.0},
                (1, 2): {"weight": 4.0},
                (2, 3): {"weight": 3.0},
                (3, 4): {"weight": 2.0},
                (4, 5): {"weight": 5.0},
                (5, 6): {"weight": 4.0},
                (6, 7): {"weight": 3.0},
                (7, 8): {"weight": 2.0},
                (8, 9): {"weight": 5.0},
                (9, 10): {"weight": 4.0},
                (10, 11): {"weight": 3.0},
                (0, 6): {"weight": 1.0},
                (3, 9): {"weight": 1.0},
            },
        }
        qpu_set = ["Q0", "Q1", "Q2"]

        def cut(partition):
            return sum(
                edge["weight"]
                for (source, target), edge in graph["edges"].items()
                if partition[source] != partition[target]
            )

        greedy = interaction_only_partition(graph, qpu_set, system)
        dqtetris = dqtetris_partition(graph, qpu_set, system)
        self.assertLessEqual(cut(dqtetris), cut(greedy))


if __name__ == "__main__":
    unittest.main()
