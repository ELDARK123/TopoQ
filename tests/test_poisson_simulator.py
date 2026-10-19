import math
import random
import unittest
from types import SimpleNamespace

from topoq.simulator import Simulator


class FakeSystem:
    def __init__(self):
        self.qpus = {"q0": {"n_qubits": 16}, "q1": {"n_qubits": 16}}
        self.nodes = {"n0": {}, "n1": {}}
        self.quantum_links = {
            frozenset(("q0", "q1")): {
                "rate_hz": 100.0,
                "max_concurrent_epr": 2,
            },
        }


def workload(tasks, edges):
    predecessors = {tid: [] for tid in tasks}
    successors = {tid: [] for tid in tasks}
    for source, target in edges:
        predecessors[target].append(source)
        successors[source].append(target)
    return SimpleNamespace(
        tasks=tasks,
        predecessors=predecessors,
        successors=successors,
        quantum_tasks=[
            tid for tid, task in tasks.items() if task["type"] == "quantum"
        ],
        classical_tasks=[
            tid for tid, task in tasks.items() if task["type"] == "classical"
        ],
    )


class PoissonEntanglementTests(unittest.TestCase):
    def setUp(self):
        self.system = FakeSystem()

    def test_inverse_cdf_matches_paper_equation(self):
        seed = 17
        expected_u = random.Random(seed).random()
        expected = -math.log1p(-expected_u) / 100.0 * 1000.0
        simulator = Simulator(
            self.system, {}, seed=seed, entanglement_mode="poisson"
        )
        actual = simulator._sample_epr_generation_ms(("q0", "q1"))
        self.assertAlmostEqual(actual, expected, places=12)

    def test_same_seed_is_reproducible(self):
        left = Simulator(self.system, {}, seed=42, entanglement_mode="poisson")
        right = Simulator(self.system, {}, seed=42, entanglement_mode="poisson")
        left_samples = [
            left._sample_epr_generation_ms(("q0", "q1")) for _ in range(20)
        ]
        right_samples = [
            right._sample_epr_generation_ms(("q0", "q1")) for _ in range(20)
        ]
        self.assertEqual(left_samples, right_samples)

    def test_doubling_rate_halves_sample_for_same_uniform_draw(self):
        fast_system = FakeSystem()
        fast_system.quantum_links[frozenset(("q0", "q1"))]["rate_hz"] = 200.0
        slow = Simulator(self.system, {}, seed=7, entanglement_mode="poisson")
        fast = Simulator(fast_system, {}, seed=7, entanglement_mode="poisson")
        slow_sample = slow._sample_epr_generation_ms(("q0", "q1"))
        fast_sample = fast._sample_epr_generation_ms(("q0", "q1"))
        self.assertAlmostEqual(fast_sample, slow_sample / 2.0, places=12)

    def test_replay_preserves_compute_durations_and_propagates_all_edge_types(self):
        tasks = {
            "q0": {"type": "quantum"},
            "q1": {"type": "quantum"},
            "c0": {"type": "classical"},
            "c1": {"type": "classical"},
            "q2": {"type": "quantum"},
        }
        graph = workload(
            tasks,
            [("q0", "q1"), ("q1", "c0"), ("c0", "c1"), ("c1", "q2")],
        )
        reservation = {
            "link": ("q0", "q1"),
            "generation_start": 5.0,
            "start": 5.0,
            "end": 8.0,
        }
        schedule = {
            "q0": {"start": 0.0, "end": 5.0, "qpu": "q0", "qpu_set": ["q0"]},
            "q1": {
                "start": 8.0,
                "end": 12.0,
                "qpu": "q1",
                "qpu_set": ["q1"],
                "entanglement_wait": 0.0,
                "dependency_transfers": [{
                    "predecessor": "q0",
                    "type": "quantum-quantum",
                    "start": 5.0,
                    "end": 8.0,
                    "entanglement_reservations": [reservation],
                }],
            },
            "c0": {
                "start": 14.0,
                "end": 17.0,
                "node": "n0",
                "comm_time": 2.0,
                "dependency_transfers": [{
                    "predecessor": "q1",
                    "type": "quantum-classical",
                    "start": 12.0,
                    "end": 14.0,
                }],
            },
            "c1": {
                "start": 20.0,
                "end": 22.0,
                "node": "n1",
                "comm_time": 3.0,
                "dependency_transfers": [{
                    "predecessor": "c0",
                    "type": "classical-classical",
                    "start": 17.0,
                    "end": 20.0,
                }],
            },
            "q2": {
                "start": 26.0,
                "end": 31.0,
                "qpu": "q0",
                "qpu_set": ["q0"],
                "entanglement_wait": 0.0,
                "dependency_transfers": [{
                    "predecessor": "c1",
                    "type": "classical-quantum",
                    "start": 22.0,
                    "end": 26.0,
                }],
            },
        }
        simulator = Simulator(
            self.system, {}, seed=3, entanglement_mode="poisson"
        )
        replayed, statistics = simulator._replay_poisson_schedule(
            schedule,
            graph,
            {"q0": "q0", "q1": "q1", "c0": "n0", "c1": "n1", "q2": "q0"},
            {},
        )

        self.assertEqual(statistics["sampled_batches"], 1)
        for tid, expected_duration in {
            "q0": 5.0, "q1": 4.0, "c0": 3.0, "c1": 2.0, "q2": 5.0
        }.items():
            self.assertAlmostEqual(
                replayed[tid]["end"] - replayed[tid]["start"],
                expected_duration,
            )
        self.assertAlmostEqual(replayed["c0"]["start"], replayed["q1"]["end"] + 2.0)
        self.assertAlmostEqual(replayed["c1"]["start"], replayed["c0"]["end"] + 3.0)
        self.assertAlmostEqual(replayed["q2"]["start"], replayed["c1"]["end"] + 4.0)

    def test_aggregate_baseline_sampling_counts_dependency_and_gate_batches(self):
        graph = workload(
            {"pred": {"type": "quantum"}, "task": {"type": "quantum"}},
            [("pred", "task")],
        )
        statistics = {"sampled_batches": 0, "sampled_generation_ms": 0.0}
        simulator = Simulator(
            self.system, {}, seed=11, entanglement_mode="poisson"
        )
        wait = simulator._sample_aggregate_quantum_wait(
            "task",
            {"entanglement_wait": 1.0},
            graph,
            {"pred": "q0", "task": "q1"},
            {"task": {"remote_gates_by_link": {("q0", "q1"): 3}}},
            statistics,
        )
        self.assertGreater(wait, 0.0)
        self.assertEqual(statistics["sampled_batches"], 3)
        self.assertAlmostEqual(statistics["sampled_generation_ms"], wait)


if __name__ == "__main__":
    unittest.main()
