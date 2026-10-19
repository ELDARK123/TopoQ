import unittest
from collections import defaultdict
from types import SimpleNamespace

from topoq.mapping import analyze_physical_routing
from topoq.resources import (
    find_quantum_capacity_start,
    reserve_quantum_capacity,
)
from topoq.simulator import Simulator
from topoq.topoq_scheduler import TopoQScheduler
from topoq.validators import validate_concurrent_qpu_capacity
from topoq.validators import validate_concurrent_physical_qubits


class QuantumSpaceSharingTests(unittest.TestCase):
    def test_equal_half_capacity_tasks_share_one_qpu(self):
        reservations = defaultdict(list)
        capacities = {"q0": 16}
        first_start, _ = find_quantum_capacity_start(
            reservations, capacities, {"q0": 8}, 0.0, 10.0
        )
        reserve_quantum_capacity(
            reservations,
            {"q0": 8},
            first_start,
            first_start + 10.0,
            "first",
            physical_qubits={"q0": range(8)},
        )
        second_start, shared_with = find_quantum_capacity_start(
            reservations,
            capacities,
            {"q0": 8},
            0.0,
            10.0,
            physical_requirements={"q0": range(8, 16)},
        )
        self.assertEqual(second_start, 0.0)
        self.assertEqual(shared_with, ["first"])

    def test_physical_overlap_waits_even_when_count_capacity_fits(self):
        reservations = defaultdict(list)
        capacities = {"q0": 16}
        reserve_quantum_capacity(
            reservations,
            {"q0": 8},
            0.0,
            10.0,
            "first",
            physical_qubits={"q0": range(8)},
        )
        start, shared_with = find_quantum_capacity_start(
            reservations,
            capacities,
            {"q0": 8},
            0.0,
            5.0,
            physical_requirements={"q0": range(8)},
        )
        self.assertEqual(start, 10.0)
        self.assertEqual(shared_with, [])

    def test_over_capacity_task_waits_for_batch_end(self):
        reservations = defaultdict(list)
        capacities = {"q0": 16}
        reserve_quantum_capacity(
            reservations, {"q0": 8}, 0.0, 10.0, "first"
        )
        start, shared_with = find_quantum_capacity_start(
            reservations, capacities, {"q0": 9}, 0.0, 5.0
        )
        self.assertEqual(start, 10.0)
        self.assertEqual(shared_with, [])

    def test_multi_qpu_task_checks_each_partition(self):
        reservations = defaultdict(list)
        capacities = {"q0": 16, "q1": 16, "q2": 16}
        reserve_quantum_capacity(
            reservations, {"q0": 8, "q1": 16}, 0.0, 10.0, "first"
        )
        start, shared_with = find_quantum_capacity_start(
            reservations, capacities, {"q0": 8, "q2": 16}, 0.0, 10.0
        )
        self.assertEqual(start, 0.0)
        self.assertEqual(shared_with, ["first"])

    def test_joint_scheduler_and_poisson_replay_preserve_space_sharing(self):
        system = SimpleNamespace(
            qpus={"q0": {
                "n_qubits": 16,
                "topology": [(index, index + 1) for index in range(15)],
                "gate_times_ns": {"2q": 334.0},
            }},
            nodes={},
            quantum_links={},
        )
        tasks = {
            "q0": {"type": "quantum", "n_qubits": 8, "estimated_time_ms": 10.0},
            "q1": {"type": "quantum", "n_qubits": 8, "estimated_time_ms": 10.0},
        }
        workload = SimpleNamespace(
            tasks=tasks,
            quantum_tasks=["q0", "q1"],
            classical_tasks=[],
            predecessors=defaultdict(list),
            successors=defaultdict(list),
        )
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.system = system
        scheduler.workload = workload
        scheduler.params = {}
        scheduler._intra_task_remote_gates = {}
        scheduler._compute_priorities = lambda: {"q0": 1.0, "q1": 1.0}
        interaction_graph = {
            "nodes": list(range(8)),
            "edges": {},
            "two_qubit_operations": None,
        }
        scheduler._interaction_graphs = {
            tid: interaction_graph for tid in tasks
        }
        mapping = {
            tid: {
                "logical_partition": {logical: "q0" for logical in range(8)},
                "physical_mapping": {logical: logical for logical in range(8)},
                "remote_gates_by_link": {},
                "routing_overhead_ms": 0.0,
            }
            for tid in tasks
        }
        assignment = {"q0": "q0", "q1": "q0"}

        schedule = scheduler._phase3_4_joint_scheduling(assignment, mapping)
        self.assertEqual(schedule["q0"]["start"], 0.0)
        self.assertEqual(schedule["q1"]["start"], 0.0)
        self.assertEqual(sum(
            int(schedule[tid]["space_shared"]) for tid in tasks
        ), 1)
        first_physical = set(schedule["q0"]["physical_qubits_by_qpu"]["q0"])
        second_physical = set(schedule["q1"]["physical_qubits_by_qpu"]["q0"])
        self.assertFalse(first_physical.intersection(second_physical))
        self.assertFalse(validate_concurrent_qpu_capacity(
            system, workload, schedule, assignment, mapping
        ))
        self.assertFalse(validate_concurrent_physical_qubits(
            system, workload, schedule, assignment, mapping
        ))

        simulator = Simulator(system, {}, seed=42, entanglement_mode="poisson")
        replayed, statistics = simulator._replay_poisson_schedule(
            schedule, workload, assignment, mapping
        )
        self.assertEqual(statistics["sampled_batches"], 0)
        self.assertEqual(replayed["q0"]["start"], 0.0)
        self.assertEqual(replayed["q1"]["start"], 0.0)
        self.assertTrue(replayed["q1"]["space_shared"])
        self.assertFalse(validate_concurrent_physical_qubits(
            system, workload, replayed, assignment, mapping
        ))

    def test_validator_reports_artificial_physical_conflict(self):
        system = SimpleNamespace(qpus={"q0": {"n_qubits": 16}})
        tasks = {
            "left": {"type": "quantum", "n_qubits": 8},
            "right": {"type": "quantum", "n_qubits": 8},
        }
        workload = SimpleNamespace(tasks=tasks, quantum_tasks=list(tasks))
        schedule = {
            tid: {
                "start": 0.0,
                "end": 10.0,
                "qpu": "q0",
                "partition_sizes": {"q0": 8},
                "physical_qubits_by_qpu": {"q0": list(range(8))},
            }
            for tid in tasks
        }
        violations = validate_concurrent_physical_qubits(
            system,
            workload,
            schedule,
            {tid: "q0" for tid in tasks},
            {},
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(
            violations[0]["constraint"], "C2_QPU_PHYSICAL_CONFLICT"
        )
        self.assertEqual(violations[0]["details"]["physical_qubits"], list(range(8)))

    def test_static_swap_routing_records_full_physical_path(self):
        system = SimpleNamespace(qpus={"q0": {
            "topology": [(0, 1), (1, 2), (2, 3)],
            "gate_times_ns": {"2q": 334.0},
        }})
        interaction_graph = {
            "nodes": [0, 1],
            "edges": {(0, 1): {"two_qubit_gates": 1}},
            "two_qubit_operations": None,
        }
        routing = analyze_physical_routing(
            interaction_graph,
            {0: "q0", 1: "q0"},
            {0: 0, 1: 3},
            system,
            routing_mode="static_aggregate",
        )
        self.assertEqual(routing["routed_edges"][0]["path"], [0, 1, 2, 3])
        self.assertEqual(routing["swap_count"], 2)
        self.assertEqual(routing["swap_gate_count"], 6)
        self.assertEqual(
            routing["physical_footprint_by_qpu"]["q0"], [0, 1, 2, 3]
        )
        self.assertEqual(routing["routing_path_incomplete_qpus"], [])

    def test_routing_fallback_is_exclusive_only_on_listed_qpu(self):
        system = SimpleNamespace(qpus={
            "q0": {"n_qubits": 16},
            "q1": {"n_qubits": 16},
        })
        tasks = {
            "left": {"type": "quantum", "n_qubits": 16},
            "right": {"type": "quantum", "n_qubits": 16},
        }
        workload = SimpleNamespace(tasks=tasks, quantum_tasks=list(tasks))
        schedule = {
            "left": {
                "start": 0.0,
                "end": 10.0,
                "qpu": "q0",
                "partition_sizes": {"q0": 8, "q1": 8},
                "space_sharing_exclusive_routing": True,
                "space_sharing_exclusive_qpus": ["q0"],
            },
            "right": {
                "start": 0.0,
                "end": 10.0,
                "qpu": "q1",
                "partition_sizes": {"q0": 8, "q1": 8},
            },
        }
        violations = validate_concurrent_qpu_capacity(
            system,
            workload,
            schedule,
            {"left": "q0", "right": "q1"},
            {},
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["resource"], "q0")


if __name__ == "__main__":
    unittest.main()
