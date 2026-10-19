import unittest
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import (
    DEFAULT_PARAMS,
    load_network_topology,
    load_sim_params,
    load_system_config,
)
from topoq.models import HybridSystem
from topoq.topoq_scheduler import TopoQScheduler


class _RemapOnlyRandom:
    def choice(self, values):
        return "remap" if "remap" in values else values[0]

    @staticmethod
    def uniform(lower, upper):
        return lower

    @staticmethod
    def random():
        return 0.0


class SAPhysicalRemapTests(unittest.TestCase):
    def _scheduler(self):
        system = HybridSystem(
            load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )
        task = {
            "id": "q0",
            "type": "quantum",
            "n_qubits": 2,
            "n_gates": 10,
            "depth": 2,
            "estimated_time_ms": 1.0,
        }
        workload = SimpleNamespace(
            id="sa-remap-test",
            tasks={"q0": task},
            quantum_tasks=["q0"],
            classical_tasks=[],
            predecessors=defaultdict(list),
            successors=defaultdict(list),
            edge_data={},
            get_shared_logical_qubits=lambda predecessor, tid: ([], False),
            get_exit_tasks=lambda: ["q0"],
        )
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.system = system
        scheduler.workload = workload
        scheduler.params = DEFAULT_PARAMS.copy()
        scheduler.ablation = {}
        scheduler.rng = _RemapOnlyRandom()
        scheduler.topology_scores = {qid: 0.5 for qid in system.qpus}
        scheduler._interaction_graphs = {
            "q0": {
                "nodes": [0, 1],
                "edges": {
                    (0, 1): {
                        "weight": 1.0,
                        "two_qubit_gates": 1,
                    }
                },
                "two_qubit_operations": None,
            }
        }
        scheduler.classical_link_timeline = {"original-classical": [1]}
        scheduler.quantum_link_timeline = {"original-quantum": [2]}
        scheduler._intra_task_remote_gates = {"q0": 0}
        scheduler._intra_task_remote_gates_by_link = {"q0": {}}
        return scheduler

    @staticmethod
    def _initial_state():
        sched = {
            "q0": {
                "start": 0.0,
                "end": 1.0,
                "qpu": "Q3",
                "qpu_set": ["Q3"],
                "partition_sizes": {"Q3": 2},
                "physical_qubits_by_qpu": {"Q3": [0, 1]},
                "mapped_physical_qubits_by_qpu": {"Q3": [0, 1]},
                "entanglement_wait": 0.0,
                "n_remote_gates": 0,
                "n_swap_gates": 0,
                "routing_overhead_ms": 0.0,
                "decoherence_penalty": 0.0,
                "space_shared": False,
                "space_shared_with": [],
                "space_sharing_exclusive_routing": False,
                "space_sharing_exclusive_qpus": [],
                "qpu_capacity_wait": 0.0,
                "dependency_transfers": [],
                "entanglement_reservations": [],
            }
        }
        assignment = {"q0": "Q3"}
        mapping = {
            "q0": {
                "primary_qpu": "Q3",
                "qpu_set": ["Q3"],
                "logical_partition": {0: "Q3", 1: "Q3"},
                "physical_mapping": {0: 0, 1: 1},
                "remote_gates_by_link": {},
            }
        }
        return sched, assignment, mapping

    def _real_remap_state(self):
        scheduler = self._scheduler()
        sched, assignment, mapping = self._initial_state()
        state = scheduler._build_sa_physical_remap_state(
            "q0",
            sched,
            assignment,
            mapping,
            scheduler._intra_task_remote_gates,
            scheduler._intra_task_remote_gates_by_link,
        )
        self.assertIsNotNone(state)
        return scheduler, sched, assignment, mapping, state

    def test_duration_weighted_physical_load_matches_formula(self):
        scheduler = self._scheduler()
        sched = {
            "q0": {
                "start": 0.0,
                "end": 9.0,
                "partition_sizes": {"Q0": 2, "Q1": 1},
            },
            "q1": {
                "start": 1.0,
                "end": 7.0,
                "qpu": "Q1",
            },
        }
        scheduler.workload.quantum_tasks = ["q0", "q1"]

        load = scheduler._quantum_qpu_load(sched)

        self.assertAlmostEqual(load["Q0"], 6.0)
        self.assertAlmostEqual(load["Q1"], 9.0)
        self.assertEqual(load["Q2"], 0.0)

    def test_real_remap_rebuilds_mapping_routing_and_caches(self):
        scheduler, _, _, mapping, state = self._real_remap_state()
        remapped = state["mapping"]["q0"]

        self.assertNotEqual(
            scheduler._physical_mapping_signature(remapped),
            scheduler._physical_mapping_signature(mapping["q0"]),
        )
        self.assertEqual(
            state["assignment"]["q0"], remapped["primary_qpu"]
        )
        self.assertEqual(
            set(remapped["logical_partition"].values()),
            set(remapped["qpu_set"]),
        )
        self.assertEqual(
            state["schedule"]["q0"]["qpu_set"], remapped["qpu_set"]
        )
        self.assertEqual(
            state["schedule"]["q0"]["n_swap_gates"],
            remapped["physical_routing"]["swap_gate_count"],
        )
        self.assertEqual(
            state["remote_gates"]["q0"],
            state["schedule"]["q0"]["n_remote_gates"],
        )
        self.assertEqual(
            state["remote_gates_by_link"]["q0"],
            remapped["remote_gates_by_link"],
        )
        self.assertEqual(
            remapped["sa_remap"]["selection_strategy"],
            "least_loaded",
        )

    def test_real_remap_preserves_all_enforced_constraints(self):
        scheduler, _, _, _, state = self._real_remap_state()
        scheduler.classical_link_timeline = state[
            "classical_link_timeline"
        ]
        scheduler.quantum_link_timeline = state["quantum_link_timeline"]
        scheduler._intra_task_remote_gates = state["remote_gates"]
        scheduler._intra_task_remote_gates_by_link = state[
            "remote_gates_by_link"
        ]

        violations = scheduler._check_constraints(
            state["schedule"], state["assignment"], state["mapping"]
        )

        self.assertEqual(
            [item for item in violations if item.get("enforced", True)],
            [],
        )

    def test_equivalent_mapping_is_not_a_candidate(self):
        scheduler = self._scheduler()
        _, _, mapping = self._initial_state()
        qpu_usage = {qid: 0.0 for qid in scheduler.system.qpus}
        first = scheduler._build_quantum_mapping_candidate(
            "q0",
            {},
            qpu_usage,
            selection_strategy="least_loaded",
            require_coherence=True,
        )
        self.assertIsNotNone(first["mapping"])
        scheduler.params["qpu_partition_max_attempts"] = 1

        repeated = scheduler._build_quantum_mapping_candidate(
            "q0",
            {"q0": first["mapping"]},
            qpu_usage,
            selection_strategy="least_loaded",
            require_coherence=True,
            exclude_equivalent_to=first["mapping"],
        )

        self.assertIsNone(repeated["mapping"])
        self.assertEqual(
            repeated["attempts"][-1]["status"],
            "rejected_equivalent",
        )

    def test_candidate_builder_preserves_c6_physical_location(self):
        scheduler = self._scheduler()
        scheduler.workload.tasks["pred"] = {
            "id": "pred",
            "type": "quantum",
            "n_qubits": 2,
        }
        scheduler.workload.predecessors["q0"] = ["pred"]
        scheduler.workload.get_shared_logical_qubits = (
            lambda predecessor, tid: ([0], False)
        )
        mapping = {
            "pred": {
                "primary_qpu": "Q1",
                "qpu_set": ["Q1"],
                "logical_partition": {0: "Q1", 1: "Q1"},
                "physical_mapping": {0: 0, 1: 1},
            },
            "q0": self._initial_state()[2]["q0"],
        }
        qpu_usage = {qid: 0.0 for qid in scheduler.system.qpus}
        qpu_usage["Q1"] = 100.0

        result = scheduler._build_quantum_mapping_candidate(
            "q0",
            mapping,
            qpu_usage,
            selection_strategy="least_loaded",
            require_coherence=True,
        )

        self.assertIsNotNone(result["mapping"])
        self.assertEqual(result["mapping"]["logical_partition"][0], "Q1")
        self.assertEqual(result["mapping"]["physical_mapping"][0], 0)
        self.assertEqual(
            result["mapping"]["mapping_continuity"][0]["predecessor"],
            "pred",
        )

    def test_failed_reschedule_restores_live_scheduler_state(self):
        scheduler = self._scheduler()
        sched, assignment, mapping = self._initial_state()
        original_assignment = deepcopy(assignment)
        original_mapping = deepcopy(mapping)
        original_classical = deepcopy(scheduler.classical_link_timeline)
        original_quantum = deepcopy(scheduler.quantum_link_timeline)
        original_remote = deepcopy(scheduler._intra_task_remote_gates)
        original_by_link = deepcopy(
            scheduler._intra_task_remote_gates_by_link
        )
        candidate_mapping = deepcopy(mapping["q0"])
        candidate_mapping["primary_qpu"] = "Q0"
        candidate_mapping["qpu_set"] = ["Q0"]
        candidate_mapping["logical_partition"] = {0: "Q0", 1: "Q0"}
        candidate_result = {
            "mapping": candidate_mapping,
            "n_remote_gates": 0,
            "remote_gates_by_link": {},
            "rank": 0,
            "attempts": [{"status": "selected"}],
        }

        def fail_after_mutation(candidate_assignment, candidate_mapping):
            scheduler.classical_link_timeline = {"polluted": [3]}
            scheduler.quantum_link_timeline = {"polluted": [4]}
            scheduler._intra_task_remote_gates = {"polluted": 5}
            scheduler._intra_task_remote_gates_by_link = {"polluted": {}}
            raise ValueError("candidate rejected")

        with patch.object(
                scheduler,
                "_build_quantum_mapping_candidate",
                return_value=candidate_result,
        ), patch.object(
                scheduler,
                "_phase3_4_joint_scheduling",
                side_effect=fail_after_mutation,
        ):
            state = scheduler._build_sa_physical_remap_state(
                "q0",
                sched,
                assignment,
                mapping,
                original_remote,
                original_by_link,
            )

        self.assertIsNone(state)
        self.assertEqual(assignment, original_assignment)
        self.assertEqual(mapping, original_mapping)
        self.assertEqual(scheduler.classical_link_timeline, original_classical)
        self.assertEqual(scheduler.quantum_link_timeline, original_quantum)
        self.assertEqual(scheduler._intra_task_remote_gates, original_remote)
        self.assertEqual(
            scheduler._intra_task_remote_gates_by_link, original_by_link
        )

    def test_best_state_returns_matching_timelines_caches_and_stats(self):
        scheduler, sched, assignment, mapping, remap_state = (
            self._real_remap_state()
        )
        scheduler.params["SA_cooling"] = 0.0

        def objective(
                candidate_sched, candidate_assignment, candidate_mapping,
                classical_timeline, quantum_timeline, reference,
                intra_remote_gates=None):
            total = 1.0 if candidate_assignment["q0"] != "Q3" else 2.0
            return {
                "total": total,
                "raw": {},
                "normalized": {},
                "weights": {},
                "violations": [],
                "enforced_violations": [],
                "reported_violations": [],
            }

        with patch.object(
                scheduler,
                "_build_sa_physical_remap_state",
                return_value=deepcopy(remap_state),
        ), patch.object(
                scheduler,
                "_compute_schedule_objective",
                side_effect=objective,
        ):
            result = scheduler._simulated_annealing(
                sched, assignment, mapping
            )

        best_sched, best_assignment, best_mapping = result[:3]
        best_classical, best_quantum, best_remote, best_by_link = result[3:]
        self.assertNotEqual(best_assignment["q0"], "Q3")
        self.assertEqual(best_sched["q0"]["qpu"], best_assignment["q0"])
        self.assertEqual(
            best_mapping["q0"]["primary_qpu"], best_assignment["q0"]
        )
        self.assertEqual(best_classical, remap_state["classical_link_timeline"])
        self.assertEqual(best_quantum, remap_state["quantum_link_timeline"])
        self.assertEqual(best_remote, remap_state["remote_gates"])
        self.assertEqual(best_by_link, remap_state["remote_gates_by_link"])
        self.assertEqual(
            scheduler.last_sa_objective["move_stats"]["remap"],
            {"attempted": 1, "feasible": 1, "accepted": 1, "best": 1},
        )


if __name__ == "__main__":
    unittest.main()
