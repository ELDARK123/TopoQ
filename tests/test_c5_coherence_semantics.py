import unittest
from collections import defaultdict
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace

from topoq.config import (
    DEFAULT_PARAMS,
    load_network_topology,
    load_sim_params,
    load_system_config,
)
from topoq.models import HybridSystem
from topoq.main import main
from topoq.topoq_scheduler import TopoQScheduler
from topoq.validators import (
    estimate_gate_batch_time_ms,
    validate_coherence_window,
)


class C5CoherenceSemanticsTests(unittest.TestCase):
    def _validation_fixture(self):
        system = SimpleNamespace(qpus={
            "Q0": {
                "coherence_time_us": 200.0,
                "gate_times_ns": {
                    "1q": 35.6, "2q": 334.0, "measure": 1200.0,
                },
            },
            "Q1": {
                "coherence_time_us": 50.0,
                "gate_times_ns": {
                    "1q": 35.6, "2q": 334.0, "measure": 1200.0,
                },
            },
        })
        workload = SimpleNamespace(
            quantum_tasks=["q0"],
            tasks={"q0": {"n_gates": 500, "depth": 1}},
        )
        sched = {
            "q0": {
                "start": 0.0,
                "end": 1.0,
                "qpu_set": ["Q0", "Q1"],
                "dependency_transfers": [],
                "entanglement_reservations": [],
            }
        }
        assignment = {"q0": "Q0"}
        mapping = {"q0": {"qpu_set": ["Q0", "Q1"]}}
        return system, workload, sched, assignment, mapping

    def test_batch_constraint_checks_every_used_qpu(self):
        system, workload, sched, assignment, mapping = (
            self._validation_fixture()
        )

        violations = validate_coherence_window(
            system,
            workload,
            sched,
            assignment,
            mapping,
            safety_fraction=0.05,
            task_window_mode="report",
        )

        batch = [
            item for item in violations
            if item["constraint"] == "C5_COHERENCE_WINDOW"
        ]
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["details"]["violating_qpus"], ["Q1"])
        self.assertTrue(batch[0]["enforced"])

    def test_task_window_is_diagnostic_in_report_and_strict_modes(self):
        system, workload, sched, assignment, mapping = (
            self._validation_fixture()
        )

        reported = validate_coherence_window(
            system, workload, sched, assignment, mapping,
            safety_fraction=0.05, task_window_mode="report",
        )
        strict = validate_coherence_window(
            system, workload, sched, assignment, mapping,
            safety_fraction=0.05, task_window_mode="strict",
        )

        report_item = next(
            item for item in reported
            if item["constraint"] == "C5_TASK_WINDOW_DIAGNOSTIC"
        )
        strict_item = next(
            item for item in strict
            if item["constraint"] == "C5_TASK_WINDOW_DIAGNOSTIC"
        )
        self.assertFalse(report_item["enforced"])
        self.assertFalse(strict_item["enforced"])
        self.assertEqual(report_item["details"]["mode"], "diagnostic")
        self.assertEqual(strict_item["details"]["mode"], "diagnostic")
        self.assertEqual(
            report_item["details"]["first_entanglement_start_ms"], 0.0
        )
        self.assertEqual(report_item["details"]["limiting_qpu"], "Q1")

    def _epr_event_fixture(self, memory_lifetime_ms, **time_overrides):
        system, workload, sched, assignment, mapping = (
            self._validation_fixture()
        )
        sched["q0"]["qpu_set"] = ["Q0"]
        mapping["q0"]["qpu_set"] = ["Q0"]
        workload.tasks["q0"] = {"n_gates": 0, "depth": 1}
        record = {
            "task": "q0",
            "link": ("Q0", "Q1"),
            "epr_count": 1,
            "start": 0.0,
            "generation_start": 0.0,
            "ready": 0.10,
            "consume": 0.30,
            "epr_fidelity": 0.90,
            "fidelity_source": "test",
            "decoherence_rate_per_ms": 0.001,
            "epr_memory_lifetime_ms": memory_lifetime_ms,
            "memory_lifetime_source": (
                "test_calibration"
                if memory_lifetime_ms is not None else "unspecified"
            ),
        }
        record.update(time_overrides)
        sched["q0"]["entanglement_reservations"] = [record]
        return system, workload, sched, assignment, mapping

    def test_calibrated_epr_holding_report_and_strict_modes(self):
        fixture = self._epr_event_fixture(0.10)
        reported = validate_coherence_window(
            *fixture, event_mode="report"
        )
        strict = validate_coherence_window(*fixture, event_mode="strict")

        report_item = next(
            item for item in reported
            if item["constraint"] == "C5_EPR_HOLDING_WINDOW"
        )
        strict_item = next(
            item for item in strict
            if item["constraint"] == "C5_EPR_HOLDING_WINDOW"
        )
        self.assertFalse(report_item["enforced"])
        self.assertTrue(strict_item["enforced"])
        self.assertAlmostEqual(report_item["details"]["hold_time_ms"], 0.20)
        self.assertAlmostEqual(
            report_item["details"]["memory_lifetime_ms"], 0.10
        )

    def test_missing_memory_lifetime_report_and_strict_modes(self):
        fixture = self._epr_event_fixture(None)
        reported = validate_coherence_window(
            *fixture, event_mode="report"
        )
        strict = validate_coherence_window(*fixture, event_mode="strict")

        report_item = next(
            item for item in reported
            if item["constraint"] ==
            "C5_EPR_MEMORY_LIFETIME_UNSPECIFIED"
        )
        strict_item = next(
            item for item in strict
            if item["constraint"] ==
            "C5_EPR_MEMORY_LIFETIME_UNSPECIFIED"
        )
        self.assertFalse(report_item["enforced"])
        self.assertTrue(strict_item["enforced"])
        self.assertEqual(report_item["details"]["event_count"], 1)
        self.assertAlmostEqual(report_item["details"]["max_hold_ms"], 0.20)

    def test_actual_event_times_override_scheduled_times(self):
        fixture = self._epr_event_fixture(
            0.10,
            ready=0.10,
            consume=0.15,
            actual_end=0.20,
            actual_consume=0.35,
        )
        violations = validate_coherence_window(
            *fixture, event_mode="strict"
        )

        item = next(
            violation for violation in violations
            if violation["constraint"] == "C5_EPR_HOLDING_WINDOW"
        )
        self.assertAlmostEqual(item["details"]["ready_time_ms"], 0.20)
        self.assertAlmostEqual(item["details"]["consume_time_ms"], 0.35)
        self.assertAlmostEqual(item["details"]["hold_time_ms"], 0.15)

    def test_legacy_task_window_mode_alias_controls_epr_events(self):
        fixture = self._epr_event_fixture(None)
        violations = validate_coherence_window(
            *fixture, task_window_mode="strict"
        )

        item = next(
            violation for violation in violations
            if violation["constraint"] ==
            "C5_EPR_MEMORY_LIFETIME_UNSPECIFIED"
        )
        self.assertTrue(item["enforced"])

    def test_conflicting_cli_names_are_rejected(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                main([
                    "--profile", "quick",
                    "--list-experiments",
                    "--coherence-event-mode", "strict",
                    "--coherence-task-window-mode", "report",
                ])

    def test_default_batch_margin_matches_c5_equation(self):
        self.assertEqual(DEFAULT_PARAMS["T_safety_frac"], 0.05)
        self.assertEqual(
            load_sim_params()["algorithm_parameters"]["T_safety_fraction"],
            0.05,
        )
        task = {"n_gates": 500, "depth": 1}
        qpu = {
            "gate_times_ns": {
                "1q": 35.6, "2q": 334.0, "measure": 1200.0,
            }
        }
        self.assertAlmostEqual(
            estimate_gate_batch_time_ms(task, qpu), 0.068236
        )

    def test_deadline_reservation_uses_deadline_as_consumption_time(self):
        system = HybridSystem(
            load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )
        scheduler = TopoQScheduler(system, SimpleNamespace(), seed=42)
        scheduler.quantum_link_timeline = defaultdict(list)
        link_key = next(iter(system.quantum_links))

        _, records = scheduler._reserve_entanglement_batches(
            link_key,
            base_ready=0.0,
            demand=1,
            task_id="q0",
            deadline=10.0,
        )

        self.assertTrue(records)
        self.assertTrue(all(record["consume"] == 10.0 for record in records))
        self.assertTrue(all(
            "epr_memory_lifetime_ms" in record for record in records
        ))

    def _repair_scheduler(self):
        system = HybridSystem(
            load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )
        task = {
            "id": "q0",
            "type": "quantum",
            "n_qubits": 1,
            "n_gates": 1000,
            "depth": 1,
            "estimated_time_ms": 1.0,
        }
        workload = SimpleNamespace(
            tasks={"q0": task},
            quantum_tasks=["q0"],
            classical_tasks=[],
            predecessors=defaultdict(list),
            successors=defaultdict(list),
            edge_data={},
            get_shared_logical_qubits=lambda predecessor, tid: ([], False),
        )
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.system = system
        scheduler.workload = workload
        scheduler.params = DEFAULT_PARAMS.copy()
        scheduler.ablation = {}
        scheduler.topology_scores = {qid: 0.5 for qid in system.qpus}
        scheduler._interaction_graphs = {
            "q0": {
                "nodes": [0],
                "edges": {},
                "two_qubit_operations": None,
            }
        }
        scheduler._intra_task_remote_gates = {"q0": 0}
        scheduler._intra_task_remote_gates_by_link = {"q0": {}}
        return scheduler

    def test_coherence_repair_rebuilds_complete_mapping(self):
        scheduler = self._repair_scheduler()
        assignment = {"q0": "Q3"}
        mapping = {"q0": {
            "primary_qpu": "Q3",
            "qpu_set": ["Q3"],
            "logical_partition": {0: "Q3"},
            "physical_mapping": {0: 0},
            "remote_gates_by_link": {},
        }}
        qpu_load = defaultdict(float, {"Q3": 1.0})

        repaired = scheduler._remap_task_for_coherence(
            "q0", assignment, mapping, qpu_load
        )

        self.assertTrue(repaired)
        self.assertNotEqual(assignment["q0"], "Q3")
        self.assertEqual(mapping["q0"]["primary_qpu"], assignment["q0"])
        self.assertEqual(
            set(mapping["q0"]["logical_partition"].values()),
            set(mapping["q0"]["qpu_set"]),
        )
        self.assertIn("physical_routing", mapping["q0"])
        self.assertIn("n_swap_gates", mapping["q0"])
        self.assertIn("routing_overhead_ms", mapping["q0"])
        self.assertIn("coherence_repair", mapping["q0"])
        self.assertEqual(scheduler._intra_task_remote_gates["q0"], 0)
        self.assertTrue(scheduler._coherence_candidate_feasible(
            scheduler.workload.tasks["q0"], mapping["q0"]["qpu_set"]
        ))

    def test_impossible_strict_window_preserves_mapping(self):
        scheduler = self._repair_scheduler()
        assignment = {"q0": "Q3"}
        mapping = {"q0": {
            "primary_qpu": "Q3",
            "qpu_set": ["Q3"],
            "logical_partition": {0: "Q3"},
            "physical_mapping": {0: 0},
        }}
        original = dict(mapping["q0"])

        repaired = scheduler._remap_task_for_coherence(
            "q0",
            assignment,
            mapping,
            defaultdict(float, {"Q3": 1.0}),
            required_window_ms=10.0,
        )

        self.assertFalse(repaired)
        self.assertEqual(assignment["q0"], "Q3")
        self.assertEqual(mapping["q0"], original)

    def test_coherence_repair_preserves_c6_shared_physical_location(self):
        scheduler = self._repair_scheduler()
        scheduler.workload.predecessors["q0"] = ["pred"]
        scheduler.workload.get_shared_logical_qubits = (
            lambda predecessor, tid: ([0], False)
        )
        assignment = {"pred": "Q0", "q0": "Q3"}
        mapping = {
            "pred": {
                "primary_qpu": "Q0",
                "qpu_set": ["Q0"],
                "logical_partition": {0: "Q0"},
                "physical_mapping": {0: 0},
            },
            "q0": {
                "primary_qpu": "Q3",
                "qpu_set": ["Q3"],
                "logical_partition": {0: "Q3"},
                "physical_mapping": {0: 0},
            },
        }

        repaired = scheduler._remap_task_for_coherence(
            "q0",
            assignment,
            mapping,
            defaultdict(float, {"Q3": 1.0}),
        )

        self.assertTrue(repaired)
        self.assertEqual(assignment["q0"], "Q0")
        self.assertEqual(mapping["q0"]["logical_partition"][0], "Q0")
        self.assertEqual(mapping["q0"]["physical_mapping"][0], 0)
        self.assertEqual(
            mapping["q0"]["mapping_continuity"][0]["predecessor"],
            "pred",
        )


if __name__ == "__main__":
    unittest.main()
