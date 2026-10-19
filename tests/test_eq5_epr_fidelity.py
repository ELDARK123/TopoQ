import math
import unittest
from types import SimpleNamespace

from topoq.fidelity import (
    build_epr_event,
    build_epr_quality_diagnostics,
    collect_task_epr_events,
    estimate_purification_scenario_fidelities,
    estimate_task_log_fidelity,
    resolve_epr_quality,
)
from topoq.purification import build_purification_plan
from topoq.baselines import GreedyScheduler
from topoq.config import (
    get_experiment,
    load_dag,
    load_network_topology,
    load_sim_params,
    load_system_config,
)
from topoq.models import HybridSystem, Workload
from topoq.simulator import Simulator


class _System(SimpleNamespace):
    def get_quantum_link(self, qpu_i, qpu_j):
        return self.quantum_links.get(frozenset((qpu_i, qpu_j)))


class Eq5EprFidelityTests(unittest.TestCase):
    @staticmethod
    def _qpu():
        return {
            "n_qubits": 16,
            "coherence_time_us": 1e12,
            "gate_times_ns": {"1q": 0.0, "2q": 0.0, "measure": 0.0},
            "gate_error_rates": {"1q": 0.0, "2q": 0.0, "measure": 0.0},
        }

    @classmethod
    def _system(cls, explicit_fidelity=0.9, purification=None):
        link = {
            "rate_hz": 1000.0,
            "decoherence_rate_per_ms": 0.02,
            "max_concurrent_epr": 4,
            "distance_km": 1.0,
        }
        if explicit_fidelity is not None:
            link["epr_fidelity"] = explicit_fidelity
        sim_params = {
            "fidelity_parameters": {
                "F_prep": 0.999,
                "F_EPR_range": [0.85, 0.95],
                "F_min": 0.99,
                "F_min_qaoa": 0.985,
            },
            "quantum_link_parameters": {
                "eta_range_per_ms": [0.01, 0.1],
            },
            "purification_policy": purification or build_purification_plan(
                0.85
            ),
        }
        return _System(
            qpus={"q0": cls._qpu()},
            quantum_links={frozenset(("q0", "q1")): link},
            sim_params=sim_params,
        )

    def test_eq5_uses_each_group_count_quality_rate_and_hold_time(self):
        system = self._system()
        intra = build_epr_event(
            system, ("q0", "q1"), 2, ready=1.0, consume=4.0,
            task_id="task", source="test_intra",
        )
        dependency = {
            **build_epr_event(
                system, ("q0", "q1"), 1, ready=2.0, consume=4.0,
                task_id="task", predecessor="pred", source="test_dependency",
            ),
            "epr_fidelity": 0.8,
            "decoherence_rate_per_ms": 0.03,
        }
        schedule_item = {
            "start": 4.0,
            "n_remote_gates": 2,
            "entanglement_reservations": [intra],
            "dependency_transfers": [{
                "type": "quantum-quantum",
                "entanglement_reservations": [dependency],
            }],
        }
        result = estimate_task_log_fidelity(
            system,
            {"n_gates": 0, "n_qubits": 1, "depth": 1},
            self._qpu(),
            schedule_item=schedule_item,
        )
        expected_log_ent = (
            2 * (math.log(0.9) - 0.02 * 3.0)
            + math.log(0.8) - 0.03 * 2.0
        )
        self.assertEqual(result["fidelity_model"], "eq5_per_epr_events")
        self.assertEqual(result["n_remote_gates"], 2)
        self.assertEqual(result["n_remote_epr_pairs"], 3)
        self.assertAlmostEqual(result["log_entanglement"], expected_log_ent)
        self.assertAlmostEqual(
            result["fidelity"], 0.999 * math.exp(expected_log_ent)
        )

    def test_dependency_events_are_collected_with_actual_times(self):
        system = self._system()
        event = build_epr_event(
            system, ("q0", "q1"), 1, ready=1.0, consume=2.0
        )
        event["actual_end"] = 3.0
        event["actual_consume"] = 5.5
        events = collect_task_epr_events(system, {
            "start": 5.0,
            "dependency_transfers": [{
                "type": "quantum-quantum",
                "entanglement_reservations": [event],
            }],
        })
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["hold_ms"], 2.5)

    def test_missing_link_fidelity_uses_global_lower_bound_with_source(self):
        system = self._system(explicit_fidelity=None)
        quality = resolve_epr_quality(
            system, system.get_quantum_link("q0", "q1")
        )
        self.assertEqual(quality["epr_fidelity"], 0.85)
        self.assertEqual(quality["fidelity_source"], "global_lower_bound")

    def test_no_events_preserve_eq7_candidate_estimate(self):
        system = self._system()
        result = estimate_task_log_fidelity(
            system,
            {"n_gates": 0, "n_qubits": 1, "depth": 1},
            self._qpu(),
            estimated_remote_gates=2,
            estimated_hold_ms=3.0,
        )
        self.assertEqual(result["fidelity_model"], "eq7_conservative_estimate")
        self.assertEqual(result["n_remote_epr_pairs"], 2)

    def test_final_task_without_remote_epr_has_empty_eq5_product(self):
        system = self._system()
        result = estimate_task_log_fidelity(
            system,
            {"n_gates": 0, "n_qubits": 1, "depth": 1},
            self._qpu(),
            schedule_item={
                "start": 1.0,
                "end": 2.0,
                "qpu": "q0",
                "n_remote_gates": 0,
                "entanglement_reservations": [],
            },
        )
        self.assertEqual(result["n_remote_epr_pairs"], 0)
        self.assertEqual(result["log_entanglement"], 0.0)
        self.assertEqual(result["fidelity_model"], "eq5_per_epr_events")

    def test_final_remote_task_without_events_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "per-EPR reservation events"):
            estimate_task_log_fidelity(
                self._system(),
                {"n_gates": 0, "n_qubits": 1, "depth": 1},
                self._qpu(),
                schedule_item={
                    "start": 1.0,
                    "end": 2.0,
                    "qpu": "q0",
                    "n_remote_gates": 1,
                    "entanglement_reservations": [],
                },
            )

    def test_purification_is_resolved_per_event_fidelity(self):
        policy = build_purification_plan(
            0.85, "bbpssw_fixed_rounds", rounds=1
        )
        system = self._system(purification=policy)
        event = build_epr_event(
            system, ("q0", "q1"), 2, ready=1.0, consume=1.0
        )
        physical = estimate_task_log_fidelity(
            system,
            {"n_gates": 0, "n_qubits": 1, "depth": 1},
            self._qpu(),
            schedule_item={
                "start": 1.0,
                "n_remote_gates": 2,
                "entanglement_reservations": [event],
            },
        )
        scenario = estimate_purification_scenario_fidelities(
            system, {"task": physical}
        )["task"]
        event_plan = build_purification_plan(
            0.9, "bbpssw_fixed_rounds", rounds=1
        )
        self.assertAlmostEqual(
            scenario["fidelity"],
            0.999 * event_plan["output_fidelity"] ** 2,
        )
        self.assertAlmostEqual(
            scenario["expected_raw_epr_pairs"],
            2 * event_plan["expected_raw_pairs_per_output"],
        )

    def test_threshold_diagnostics_record_manuscript_conflict(self):
        system = self._system()
        purified = build_purification_plan(
            0.85, "bbpssw_target", target=0.95
        )
        diagnostics = build_epr_quality_diagnostics(
            system.sim_params, purified, 0.99
        )
        self.assertAlmostEqual(
            diagnostics["optimistic_raw_min_single_remote_ceiling"],
            0.84915,
        )
        self.assertAlmostEqual(
            diagnostics["optimistic_raw_max_single_remote_ceiling"],
            0.94905,
        )
        self.assertAlmostEqual(
            diagnostics["optimistic_purified_single_remote_ceiling"],
            0.999 * purified["output_fidelity"],
        )
        self.assertTrue(diagnostics["threshold_conflict"])
        self.assertFalse(diagnostics["purified_can_meet_threshold"])

    def test_simulator_without_injected_policy_reports_raw_output_fidelity(self):
        system = self._system()
        del system.sim_params["purification_policy"]
        workload = SimpleNamespace(
            quantum_tasks=["task"],
            tasks={"task": {"n_gates": 0, "n_qubits": 1, "depth": 1}},
        )
        metrics = Simulator(system, system.sim_params)._compute_fidelity_metrics(
            {"task": {
                "start": 0.0,
                "end": 1.0,
                "qpu": "q0",
                "n_remote_gates": 0,
            }},
            {"task": "q0"},
            workload,
        )
        self.assertEqual(metrics["epr_purification_raw_fidelity"], 0.85)
        self.assertEqual(metrics["epr_purification_output_fidelity"], 0.85)

    def test_real_greedy_schedule_emits_common_format_epr_events(self):
        spec = get_experiment("VQE-50_Small")
        system = HybridSystem(
            load_system_config(spec.system_config),
            load_network_topology(spec.network_topology),
            load_sim_params(),
        )
        workload = Workload(load_dag(spec.dag_id))
        schedule, _, _ = GreedyScheduler(system, workload, seed=42).schedule()
        events = [
            event
            for tid in workload.quantum_tasks
            for event in collect_task_epr_events(system, schedule[tid])
        ]

        self.assertTrue(events)
        self.assertTrue(all(
            event["event_source"] == "baseline_sequential_generation"
            for event in events
        ))
        self.assertTrue(all(event["epr_count"] > 0 for event in events))


if __name__ == "__main__":
    unittest.main()
