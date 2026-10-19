import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import (
    EPR_PURIFICATION_MODES,
    get_experiment,
    validate_purification_policy,
)
from topoq.fidelity import (
    estimate_purification_scenario_fidelities,
    estimate_task_log_fidelity,
)
from topoq.purification import bbpssw_round, build_purification_plan
from topoq.runner import run_experiment_spec, run_with_repetitions
from topoq.simulator import Simulator


class EprPurificationReportingTests(unittest.TestCase):
    def test_modes_and_strict_parameter_validation(self):
        self.assertEqual(EPR_PURIFICATION_MODES, (
            "none",
            "bbpssw_fixed_rounds",
            "bbpssw_target",
        ))
        self.assertEqual(validate_purification_policy(), {
            "mode": "none",
            "rounds": None,
            "target_fidelity": None,
            "max_rounds": 8,
        })
        self.assertEqual(
            validate_purification_policy(
                "bbpssw_fixed_rounds", 2, None, 6
            ),
            {
                "mode": "bbpssw_fixed_rounds",
                "rounds": 2,
                "target_fidelity": None,
                "max_rounds": 6,
            },
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_purification_policy("bbpssw_fixed_rounds")
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_purification_policy("bbpssw_target")
        with self.assertRaisesRegex(ValueError, "does not accept"):
            validate_purification_policy(
                "bbpssw_target", rounds=1, target=0.95
            )
        with self.assertRaisesRegex(ValueError, "explicit BBPSSW"):
            validate_purification_policy("none", rounds=1)
        for target in (0.5, 1.01, math.inf, "bad"):
            with self.assertRaisesRegex(ValueError, "target"):
                validate_purification_policy(
                    "bbpssw_target", target=target
                )
        for invalid in (math.inf, math.nan, True):
            with self.assertRaisesRegex(ValueError, "rounds"):
                validate_purification_policy(
                    "bbpssw_fixed_rounds", rounds=invalid
                )
            with self.assertRaisesRegex(ValueError, "max_rounds"):
                validate_purification_policy(
                    "bbpssw_target", target=0.95,
                    max_rounds=invalid,
                )
        with self.assertRaisesRegex(ValueError, "target"):
            validate_purification_policy(
                "bbpssw_target", target=True
            )

    def test_bbpssw_round_and_target_costs_match_reference_values(self):
        output, success = bbpssw_round(0.85)
        self.assertAlmostEqual(output, 0.8841463414634146)
        self.assertAlmostEqual(success, 0.82)

        plan = build_purification_plan(
            0.85,
            "bbpssw_target",
            target=0.95,
            max_rounds=8,
        )
        self.assertEqual(plan["rounds"], 4)
        self.assertTrue(plan["target_met"])
        self.assertAlmostEqual(
            plan["output_fidelity"], 0.9553522406583336
        )
        self.assertAlmostEqual(
            plan["expected_raw_pairs_per_output"],
            27.76503755564679,
        )
        self.assertAlmostEqual(
            plan["expected_purification_attempts_per_output"],
            23.101991505544845,
        )

    def test_fixed_rounds_and_none_are_explicitly_separate(self):
        none = build_purification_plan(0.85)
        fixed = build_purification_plan(
            0.85, "bbpssw_fixed_rounds", rounds=2
        )
        self.assertEqual(none["rounds"], 0)
        self.assertEqual(none["output_fidelity"], 0.85)
        self.assertEqual(none["expected_raw_pairs_per_output"], 1.0)
        self.assertEqual(fixed["rounds"], 2)
        self.assertAlmostEqual(fixed["output_fidelity"], 0.9134033532472536)

    def test_physical_fidelity_is_unchanged_and_scenario_replaces_only_epr(self):
        qpu = {
            "coherence_time_us": 100.0,
            "gate_times_ns": {"1q": 10.0, "2q": 20.0, "measure": 30.0},
            "gate_error_rates": {"1q": 0.0, "2q": 0.0, "measure": 0.0},
        }
        task = {"n_gates": 0, "n_qubits": 1, "depth": 1}
        plan = build_purification_plan(
            0.85, "bbpssw_target", target=0.95
        )
        system = SimpleNamespace(
            sim_params={
                "fidelity_parameters": {
                    "F_prep": 0.999,
                    "F_EPR_range": [0.85, 0.95],
                },
                "quantum_link_parameters": {
                    "eta_range_per_ms": [0.01, 0.1],
                },
                "purification_policy": plan,
            }
        )
        physical = estimate_task_log_fidelity(
            system,
            task,
            qpu,
            schedule_item={
                "n_remote_gates": 2,
                "entanglement_reservations": [{
                    "task": "task",
                    "link": ("q0", "q1"),
                    "epr_count": 2,
                    "ready": 0.0,
                    "consume": 0.0,
                    "epr_fidelity": 0.85,
                    "fidelity_source": "test_fixture",
                    "decoherence_rate_per_ms": 0.0,
                    "event_source": "test_fixture",
                }],
            },
        )
        scenario = estimate_purification_scenario_fidelities(
            system, {"task": physical}
        )["task"]

        self.assertAlmostEqual(physical["fidelity"], 0.999 * 0.85 ** 2)
        self.assertAlmostEqual(
            scenario["fidelity"], 0.999 * plan["output_fidelity"] ** 2
        )
        self.assertAlmostEqual(
            physical["fidelity"],
            math.exp(physical["log_fidelity"]),
        )

    def test_simulator_reports_expected_cost_without_changing_physical_metric(self):
        plan = build_purification_plan(
            0.85, "bbpssw_fixed_rounds", rounds=1
        )
        qpu = {
            "n_qubits": 16,
            "coherence_time_us": 100.0,
            "gate_times_ns": {"1q": 10.0, "2q": 20.0, "measure": 30.0},
            "gate_error_rates": {"1q": 0.0, "2q": 0.0, "measure": 0.0},
        }
        system = SimpleNamespace(
            qpus={"q0": qpu},
            sim_params={
                "fidelity_parameters": {
                    "F_prep": 0.999,
                    "F_EPR_range": [0.85, 0.95],
                },
                "quantum_link_parameters": {
                    "eta_range_per_ms": [0.01, 0.1],
                },
                "purification_policy": plan,
            },
        )
        workload = SimpleNamespace(
            quantum_tasks=["task"],
            tasks={"task": {"n_gates": 0, "n_qubits": 1, "depth": 1}},
        )
        sched = {
            "task": {
                "start": 0.0,
                "end": 1.0,
                "qpu": "q0",
                "n_remote_gates": 3,
                "entanglement_reservations": [{
                    "task": "task",
                    "link": ("q0", "q1"),
                    "epr_count": 3,
                    "ready": 0.0,
                    "consume": 0.0,
                    "epr_fidelity": 0.85,
                    "fidelity_source": "test_fixture",
                    "decoherence_rate_per_ms": 0.0,
                    "event_source": "test_fixture",
                }],
            }
        }
        metrics = Simulator(system, system.sim_params)._compute_fidelity_metrics(
            sched, {"task": "q0"}, workload
        )

        self.assertAlmostEqual(metrics["fidelity"], 0.999 * 0.85 ** 3)
        self.assertGreater(
            metrics["purification_adjusted_fidelity"], metrics["fidelity"]
        )
        self.assertAlmostEqual(
            metrics["epr_purification_expected_raw_epr_pairs"],
            3 * plan["expected_raw_pairs_per_output"],
        )
        self.assertAlmostEqual(
            metrics["epr_purification_expected_attempts"],
            3 * plan["expected_purification_attempts_per_output"],
        )

    def test_runner_metadata_records_equal_policy_for_all_algorithms(self):
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0},
        ):
            result = run_experiment_spec(
                get_experiment("VQE-50_Small"),
                profile="quick",
                repetitions=1,
                algorithms=("Greedy", "TopoQ"),
                params={
                    "epr_purification_mode": "bbpssw_target",
                    "epr_purification_target": 0.95,
                },
            )

        metadata = result["metadata"]["epr_purification"]
        self.assertEqual(metadata["mode"], "bbpssw_target")
        self.assertEqual(metadata["rounds"], 4)
        self.assertTrue(metadata["physical_fidelity_unchanged"])
        self.assertEqual(
            metadata["scenario_metric"], "purification_adjusted_fidelity"
        )

    def test_repetition_averaging_preserves_purification_mode_string(self):
        sample = {
            "epr_purification_mode": "bbpssw_target",
            "fidelity": 0.8,
        }
        with patch(
            "topoq.runner.run_single_experiment", return_value=sample
        ):
            result = run_with_repetitions(
                "dag", "system", "network", "Greedy", n_rep=2
            )

        self.assertEqual(result["epr_purification_mode"], "bbpssw_target")
        self.assertEqual(result["fidelity"], 0.8)


if __name__ == "__main__":
    unittest.main()
