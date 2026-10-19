import math
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import (
    DEFAULT_PARAMS,
    get_experiment,
    validate_phase5_objective_weights,
)
from topoq.experiments import (
    DEFAULT_OBJECTIVE_WEIGHT_CONFIGS,
    experiment_parameter_sensitivity,
)
from topoq.main import main
from topoq.runner import run_experiment_spec
from topoq.topoq_scheduler import TopoQScheduler


class Phase5ObjectiveTests(unittest.TestCase):
    def test_weight_validation_accepts_defaults_and_rejects_invalid_sets(self):
        weights = validate_phase5_objective_weights()
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(
            weights["objective_w_makespan"],
            DEFAULT_PARAMS["objective_w_makespan"],
        )

        invalid_sets = (
            {"objective_w_makespan": -0.1},
            {"objective_w_makespan": float("inf")},
            {"objective_w_makespan": True},
            {
                "objective_w_makespan": 0.8,
                "objective_w_quantum_comm": 0.15,
                "objective_w_classical_comm": 0.10,
                "objective_w_fidelity": 0.20,
            },
        )
        for params in invalid_sets:
            with self.subTest(params=params), self.assertRaises(ValueError):
                validate_phase5_objective_weights(params)

    def test_all_default_sensitivity_weight_sets_are_valid(self):
        for config in DEFAULT_OBJECTIVE_WEIGHT_CONFIGS:
            with self.subTest(config=config):
                validated = validate_phase5_objective_weights(config)
                self.assertAlmostEqual(sum(validated.values()), 1.0)

    def test_zero_reference_normalization_uses_explicit_native_scale(self):
        normalize = TopoQScheduler._normalize_objective_component
        self.assertEqual(normalize(0.0, 0.0, 1.0), 0.0)
        self.assertEqual(normalize(3.0, 0.0, 1.0), 3.0)
        self.assertEqual(normalize(5.0, 10.0, 1.0), 0.5)

    def test_joint_objective_uses_normalized_weighted_components(self):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.phase5_objective_weights = (
            validate_phase5_objective_weights()
        )
        scheduler.workload = SimpleNamespace(quantum_tasks=[])
        scheduler.system = SimpleNamespace()
        scheduler.params = {}
        scheduler._intra_task_remote_gates = {}
        scheduler._objective_raw_components = lambda sched, assignment: {
            "makespan": 50.0,
            "entanglement_wait": 4.0,
            "remote_gates": 3.0,
            "classical_comm": 2.0,
            "fidelity_loss": 1.0,
        }
        reference = {
            "makespan": 100.0,
            "entanglement_wait": 2.0,
            "remote_gates": 6.0,
            "classical_comm": 4.0,
            "fidelity_loss": 2.0,
        }

        with patch(
            "topoq.topoq_scheduler.validate_schedule", return_value=[]
        ):
            objective = scheduler._compute_schedule_objective(
                {}, {}, {}, {}, {}, reference
            )

        expected_quantum = 0.70 * 2.0 + 0.30 * 0.5
        expected_total = (
            0.55 * 0.5
            + 0.15 * expected_quantum
            + 0.10 * 0.5
            + 0.20 * 0.5
        )
        self.assertAlmostEqual(
            objective["normalized"]["quantum_comm"], expected_quantum
        )
        self.assertAlmostEqual(objective["total"], expected_total)

    def test_acceptance_probability_is_standard_metropolis(self):
        probability = TopoQScheduler._sa_acceptance_probability(0.2, 0.1)
        self.assertAlmostEqual(probability, math.exp(-2.0))
        self.assertEqual(
            TopoQScheduler._sa_acceptance_probability(-0.1, 0.1), 1.0
        )
        self.assertEqual(
            TopoQScheduler._sa_acceptance_probability(0.1, 0.0), 0.0
        )

    def test_runner_metadata_records_joint_objective_policy(self):
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

        policy = result["metadata"]["phase5_objective"]
        self.assertEqual(
            policy["acceptance_probability"],
            "exp(-(J_new-J_current)/T_SA)",
        )
        self.assertFalse(policy["gamma_modulates_acceptance"])
        self.assertEqual(policy["applies_to"], ["TopoQ"])
        self.assertAlmostEqual(sum(policy["weights"].values()), 1.0)

    def test_cli_accepts_complete_weights_and_rejects_invalid_sum(self):
        valid_args = [
            "--profile", "quick",
            "--list-experiments",
            "--objective-w-makespan", "0.70",
            "--objective-w-quantum-comm", "0.10",
            "--objective-w-classical-comm", "0.05",
            "--objective-w-fidelity", "0.15",
        ]
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = main(valid_args)
        self.assertEqual(result, {})

        invalid_args = valid_args[:-1] + ["0.20"]
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                main(invalid_args)

    def test_objective_weight_sensitivity_has_separate_public_interface(self):
        metrics = {
            "makespan": 10.0,
            "comm_total": 4.0,
            "comm_entanglement": 2.5,
            "comm_classical_net": 1.0,
            "fidelity": 0.8,
            "sa_best_objective": 0.75,
        }
        objective_config = {
            "objective_w_makespan": 0.25,
            "objective_w_quantum_comm": 0.25,
            "objective_w_classical_comm": 0.25,
            "objective_w_fidelity": 0.25,
        }
        with patch(
            "topoq.experiments._run_spec", return_value=metrics
        ) as run:
            with redirect_stdout(StringIO()):
                result = experiment_parameter_sensitivity(
                    experiment="VQE-50_Small",
                    weight_configs=({
                        "w1": 0.35, "w2": 0.25,
                        "w3": 0.20, "w4": 0.20,
                    },),
                    objective_weight_configs=(objective_config,),
                    beta_values=(0.5,),
                    gamma_values=(0.3,),
                    sa_temperature_values=(0.15,),
                    profile="quick",
                    repetitions=1,
                )

        self.assertEqual(len(result["objective_weight_configs"]), 1)
        item = result["objective_weight_configs"][0]
        self.assertEqual(item["config"], objective_config)
        self.assertEqual(item["metrics"]["sa_best_objective"], 0.75)
        objective_calls = [
            call for call in run.call_args_list
            if call.kwargs.get("params", {}).get(
                "objective_w_makespan"
            ) == 0.25
        ]
        self.assertEqual(len(objective_calls), 1)


if __name__ == "__main__":
    unittest.main()
