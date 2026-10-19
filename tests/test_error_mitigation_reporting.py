import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import (
    ERROR_MITIGATION_MODES,
    get_experiment,
    validate_error_mitigation_policy,
)
from topoq.fidelity import estimate_task_log_fidelity
from topoq.mitigation import (
    estimate_error_mitigation_metrics,
    richardson_coefficients,
    zne_sampling_overhead,
)
from topoq.runner import run_experiment_spec, run_with_repetitions


class ErrorMitigationReportingTests(unittest.TestCase):
    @staticmethod
    def _task_estimates(fidelity=0.8):
        return {
            "task": {
                "fidelity": fidelity,
                "log_fidelity": math.log(fidelity),
            }
        }

    @staticmethod
    def _schedule(duration_ms=10.0):
        return {"task": {"start": 0.0, "end": duration_ms}}

    def test_modes_and_strict_parameter_validation(self):
        self.assertEqual(ERROR_MITIGATION_MODES, (
            "none",
            "symmetry_verification",
            "zne",
            "zne_symmetry",
        ))
        self.assertEqual(validate_error_mitigation_policy(), {
            "mode": "none",
            "symmetry_error_detection_fraction": None,
            "zne_noise_scales": None,
        })
        self.assertEqual(
            validate_error_mitigation_policy(
                "zne_symmetry", 0.9, [1, 3, 5]
            ),
            {
                "mode": "zne_symmetry",
                "symmetry_error_detection_fraction": 0.9,
                "zne_noise_scales": (1.0, 3.0, 5.0),
            },
        )
        for invalid in (-0.1, 1.1, math.inf, "bad"):
            with self.assertRaisesRegex(
                ValueError, "symmetry_error_detection_fraction"
            ):
                validate_error_mitigation_policy(
                    "symmetry_verification", invalid, None
                )
        for invalid in (None, [1], [1, 1], [3, 5], [1, math.inf]):
            with self.assertRaisesRegex(ValueError, "zne_noise_scales"):
                validate_error_mitigation_policy("zne", None, invalid)
        with self.assertRaisesRegex(ValueError, "may only be supplied"):
            validate_error_mitigation_policy("none", 0.5, None)
        with self.assertRaisesRegex(ValueError, "may only be supplied"):
            validate_error_mitigation_policy("none", None, [1, 3])

    def test_richardson_coefficients_and_sampling_overhead(self):
        scales = (1.0, 3.0, 5.0)
        coefficients = richardson_coefficients(scales)
        self.assertEqual(coefficients, (1.875, -1.25, 0.375))
        self.assertAlmostEqual(sum(coefficients), 1.0)
        self.assertAlmostEqual(sum(
            coefficient * scale
            for coefficient, scale in zip(coefficients, scales)
        ), 0.0)
        self.assertAlmostEqual(sum(
            coefficient * scale * scale
            for coefficient, scale in zip(coefficients, scales)
        ), 0.0)
        _, overhead = zne_sampling_overhead(scales)
        self.assertAlmostEqual(overhead, 12.25)

    def test_symmetry_verification_reports_conditional_quality_and_cost(self):
        policy = validate_error_mitigation_policy(
            "symmetry_verification", 0.75, None
        )
        metrics = estimate_error_mitigation_metrics(
            self._task_estimates(), self._schedule(), policy
        )

        acceptance = 1.0 - 0.75 * (1.0 - 0.8)
        self.assertAlmostEqual(acceptance, 0.85)
        self.assertAlmostEqual(
            metrics["postselected_fidelity"], 0.8 / acceptance
        )
        self.assertAlmostEqual(
            metrics["symmetry_acceptance_probability"], acceptance
        )
        self.assertAlmostEqual(
            metrics["mitigation_mean_shot_multiplier"], 1.0 / acceptance
        )
        self.assertAlmostEqual(
            metrics["mitigation_adjusted_quantum_work_ms"],
            10.0 / acceptance,
        )

    def test_zne_changes_sampling_work_not_physical_or_postselected_fidelity(self):
        policy = validate_error_mitigation_policy("zne", None, [1, 3, 5])
        metrics = estimate_error_mitigation_metrics(
            self._task_estimates(), self._schedule(), policy
        )

        self.assertEqual(metrics["postselected_fidelity"], 0.8)
        self.assertEqual(metrics["symmetry_acceptance_probability"], 1.0)
        self.assertAlmostEqual(metrics["zne_sampling_overhead"], 12.25)
        self.assertAlmostEqual(
            metrics["mitigation_adjusted_quantum_work_ms"], 122.5
        )

    def test_none_preserves_unfloored_physical_log_product(self):
        estimates = {
            "task": {
                "fidelity": 1e-300,
                "log_fidelity": -1000.0,
            }
        }
        metrics = estimate_error_mitigation_metrics(
            estimates, self._schedule(),
            validate_error_mitigation_policy("none", None, None),
        )

        self.assertAlmostEqual(metrics["postselected_fidelity"], 1e-300)
        self.assertAlmostEqual(
            metrics["postselected_fidelity_log10_product"],
            -1000.0 / math.log(10.0),
        )

    def test_combined_overhead_multiplies_zne_and_postselection_cost(self):
        policy = validate_error_mitigation_policy(
            "zne_symmetry", 0.75, [1, 3, 5]
        )
        metrics = estimate_error_mitigation_metrics(
            self._task_estimates(), self._schedule(), policy
        )

        acceptance = 0.85
        expected_multiplier = 12.25 / acceptance
        self.assertAlmostEqual(
            metrics["mitigation_mean_shot_multiplier"], expected_multiplier
        )
        self.assertAlmostEqual(
            metrics["mitigation_adjusted_quantum_work_ms"],
            10.0 * expected_multiplier,
        )

    def test_legacy_mitigation_factors_cannot_change_physical_fidelity(self):
        qpu = {
            "coherence_time_us": 100.0,
            "gate_times_ns": {"1q": 10.0, "2q": 20.0, "measure": 30.0},
            "gate_error_rates": {"1q": 0.01, "2q": 0.02, "measure": 0.03},
        }
        task = {"n_gates": 10, "n_qubits": 2, "depth": 2}
        base_fidelity = {
            "F_prep": 0.999,
            "F_EPR_range": [0.9, 0.95],
        }
        link_params = {"eta_range_per_ms": [0.01, 0.1]}
        plain = SimpleNamespace(sim_params={
            "fidelity_parameters": dict(base_fidelity),
            "quantum_link_parameters": link_params,
        })
        legacy = SimpleNamespace(sim_params={
            "fidelity_parameters": {
                **base_fidelity,
                "mitigation_1q": 0.01,
                "mitigation_2q": 0.01,
            },
            "quantum_link_parameters": link_params,
        })

        plain_result = estimate_task_log_fidelity(plain, task, qpu)
        legacy_result = estimate_task_log_fidelity(legacy, task, qpu)
        self.assertEqual(plain_result["log_gate"], legacy_result["log_gate"])
        self.assertEqual(plain_result["fidelity"], legacy_result["fidelity"])

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
                    "error_mitigation_mode": "zne_symmetry",
                    "symmetry_error_detection_fraction": 0.9,
                    "zne_noise_scales": [1, 3, 5],
                },
            )

        metadata = result["metadata"]["error_mitigation"]
        self.assertEqual(metadata["applies_to"], ["Greedy", "TopoQ"])
        self.assertTrue(metadata["physical_fidelity_unchanged_by_zne"])
        self.assertEqual(
            metadata["zne_richardson_coefficients"],
            (1.875, -1.25, 0.375),
        )
        self.assertEqual(metadata["zne_sampling_overhead"], 12.25)

    def test_repetition_averaging_preserves_mitigation_mode_string(self):
        sample = {
            "error_mitigation_mode": "zne",
            "fidelity": 0.8,
        }
        with patch(
            "topoq.runner.run_single_experiment", return_value=sample
        ):
            result = run_with_repetitions(
                "dag", "system", "network", "Greedy", n_rep=2
            )

        self.assertEqual(result["error_mitigation_mode"], "zne")
        self.assertEqual(result["fidelity"], 0.8)


if __name__ == "__main__":
    unittest.main()
