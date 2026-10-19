import unittest
from unittest.mock import patch

from topoq.config import EXPERIMENT_REGISTRY
from topoq.runner import run_experiment_spec


class C4SolutionClassificationTests(unittest.TestCase):
    def _metadata(self, mode, threshold=None):
        params = {"fidelity_constraint_mode": mode}
        if threshold is not None:
            params["fidelity_threshold"] = threshold
        spec = EXPERIMENT_REGISTRY["VQE-50_Small"]
        with patch("topoq.runner.run_with_repetitions", return_value={}):
            result = run_experiment_spec(
                spec,
                profile="quick",
                repetitions=1,
                algorithms=("TopoQ",),
                params=params,
            )
        self.assertEqual(result["status"], "completed")
        return result["metadata"]["fidelity_constraint"]

    def test_report_only_is_diagnostic_not_strictly_certified(self):
        metadata = self._metadata("report_only")

        self.assertEqual(
            metadata["solution_class"],
            "scheduling_result_with_fidelity_diagnostics",
        )
        self.assertIsNone(metadata["strict_tahs_feasible"])
        self.assertEqual(metadata["c4_certification_scope"], "not_certified")
        self.assertEqual(metadata["applies_to"], ["TopoQ"])

    def test_paper_strict_success_is_a_strict_candidate(self):
        metadata = self._metadata("paper_strict")

        self.assertEqual(metadata["solution_class"], "strict_tahs_candidate")
        self.assertIs(metadata["strict_tahs_feasible"], True)
        self.assertEqual(metadata["c4_certification_scope"], "enforced")

    def test_hardware_explicit_success_is_a_strict_candidate(self):
        metadata = self._metadata("hardware_explicit", threshold=0.8)

        self.assertEqual(metadata["solution_class"], "strict_tahs_candidate")
        self.assertIs(metadata["strict_tahs_feasible"], True)
        self.assertEqual(metadata["c4_certification_scope"], "enforced")
        self.assertEqual(metadata["threshold"], 0.8)
        self.assertEqual(metadata["threshold_source"], "user")


if __name__ == "__main__":
    unittest.main()
