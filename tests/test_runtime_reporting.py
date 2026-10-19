import unittest
from unittest.mock import patch

from topoq.config import get_experiment
from topoq.runner import run_experiment_spec, run_with_repetitions


class RuntimeReportingTests(unittest.TestCase):
    def test_repetition_runtime_uses_mean_and_population_std(self):
        samples = [{"makespan": 10.0}, {"makespan": 14.0}]
        with patch(
            "topoq.runner.run_single_experiment", side_effect=samples
        ), patch(
            "topoq.runner.time.perf_counter",
            side_effect=[100.0, 101.5, 200.0, 202.5],
        ):
            result = run_with_repetitions(
                "dag", "system", "network", "TopoQ", n_rep=2
            )

        self.assertAlmostEqual(result["runtime_seconds"], 2.0)
        self.assertAlmostEqual(result["runtime_seconds_std"], 0.5)
        self.assertAlmostEqual(result["makespan"], 12.0)
        self.assertAlmostEqual(result["makespan_std"], 2.0)

    def test_single_repetition_runtime_std_is_zero(self):
        with patch(
            "topoq.runner.run_single_experiment",
            return_value={"makespan": 10.0},
        ), patch(
            "topoq.runner.time.perf_counter", side_effect=[10.0, 10.25]
        ):
            result = run_with_repetitions(
                "dag", "system", "network", "Greedy", n_rep=1
            )

        self.assertAlmostEqual(result["runtime_seconds"], 0.25)
        self.assertEqual(result["runtime_seconds_std"], 0.0)

    def test_experiment_metadata_describes_runtime_scope(self):
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0, "runtime_seconds": 0.1},
        ):
            result = run_experiment_spec(
                get_experiment("VQE-50_Small"),
                profile="quick",
                repetitions=1,
                algorithms=("Greedy", "TopoQ"),
            )

        metadata = result["metadata"]["runtime_measurement"]
        self.assertEqual(metadata["timer"], "time.perf_counter")
        self.assertEqual(
            metadata["scope"],
            "per_algorithm_repetition_including_setup_scheduler_and_simulator",
        )
        self.assertEqual(
            metadata["aggregation"],
            "mean_and_population_standard_deviation",
        )
        self.assertFalse(metadata["included_in_objective"])
        self.assertEqual(metadata["applies_to"], ["Greedy", "TopoQ"])


if __name__ == "__main__":
    unittest.main()
