import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import (
    DEFAULT_PARAMS,
    TOPOLOGY_INVERSE_MODES,
    get_experiment,
    load_network_topology,
    load_sim_params,
    load_system_config,
    validate_topology_inverse_policy,
)
from topoq.models import HybridSystem
from topoq.runner import run_experiment_spec
from topoq.topoq_scheduler import TopoQScheduler


class Phase1TopologyInverseTests(unittest.TestCase):
    @staticmethod
    def _scheduler(params=None):
        configured = DEFAULT_PARAMS.copy()
        configured.update(params or {})
        system = HybridSystem(
            load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )
        return TopoQScheduler(
            system, SimpleNamespace(), configured, {}, 42
        )

    def test_default_bounded_complement_recombines_score(self):
        scheduler = self._scheduler()
        metrics = scheduler.topology_metric_components

        self.assertEqual(TOPOLOGY_INVERSE_MODES, (
            "bounded_complement",
            "normalized_reciprocal",
            "paper_raw_strict",
        ))
        self.assertEqual(metrics["topology_inverse_mode"], "bounded_complement")
        self.assertIsNone(metrics["topology_reciprocal_epsilon"])
        for qid, normalized in metrics[
                "interface_latency_normalized"].items():
            self.assertAlmostEqual(
                metrics["interface_latency_benefit"][qid],
                1.0 - normalized,
            )
            expected = (
                DEFAULT_PARAMS["w1"]
                * metrics["quantum_centrality_normalized"][qid]
                + DEFAULT_PARAMS["w2"]
                * metrics["interface_latency_benefit"][qid]
                + DEFAULT_PARAMS["w3"]
                * metrics["qpu_capacity_normalized"][qid]
                + DEFAULT_PARAMS["w4"]
                * metrics["controller_betweenness_benefit"][qid]
            )
            self.assertAlmostEqual(scheduler.topology_scores[qid], expected)

    def test_normalized_reciprocal_formula_and_endpoints(self):
        values = {"low": 0.0, "mid": 0.5, "high": 1.0}
        benefits = TopoQScheduler._inverse_cost_benefit(
            values, "normalized_reciprocal", 1.0, "test_cost"
        )

        self.assertEqual(benefits["low"], 1.0)
        self.assertAlmostEqual(benefits["mid"], 1.0 / 3.0)
        self.assertEqual(benefits["high"], 0.0)

        scheduler = self._scheduler({
            "topology_inverse_mode": "normalized_reciprocal",
            "topology_reciprocal_epsilon": 1.0,
        })
        metrics = scheduler.topology_metric_components
        self.assertEqual(
            metrics["topology_inverse_mode"], "normalized_reciprocal"
        )
        self.assertEqual(metrics["topology_reciprocal_epsilon"], 1.0)
        self.assertTrue(all(
            0.0 <= value <= 1.0
            for value in metrics["interface_latency_benefit"].values()
        ))

    def test_epsilon_controls_reciprocal_curvature(self):
        values = {"mid": 0.5}
        epsilon_half = TopoQScheduler._inverse_cost_benefit(
            values, "normalized_reciprocal", 0.5, "test_cost"
        )["mid"]
        epsilon_two = TopoQScheduler._inverse_cost_benefit(
            values, "normalized_reciprocal", 2.0, "test_cost"
        )["mid"]

        self.assertAlmostEqual(epsilon_half, 0.25)
        self.assertAlmostEqual(epsilon_two, 0.4)
        self.assertLess(epsilon_half, epsilon_two)

    def test_paper_raw_strict_matches_reciprocal_and_rejects_zero(self):
        benefits = TopoQScheduler._inverse_cost_benefit(
            {"a": 0.25, "b": 0.5, "c": 1.0},
            "paper_raw_strict",
            None,
            "test_cost",
        )
        self.assertEqual(benefits, {"a": 4.0, "b": 2.0, "c": 1.0})

        with self.assertRaisesRegex(ValueError, "undefined.*zero"):
            TopoQScheduler._inverse_cost_benefit(
                {"minimum": 0.0},
                "paper_raw_strict",
                None,
                "test_cost",
            )
        with self.assertRaisesRegex(ValueError, "interface_latency"):
            self._scheduler({"topology_inverse_mode": "paper_raw_strict"})

    def test_invalid_mode_epsilon_and_normalized_value_fail(self):
        with self.assertRaises(ValueError):
            validate_topology_inverse_policy("unknown", 1.0)
        for epsilon in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "positive finite"):
                validate_topology_inverse_policy(
                    "normalized_reciprocal", epsilon
                )
        with self.assertRaisesRegex(ValueError, "must be in"):
            TopoQScheduler._inverse_cost_benefit(
                {"invalid": 1.1},
                "bounded_complement",
                None,
                "test_cost",
            )

    def test_runner_metadata_records_inverse_policy(self):
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0},
        ):
            result = run_experiment_spec(
                get_experiment("VQE-50_Small"),
                profile="quick",
                repetitions=1,
                algorithms=("TopoQ",),
                params={
                    "topology_inverse_mode": "normalized_reciprocal",
                    "topology_reciprocal_epsilon": 0.5,
                },
            )

        topology = result["metadata"]["topology_characterization"]
        self.assertEqual(
            topology["topology_inverse_mode"], "normalized_reciprocal"
        )
        self.assertEqual(topology["topology_reciprocal_epsilon"], 0.5)


if __name__ == "__main__":
    unittest.main()
