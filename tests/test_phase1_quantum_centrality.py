import math
import unittest
from types import SimpleNamespace

from topoq.config import (
    DEFAULT_PARAMS,
    QUANTUM_CENTRALITY_MODES,
    load_network_topology,
    load_sim_params,
    load_system_config,
    validate_quantum_centrality_mode,
)
from topoq.models import HybridSystem
from topoq.topoq_scheduler import TopoQScheduler


class Phase1QuantumCentralityTests(unittest.TestCase):
    def _scheduler(self, params=None):
        system = HybridSystem(
            load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )
        merged = DEFAULT_PARAMS.copy()
        merged.update(params or {})
        return TopoQScheduler(
            system,
            SimpleNamespace(),
            merged,
            {},
            42,
        )

    def test_default_mode_is_paper_eq22(self):
        scheduler = self._scheduler()
        self.assertEqual(DEFAULT_PARAMS["quantum_centrality_mode"], "paper_eq22")
        self.assertEqual(
            scheduler.topology_metric_components["quantum_centrality_mode"],
            "paper_eq22",
        )

    def test_paper_mode_uses_product_and_dynamic_b_max(self):
        scheduler = self._scheduler()
        system = scheduler.system
        first_key = next(iter(system.quantum_links))
        system.quantum_links[first_key]["max_concurrent_epr"] = 64
        scheduler._compute_topology_metrics()

        qid = "Q0"
        b_max = max(
            link["max_concurrent_epr"]
            for link in system.quantum_links.values()
        )
        expected = 0.0
        for other in system.qpus:
            if other == qid:
                continue
            link = system.get_quantum_link(qid, other)
            if link:
                expected += (
                    link["rate_hz"] / link["distance_km"]
                    * link["max_concurrent_epr"] / b_max
                )

        actual = scheduler.topology_metric_components[
            "quantum_centrality_raw"
        ][qid]
        self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            scheduler.topology_metric_components[
                "quantum_centrality_parameters"
            ]["b_max"],
            64.0,
        )

    def test_effective_service_matches_batch_generation_model(self):
        scheduler = self._scheduler({
            "quantum_centrality_mode": "effective_service",
        })
        system = scheduler.system
        qid = "Q0"
        p_threshold = scheduler.params["p_threshold"]
        expected = 0.0
        for other in system.qpus:
            if other == qid:
                continue
            link = system.get_quantum_link(qid, other)
            if not link:
                continue
            generation_ms = (
                -math.log1p(-p_threshold) / link["rate_hz"] * 1000.0
            )
            expected += link["max_concurrent_epr"] / generation_ms

        actual = scheduler.topology_metric_components[
            "quantum_centrality_raw"
        ][qid]
        self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            scheduler.topology_metric_components[
                "quantum_centrality_parameters"
            ]["raw_unit"],
            "epr_pairs_per_ms",
        )
        self.assertEqual(
            scheduler.topology_metric_components[
                "quantum_centrality_parameters"
            ]["quantum_link_latency_mode"],
            "delivered_pair_rate",
        )

    def test_modes_produce_finite_scores_and_can_be_selected(self):
        paper = self._scheduler({
            "quantum_centrality_mode": "paper_eq22",
        })
        service = self._scheduler({
            "quantum_centrality_mode": "effective_service",
        })
        self.assertEqual(
            QUANTUM_CENTRALITY_MODES,
            ("paper_eq22", "effective_service"),
        )
        self.assertTrue(
            all(math.isfinite(value) for value in paper.topology_scores.values())
        )
        self.assertTrue(
            all(math.isfinite(value) for value in service.topology_scores.values())
        )
        self.assertNotEqual(
            paper.topology_metric_components["quantum_centrality_raw"],
            service.topology_metric_components["quantum_centrality_raw"],
        )

    def test_invalid_mode_and_service_parameters_fail_explicitly(self):
        with self.assertRaises(ValueError):
            validate_quantum_centrality_mode("unknown")
        with self.assertRaises(ValueError):
            self._scheduler({
                "quantum_centrality_mode": "effective_service",
                "p_threshold": 1.0,
            })


if __name__ == "__main__":
    unittest.main()
