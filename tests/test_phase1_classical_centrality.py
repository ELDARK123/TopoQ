import unittest
from types import SimpleNamespace

from topoq.config import (
    DEFAULT_PARAMS,
    load_network_topology,
    load_sim_params,
    load_system_config,
)
from topoq.models import HybridSystem
from topoq.topoq_scheduler import TopoQScheduler


class Phase1ClassicalCentralityTests(unittest.TestCase):
    @staticmethod
    def _system(config="small", network="fattree_k4"):
        return HybridSystem(
            load_system_config(config),
            load_network_topology(network),
            load_sim_params(),
        )

    @staticmethod
    def _scheduler(system):
        return TopoQScheduler(
            system,
            SimpleNamespace(),
            DEFAULT_PARAMS.copy(),
            {},
            42,
        )

    def test_brandes_path_has_known_raw_centrality(self):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.system = SimpleNamespace(
            net_nodes=[{"id": "A"}, {"id": "B"}, {"id": "C"}],
            net_adj={
                "A": [("B", 1.0, 1.0)],
                "B": [("A", 1.0, 1.0), ("C", 1.0, 1.0)],
                "C": [("B", 1.0, 1.0)],
            },
        )

        centrality = scheduler._compute_betweenness_centrality()

        self.assertEqual(centrality, {"A": 0.0, "B": 1.0, "C": 0.0})

    def test_leaf_controller_uses_real_zero_not_missing_fallback(self):
        scheduler = self._scheduler(self._system())
        metrics = scheduler.topology_metric_components

        for qid, controller in metrics["controller_resource"].items():
            endpoint = scheduler.system.get_network_node_id(controller)
            self.assertEqual(
                metrics["controller_network_node"][qid], endpoint
            )
            self.assertEqual(metrics["controller_betweenness_raw"][qid], 0.0)
            self.assertEqual(
                metrics["controller_betweenness_normalized"][qid], 0.0
            )
            self.assertEqual(
                metrics["controller_betweenness_benefit"][qid], 1.0
            )

    def test_internal_controller_has_positive_centrality_and_lower_score(self):
        original = self._scheduler(self._system())
        bottleneck = max(
            original.topology_metric_components[
                "classical_betweenness_raw"
            ],
            key=original.topology_metric_components[
                "classical_betweenness_raw"
            ].get,
        )
        internal_system = self._system()
        internal_system.nodes["P0"]["network_node_id"] = bottleneck
        internal = self._scheduler(internal_system)

        self.assertEqual(
            internal.topology_metric_components["controller_network_node"][
                "Q0"
            ],
            bottleneck,
        )
        self.assertGreater(
            internal.topology_metric_components["controller_betweenness_raw"][
                "Q0"
            ],
            0.0,
        )
        self.assertLess(
            internal.topology_scores["Q0"], original.topology_scores["Q0"]
        )

    def test_missing_interface_or_controller_fails_explicitly(self):
        missing_interface = self._system()
        del missing_interface.interface_links["Q0"]
        with self.assertRaisesRegex(ValueError, "no classical interface"):
            self._scheduler(missing_interface)

        unknown_controller = self._system()
        unknown_controller.interface_links["Q0"][
            "classical_node_id"
        ] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unknown classical controller"):
            self._scheduler(unknown_controller)

        unknown_endpoint = self._system()
        unknown_endpoint.nodes["P0"]["network_node_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unknown network node"):
            self._scheduler(unknown_endpoint)

    def test_all_real_configs_resolve_controllers_deterministically(self):
        configurations = (
            ("small", "fattree_k4"),
            ("medium", "dragonfly_4x4x2"),
            ("large", "dragonflyplus_8x8x4"),
        )
        for config, network in configurations:
            first = self._scheduler(self._system(config, network))
            second = self._scheduler(self._system(config, network))
            first_metrics = first.topology_metric_components
            second_metrics = second.topology_metric_components
            self.assertEqual(
                len(first_metrics["controller_network_node"]),
                len(first.system.qpus),
            )
            self.assertEqual(
                first_metrics["classical_betweenness_raw"],
                second_metrics["classical_betweenness_raw"],
            )
            self.assertEqual(first.topology_scores, second.topology_scores)


if __name__ == "__main__":
    unittest.main()
