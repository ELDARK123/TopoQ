import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import (
    DEFAULT_PARAMS,
    INTERFACE_LATENCY_MODES,
    get_experiment,
    load_network_topology,
    load_sim_params,
    load_system_config,
    validate_interface_latency_mode,
)
from topoq.models import HybridSystem
from topoq.runner import run_experiment_spec
from topoq.topoq_scheduler import TopoQScheduler


class Phase1InterfaceLatencyTests(unittest.TestCase):
    @staticmethod
    def _config_with_distances():
        config = copy.deepcopy(load_system_config("small"))
        for link in config["interface_links"]:
            link["distance_km"] = 0.0
        return config

    @staticmethod
    def _system(config=None):
        return HybridSystem(
            config or load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )

    @staticmethod
    def _scheduler(system, params=None):
        configured = DEFAULT_PARAMS.copy()
        configured.update(params or {})
        workload = SimpleNamespace(edge_data={
            ("quantum", "classical"): {
                "data_volume_bits": 8_000_000,
            },
        })
        return TopoQScheduler(system, workload, configured, {}, 42)

    def test_default_proxy_preserves_configured_controller(self):
        scheduler = self._scheduler(self._system())
        metrics = scheduler.topology_metric_components

        self.assertEqual(INTERFACE_LATENCY_MODES, (
            "readout_proxy", "paper_eq24"
        ))
        self.assertEqual(
            validate_interface_latency_mode(), "readout_proxy"
        )
        self.assertEqual(metrics["interface_latency_mode"], "readout_proxy")
        self.assertEqual(metrics["controller_resource"]["Q0"], "P0")
        self.assertAlmostEqual(metrics["interface_latency_us"]["Q0"], 3.2)
        self.assertIsNone(metrics["interface_distance_km"]["Q0"])
        self.assertIsNone(metrics["interface_signal_speed_m_s"])

    def test_paper_eq24_selects_minimum_and_matches_units(self):
        config = self._config_with_distances()
        q0 = next(
            link for link in config["interface_links"]
            if link["qpu_id"] == "Q0"
        )
        q0["distance_km"] = 0.4
        config["interface_links"].append({
            "qpu_id": "Q0",
            "classical_node_id": "P4",
            "readout_latency_us": 3.2,
            "control_bandwidth_gbps": 10.0,
            "distance_km": 0.2,
        })
        system = self._system(config)
        scheduler = self._scheduler(system, {
            "interface_latency_mode": "paper_eq24",
            "interface_signal_speed_m_s": 200_000_000.0,
        })
        metrics = scheduler.topology_metric_components

        self.assertEqual(len(system.interface_candidates["Q0"]), 2)
        self.assertEqual(system.interface_links["Q0"]["classical_node_id"], "P0")
        self.assertEqual(metrics["controller_resource"]["Q0"], "P4")
        self.assertEqual(metrics["interface_candidate_count"]["Q0"], 2)
        self.assertAlmostEqual(
            metrics["interface_propagation_latency_us"]["Q0"], 1.0
        )
        self.assertAlmostEqual(metrics["interface_latency_us"]["Q0"], 4.2)
        self.assertEqual(
            scheduler._get_interface_link("Q0")["classical_node_id"], "P4"
        )
        self.assertAlmostEqual(
            scheduler._interface_transfer_ms(
                "Q0", "quantum", "classical"
            ),
            0.8042,
        )

    def test_equal_latency_uses_controller_id_tie_break(self):
        config = self._config_with_distances()
        config["interface_links"] = [
            link for link in config["interface_links"]
            if link["qpu_id"] != "Q0"
        ] + [
            {
                "qpu_id": "Q0",
                "classical_node_id": "P5",
                "readout_latency_us": 3.2,
                "control_bandwidth_gbps": 10.0,
                "distance_km": 0.2,
            },
            {
                "qpu_id": "Q0",
                "classical_node_id": "P4",
                "readout_latency_us": 3.2,
                "control_bandwidth_gbps": 10.0,
                "distance_km": 0.2,
            },
        ]
        scheduler = self._scheduler(self._system(config), {
            "interface_latency_mode": "paper_eq24",
            "interface_signal_speed_m_s": 200_000_000.0,
        })

        self.assertEqual(
            scheduler.topology_metric_components["controller_resource"]["Q0"],
            "P4",
        )

    def test_paper_mode_rejects_missing_speed_or_distance(self):
        with self.assertRaisesRegex(
                ValueError, "requires interface_signal_speed_m_s"):
            self._scheduler(self._system(), {
                "interface_latency_mode": "paper_eq24",
            })

        with self.assertRaisesRegex(ValueError, "requires distance_km"):
            self._scheduler(self._system(), {
                "interface_latency_mode": "paper_eq24",
                "interface_signal_speed_m_s": 200_000_000.0,
            })

    def test_invalid_mode_distance_and_speed_fail_explicitly(self):
        with self.assertRaises(ValueError):
            validate_interface_latency_mode("unknown")

        config = self._config_with_distances()
        config["interface_links"][0]["distance_km"] = -0.1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._scheduler(self._system(config), {
                "interface_latency_mode": "paper_eq24",
                "interface_signal_speed_m_s": 200_000_000.0,
            })

        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._scheduler(self._system(self._config_with_distances()), {
                "interface_latency_mode": "paper_eq24",
                "interface_signal_speed_m_s": 0.0,
            })

        inconsistent = self._config_with_distances()
        inconsistent["interface_links"].append({
            "qpu_id": "Q0",
            "classical_node_id": "P4",
            "readout_latency_us": 1.0,
            "control_bandwidth_gbps": 10.0,
            "distance_km": 0.2,
        })
        with self.assertRaisesRegex(ValueError, "one QPU-level rho_i"):
            self._scheduler(self._system(inconsistent), {
                "interface_latency_mode": "paper_eq24",
                "interface_signal_speed_m_s": 200_000_000.0,
            })

    def test_runner_metadata_records_interface_policy(self):
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0},
        ):
            result = run_experiment_spec(
                get_experiment("VQE-50_Small"),
                profile="quick",
                repetitions=1,
                algorithms=("TopoQ",),
                params={"interface_latency_mode": "readout_proxy"},
            )

        topology = result["metadata"]["topology_characterization"]
        self.assertEqual(topology["interface_latency_mode"], "readout_proxy")
        self.assertIsNone(topology["interface_signal_speed_m_s"])


if __name__ == "__main__":
    unittest.main()
