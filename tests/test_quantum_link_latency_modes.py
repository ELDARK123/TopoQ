import math
import random
import unittest
from collections import defaultdict
from copy import deepcopy

from topoq.baselines import RandomScheduler
from topoq.config import (
    QUANTUM_LINK_LATENCY_MODES,
    load_network_topology,
    load_sim_params,
    load_system_config,
    validate_quantum_link_latency_policy,
)
from topoq.entanglement import (
    deterministic_epr_service_window_ms,
    resolve_channel_latency_components,
    resolve_quantum_link_latency_policy,
    resolve_service_latency,
)
from topoq.models import HybridSystem
from topoq.simulator import Simulator
from topoq.topoq_scheduler import TopoQScheduler


class QuantumLinkLatencyModeTests(unittest.TestCase):
    def _system(self):
        return HybridSystem(
            load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )

    @staticmethod
    def _physical_params():
        return {
            "quantum_link_latency_mode": "physical_stage",
            "quantum_link_bsm_latency_ms": 0.20,
            "quantum_link_switch_latency_ms": 0.30,
            "quantum_link_signal_speed_m_s": 200_000_000.0,
            "quantum_link_signal_traversals": 2,
        }

    def test_public_modes_and_delivered_rate_default(self):
        self.assertEqual(QUANTUM_LINK_LATENCY_MODES, (
            "delivered_pair_rate", "physical_stage",
        ))
        policy = validate_quantum_link_latency_policy()
        self.assertEqual(policy["mode"], "delivered_pair_rate")
        self.assertEqual(policy["bsm_latency_ms"], None)
        with self.assertRaisesRegex(ValueError, "require.*physical_stage"):
            validate_quantum_link_latency_policy(
                "delivered_pair_rate", bsm_latency_ms=0.1
            )

    def test_physical_stage_requires_complete_channel_calibration(self):
        policy = resolve_quantum_link_latency_policy(
            {"quantum_link_latency_mode": "physical_stage"},
            load_sim_params(),
        )
        channel = {
            "channel_id": "test",
            "distance_km": 10.0,
        }

        with self.assertRaisesRegex(
            ValueError, "bsm_latency_ms, switch_latency_ms, signal_traversals"
        ):
            resolve_channel_latency_components(
                channel, policy, load_sim_params()
            )

    def test_physical_stage_formula(self):
        policy = resolve_quantum_link_latency_policy(
            self._physical_params(), load_sim_params()
        )
        channel = {"channel_id": "test", "distance_km": 10.0}
        components = resolve_channel_latency_components(
            channel, policy, load_sim_params()
        )

        self.assertAlmostEqual(components["propagation_latency_ms"], 0.10)
        self.assertAlmostEqual(components["stage_overhead_ms"], 0.60)

    def test_multi_channel_stage_uses_conservative_maximum(self):
        policy = resolve_quantum_link_latency_policy(
            self._physical_params(), load_sim_params()
        )
        service = resolve_service_latency((
            {"channel_id": "near", "distance_km": 1.0},
            {"channel_id": "far", "distance_km": 11.0},
        ), policy, load_sim_params())

        self.assertEqual(
            service["multi_channel_stage_aggregation"], "conservative_max"
        )
        self.assertAlmostEqual(service["stage_overhead_ms"], 0.61)

    def test_scheduler_and_baseline_share_physical_service_window(self):
        system = self._system()
        link_key, link = next(iter(system.quantum_links.items()))
        params = {"p_threshold": 0.95, **self._physical_params()}
        policy = resolve_quantum_link_latency_policy(
            params, system.sim_params
        )

        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.system = system
        scheduler.params = params
        scheduler.ablation = {}
        scheduler.quantum_epr_routing_mode = "single_channel"
        scheduler.quantum_epr_routing_policy = {
            "mode": "single_channel", "reliability_window_ms": None,
        }
        scheduler.quantum_link_latency_policy = policy
        scheduler.quantum_link_timeline = defaultdict(list)
        service = scheduler._quantum_epr_service(link_key)
        topoq_window = scheduler._entanglement_generation_window_ms(service)

        baseline = RandomScheduler.__new__(RandomScheduler)
        baseline.system = system
        baseline.quantum_link_latency_policy = policy
        baseline_window = baseline._compute_ent_time(link)

        self.assertAlmostEqual(topoq_window, baseline_window)
        expected = deterministic_epr_service_window_ms(
            link["rate_hz"], 0.95, 1.0, service["stage_overhead_ms"]
        )
        self.assertAlmostEqual(topoq_window, expected)

    def test_poisson_replay_separates_generation_and_fixed_stages(self):
        system = self._system()
        link_key, link = next(iter(system.quantum_links.items()))
        sim_params = deepcopy(system.sim_params)
        sim_params["quantum_link_latency_policy"] = (
            resolve_quantum_link_latency_policy(
                self._physical_params(), sim_params
            )
        )
        simulator = Simulator(system, sim_params, seed=7)
        service = resolve_service_latency(
            (link,), simulator.quantum_link_latency_policy, sim_params
        )
        record = {
            "link": tuple(link_key),
            "generation_start": 1.0,
            "effective_rate_hz": link["rate_hz"],
            "duration_scale": 1.0,
            "stage_overhead_ms": service["stage_overhead_ms"],
        }
        statistics = {
            "sampled_batches": 0,
            "sampled_generation_ms": 0.0,
            "sampled_stage_overhead_ms": 0.0,
            "sampled_service_ms": 0.0,
        }
        expected_u = random.Random(7).random()
        expected_generation = (
            -math.log1p(-expected_u) / link["rate_hz"] * 1000.0
        )

        end = simulator._sample_reservation_end(record, {}, statistics)

        self.assertAlmostEqual(
            statistics["sampled_generation_ms"], expected_generation
        )
        self.assertAlmostEqual(
            statistics["sampled_stage_overhead_ms"],
            service["stage_overhead_ms"],
        )
        self.assertAlmostEqual(
            end,
            1.0 + expected_generation + service["stage_overhead_ms"],
        )

    def test_system_rejects_invalid_link_stage_fields(self):
        invalid_cases = {
            "bsm_latency_ms": -0.1,
            "switch_latency_ms": math.inf,
            "signal_speed_m_s": 0.0,
            "signal_traversals": 1.5,
        }
        for field, value in invalid_cases.items():
            config = deepcopy(load_system_config("small"))
            config["quantum_links"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    HybridSystem(
                        config,
                        load_network_topology("fattree_k4"),
                        load_sim_params(),
                    )


if __name__ == "__main__":
    unittest.main()
