import math
import unittest
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import (
    QUANTUM_EPR_ROUTING_MODES,
    get_experiment,
    validate_quantum_epr_routing_mode,
    validate_quantum_epr_routing_policy,
)
from topoq.models import HybridSystem
from topoq.runner import run_experiment_spec
from topoq.simulator import Simulator
from topoq.topoq_scheduler import TopoQScheduler
from topoq.validators import validate_entanglement_capacity


class QuantumEprRoutingTests(unittest.TestCase):
    @staticmethod
    def _config(second_channel=True):
        links = [{
            "channel_id": "primary",
            "qpu_i": "Q0",
            "qpu_j": "Q1",
            "entanglement_rate_hz": 100.0,
            "decoherence_rate_per_ms": 0.01,
            "max_concurrent_epr": 4,
            "memory_capacity_epr": 5,
            "distance_km": 1.0,
        }]
        if second_channel:
            links.append({
                "channel_id": "secondary",
                "qpu_i": "Q0",
                "qpu_j": "Q1",
                "entanglement_rate_hz": 300.0,
                "decoherence_rate_per_ms": 0.02,
                "max_concurrent_epr": 6,
                "memory_capacity_epr": 7,
                "distance_km": 1.5,
            })
        return {
            "name": "two-channel-test",
            "n_qpus": 2,
            "n_classical_nodes": 1,
            "qpu_instances": [
                {
                    "qpu_id": qpu_id,
                    "n_qubits": 4,
                    "coherence_time_us": 100.0,
                    "gate_times_ns": {},
                    "gate_error_rates": {},
                    "topology": [],
                }
                for qpu_id in ("Q0", "Q1")
            ],
            "classical_instances": [{
                "node_id": "P0",
                "type": "CPU",
                "compute_speed_tflops": 1.0,
                "memory_gb": 1.0,
            }],
            "quantum_links": links,
            "interface_links": [
                {
                    "qpu_id": qpu_id,
                    "classical_node_id": "P0",
                    "readout_latency_us": 1.0,
                    "control_bandwidth_gbps": 1.0,
                }
                for qpu_id in ("Q0", "Q1")
            ],
        }

    @staticmethod
    def _network():
        return {
            "nodes": ["host_0", "switch_0"],
            "edges": [{
                "source": "host_0",
                "target": "switch_0",
                "bandwidth_gbps": 10.0,
                "latency_us": 1.0,
            }],
        }

    def _system(self, second_channel=True):
        return HybridSystem(
            self._config(second_channel), self._network(), {
                "fidelity_parameters": {
                    "F_EPR_range": [0.85, 0.95],
                },
            }
        )

    def _scheduler(self, mode, second_channel=True, no_multipath=False,
                   reliability_window_ms=None):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.system = self._system(second_channel)
        scheduler.params = {
            "p_threshold": 0.95,
            "quantum_epr_reliability_window_ms": reliability_window_ms,
        }
        scheduler.ablation = {"no_multipath": True} if no_multipath else {}
        scheduler.quantum_epr_routing_mode = mode
        scheduler.quantum_epr_routing_policy = {
            "mode": mode,
            "reliability_window_ms": reliability_window_ms,
        }
        scheduler.quantum_link_timeline = defaultdict(list)
        return scheduler

    def test_public_modes_and_validation(self):
        self.assertEqual(QUANTUM_EPR_ROUTING_MODES, (
            "single_channel",
            "parallel_channels",
            "deadline_direct_backup",
            "paper_fixed_25",
            "paper_algorithm_145",
        ))
        self.assertEqual(
            validate_quantum_epr_routing_mode(), "single_channel"
        )
        with self.assertRaisesRegex(ValueError, "Unknown quantum EPR"):
            validate_quantum_epr_routing_mode("unknown")

    def test_deadline_policy_requires_one_positive_finite_window(self):
        self.assertEqual(
            validate_quantum_epr_routing_policy(
                "deadline_direct_backup", 10
            ),
            {
                "mode": "deadline_direct_backup",
                "reliability_window_ms": 10.0,
            },
        )
        for invalid in (None, 0, -1, math.inf, "bad"):
            with self.assertRaisesRegex(
                ValueError, "quantum_epr_reliability_window_ms"
            ):
                validate_quantum_epr_routing_policy(
                    "deadline_direct_backup", invalid
                )
        with self.assertRaisesRegex(ValueError, "may only be supplied"):
            validate_quantum_epr_routing_policy("single_channel", 10)

    def test_system_preserves_parallel_channels_and_primary_interface(self):
        system = self._system()
        channels = system.get_quantum_link_channels("Q0", "Q1")

        self.assertEqual(len(channels), 2)
        self.assertEqual([item["channel_id"] for item in channels], [
            "primary", "secondary",
        ])
        self.assertIs(system.get_quantum_link("Q0", "Q1"), channels[0])

    def test_physical_parallel_service_aggregates_rate_and_capacity(self):
        scheduler = self._scheduler("parallel_channels")
        service = scheduler._quantum_epr_service(("Q0", "Q1"))

        self.assertEqual(service["service_mode"], "parallel_channels")
        self.assertEqual(service["channel_count"], 2)
        self.assertEqual(service["rate_hz"], 400.0)
        self.assertEqual(service["max_concurrent_epr"], 10)
        self.assertEqual(service["memory_capacity_epr"], 12)
        expected = -math.log(0.05) / 400.0 * 1000.0
        self.assertAlmostEqual(
            scheduler._entanglement_generation_window_ms(service), expected
        )

    def test_parallel_mode_degenerates_when_only_one_channel_exists(self):
        scheduler = self._scheduler(
            "parallel_channels", second_channel=False
        )
        service = scheduler._quantum_epr_service(("Q0", "Q1"))

        self.assertEqual(service["service_mode"], "single_channel")
        self.assertEqual(service["channel_count"], 1)
        self.assertEqual(service["rate_hz"], 100.0)
        self.assertEqual(service["max_concurrent_epr"], 4)
        self.assertEqual(service["memory_capacity_epr"], 5)

    def test_parallel_memory_capacity_requires_all_selected_channels(self):
        scheduler = self._scheduler("parallel_channels")
        channels = scheduler.system.get_quantum_link_channels("Q0", "Q1")
        del channels[1]["memory_capacity_epr"]

        service = scheduler._quantum_epr_service(("Q0", "Q1"))

        self.assertIsNone(service["memory_capacity_epr"])
        self.assertEqual(
            service["memory_capacity_source"], "unspecified_channel"
        )

    def test_deadline_backup_uses_primary_when_it_already_meets_target(self):
        scheduler = self._scheduler(
            "deadline_direct_backup", reliability_window_ms=30.0
        )
        service = scheduler._quantum_epr_service(("Q0", "Q1"))

        expected_probability = 1.0 - math.exp(-100.0 * 30.0 / 1000.0)
        self.assertEqual(service["service_mode"], "single_channel")
        self.assertEqual(service["channel_ids"], ("primary",))
        self.assertEqual(service["backup_channel_count"], 0)
        self.assertAlmostEqual(
            service["planned_success_probability"], expected_probability
        )
        self.assertGreaterEqual(
            service["planned_success_probability"], 0.95
        )

    def test_deadline_backup_adds_minimum_fastest_direct_channel(self):
        scheduler = self._scheduler(
            "deadline_direct_backup", reliability_window_ms=10.0
        )
        service = scheduler._quantum_epr_service(("Q0", "Q1"))

        primary_probability = 1.0 - math.exp(-100.0 * 10.0 / 1000.0)
        combined_probability = 1.0 - math.exp(-400.0 * 10.0 / 1000.0)
        self.assertLess(primary_probability, 0.95)
        self.assertEqual(service["service_mode"], "deadline_direct_backup")
        self.assertEqual(service["channel_ids"], ("primary", "secondary"))
        self.assertEqual(service["backup_channel_count"], 1)
        self.assertEqual(service["rate_hz"], 400.0)
        self.assertEqual(service["max_concurrent_epr"], 10)
        self.assertAlmostEqual(
            service["planned_success_probability"], combined_probability
        )
        self.assertGreaterEqual(combined_probability, 0.95)

    def test_deadline_backup_fails_when_direct_channels_cannot_meet_target(self):
        scheduler = self._scheduler(
            "deadline_direct_backup", reliability_window_ms=1.0
        )
        with self.assertRaisesRegex(
            ValueError, "cannot reach p_threshold=0.95"
        ):
            scheduler._quantum_epr_service(("Q0", "Q1"))

    def test_no_multipath_disables_required_direct_backup(self):
        scheduler = self._scheduler(
            "deadline_direct_backup",
            no_multipath=True,
            reliability_window_ms=10.0,
        )
        with self.assertRaisesRegex(
            ValueError, "no_multipath disables direct backup channels"
        ):
            scheduler._quantum_epr_service(("Q0", "Q1"))

    def test_paper_proxy_scales_match_the_two_manuscript_definitions(self):
        fixed = self._scheduler("paper_fixed_25")
        fixed_disabled = self._scheduler(
            "paper_fixed_25", no_multipath=True
        )
        algorithm = self._scheduler("paper_algorithm_145")
        algorithm_disabled = self._scheduler(
            "paper_algorithm_145", no_multipath=True
        )

        self.assertEqual(
            fixed._quantum_epr_service(("Q0", "Q1"))["duration_scale"],
            0.75,
        )
        self.assertEqual(
            fixed_disabled._quantum_epr_service(
                ("Q0", "Q1")
            )["duration_scale"],
            1.0,
        )
        self.assertEqual(
            algorithm._quantum_epr_service(
                ("Q0", "Q1")
            )["duration_scale"],
            1.0,
        )
        self.assertEqual(
            algorithm_disabled._quantum_epr_service(
                ("Q0", "Q1")
            )["duration_scale"],
            1.45,
        )

    def test_reservation_uses_aggregate_capacity_and_records_diagnostics(self):
        scheduler = self._scheduler("parallel_channels")
        ready, records = scheduler._reserve_entanglement_batches(
            ("Q0", "Q1"), 0.0, 9, "task"
        )

        self.assertEqual(len(records), 1)
        self.assertGreater(ready, 0.0)
        self.assertEqual(records[0]["capacity_epr"], 10)
        self.assertEqual(records[0]["generation_capacity_epr"], 10)
        self.assertEqual(records[0]["memory_capacity_epr"], 12)
        self.assertEqual(records[0]["effective_rate_hz"], 400.0)
        self.assertEqual(records[0]["quantum_channel_count"], 2)
        self.assertEqual(records[0]["quantum_epr_service_mode"],
                         "parallel_channels")

    def test_poisson_replay_uses_effective_rate_and_paper_scale(self):
        system = self._system()
        raw = Simulator(system, {}, seed=7)._sample_epr_generation_ms(
            ("Q0", "Q1")
        )
        adjusted = Simulator(system, {}, seed=7)._sample_epr_generation_ms(
            ("Q0", "Q1"), effective_rate_hz=400.0,
            duration_scale=0.75,
        )

        self.assertAlmostEqual(adjusted, raw * 100.0 / 400.0 * 0.75)

    def test_c9_uses_effective_parallel_capacity_from_reservations(self):
        system = self._system()
        base = {
            "start": 0.0,
            "end": 1.0,
            "deadline": 1.0,
            "epr_count": 8,
            "capacity_epr": 10,
            "task": "task",
        }
        timeline = {frozenset(("Q0", "Q1")): [base]}
        self.assertEqual(validate_entanglement_capacity(
            system, None, None, None, None,
            quantum_link_timeline=timeline,
        ), [])

        overloaded = dict(base, task="task-2", epr_count=3)
        timeline[frozenset(("Q0", "Q1"))].append(overloaded)
        violations = validate_entanglement_capacity(
            system, None, None, None, None,
            quantum_link_timeline=timeline,
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(
            violations[0]["constraint"], "C9_EPR_GENERATION_CAPACITY"
        )

    def test_runner_metadata_records_physical_degeneration_and_proxy(self):
        experiment = get_experiment("VQE-50_Small")
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0},
        ):
            physical = run_experiment_spec(
                experiment,
                profile="quick",
                repetitions=1,
                algorithms=("TopoQ",),
                params={"quantum_epr_routing_mode": "parallel_channels"},
            )
            algorithm_proxy = run_experiment_spec(
                replace(experiment, ablation={"no_multipath": True}),
                profile="quick",
                repetitions=1,
                algorithms=("TopoQ",),
                params={
                    "quantum_epr_routing_mode": "paper_algorithm_145"
                },
            )
            deadline = run_experiment_spec(
                experiment,
                profile="quick",
                repetitions=1,
                algorithms=("TopoQ",),
                params={
                    "quantum_epr_routing_mode": "deadline_direct_backup",
                    "quantum_epr_reliability_window_ms": 30.0,
                },
            )

        physical_meta = physical["metadata"]["quantum_epr_routing"]
        self.assertEqual(
            physical_meta["effective_physical_service"], "single_channel"
        )
        self.assertEqual(physical_meta["maximum_channel_count"], 1)
        proxy_meta = algorithm_proxy["metadata"]["quantum_epr_routing"]
        self.assertEqual(proxy_meta["paper_duration_scale"], 1.45)
        self.assertTrue(proxy_meta["multipath_disabled"])
        deadline_meta = deadline["metadata"]["quantum_epr_routing"]
        self.assertEqual(
            deadline_meta["effective_physical_service"],
            "reservation_dependent_direct_channels",
        )
        self.assertEqual(deadline_meta["reliability_window_ms"], 30.0)
        self.assertEqual(
            deadline_meta["backup_selection"],
            "primary_then_descending_rate_until_threshold",
        )
        self.assertEqual(
            physical["metadata"]["entanglement_capacity_constraint"]
            ["missing_memory_capacity_mode"],
            "report",
        )


if __name__ == "__main__":
    unittest.main()
