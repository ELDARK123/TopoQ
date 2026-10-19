import unittest
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from topoq.communication import (
    reserve_classical_path,
    reserve_classical_route,
    select_classical_route,
    select_edge_disjoint_paths,
    split_data_across_paths,
)
from topoq.config import (
    CLASSICAL_BANDWIDTH_SHARING_MODES,
    CLASSICAL_ROUTING_MODES,
    get_experiment,
    validate_classical_bandwidth_sharing_mode,
    validate_classical_routing_policy,
)
from topoq.runner import run_experiment_spec
from topoq.main import main
from topoq.topoq_scheduler import TopoQScheduler
from topoq.validators import validate_classical_bandwidth


class ClassicalMultipathRoutingTests(unittest.TestCase):
    @staticmethod
    def _system(include_second_path=True):
        edges = [
            ("A", "C", 20.0, 1.0),
            ("C", "D", 20.0, 1.0),
        ]
        if include_second_path:
            edges.extend([
                ("A", "B", 10.0, 1.0),
                ("B", "D", 10.0, 1.0),
            ])
        adjacency = defaultdict(list)
        edge_info = {}
        for left, right, bandwidth, latency in edges:
            adjacency[left].append((right, bandwidth, latency))
            adjacency[right].append((left, bandwidth, latency))
            edge_info[frozenset((left, right))] = {
                "bw": bandwidth,
                "lat": latency,
            }
        return SimpleNamespace(
            net_adj=adjacency,
            net_edge_info=edge_info,
            get_network_node_id=lambda node: node,
        )

    def test_policy_modes_and_parameter_validation(self):
        self.assertEqual(CLASSICAL_ROUTING_MODES, (
            "single_widest",
            "paper_fixed_15",
            "edge_disjoint_split",
        ))
        self.assertEqual(
            CLASSICAL_BANDWIDTH_SHARING_MODES,
            ("exclusive_reservation",),
        )
        self.assertEqual(
            validate_classical_bandwidth_sharing_mode(),
            "exclusive_reservation",
        )
        with self.assertRaisesRegex(
            ValueError, "Unknown classical bandwidth sharing mode"
        ):
            validate_classical_bandwidth_sharing_mode("unknown")
        self.assertEqual(validate_classical_routing_policy(), {
            "mode": "single_widest",
            "multipath_k": None,
        })
        self.assertEqual(
            validate_classical_routing_policy("edge_disjoint_split", 3),
            {"mode": "edge_disjoint_split", "multipath_k": 3},
        )
        with self.assertRaises(ValueError):
            validate_classical_routing_policy("unknown", 2)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                validate_classical_routing_policy(
                    "edge_disjoint_split", invalid
                )

    def test_selects_widest_edge_disjoint_paths_and_water_fills(self):
        paths = select_edge_disjoint_paths(self._system(), "A", "D", 2)
        allocations = split_data_across_paths(paths, 30.0)

        self.assertEqual([path["path"] for path in paths], [
            ["A", "C", "D"],
            ["A", "B", "D"],
        ])
        self.assertEqual(
            [path["bottleneck_gbps"] for path in paths], [20.0, 10.0]
        )
        self.assertAlmostEqual(allocations[0], 20.0)
        self.assertAlmostEqual(allocations[1], 10.0)

    def test_cli_accepts_explicit_bandwidth_sharing_mode(self):
        args = [
            "--profile", "quick",
            "--list-experiments",
            "--classical-bandwidth-sharing-mode",
            "exclusive_reservation",
        ]
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = main(args)
        self.assertEqual(result, {})

    def test_real_split_reserves_subflows_and_obeys_contention(self):
        system = self._system()
        route = select_classical_route(
            system, "A", "D", "edge_disjoint_split", 2
        )
        timeline = defaultdict(list)
        first = reserve_classical_route(
            system, timeline, route, 0.0, 30.0, "t1", "p1"
        )
        second = reserve_classical_route(
            system, timeline, route, 0.0, 30.0, "t2", "p2"
        )

        self.assertEqual(first["path_count"], 2)
        self.assertAlmostEqual(first["data_mb"], 30.0)
        self.assertAlmostEqual(first["end"], 8.002)
        self.assertAlmostEqual(second["start"], first["end"])
        self.assertAlmostEqual(second["end"], 16.004)
        self.assertEqual(
            first["bandwidth_sharing_mode"], "exclusive_reservation"
        )
        self.assertTrue(all(
            len(reservations) == 2 for reservations in timeline.values()
        ))

    def test_edge_disjoint_paths_can_overlap(self):
        system = self._system()
        paths = select_edge_disjoint_paths(system, "A", "D", 2)
        timeline = defaultdict(list)

        first = reserve_classical_path(
            system, timeline, paths[0], 0.0, 10.0, "t1", "p1"
        )
        second = reserve_classical_path(
            system, timeline, paths[1], 0.0, 10.0, "t2", "p2"
        )

        self.assertEqual(first["start"], 0.0)
        self.assertEqual(second["start"], 0.0)
        self.assertGreater(first["end"], 0.0)
        self.assertGreater(second["end"], 0.0)

    def test_paper_fixed_15_scales_single_path_duration(self):
        system = self._system()
        route = select_classical_route(
            system, "A", "D", "paper_fixed_15"
        )
        reservation = reserve_classical_route(
            system, defaultdict(list), route, 0.0, 30.0, "t", "p"
        )

        self.assertEqual(reservation["routing_mode"], "paper_fixed_15")
        self.assertEqual(reservation["path_count"], 1)
        self.assertAlmostEqual(reservation["duration"], 12.002 * 0.85)
        self.assertEqual(reservation["duration_scale"], 0.85)

    def test_multipath_gracefully_degenerates_to_one_path(self):
        system = self._system(include_second_path=False)
        route = select_classical_route(
            system, "A", "D", "edge_disjoint_split", 4
        )
        reservation = reserve_classical_route(
            system, defaultdict(list), route, 0.0, 10.0, "t", "p"
        )

        self.assertEqual(len(route["paths"]), 1)
        self.assertEqual(reservation["path_count"], 1)
        self.assertAlmostEqual(reservation["data_mb"], 10.0)

    def test_c8_detects_three_flow_cumulative_overload(self):
        system = SimpleNamespace(net_edge_info={
            frozenset(("A", "B")): {"bw": 10.0},
        })
        timeline = {
            frozenset(("A", "B")): [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "bandwidth_gbps": 4.0,
                    "task": f"t{index}",
                }
                for index in range(3)
            ],
        }
        violations = validate_classical_bandwidth(
            system, None, None, None, None,
            classical_link_timeline=timeline,
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["constraint"], "C8_CLASSICAL_BANDWIDTH")
        self.assertEqual(
            violations[0]["details"]["used_bandwidth_gbps"], 12.0
        )

    def test_no_multipath_ablation_forces_single_classical_path(self):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.system = self._system()
        scheduler.classical_routing_policy = {
            "mode": "edge_disjoint_split",
            "multipath_k": 2,
        }
        scheduler.ablation = {"no_multipath": True}

        route = scheduler._select_classical_path("A", "D", 30.0)

        self.assertEqual(route["routing_mode"], "single_widest")
        self.assertEqual(len(route["paths"]), 1)

    def test_runner_metadata_records_classical_routing(self):
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
                    "classical_routing_mode": "edge_disjoint_split",
                    "classical_multipath_k": 3,
                },
            )

        routing = result["metadata"]["classical_routing"]
        self.assertEqual(routing["mode"], "edge_disjoint_split")
        self.assertEqual(routing["multipath_k"], 3)
        self.assertIsNone(routing["paper_fixed_reduction_fraction"])
        sharing = result["metadata"]["classical_bandwidth_sharing"]
        self.assertEqual(sharing["mode"], "exclusive_reservation")
        self.assertEqual(
            sharing["rate_allocation"], "fixed_path_bottleneck"
        )
        self.assertEqual(
            sharing["shared_edge_policy"],
            "serialize_conflicting_intervals",
        )
        self.assertTrue(sharing["edge_disjoint_subflows_can_overlap"])
        self.assertEqual(sharing["applies_to"], ["Greedy", "TopoQ"])


if __name__ == "__main__":
    unittest.main()
