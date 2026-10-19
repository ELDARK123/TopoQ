import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from topoq.config import get_experiment, validate_communication_overlap_mode
from topoq.runner import run_experiment_spec, run_with_repetitions
from topoq.simulator import Simulator
from topoq.topoq_scheduler import TopoQScheduler


class CommunicationOverlapReportingTests(unittest.TestCase):
    def _scheduler(self, mode="paper_fixed_report", ablation=None):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.communication_overlap_mode = mode
        scheduler.ablation = dict(ablation or {})
        return scheduler

    def test_mode_validation(self):
        self.assertEqual(
            validate_communication_overlap_mode("paper_fixed_report"),
            "paper_fixed_report",
        )
        with self.assertRaisesRegex(ValueError, "communication overlap mode"):
            validate_communication_overlap_mode("physical_shortening")

    def test_apply_overlap_preserves_physical_schedule_and_raw_waits(self):
        scheduler = self._scheduler()
        schedule = {
            "q0": {
                "start": 10.0,
                "end": 35.0,
                "entanglement_wait": 12.0,
                "qpu": "Q0",
            },
            "c0": {
                "start": 40.0,
                "end": 46.0,
                "comm_time": 4.0,
                "node": "P0",
            },
        }
        before = deepcopy(schedule)

        result = scheduler._apply_overlap(schedule, {})

        for task_id in schedule:
            self.assertEqual(result[task_id]["start"], before[task_id]["start"])
            self.assertEqual(result[task_id]["end"], before[task_id]["end"])
            self.assertEqual(
                result[task_id]["end"] - result[task_id]["start"],
                before[task_id]["end"] - before[task_id]["start"],
            )
        self.assertEqual(result["q0"]["entanglement_wait"], 12.0)
        self.assertEqual(result["c0"]["comm_time"], 4.0)
        self.assertAlmostEqual(
            result["q0"]["paper_entanglement_overlap_credit_ms"], 3.6
        )
        self.assertAlmostEqual(
            result["q0"]["paper_entanglement_effective_wait_ms"], 8.4
        )
        self.assertAlmostEqual(
            result["c0"]["paper_classical_overlap_credit_ms"], 1.0
        )
        self.assertAlmostEqual(
            result["c0"]["paper_classical_effective_comm_ms"], 3.0
        )

    def test_disabled_ablation_reports_zero_without_timing_change(self):
        scheduler = self._scheduler(ablation={"no_comm_overlap": True})
        schedule = {
            "q0": {
                "start": 1.0,
                "end": 9.0,
                "entanglement_wait": 5.0,
                "comm_time": 2.0,
            }
        }

        result = scheduler._apply_overlap(schedule, {})

        self.assertEqual((result["q0"]["start"], result["q0"]["end"]), (1.0, 9.0))
        self.assertEqual(result["q0"]["entanglement_wait"], 5.0)
        self.assertEqual(result["q0"]["comm_time"], 2.0)
        self.assertEqual(
            result["q0"]["paper_entanglement_overlap_credit_ms"], 0.0
        )
        self.assertEqual(
            result["q0"]["paper_classical_overlap_credit_ms"], 0.0
        )

    def test_classical_candidate_rebuild_does_not_shorten_task(self):
        scheduler = self._scheduler()
        scheduler.workload = SimpleNamespace(quantum_tasks=set())
        scheduler.classical_link_timeline = {"old": []}
        scheduler._phase4_classical_scheduling = lambda *args, **kwargs: {
            "c0": {
                "start": 4.0,
                "end": 9.0,
                "comm_time": 3.0,
                "node": "P0",
            }
        }
        scheduler._fix_dag_precedence = lambda schedule: schedule

        rebuilt, _ = scheduler._rebuild_classical_candidate({}, {"c0": "P0"})

        self.assertEqual(rebuilt["c0"]["start"], 4.0)
        self.assertEqual(rebuilt["c0"]["end"], 9.0)
        self.assertEqual(rebuilt["c0"]["comm_time"], 3.0)
        self.assertAlmostEqual(
            rebuilt["c0"]["paper_classical_overlap_credit_ms"], 0.75
        )

    def test_simulator_aggregates_separate_proxy_metrics(self):
        workload = SimpleNamespace(
            tasks={
                "q0": {"type": "quantum"},
                "c0": {"type": "classical"},
                "c1": {"type": "classical"},
            },
            predecessors={"q0": [], "c0": ["q0"], "c1": ["c0"]},
            quantum_tasks=["q0"],
            classical_tasks=["c0", "c1"],
        )
        schedule = {
            "q0": {
                "start": 0.0,
                "end": 10.0,
                "entanglement_wait": 5.0,
                "qpu": "Q0",
            },
            "c0": {
                "start": 10.0,
                "end": 14.0,
                "comm_time": 2.0,
                "node": "P0",
            },
            "c1": {
                "start": 14.0,
                "end": 20.0,
                "comm_time": 4.0,
                "node": "P1",
            },
        }
        simulator = Simulator(
            SimpleNamespace(),
            {},
            communication_overlap_mode="paper_fixed_report",
        )
        simulator._compute_fidelity_metrics = lambda *args: {"fidelity": 1.0}
        simulator._compute_utilization = lambda *args: 0.0

        metrics = simulator.simulate(
            schedule,
            {"q0": "Q0", "c0": "P0", "c1": "P1"},
            workload,
            {},
        )

        self.assertEqual(metrics["makespan"], 20.0)
        self.assertEqual(metrics["comm_total"], 11.0)
        self.assertAlmostEqual(
            metrics["paper_entanglement_overlap_credit_ms"], 1.5
        )
        self.assertAlmostEqual(metrics["paper_qc_overlap_credit_ms"], 0.5)
        self.assertAlmostEqual(
            metrics["paper_classical_overlap_credit_ms"], 1.0
        )
        self.assertAlmostEqual(metrics["paper_overlap_credit_total_ms"], 3.0)
        self.assertAlmostEqual(metrics["paper_effective_comm_total_ms"], 8.0)

    def test_runner_metadata_records_reporting_semantics_and_factors(self):
        with patch(
            "topoq.runner.run_with_repetitions",
            return_value={"makespan": 1.0},
        ):
            result = run_experiment_spec(
                get_experiment("VQE-50_Small"),
                profile="quick",
                repetitions=1,
                algorithms=("Greedy", "TopoQ"),
                params={"communication_overlap_mode": "paper_fixed_report"},
            )

        metadata = result["metadata"]["communication_overlap"]
        self.assertEqual(metadata["effective_mode"], "paper_fixed_report")
        self.assertTrue(metadata["reporting_only"])
        self.assertTrue(metadata["physical_schedule_unchanged"])
        self.assertEqual(metadata["paper_entanglement_effective_factor"], 0.7)
        self.assertEqual(metadata["paper_classical_effective_factor"], 0.75)
        self.assertEqual(metadata["applies_to"], ["TopoQ"])

    def test_repetition_averaging_preserves_overlap_mode_string(self):
        sample = {
            "communication_overlap_mode": "paper_fixed_report",
            "makespan": 12.0,
        }
        with patch("topoq.runner.run_single_experiment", return_value=sample):
            result = run_with_repetitions(
                "dag", "system", "network", "TopoQ", n_rep=2
            )

        self.assertEqual(
            result["communication_overlap_mode"], "paper_fixed_report"
        )
        self.assertEqual(result["makespan"], 12.0)


if __name__ == "__main__":
    unittest.main()
