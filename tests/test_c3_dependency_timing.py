import unittest
from collections import defaultdict
from types import SimpleNamespace

from topoq.topoq_scheduler import TopoQScheduler
from topoq.validators import validate_dag_precedence


class C3DependencyTimingTests(unittest.TestCase):
    def _workload(self, pred_type="quantum", task_type="classical"):
        return SimpleNamespace(
            tasks={
                "pred": {"type": pred_type},
                "task": {"type": task_type},
            },
            edge_data={("pred", "task"): {}},
            predecessors=defaultdict(list, {"task": ["pred"]}),
            successors=defaultdict(list, {"pred": ["task"]}),
        )

    def test_valid_transfer_satisfies_paper_c3(self):
        workload = self._workload()
        sched = {
            "pred": {"start": 0.0, "end": 5.0},
            "task": {
                "start": 7.0,
                "end": 9.0,
                "dependency_transfers": [{
                    "predecessor": "pred",
                    "type": "quantum-classical",
                    "source_resource": "Q0",
                    "target_resource": "P0",
                    "start": 5.0,
                    "end": 7.0,
                }],
            },
        }
        violations = validate_dag_precedence(
            None, workload, sched, {"pred": "Q0", "task": "P0"}, {}
        )
        self.assertEqual(violations, [])

    def test_missing_transfer_is_a_c3_violation(self):
        workload = self._workload()
        sched = {
            "pred": {"start": 0.0, "end": 5.0},
            "task": {"start": 7.0, "end": 9.0},
        }
        violations = validate_dag_precedence(
            None, workload, sched, {"pred": "Q0", "task": "P0"}, {}
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(
            violations[0]["details"]["violation_kind"], "missing_transfer"
        )

    def test_transfer_and_successor_timing_are_checked_separately(self):
        workload = self._workload()
        sched = {
            "pred": {"start": 0.0, "end": 5.0},
            "task": {
                "start": 5.5,
                "end": 8.0,
                "dependency_transfers": [{
                    "predecessor": "pred",
                    "type": "quantum-classical",
                    "source_resource": "Q0",
                    "target_resource": "P0",
                    "start": 4.0,
                    "end": 6.0,
                }],
            },
        }
        violations = validate_dag_precedence(
            None, workload, sched, {"pred": "Q0", "task": "P0"}, {}
        )
        self.assertEqual(
            {item["details"]["violation_kind"] for item in violations},
            {"transfer_before_predecessor_end", "task_before_transfer_end"},
        )

    def test_stale_resource_metadata_is_rejected(self):
        workload = self._workload("quantum", "quantum")
        sched = {
            "pred": {"start": 0.0, "end": 5.0},
            "task": {
                "start": 6.0,
                "end": 8.0,
                "dependency_transfers": [{
                    "predecessor": "pred",
                    "source_resource": "Q0",
                    "target_resource": "Q0",
                    "start": 5.0,
                    "end": 5.0,
                }],
            },
        }
        violations = validate_dag_precedence(
            None, workload, sched, {"pred": "Q0", "task": "Q1"}, {}
        )
        self.assertIn(
            "stale_transfer_metadata",
            {item["details"]["violation_kind"] for item in violations},
        )

    def test_precedence_repair_waits_for_transfer_end(self):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.workload = self._workload()
        sched = {
            "pred": {"start": 0.0, "end": 5.0},
            "task": {
                "start": 5.0,
                "end": 7.0,
                "dependency_transfers": [{
                    "predecessor": "pred",
                    "start": 5.0,
                    "end": 8.0,
                }],
            },
        }

        repaired = scheduler._fix_dag_precedence(sched)

        self.assertEqual(repaired["task"]["start"], 8.0)
        self.assertEqual(repaired["task"]["end"], 10.0)

    def test_phase4_deduplicates_repeated_predecessor_entries(self):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.system = SimpleNamespace(
            nodes={
                "P0": {
                    "memory_gb": 16.0,
                    "compute_speed_tflops": 1.0,
                }
            },
            qpus={"Q0": {}},
            interface_links={"Q0": {"classical_node_id": "P0"}},
            get_node_ids=lambda: ["P0"],
        )
        scheduler.workload = self._workload()
        scheduler.workload.classical_tasks = ["task"]
        scheduler.workload.predecessors["task"] = ["pred", "pred"]
        scheduler.workload.tasks["task"].update({
            "estimated_time_ms": 2.0,
            "memory_mb": 1.0,
        })
        scheduler.ablation = {}
        scheduler._compute_priorities = lambda: {"task": 1.0}
        scheduler._interface_transfer_ms = (
            lambda qpu_id, predecessor, successor: 1.5
        )
        assignment = {"pred": "Q0", "task": "P0"}

        sched = scheduler._phase4_classical_scheduling(
            assignment,
            {"pred": {"start": 0.0, "end": 5.0}},
            preserve_assignment=True,
        )

        self.assertEqual(len(sched["task"]["dependency_transfers"]), 1)
        self.assertEqual(
            sched["task"]["dependency_transfers"][0]["end"], 6.5
        )
        self.assertEqual(sched["task"]["start"], 6.5)


if __name__ == "__main__":
    unittest.main()
