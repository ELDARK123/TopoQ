import unittest

from topoq.topoq_scheduler import TopoQScheduler


class FeedbackStateConsistencyTests(unittest.TestCase):
    def _scheduler(self, max_iterations=2):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.params = {
            "max_iterations": max_iterations,
            "fidelity_constraint_mode": "report_only",
        }
        scheduler.ablation = {}
        scheduler._phase2_partition_and_map = lambda: (
            {"q0": "Q0"},
            {"q0": {"state": "mapping-0"}},
        )
        scheduler._assign_classical_nodes = lambda assignment: None

        def phase3_4(assignment, mapping):
            state = "0" if assignment["q0"] == "Q0" else "1"
            scheduler.classical_link_timeline = {"classical": [state]}
            scheduler.quantum_link_timeline = {"quantum": [state]}
            scheduler._intra_task_remote_gates = {"q0": int(state)}
            scheduler._intra_task_remote_gates_by_link = {
                "q0": {("Q0", "Q1"): int(state)}
            }
            end = 5.0 if state == "0" else 8.0
            return {"q0": {"start": 0.0, "end": end}}

        def phase5(sched, assignment, mapping):
            scheduler.last_sa_objective = {
                "state": assignment["q0"],
                "best_makespan": sched["q0"]["end"],
            }
            return sched

        scheduler._phase3_4_joint_scheduling = phase3_4
        scheduler._phase5_joint_optimize = phase5
        scheduler._repair_mapping = lambda assignment, mapping, violations: (
            {"q0": "Q1"},
            {"q0": {"state": "mapping-1"}},
        )
        return scheduler

    def test_returns_complete_state_from_slower_feasible_iteration(self):
        scheduler = self._scheduler()
        scheduler._check_constraints = lambda sched, assignment, mapping: (
            [{"constraint": "C2_QPU_CAPACITY", "enforced": True}]
            if assignment["q0"] == "Q0" else []
        )

        sched, assignment, mapping = scheduler.schedule()

        self.assertEqual(sched["q0"]["end"], 8.0)
        self.assertEqual(assignment, {"q0": "Q1"})
        self.assertEqual(mapping, {"q0": {"state": "mapping-1"}})
        self.assertEqual(
            scheduler.classical_link_timeline, {"classical": ["1"]}
        )
        self.assertEqual(
            scheduler.quantum_link_timeline, {"quantum": ["1"]}
        )
        self.assertEqual(scheduler._intra_task_remote_gates, {"q0": 1})
        self.assertEqual(
            scheduler._intra_task_remote_gates_by_link,
            {"q0": {("Q0", "Q1"): 1}},
        )
        self.assertEqual(scheduler.last_sa_objective["state"], "Q1")
        self.assertEqual(scheduler.last_constraint_report, [])
        self.assertEqual(scheduler.last_enforced_violations, [])
        self.assertEqual(scheduler.last_feedback_iteration, 2)
        self.assertEqual(scheduler.last_feedback_makespan, 8.0)

    def test_all_enforced_violations_fail_with_grouped_summary(self):
        scheduler = self._scheduler()
        scheduler._check_constraints = lambda sched, assignment, mapping: [
            {"constraint": "C3_PRECEDENCE", "enforced": True},
            {"constraint": "C2_QPU_CAPACITY", "enforced": True},
            {"constraint": "C3_PRECEDENCE", "enforced": True},
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"C2_QPU_CAPACITY=1, C3_PRECEDENCE=2",
        ):
            scheduler.schedule()

    def test_report_only_c4_is_returned_and_preserved(self):
        scheduler = self._scheduler(max_iterations=1)
        report = {
            "constraint": "C4_FIDELITY_THRESHOLD",
            "enforced": False,
            "task": "q0",
        }
        scheduler._check_constraints = lambda sched, assignment, mapping: [report]

        sched, assignment, mapping = scheduler.schedule()

        self.assertEqual(sched["q0"]["end"], 5.0)
        self.assertEqual(assignment["q0"], "Q0")
        self.assertEqual(mapping["q0"]["state"], "mapping-0")
        self.assertEqual(scheduler.last_constraint_report, [report])
        self.assertEqual(scheduler.last_enforced_violations, [])

    def test_paper_strict_c4_is_an_enforced_failure(self):
        scheduler = self._scheduler(max_iterations=1)
        scheduler.params["fidelity_constraint_mode"] = "paper_strict"
        scheduler._check_constraints = lambda sched, assignment, mapping: [{
            "constraint": "C4_FIDELITY_THRESHOLD",
            "enforced": True,
            "task": "q0",
        }]

        with self.assertRaisesRegex(
            ValueError,
            r"C4_FIDELITY_THRESHOLD=1",
        ):
            scheduler.schedule()


if __name__ == "__main__":
    unittest.main()
