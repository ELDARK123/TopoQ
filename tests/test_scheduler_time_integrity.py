import unittest
from collections import defaultdict
from inspect import getsource
from types import SimpleNamespace

from topoq.topoq_scheduler import TopoQScheduler


class SchedulerTimeIntegrityTests(unittest.TestCase):
    @staticmethod
    def _scheduler(tasks, edges):
        predecessors = defaultdict(list)
        successors = defaultdict(list)
        for source, target in edges:
            successors[source].append(target)
            predecessors[target].append(source)
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.workload = SimpleNamespace(
            tasks={tid: {"id": tid} for tid in tasks},
            predecessors=predecessors,
            successors=successors,
        )
        durations = {tid: float(index + 1) for index, tid in enumerate(tasks)}
        scheduler._reference_duration = lambda tid: durations[tid]
        return scheduler, durations

    def test_priority_uses_every_successor_in_branched_dag(self):
        scheduler, durations = self._scheduler(
            ["root", "short", "long", "join"],
            [
                ("root", "short"),
                ("root", "long"),
                ("short", "join"),
                ("long", "join"),
            ],
        )

        priorities = scheduler._compute_priorities()

        self.assertEqual(priorities["join"], durations["join"])
        self.assertEqual(
            priorities["root"],
            durations["root"] + durations["long"] + durations["join"],
        )

    def test_priority_rejects_cycle(self):
        scheduler, _ = self._scheduler(
            ["left", "right"],
            [("left", "right"), ("right", "left")],
        )

        with self.assertRaisesRegex(ValueError, "must form a DAG"):
            scheduler._compute_priorities()

    def test_non_physical_duration_reduction_moves_are_inactive(self):
        source = getsource(TopoQScheduler._simulated_annealing)
        active_moves = source.split("moves = [", 1)[1].split("]", 1)[0]
        self.assertNotIn("entanglement_opt", active_moves)
        self.assertNotIn("critical_path_opt", active_moves)
        self.assertIn("reorder", active_moves)
        self.assertIn("migration", active_moves)
        self.assertIn("remap", active_moves)


if __name__ == "__main__":
    unittest.main()
