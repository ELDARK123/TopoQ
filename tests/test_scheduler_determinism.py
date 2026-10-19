import os
import subprocess
import sys
import textwrap
import unittest


SCRIPT = textwrap.dedent(
    """
    from collections import defaultdict
    from types import SimpleNamespace
    from topoq.topoq_scheduler import TopoQScheduler

    scheduler = TopoQScheduler.__new__(TopoQScheduler)
    scheduler.system = SimpleNamespace(
        nodes={"n0": {"compute_speed_tflops": 1.0, "memory_gb": 1.0}},
        qpus={},
        interface_links={},
        get_node_ids=lambda: ["n0"],
    )
    scheduler.workload = SimpleNamespace(
        classical_tasks=["c0", "c1", "c2"],
        quantum_tasks=[],
        tasks={
            "c0": {"type": "classical", "estimated_time_ms": 1.0},
            "c1": {"type": "classical", "estimated_time_ms": 1.0},
            "c2": {"type": "classical", "estimated_time_ms": 1.0},
        },
        predecessors=defaultdict(list),
    )
    scheduler.ablation = {"no_comm_overlap": True}
    scheduler._compute_priorities = lambda: {"c0": 1.0, "c1": 1.0, "c2": 1.0}
    assignment = {"c0": "n0", "c1": "n0", "c2": "n0"}
    result = scheduler._phase4_classical_scheduling(
        assignment, {}, preserve_assignment=True
    )
    print(",".join(sorted(result, key=lambda tid: result[tid]["start"])))
    """
)


class SchedulerDeterminismTests(unittest.TestCase):
    def test_phase4_tie_order_is_independent_of_python_hash_seed(self):
        outputs = []
        for hash_seed in ("1", "3"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = hash_seed
            completed = subprocess.run(
                [sys.executable, "-c", SCRIPT],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs, ["c0,c1,c2", "c0,c1,c2"])


if __name__ == "__main__":
    unittest.main()
