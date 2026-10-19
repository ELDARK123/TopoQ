import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topoq.baselines import (
    DQTetrisScheduler,
    GreedyScheduler,
    QOnlyScheduler,
    RandomScheduler,
)
from topoq.config import validate_execution_time_policy
from topoq.timing import (
    classical_task_duration_ms,
    paper_quantum_duration_ms,
    quantum_task_duration_ms,
)
from topoq.topoq_scheduler import TopoQScheduler


class ExecutionTimeModelTests(unittest.TestCase):
    def setUp(self):
        self.fast_qpu = {
            "gate_times_ns": {"1q": 10.0, "2q": 30.0, "measure": 100.0}
        }
        self.slow_qpu = {
            "gate_times_ns": {"1q": 20.0, "2q": 60.0, "measure": 100.0}
        }
        self.quantum_task = {
            "id": "q0",
            "type": "quantum",
            "estimated_time_ms": 7.0,
            "paper_n_1q_gates": 2,
            "paper_n_2q_gates": 2,
            "paper_depth": 10,
            "paper_measurement_count": 1,
            "shot_scaled": True,
        }

    def test_policy_validation_and_canonical_values(self):
        policy = validate_execution_time_policy(
            "paper_equation", 1000.0, 0.75
        )
        self.assertEqual(policy["mode"], "paper_equation")
        self.assertEqual(policy["quantum_shots"], 1000)
        self.assertEqual(policy["classical_compute_efficiency"], 0.75)
        with self.assertRaises(ValueError):
            validate_execution_time_policy("unknown", 1000, 1.0)
        with self.assertRaises(ValueError):
            validate_execution_time_policy("paper_equation", 0, 1.0)

    def test_weighted_gate_time_and_shot_scaling(self):
        # tau_bar=(2*10+2*30)/4=20 ns; one shot is 10*20+100=300 ns.
        self.assertAlmostEqual(
            paper_quantum_duration_ms(
                self.quantum_task, self.fast_qpu, quantum_shots=1000
            ),
            0.3,
        )
        one_execution = dict(self.quantum_task, shot_scaled=False)
        self.assertAlmostEqual(
            paper_quantum_duration_ms(
                one_execution, self.fast_qpu, quantum_shots=1000
            ),
            0.0003,
        )

    def test_legacy_quantum_proxy_and_single_routing_addition(self):
        system = SimpleNamespace(qpus={"Q0": self.fast_qpu})
        self.assertAlmostEqual(
            quantum_task_duration_ms(
                self.quantum_task,
                system,
                ["Q0"],
                {"execution_time_mode": "legacy_proxy"},
                routing_overhead_ms=2.0,
            ),
            9.0,
        )

    def test_multi_qpu_duration_uses_conservative_maximum(self):
        system = SimpleNamespace(
            qpus={"Q0": self.fast_qpu, "Q1": self.slow_qpu}
        )
        duration = quantum_task_duration_ms(
            self.quantum_task,
            system,
            ["Q0", "Q1"],
            {
                "execution_time_mode": "paper_equation",
                "quantum_shots": 1000,
            },
        )
        self.assertAlmostEqual(duration, 0.5)

    def test_classical_paper_equation_and_legacy_compatibility(self):
        task = {
            "id": "c0",
            "type": "classical",
            "workload_flops": 2e12,
            "estimated_time_ms": 8.0,
        }
        node = {"compute_speed_tflops": 4.0}
        self.assertAlmostEqual(
            classical_task_duration_ms(
                task,
                node,
                {
                    "execution_time_mode": "paper_equation",
                    "classical_compute_efficiency": 0.5,
                },
            ),
            1000.0,
        )
        self.assertAlmostEqual(
            classical_task_duration_ms(
                task, node, {"execution_time_mode": "legacy_proxy"}
            ),
            2.0,
        )

    def test_all_schedulers_store_the_same_timing_policy(self):
        params = {
            "execution_time_mode": "paper_equation",
            "quantum_shots": 321,
            "classical_compute_efficiency": 0.8,
        }
        system = SimpleNamespace(
            sim_params={},
            qpus={"Q0": self.fast_qpu},
            nodes={"P0": {"compute_speed_tflops": 1.0}},
        )
        workload = SimpleNamespace(tasks={})
        schedulers = [
            scheduler_type(system, workload, seed=42, params=params)
            for scheduler_type in (
                RandomScheduler,
                GreedyScheduler,
                QOnlyScheduler,
                DQTetrisScheduler,
            )
        ]
        with patch.object(TopoQScheduler, "_compute_topology_metrics"):
            schedulers.append(
                TopoQScheduler(system, workload, params=params, seed=42)
            )

        for scheduler in schedulers:
            self.assertEqual(
                scheduler.params["execution_time_mode"], "paper_equation"
            )
            self.assertEqual(scheduler.params["quantum_shots"], 321)
            self.assertEqual(
                scheduler.params["classical_compute_efficiency"], 0.8
            )


if __name__ == "__main__":
    unittest.main()
