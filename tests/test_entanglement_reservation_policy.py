import unittest
from types import SimpleNamespace

from topoq.config import validate_entanglement_reservation_policy
from topoq.fidelity import estimate_task_log_fidelity
from topoq.topoq_scheduler import TopoQScheduler


class EntanglementReservationPolicyTests(unittest.TestCase):
    def test_policy_validation_exposes_all_modes(self):
        for mode in ("proactive", "on_demand", "just_in_time", "adaptive"):
            policy = validate_entanglement_reservation_policy(mode=mode)
            self.assertEqual(policy["mode"], mode)

        with self.assertRaises(ValueError):
            validate_entanglement_reservation_policy(mode="unknown")
        with self.assertRaises(ValueError):
            validate_entanglement_reservation_policy(
                mode="adaptive", adaptive_2q_gate_threshold=-1
            )

    def test_adaptive_mode_uses_revised_paper_two_qubit_gate_count(self):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.entanglement_reservation_policy = {
            "mode": "adaptive",
            "proactive_lead_fraction": 0.90,
            "jit_memory_fraction": 0.25,
            "adaptive_2q_gate_threshold": 5000,
            "adaptive_high_gate_mode": "just_in_time",
        }
        scheduler.workload = SimpleNamespace(tasks={
            "dense": {
                "n_gates": 200,
                "paper_n_1q_gates": 400,
                "paper_n_2q_gates": 9811,
                "paper_depth": 344,
            },
            "small": {
                "n_gates": 200,
                "paper_n_1q_gates": 100,
                "paper_n_2q_gates": 100,
                "paper_depth": 20,
            },
        })

        self.assertEqual(
            scheduler._resolve_entanglement_reservation_mode("dense"),
            "just_in_time",
        )
        self.assertEqual(
            scheduler._resolve_entanglement_reservation_mode("small"),
            "proactive",
        )

    def test_jit_lead_is_limited_by_entanglement_memory(self):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.entanglement_reservation_policy = {
            "mode": "just_in_time",
            "proactive_lead_fraction": 0.90,
            "jit_memory_fraction": 0.25,
            "adaptive_2q_gate_threshold": 5000,
            "adaptive_high_gate_mode": "just_in_time",
        }
        service = {"epr_memory_lifetime_ms": 40.0}
        self.assertEqual(
            scheduler._entanglement_reservation_lead_ms(
                "just_in_time", service, total_generation=20.0,
                batch_duration=2.0,
            ),
            10.0,
        )
        self.assertEqual(
            scheduler._entanglement_reservation_lead_ms(
                "on_demand", service, total_generation=20.0,
                batch_duration=2.0,
            ),
            0.0,
        )

    def test_phase5_uses_unclipped_log_fidelity_and_swap_count(self):
        scheduler = TopoQScheduler.__new__(TopoQScheduler)
        scheduler.workload = SimpleNamespace(
            quantum_tasks=["q"], tasks={"q": {"n_gates": 1}}
        )
        scheduler.system = SimpleNamespace(qpus={"Q0": {}})
        observed = {}

        def estimate(*args, **kwargs):
            observed.update(kwargs)
            return {"fidelity": 1e-300, "log_fidelity": -1234.5}

        scheduler._estimate_fidelity_details = estimate
        raw = scheduler._objective_raw_components({
            "q": {
                "start": 0.0,
                "end": 1.0,
                "n_remote_gates": 7,
                "n_swap_gates": 5415,
                "entanglement_wait": 0.25,
            }
        }, {"q": "Q0"})

        self.assertEqual(observed["estimated_swap_gates"], 5415)
        self.assertEqual(observed["estimated_remote_gates"], 7)
        self.assertEqual(raw["fidelity_loss"], 1234.5)

    def test_fidelity_uses_public_paper_gate_profile(self):
        system = SimpleNamespace(sim_params={
            "fidelity_parameters": {"F_prep": 0.999, "F_EPR_range": [0.85, 0.95]},
            "quantum_link_parameters": {"eta_range_per_ms": [0.001, 0.01]},
        })
        qpu = {
            "gate_error_rates": {"1q": 0.001, "2q": 0.01, "measure": 0.02},
            "gate_times_ns": {"1q": 35.6, "2q": 334.0, "measure": 1200.0},
            "coherence_time_us": 100.0,
        }
        task = {
            "n_qubits": 200,
            "n_gates": 200,
            "paper_n_1q_gates": 400,
            "paper_n_2q_gates": 9811,
            "paper_depth": 344,
            "paper_measurement_count": 200,
        }
        result = estimate_task_log_fidelity(
            system, task, qpu, estimated_remote_gates=0,
            estimated_swap_gates=5415,
        )

        self.assertEqual(result["n_1q_gates"], 400)
        self.assertEqual(result["n_2q_gates"], 15226)
        self.assertEqual(result["measurement_count"], 200)
        self.assertLess(result["log_fidelity"], -100.0)


if __name__ == "__main__":
    unittest.main()
