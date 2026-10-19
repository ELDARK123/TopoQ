import math
import unittest
from types import SimpleNamespace

from topoq.calibration import (
    load_ibm_chip_calibration,
    resolve_cx_calibration,
    resolve_qpu_calibration_profile,
)
from topoq.config import (
    load_network_topology,
    load_sim_params,
    load_system_config,
    validate_qpu_calibration_policy,
)
from topoq.fidelity import estimate_task_log_fidelity
from topoq.models import HybridSystem


class IbmPhysicalCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.policy = validate_qpu_calibration_policy(
            "ibm_physical_qubit", "conservative_max", "error"
        )
        self.system = HybridSystem(
            load_system_config("small"),
            load_network_topology("fattree_k4"),
            load_sim_params(),
        )
        self.system.sim_params["qpu_calibration_policy"] = self.policy

    def test_snapshot_units_are_standardized(self):
        calibration = load_ibm_chip_calibration("ibm_guadalupe")
        self.assertEqual(calibration["units"]["t1_us"], "us")
        self.assertEqual(calibration["units"]["t2_us"], "us")
        self.assertEqual(calibration["units"]["gate_length"], "ns")
        self.assertEqual(calibration["units"]["readout_length_ns"], "ns")

    def test_physical_profile_records_selected_qubits_and_snapshot(self):
        profile = resolve_qpu_calibration_profile(
            self.system.qpus["Q0"],
            self.policy,
            physical_qubits=[0, 1],
            two_qubit_pairs=[((0, 1), 4)],
        )

        provenance = profile["provenance"]
        self.assertEqual(provenance["mode"], "ibm_physical_qubit")
        self.assertEqual(provenance["selected_physical_qubits"], [0, 1])
        self.assertEqual(provenance["two_qubit_resolution"], "mapped_interaction_edges")
        self.assertGreater(profile["gate_times_ns"]["2q"], 0.0)
        self.assertGreater(profile["coherence_time_us"], 0.0)

    def test_conservative_cx_uses_worst_direction(self):
        qpu = self.system.qpus["Q0"]
        calibration = qpu["ibm_calibration"]
        forward = calibration["gates"][("cx", (0, 1))]
        reverse = calibration["gates"][("cx", (1, 0))]

        resolved = resolve_cx_calibration(qpu, 0, 1, self.policy)

        self.assertEqual(
            resolved["length_ns"],
            max(forward["length_ns"], reverse["length_ns"]),
        )
        self.assertEqual(
            resolved["error"], max(forward["error"], reverse["error"])
        )

    def test_multi_qpu_log_fidelity_accumulates_both_partitions(self):
        system = SimpleNamespace(sim_params={
            "fidelity_parameters": {
                "F_prep": 1.0,
                "F_EPR_range": [0.9, 0.95],
            },
            "quantum_link_parameters": {"eta_range_per_ms": [0.0, 0.0]},
        })
        qpu = {
            "id": "Q0",
            "gate_error_rates": {"1q": 0.0, "2q": 0.0, "measure": 0.0},
            "gate_times_ns": {"1q": 10.0, "2q": 20.0, "measure": 30.0},
            "coherence_time_us": 1000.0,
        }
        task = {
            "n_qubits": 4,
            "n_gates": 4,
            "paper_n_1q_gates": 0,
            "paper_n_2q_gates": 4,
            "paper_depth": 4,
            "paper_measurement_count": 0,
        }
        schedule = {
            "n_remote_gates": 0,
            "n_swap_gates": 0,
            "entanglement_reservations": [],
            "quantum_timing": {"qpu_partitions": {
                "Q0": {
                    "gate_times_ns": qpu["gate_times_ns"],
                    "gate_error_rates": qpu["gate_error_rates"],
                    "coherence_time_us": 1000.0,
                    "logical_qubit_count": 2,
                    "swap_gate_count": 0,
                },
                "Q1": {
                    "gate_times_ns": qpu["gate_times_ns"],
                    "gate_error_rates": {
                        "1q": 0.0, "2q": 0.1, "measure": 0.0,
                    },
                    "coherence_time_us": 1000.0,
                    "logical_qubit_count": 2,
                    "swap_gate_count": 0,
                },
            }},
        }

        result = estimate_task_log_fidelity(
            system, task, qpu, schedule_item=schedule
        )

        self.assertAlmostEqual(result["log_gate"], 2.0 * math.log(0.9))
        self.assertEqual(len(result["partition_fidelity_components"]), 2)


if __name__ == "__main__":
    unittest.main()
