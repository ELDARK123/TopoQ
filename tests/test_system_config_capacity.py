import unittest

from topoq.config import load_system_config


EXPECTED_CONFIGS = {
    "small": {
        "chip_type": "ibm_guadalupe",
        "n_qpus": 4,
        "qubits_per_qpu": 16,
        "total_qubits": 64,
    },
    "medium": {
        "chip_type": "ibm_sydney",
        "n_qpus": 8,
        "qubits_per_qpu": 27,
        "total_qubits": 216,
    },
    "large": {
        "chip_type": "ibm_washington",
        "n_qpus": 16,
        "qubits_per_qpu": 127,
        "total_qubits": 2032,
    },
}


class SystemConfigCapacityTests(unittest.TestCase):
    def test_configs_use_only_physical_instance_capacities(self):
        for name, expected in EXPECTED_CONFIGS.items():
            with self.subTest(config=name):
                config = load_system_config(name)
                instances = config["qpu_instances"]

                self.assertNotIn("qpu_qubits_override", config)
                self.assertEqual(config["qpu_type"], expected["chip_type"])
                self.assertEqual(config["n_qpus"], expected["n_qpus"])
                self.assertEqual(len(instances), expected["n_qpus"])
                self.assertEqual(
                    {qpu["n_qubits"] for qpu in instances},
                    {expected["qubits_per_qpu"]},
                )
                self.assertEqual(
                    sum(qpu["n_qubits"] for qpu in instances),
                    expected["total_qubits"],
                )

    def test_coupling_vertices_fit_declared_capacity(self):
        for name in EXPECTED_CONFIGS:
            config = load_system_config(name)
            for qpu in config["qpu_instances"]:
                with self.subTest(config=name, qpu=qpu["qpu_id"]):
                    vertices = {
                        vertex
                        for edge in qpu["topology"]
                        for vertex in edge
                    }
                    self.assertTrue(vertices)
                    self.assertGreaterEqual(min(vertices), 0)
                    self.assertLess(max(vertices), qpu["n_qubits"])


if __name__ == "__main__":
    unittest.main()
