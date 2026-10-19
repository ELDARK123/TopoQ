import math
import random
import unittest

from topoq.entanglement import deterministic_epr_service_window_ms
from topoq.simulator import Simulator


class _System:
    quantum_links = {
        frozenset(("Q0", "Q1")): {
            "rate_hz": 100.0,
            "max_concurrent_epr": 8,
        }
    }


class EprBatchCountModelTests(unittest.TestCase):
    def test_single_pair_keeps_exponential_quantile(self):
        expected = -math.log1p(-0.95) / 100.0 * 1000.0
        actual = deterministic_epr_service_window_ms(
            100.0, 0.95, epr_count=1
        )
        self.assertAlmostEqual(actual, expected, places=10)

    def test_more_pairs_require_a_longer_reliability_window(self):
        one = deterministic_epr_service_window_ms(
            100.0, 0.95, epr_count=1
        )
        four = deterministic_epr_service_window_ms(
            100.0, 0.95, epr_count=4
        )
        self.assertGreater(four, one)

    def test_random_batch_is_sum_of_pair_waits(self):
        seed = 31
        expected_rng = random.Random(seed)
        expected = sum(
            -math.log1p(-expected_rng.random()) for _ in range(3)
        ) / 100.0 * 1000.0
        simulator = Simulator(_System(), {}, seed=seed)

        actual = simulator._sample_epr_generation_ms(
            ("Q0", "Q1"), epr_count=3
        )

        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
