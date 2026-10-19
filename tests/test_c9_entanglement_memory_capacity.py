import math
import unittest
from copy import deepcopy
from types import SimpleNamespace

from topoq.config import (
    ENTANGLEMENT_MEMORY_CAPACITY_MODES,
    load_network_topology,
    load_sim_params,
    load_system_config,
    validate_entanglement_memory_capacity_mode,
)
from topoq.models import HybridSystem
from topoq.validators import validate_entanglement_capacity


class C9EntanglementMemoryCapacityTests(unittest.TestCase):
    link_key = frozenset(("Q0", "Q1"))

    @classmethod
    def _system(cls, generation_capacity=2, memory_capacity=3):
        link = {"max_concurrent_epr": generation_capacity}
        if memory_capacity is not None:
            link["memory_capacity_epr"] = memory_capacity
        return SimpleNamespace(quantum_links={cls.link_key: link})

    @classmethod
    def _record(cls, task, start, ready, consume, count,
                generation_capacity=2, memory_capacity=3, **extra):
        record = {
            "task": task,
            "link": ("Q0", "Q1"),
            "generation_start": start,
            "start": start,
            "ready": ready,
            "end": ready,
            "consume": consume,
            "deadline": consume,
            "epr_count": count,
            "generation_capacity_epr": generation_capacity,
            "memory_capacity_epr": memory_capacity,
        }
        record.update(extra)
        return record

    def _validate(self, system, records, mode="report"):
        return validate_entanglement_capacity(
            system, None, None, None, None,
            quantum_link_timeline={self.link_key: records},
            memory_capacity_mode=mode,
        )

    def test_generation_parallelism_is_distinct_from_memory_capacity(self):
        records = [
            self._record("a", 0.0, 2.0, 2.0, 2, memory_capacity=10),
            self._record("b", 0.0, 2.0, 2.0, 1, memory_capacity=10),
        ]
        violations = self._validate(self._system(2, 10), records)

        self.assertEqual(
            [item["constraint"] for item in violations],
            ["C9_EPR_GENERATION_CAPACITY"],
        )

    def test_nonoverlapping_generation_can_overflow_epr_memory(self):
        records = [
            self._record("a", 0.0, 1.0, 4.0, 2),
            self._record("b", 1.0, 2.0, 4.0, 2),
        ]
        violations = self._validate(self._system(), records)

        self.assertEqual(
            [item["constraint"] for item in violations],
            ["C9_EPR_MEMORY_CAPACITY"],
        )
        self.assertEqual(violations[0]["details"]["stored_epr"], 4)
        self.assertTrue(violations[0]["enforced"])
        self.assertEqual(
            violations[0]["details"]["memory_capacity_epr"], 3
        )

    def test_memory_release_precedes_ready_at_the_same_time(self):
        records = [
            self._record("a", 0.0, 1.0, 2.0, 2, memory_capacity=2),
            self._record("b", 1.0, 2.0, 3.0, 2, memory_capacity=2),
        ]

        self.assertEqual(self._validate(self._system(2, 2), records), [])

    def test_missing_memory_capacity_is_reported_or_enforced(self):
        record = self._record(
            "a", 0.0, 1.0, 2.0, 1, memory_capacity=None
        )
        system = self._system(memory_capacity=None)
        reported = self._validate(system, [record], "report")
        strict = self._validate(system, [record], "strict")

        self.assertEqual(
            reported[0]["constraint"],
            "C9_EPR_MEMORY_CAPACITY_UNSPECIFIED",
        )
        self.assertFalse(reported[0]["enforced"])
        self.assertTrue(strict[0]["enforced"])
        self.assertEqual(
            reported[0]["details"]["maximum_observed_stored_epr"], 1
        )

    def test_zero_length_storage_needs_no_memory_calibration(self):
        record = self._record(
            "a", 0.0, 1.0, 1.0, 1, memory_capacity=None
        )

        self.assertEqual(self._validate(
            self._system(memory_capacity=None), [record], "strict"
        ), [])

    def test_actual_event_times_override_planned_memory_intervals(self):
        records = [
            self._record("a", 0.0, 1.0, 2.0, 2, memory_capacity=3),
            self._record(
                "b", 1.0, 2.0, 3.0, 2, memory_capacity=3,
                actual_end=1.5, actual_consume=3.0,
            ),
        ]
        violations = self._validate(self._system(2, 3), records)

        memory = next(
            item for item in violations
            if item["constraint"] == "C9_EPR_MEMORY_CAPACITY"
        )
        self.assertEqual(memory["details"]["stored_epr"], 4)
        self.assertAlmostEqual(memory["details"]["time_ms"], 1.5)

    def test_public_memory_capacity_modes(self):
        self.assertEqual(
            ENTANGLEMENT_MEMORY_CAPACITY_MODES, ("report", "strict")
        )
        self.assertEqual(
            validate_entanglement_memory_capacity_mode(), "report"
        )
        with self.assertRaisesRegex(ValueError, "memory capacity mode"):
            validate_entanglement_memory_capacity_mode("unknown")

    def test_system_rejects_non_positive_or_fractional_memory_capacity(self):
        for value in (0, -1, 1.5, math.inf):
            config = deepcopy(load_system_config("small"))
            config["quantum_links"][0]["memory_capacity_epr"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "memory_capacity_epr"
                ):
                    HybridSystem(
                        config,
                        load_network_topology("fattree_k4"),
                        load_sim_params(),
                    )


if __name__ == "__main__":
    unittest.main()
