"""Shared interval-capacity helpers for quantum resource scheduling."""


TIME_TOLERANCE_MS = 1e-12


def _overlaps(left_start, left_end, right_start, right_end):
    return (
        left_start < right_end - TIME_TOLERANCE_MS
        and right_start < left_end - TIME_TOLERANCE_MS
    )


def _conflicting_ends(reservations, capacity, requirement, start, end):
    """Return reservation end times when one QPU exceeds capacity."""
    active = 0
    events = []
    relevant = []
    for reservation in reservations:
        if not _overlaps(
                start, end, reservation["start"], reservation["end"]):
            continue
        relevant.append(reservation)
        qubits = reservation["qubits"]
        if reservation["start"] <= start + TIME_TOLERANCE_MS:
            active += qubits
        else:
            events.append((reservation["start"], 1, qubits))
        if reservation["end"] < end - TIME_TOLERANCE_MS:
            events.append((reservation["end"], 0, -qubits))

    if active + requirement > capacity:
        return [item["end"] for item in relevant if item["end"] > start]

    for _, _, delta in sorted(events):
        active += delta
        if active + requirement > capacity:
            return [item["end"] for item in relevant if item["end"] > start]
    return []


def overlapping_quantum_reservations(reservations, qpu_ids, start, end):
    """Return reservations overlapping one half-open interval by QPU."""
    return {
        qid: [
            reservation
            for reservation in reservations.get(qid, [])
            if _overlaps(
                start,
                end,
                reservation["start"],
                reservation["end"],
            )
        ]
        for qid in qpu_ids
    }


def physical_qubits_by_qpu(logical_partition, physical_mapping):
    """Group one task's mapped physical qubits by partition QPU."""
    grouped = {}
    for logical, qid in logical_partition.items():
        if logical not in physical_mapping:
            continue
        grouped.setdefault(qid, set()).add(physical_mapping[logical])
    return grouped


def find_quantum_capacity_start(reservations, capacities, requirements,
                                earliest_start, duration,
                                physical_requirements=None):
    """Find the earliest interval satisfying count and physical C2 capacity."""
    requirements = {
        qid: int(qubits) for qid, qubits in requirements.items()
        if int(qubits) > 0
    }
    physical_requirements = {
        qid: set(physical)
        for qid, physical in (physical_requirements or {}).items()
    }
    for qid, qubits in requirements.items():
        if qid not in capacities:
            raise ValueError(f"Unknown QPU in quantum requirements: {qid}")
        if qubits > capacities[qid]:
            raise ValueError(
                f"Quantum requirement {qubits} exceeds {qid} capacity "
                f"{capacities[qid]}"
            )

    candidate = max(float(earliest_start), 0.0)
    duration = max(float(duration), 0.0)
    if duration <= TIME_TOLERANCE_MS or not requirements:
        return candidate, []

    while True:
        candidate_end = candidate + duration
        conflicting_ends = []
        for qid, qubits in requirements.items():
            conflicting_ends.extend(_conflicting_ends(
                reservations.get(qid, []), capacities[qid], qubits,
                candidate, candidate_end,
            ))
            required_physical = physical_requirements.get(qid, set())
            if required_physical:
                conflicting_ends.extend(
                    reservation["end"]
                    for reservation in reservations.get(qid, [])
                    if _overlaps(
                        candidate,
                        candidate_end,
                        reservation["start"],
                        reservation["end"],
                    )
                    and required_physical.intersection(
                        reservation.get("physical_qubits", [])
                    )
                )
        if not conflicting_ends:
            break
        next_candidate = min(conflicting_ends)
        if next_candidate <= candidate + TIME_TOLERANCE_MS:
            raise ValueError("Quantum capacity search did not make progress")
        candidate = next_candidate

    candidate_end = candidate + duration
    shared_with = {
        reservation["task"]
        for qid in requirements
        for reservation in reservations.get(qid, [])
        if _overlaps(
            candidate, candidate_end,
            reservation["start"], reservation["end"],
        )
    }
    return candidate, sorted(shared_with, key=str)


def reserve_quantum_capacity(reservations, requirements, start, end, task_id,
                             physical_qubits=None):
    """Record one task's per-QPU qubit occupancy interval."""
    physical_qubits = physical_qubits or {}
    for qid, qubits in requirements.items():
        if qubits <= 0:
            continue
        reservations.setdefault(qid, []).append({
            "task": task_id,
            "start": start,
            "end": end,
            "qubits": int(qubits),
            "physical_qubits": sorted(
                set(physical_qubits.get(qid, []))
            ),
        })
