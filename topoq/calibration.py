"""IBM calibration loading and mapped-physical calibration aggregation."""

import json
import math
import os
from functools import lru_cache


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IBM_CHIP_DIR = os.path.join(_PROJECT_ROOT, "dataset", "ibm_chips")
_NON_VIRTUAL_1Q_GATES = frozenset(("id", "sx", "x"))


def _finite(value):
    return value is not None and math.isfinite(float(value))


@lru_cache(maxsize=None)
def load_ibm_chip_calibration(chip_type):
    """Load one standardized historical IBM calibration snapshot."""
    path = os.path.join(_IBM_CHIP_DIR, f"{chip_type}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"IBM calibration snapshot not found for {chip_type!r}: {path}"
        )
    with open(path, encoding="utf-8") as source:
        raw = json.load(source)

    units = raw.get("calibration_metadata", {}).get(
        "standardized_units", {}
    )
    expected_units = {
        "t1_us": "us",
        "t2_us": "us",
        "freq_ghz": "GHz",
        "readout_error": "dimensionless",
        "readout_length_ns": "ns",
        "gate_error": "dimensionless",
        "gate_length": "ns",
    }
    wrong_units = {
        key: units.get(key)
        for key, expected in expected_units.items()
        if units.get(key) != expected
    }
    if wrong_units:
        raise ValueError(
            f"IBM calibration {chip_type!r} has unexpected units: "
            f"{wrong_units}"
        )

    qubits = {
        int(item["qubit"]): {
            "t1_us": float(item["t1_us"]),
            "t2_us": float(item["t2_us"]),
            "freq_ghz": float(item["freq_ghz"]),
            "readout_error": float(item["readout_error"]),
            "readout_length_ns": (
                float(item["readout_length_ns"])
                if _finite(item.get("readout_length_ns")) else None
            ),
        }
        for item in raw["qubit_properties"]
    }
    gates = {}
    one_qubit_gates_by_qubit = {}
    cx_edges = set()
    for item in raw["gate_properties"]:
        parameters = item.get("parameters", {})
        length = parameters.get("gate_length")
        error = parameters.get("gate_error")
        if not _finite(length) or not _finite(error):
            continue
        gate = item["gate"]
        qubits_tuple = tuple(map(int, item["qubits"]))
        record = {
            "length_ns": float(length),
            "error": float(error),
        }
        gates[(gate, qubits_tuple)] = record
        if gate in _NON_VIRTUAL_1Q_GATES and len(qubits_tuple) == 1:
            one_qubit_gates_by_qubit.setdefault(
                qubits_tuple[0], []
            ).append(record)
        if gate == "cx" and len(qubits_tuple) == 2:
            cx_edges.add(tuple(sorted(qubits_tuple)))

    return {
        "name": raw["name"],
        "n_qubits": int(raw["n_qubits"]),
        "source": raw.get("source"),
        "last_update_date": raw.get("calibration_metadata", {}).get(
            "last_update_date"
        ),
        "units": dict(units),
        "stats": dict(raw.get("stats", {})),
        "qubits": qubits,
        "gates": gates,
        "one_qubit_gates_by_qubit": one_qubit_gates_by_qubit,
        "cx_edges": tuple(sorted(cx_edges)),
        "profile_cache": {},
        "path": path,
    }


def _configured_profile(qpu, reason="configured_qpu_average"):
    gate_error_rates = qpu.get(
        "gate_error_rates", {"1q": 0.0, "2q": 0.0, "measure": 0.0}
    )
    return {
        "gate_times_ns": {
            key: float(value)
            for key, value in qpu["gate_times_ns"].items()
        },
        "gate_error_rates": {
            key: float(value)
            for key, value in gate_error_rates.items()
        },
        "coherence_time_us": float(qpu.get("coherence_time_us", math.inf)),
        "qubit_coherence_us": {},
        "provenance": {
            "mode": "configured_qpu_average",
            "resolution": reason,
            "chip_type": qpu.get("chip_type"),
            "calibration_source": "system_config",
        },
    }


def _missing_or_fallback(qpu, policy, message):
    if policy["missing_calibration_policy"] == "configured_fallback":
        profile = _configured_profile(qpu, reason="explicit_configured_fallback")
        profile["provenance"]["fallback_reason"] = message
        return profile
    raise ValueError(message)


def _mean(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty IBM calibration selection.")
    return sum(values) / len(values)


def _directional_cx(calibration, left, right, direction_policy):
    records = [
        calibration["gates"].get(("cx", (left, right))),
        calibration["gates"].get(("cx", (right, left))),
    ]
    records = [item for item in records if item is not None]
    if not records:
        return None
    if direction_policy == "conservative_max":
        return {
            "length_ns": max(item["length_ns"] for item in records),
            "error": max(item["error"] for item in records),
        }
    if direction_policy == "mean":
        return {
            "length_ns": _mean(item["length_ns"] for item in records),
            "error": _mean(item["error"] for item in records),
        }
    return min(records, key=lambda item: (item["error"], item["length_ns"]))


def resolve_cx_calibration(qpu, left, right, policy):
    """Return mapped CX length/error for one undirected physical edge."""
    if policy["mode"] == "configured_qpu_average":
        return {
            "length_ns": float(qpu["gate_times_ns"]["2q"]),
            "error": float(qpu.get("gate_error_rates", {}).get("2q", 0.0)),
            "source": "system_config",
        }
    calibration = qpu.get("ibm_calibration")
    if calibration is None:
        fallback = _missing_or_fallback(
            qpu,
            policy,
            f"QPU {qpu['id']} has no IBM calibration snapshot.",
        )
        return {
            "length_ns": fallback["gate_times_ns"]["2q"],
            "error": fallback["gate_error_rates"]["2q"],
            "source": fallback["provenance"]["resolution"],
        }
    record = _directional_cx(
        calibration,
        int(left),
        int(right),
        policy["cx_direction_policy"],
    )
    if record is None:
        fallback = _missing_or_fallback(
            qpu,
            policy,
            f"IBM calibration for {qpu['id']} has no CX data for "
            f"physical edge ({left}, {right}).",
        )
        return {
            "length_ns": fallback["gate_times_ns"]["2q"],
            "error": fallback["gate_error_rates"]["2q"],
            "source": fallback["provenance"]["resolution"],
        }
    return {**record, "source": "ibm_physical_edge"}


def resolve_qpu_calibration_profile(qpu, policy, physical_qubits=None,
                                    two_qubit_pairs=None,
                                    allow_unmapped=False):
    """Aggregate the calibration used by a mapped circuit partition."""
    if policy["mode"] == "configured_qpu_average":
        return _configured_profile(qpu)

    calibration = qpu.get("ibm_calibration")
    if calibration is None:
        return _missing_or_fallback(
            qpu,
            policy,
            f"QPU {qpu['id']} has no IBM calibration snapshot.",
        )

    mode = policy["mode"]
    physical_qubits = sorted(set(map(int, physical_qubits or ())))
    if mode == "ibm_physical_qubit" and not physical_qubits:
        if not allow_unmapped:
            return _missing_or_fallback(
                qpu,
                policy,
                f"IBM physical-qubit mode requires a final physical mapping "
                f"for QPU {qpu['id']}.",
            )
        selected_qubits = sorted(calibration["qubits"])
        resolution = "provisional_chip_average_before_mapping"
    elif mode == "ibm_chip_average":
        selected_qubits = sorted(calibration["qubits"])
        resolution = "ibm_chip_average"
    else:
        missing_qubits = [
            qubit for qubit in physical_qubits
            if qubit not in calibration["qubits"]
        ]
        if missing_qubits:
            return _missing_or_fallback(
                qpu,
                policy,
                f"IBM calibration for {qpu['id']} lacks physical qubits "
                f"{missing_qubits}.",
            )
        selected_qubits = physical_qubits
        resolution = "ibm_mapped_physical_qubits"

    pair_inputs = list(two_qubit_pairs or ())
    normalized_pair_inputs = []
    for item in pair_inputs:
        if len(item) == 2 and isinstance(item[1], (int, float)):
            pair, weight = item
        else:
            pair, weight = item, 1
        normalized_pair_inputs.append((
            tuple(sorted(map(int, pair))), max(float(weight), 0.0)
        ))
    normalized_pair_inputs.sort()
    cache_key = (
        mode,
        tuple(selected_qubits),
        tuple(normalized_pair_inputs),
        policy["cx_direction_policy"],
        resolution,
    )
    cached = calibration["profile_cache"].get(cache_key)
    if cached is not None:
        return cached

    one_qubit_records = [
        record
        for qubit in selected_qubits
        for record in calibration["one_qubit_gates_by_qubit"].get(
            qubit, ()
        )
    ]
    if not one_qubit_records:
        return _missing_or_fallback(
            qpu,
            policy,
            f"IBM calibration for {qpu['id']} has no non-virtual 1Q data "
            f"for selected qubits {selected_qubits}.",
        )

    pair_weights = []
    for pair, weight in normalized_pair_inputs:
        left, right = pair
        record = _directional_cx(
            calibration, left, right, policy["cx_direction_policy"]
        )
        if record is None:
            return _missing_or_fallback(
                qpu,
                policy,
                f"IBM calibration for {qpu['id']} has no CX data for "
                f"physical edge ({left}, {right}).",
            )
        pair_weights.append((record, max(float(weight), 0.0)))

    if not pair_weights:
        selected = set(selected_qubits)
        undirected = set()
        for left, right in calibration["cx_edges"]:
            if mode == "ibm_chip_average" or left in selected or right in selected:
                undirected.add(tuple(sorted((left, right))))
        for left, right in sorted(undirected):
            record = _directional_cx(
                calibration, left, right, policy["cx_direction_policy"]
            )
            if record is not None:
                pair_weights.append((record, 1.0))
        pair_resolution = "adjacent_edge_proxy"
    else:
        pair_resolution = "mapped_interaction_edges"

    if not pair_weights:
        return _missing_or_fallback(
            qpu,
            policy,
            f"IBM calibration for {qpu['id']} has no usable CX records.",
        )
    total_pair_weight = sum(weight for _, weight in pair_weights)
    if total_pair_weight <= 0.0:
        total_pair_weight = float(len(pair_weights))
        pair_weights = [(record, 1.0) for record, _ in pair_weights]

    qubit_records = [calibration["qubits"][qubit] for qubit in selected_qubits]
    measurement_lengths = [
        item["readout_length_ns"] for item in qubit_records
        if item["readout_length_ns"] is not None
    ]
    if not measurement_lengths:
        return _missing_or_fallback(
            qpu,
            policy,
            f"IBM calibration for {qpu['id']} has no readout lengths for "
            f"selected qubits {selected_qubits}.",
        )

    coherence_by_qubit = {
        qubit: min(
            calibration["qubits"][qubit]["t1_us"],
            calibration["qubits"][qubit]["t2_us"],
        )
        for qubit in selected_qubits
    }
    result = {
        "gate_times_ns": {
            "1q": _mean(item["length_ns"] for item in one_qubit_records),
            "2q": sum(
                item["length_ns"] * weight for item, weight in pair_weights
            ) / total_pair_weight,
            # Measurements are assumed parallel across mapped qubits.
            "measure": max(measurement_lengths),
        },
        "gate_error_rates": {
            "1q": _mean(item["error"] for item in one_qubit_records),
            "2q": sum(
                item["error"] * weight for item, weight in pair_weights
            ) / total_pair_weight,
            "measure": _mean(item["readout_error"] for item in qubit_records),
        },
        "coherence_time_us": min(coherence_by_qubit.values()),
        "qubit_coherence_us": coherence_by_qubit,
        "provenance": {
            "mode": mode,
            "resolution": resolution,
            "two_qubit_resolution": pair_resolution,
            "one_qubit_aggregation": "nonvirtual_id_sx_x_mean",
            "measurement_aggregation": "parallel_max_length_mean_error",
            "cx_direction_policy": policy["cx_direction_policy"],
            "chip_type": qpu.get("chip_type"),
            "calibration_source": calibration["source"],
            "calibration_last_update": calibration["last_update_date"],
            "selected_physical_qubits": selected_qubits,
        },
    }
    calibration["profile_cache"][cache_key] = result
    return result
