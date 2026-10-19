"""Shared task execution-time models for TopoQ and all baselines.

``legacy_proxy`` preserves the historical public-DAG timing fields.
``paper_equation`` evaluates the revised manuscript estimators on the
actually assigned QPU or classical node. Communication, EPR waiting, and
physical-routing overhead remain separate schedule components.
"""

import math
from collections import defaultdict

from .calibration import resolve_qpu_calibration_profile
from .config import (
    validate_execution_time_policy,
    validate_qpu_calibration_policy,
)


def resolve_task_gate_profile(task):
    """Return the gate/depth profile used by the revised-paper estimator."""
    n_gates = max(int(task.get("n_gates", 0)), 0)
    fallback_1q = int(n_gates * 0.67)
    n_1q = max(int(task.get("paper_n_1q_gates", fallback_1q)), 0)
    n_2q = max(int(task.get(
        "paper_n_2q_gates", max(n_gates - fallback_1q, 0)
    )), 0)
    depth = max(int(task.get("paper_depth", task.get("depth", 1))), 1)
    measurement_count = max(int(task.get("paper_measurement_count", 0)), 0)
    return {
        "n_1q_gates": n_1q,
        "n_2q_gates": n_2q,
        "depth": depth,
        "measurement_count": measurement_count,
    }


def paper_quantum_duration_ms(task, qpu, quantum_shots=1000,
                              gate_times_ns=None):
    """Evaluate N_shots * (depth * tau_bar + measurement latency)."""
    profile = resolve_task_gate_profile(task)
    gate_times = gate_times_ns or qpu.get(
        "gate_times_ns",
        {"1q": 35.6, "2q": 334.0, "measure": 1200.0},
    )
    n_1q = profile["n_1q_gates"]
    n_2q = profile["n_2q_gates"]
    counted_gates = n_1q + n_2q
    if counted_gates:
        average_gate_ns = (
            n_1q * float(gate_times["1q"])
            + n_2q * float(gate_times["2q"])
        ) / counted_gates
        circuit_ns = profile["depth"] * average_gate_ns
    else:
        average_gate_ns = 0.0
        circuit_ns = 0.0

    measurement_ns = (
        float(gate_times["measure"])
        if profile["measurement_count"] > 0 else 0.0
    )
    repetitions = int(quantum_shots) if task.get("shot_scaled", False) else 1
    duration_ms = repetitions * (circuit_ns + measurement_ns) / 1e6
    if not math.isfinite(duration_ms) or duration_ms < 0.0:
        raise ValueError("Quantum task duration must be finite and non-negative.")
    return duration_ms


def mapped_two_qubit_pairs_by_qpu(interaction_graph, logical_partition,
                                  physical_mapping, physical_routing=None):
    """Return calibrated physical CX-edge weights for mapped interactions."""
    weighted = defaultdict(lambda: defaultdict(float))
    if not interaction_graph or not logical_partition or not physical_mapping:
        return {}

    routing = physical_routing or {}
    sequence_operations = routing.get("routed_operations", [])
    if sequence_operations:
        for item in sequence_operations:
            pair = tuple(map(int, item["physical_pair_after"]))
            weighted[item["qpu"]][tuple(sorted(pair))] += 1.0
        return {
            qid: [(pair, weight) for pair, weight in sorted(items.items())]
            for qid, items in weighted.items()
        }

    routed_paths = {
        tuple(item["logical_pair"]): item.get("path", [])
        for item in routing.get("routed_edges", [])
    }
    routed_paths.update({
        tuple(reversed(item["logical_pair"])): item.get("path", [])
        for item in routing.get("routed_edges", [])
    })
    for logical_pair, edge in interaction_graph.get("edges", {}).items():
        left, right = logical_pair
        qid = logical_partition[left]
        if logical_partition[right] != qid:
            continue
        path = routed_paths.get(tuple(logical_pair))
        if path and len(path) >= 2:
            physical_pair = (path[-2], path[-1])
        else:
            physical_pair = (
                physical_mapping[left], physical_mapping[right]
            )
        weighted[qid][tuple(sorted(map(int, physical_pair)))] += max(
            float(edge.get("two_qubit_gates", 1)), 0.0
        )
    return {
        qid: [(pair, weight) for pair, weight in sorted(items.items())]
        for qid, items in weighted.items()
    }


def quantum_task_timing_details(task, system, qpu_ids, params=None,
                                routing_overhead_ms=0.0,
                                logical_partition=None,
                                physical_mapping=None,
                                interaction_graph=None,
                                physical_routing=None):
    """Return mapped quantum duration and calibration provenance."""
    params = params or {}
    policy = validate_execution_time_policy(
        params.get("execution_time_mode"),
        params.get("quantum_shots"),
        params.get("classical_compute_efficiency"),
    )
    routing_overhead_ms = max(float(routing_overhead_ms), 0.0)
    if policy["mode"] == "legacy_proxy":
        base_duration = max(float(task.get("estimated_time_ms", 10.0)), 0.0)
        return {
            "duration_ms": base_duration + routing_overhead_ms,
            "base_duration_ms": base_duration,
            "routing_overhead_ms": routing_overhead_ms,
            "qpu_partitions": {},
            "execution_time_mode": policy["mode"],
        }

    selected = [qid for qid in dict.fromkeys(qpu_ids or ()) if qid in system.qpus]
    if not selected:
        raise ValueError("Paper-equation quantum timing requires an assigned QPU.")
    calibration_policy = validate_qpu_calibration_policy(
        params.get("qpu_calibration_mode"),
        params.get("ibm_cx_direction_policy"),
        params.get("ibm_missing_calibration_policy"),
    )
    physical_by_qpu = defaultdict(list)
    logical_count_by_qpu = defaultdict(int)
    if logical_partition and physical_mapping:
        for logical, qid in logical_partition.items():
            logical_count_by_qpu[qid] += 1
            if logical in physical_mapping:
                physical_by_qpu[qid].append(physical_mapping[logical])
    pair_weights = mapped_two_qubit_pairs_by_qpu(
        interaction_graph,
        logical_partition,
        physical_mapping,
        physical_routing,
    )
    swap_gates_by_qpu = defaultdict(int)
    for item in (physical_routing or {}).get("routed_edges", ()):
        swap_gates_by_qpu[item["qpu"]] += int(
            item.get("swap_gate_count", 0)
        )
    for item in (physical_routing or {}).get("routed_operations", ()):
        swap_gates_by_qpu[item["qpu"]] += int(
            item.get("swap_gate_count", 0)
        )
    qpu_partitions = {}
    for qid in selected:
        calibration = resolve_qpu_calibration_profile(
            system.qpus[qid],
            calibration_policy,
            physical_qubits=physical_by_qpu.get(qid),
            two_qubit_pairs=pair_weights.get(qid),
            allow_unmapped=True,
        )
        duration = paper_quantum_duration_ms(
            task,
            system.qpus[qid],
            policy["quantum_shots"],
            gate_times_ns=calibration["gate_times_ns"],
        )
        qpu_partitions[qid] = {
            "duration_ms": duration,
            "gate_times_ns": calibration["gate_times_ns"],
            "gate_error_rates": calibration["gate_error_rates"],
            "coherence_time_us": calibration["coherence_time_us"],
            "qubit_coherence_us": calibration["qubit_coherence_us"],
            "logical_qubit_count": logical_count_by_qpu.get(qid, 0),
            "mapped_two_qubit_weight": sum(
                weight for _, weight in pair_weights.get(qid, ())
            ),
            "swap_gate_count": swap_gates_by_qpu.get(qid, 0),
            "calibration_provenance": calibration["provenance"],
        }
    base_duration = max(
        item["duration_ms"] for item in qpu_partitions.values()
    )
    return {
        "duration_ms": base_duration + routing_overhead_ms,
        "base_duration_ms": base_duration,
        "routing_overhead_ms": routing_overhead_ms,
        "qpu_partitions": qpu_partitions,
        "execution_time_mode": policy["mode"],
        "qpu_calibration_policy": calibration_policy,
    }


def quantum_task_duration_ms(task, system, qpu_ids, params=None,
                             routing_overhead_ms=0.0,
                             logical_partition=None,
                             physical_mapping=None,
                             interaction_graph=None,
                             physical_routing=None):
    """Return quantum execution time on the selected QPU set."""
    return quantum_task_timing_details(
        task,
        system,
        qpu_ids,
        params,
        routing_overhead_ms=routing_overhead_ms,
        logical_partition=logical_partition,
        physical_mapping=physical_mapping,
        interaction_graph=interaction_graph,
        physical_routing=physical_routing,
    )["duration_ms"]


def classical_task_duration_ms(task, node, params=None):
    """Return historical proxy time or revised-paper w/(kappa p) time."""
    params = params or {}
    policy = validate_execution_time_policy(
        params.get("execution_time_mode"),
        params.get("quantum_shots"),
        params.get("classical_compute_efficiency"),
    )
    speed = max(float(node["compute_speed_tflops"]), 1e-12)
    if policy["mode"] == "legacy_proxy":
        if task.get("estimated_time_ms") is not None:
            return max(float(task["estimated_time_ms"]), 0.0) / max(speed, 0.5)
        flops = max(float(task.get("workload_flops", 1e9)), 0.0)
        return flops / (speed * 1e12) * 1000.0

    flops = task.get("workload_flops")
    if flops is None:
        raise ValueError(
            f"Paper-equation classical timing requires workload_flops for "
            f"task {task.get('id', '<unknown>')}."
        )
    flops = max(float(flops), 0.0)
    return (
        flops
        / (policy["classical_compute_efficiency"] * speed * 1e12)
        * 1000.0
    )


def reference_task_duration_ms(task, system, params=None):
    """Return a deterministic resource-aware reference for ranking/load only."""
    if task.get("type") == "quantum":
        qpu_ids = sorted(system.qpus)
        return min(
            quantum_task_duration_ms(task, system, [qid], params)
            for qid in qpu_ids
        )
    node_ids = sorted(system.nodes)
    return min(
        classical_task_duration_ms(task, system.nodes[nid], params)
        for nid in node_ids
    )
