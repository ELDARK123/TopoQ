"""Constraint checks used by the TopoQ feedback loop.

Each validator corresponds to a constraint in the paper and returns
structured violation records.  The module is intentionally independent of
``TopoQScheduler`` so constraint logic is testable without running a complete
experiment; baseline schedulers do not invoke these TopoQ-specific checks.
"""

import math

from .config import (
    resolve_coherence_event_mode,
    validate_entanglement_memory_capacity_mode,
)
from .fidelity import collect_task_epr_events
from .resources import physical_qubits_by_qpu


TIME_TOLERANCE_MS = 1e-9


def _violation(constraint, task, reason, resource=None, enforced=True, **details):
    """Create one consistently shaped constraint violation record."""
    return {
        "constraint": constraint,
        "task": task,
        "resource": resource,
        "reason": reason,
        "enforced": enforced,
        "details": details,
    }


def validate_task_assignment(system, workload, sched, assignment, mapping):
    """Validate C1: every task is scheduled on a resource of its type."""
    violations = []
    for tid, task in workload.tasks.items():
        if tid not in sched:
            violations.append(_violation(
                "C1_TASK_ASSIGNMENT",
                tid,
                "Task is missing from the schedule.",
            ))
            continue

        resource = assignment.get(tid)
        valid_resources = system.qpus if task["type"] == "quantum" else system.nodes
        if resource not in valid_resources:
            violations.append(_violation(
                "C1_TASK_ASSIGNMENT",
                tid,
                f"{task['type'].capitalize()} task is assigned to an invalid resource.",
                resource=resource,
                task_type=task["type"],
                assigned_resource=resource,
            ))
            continue

        schedule_field = "qpu" if task["type"] == "quantum" else "node"
        scheduled_resource = sched[tid].get(schedule_field)
        if scheduled_resource != resource:
            violations.append(_violation(
                "C1_TASK_ASSIGNMENT",
                tid,
                "Schedule resource does not match the assignment.",
                resource=resource,
                assignment_resource=resource,
                schedule_resource=scheduled_resource,
            ))
    return violations


def validate_qpu_capacity(system, workload, sched, assignment, mapping):
    """Validate C2: physical mappings fit and cover the logical qubits."""
    violations = []
    for tid in workload.quantum_tasks:
        qid = assignment.get(tid)
        if qid not in system.qpus:
            continue  # C1 reports the invalid assignment.

        task = workload.tasks[tid]
        required = task.get("n_qubits", 0)
        capacity = system.qpus[qid]["n_qubits"]
        task_mapping = mapping.get(tid, {})
        logical_partition = task_mapping.get("logical_partition", {})
        physical_mapping = task_mapping.get("physical_mapping", {})

        if len(logical_partition) != required or len(physical_mapping) != required:
            violations.append(_violation(
                "C2_QPU_CAPACITY",
                tid,
                "Partition and physical mapping must cover every logical qubit.",
                resource=qid,
                required_qubits=required,
                partitioned_qubits=len(logical_partition),
                mapped_qubits=len(physical_mapping),
            ))

        partition_counts = {}
        for logical, part_qid in logical_partition.items():
            partition_counts[part_qid] = partition_counts.get(part_qid, 0) + 1
        for part_qid, count in partition_counts.items():
            if part_qid not in system.qpus:
                violations.append(_violation(
                    "C2_QPU_CAPACITY", tid, "Partition references an unknown QPU.",
                    resource=part_qid,
                ))
                continue
            part_capacity = system.qpus[part_qid]["n_qubits"]
            if count > part_capacity:
                violations.append(_violation(
                    "C2_QPU_CAPACITY", tid, "Logical partition exceeds QPU capacity.",
                    resource=part_qid, partition_size=count, qpu_capacity=part_capacity,
                ))

            physical_qubits = [physical_mapping[logical]
                               for logical, owner in logical_partition.items()
                               if owner == part_qid and logical in physical_mapping]
            if len(set(physical_qubits)) != len(physical_qubits):
                violations.append(_violation(
                    "C2_QPU_CAPACITY", tid,
                    "Multiple logical qubits on one QPU map to the same physical qubit.",
                    resource=part_qid, physical_qubits=physical_qubits,
                ))
            invalid = [physical for physical in physical_qubits
                       if not isinstance(physical, int)
                       or physical < 0 or physical >= part_capacity]
            if invalid:
                violations.append(_violation(
                    "C2_QPU_CAPACITY", tid,
                    "Physical mapping contains an out-of-range qubit index.",
                    resource=part_qid, invalid_physical_qubits=invalid,
                    qpu_capacity=part_capacity,
                ))
    return violations


def validate_concurrent_qpu_capacity(system, workload, sched, assignment,
                                     mapping):
    """Validate C2 at every time point for concurrent quantum tasks."""
    violations = []
    reservations = {}
    for tid in workload.quantum_tasks:
        if tid not in sched:
            continue
        item = sched[tid]
        start, end = item["start"], item["end"]
        if end <= start + TIME_TOLERANCE_MS:
            continue
        partition_sizes = item.get("partition_sizes")
        if not partition_sizes:
            qid = assignment.get(tid)
            partition_sizes = {qid: workload.tasks[tid].get("n_qubits", 0)}
        exclusive_qpus = item.get("space_sharing_exclusive_qpus")
        if exclusive_qpus is None:
            exclusive_qpus = (
                list(partition_sizes)
                if item.get("space_sharing_exclusive_routing") else []
            )
        partition_sizes = {
            qid: (
                system.qpus[qid]["n_qubits"]
                if qid in exclusive_qpus and qid in system.qpus else qubits
            )
            for qid, qubits in partition_sizes.items()
        }
        for qid, qubits in partition_sizes.items():
            if qid not in system.qpus:
                continue  # C1/C2 mapping checks report unknown resources.
            reservations.setdefault(qid, []).extend([
                (start, 1, qubits, tid),
                (end, 0, -qubits, tid),
            ])

    for qid, events in reservations.items():
        capacity = system.qpus[qid]["n_qubits"]
        active = {}
        used = 0
        reported_at = set()
        for time_ms, event_type, delta, tid in sorted(events):
            used += delta
            if event_type == 0:
                active.pop(tid, None)
            else:
                active[tid] = delta
                if used > capacity and time_ms not in reported_at:
                    reported_at.add(time_ms)
                    violations.append(_violation(
                        "C2_QPU_CONCURRENT_CAPACITY",
                        tid,
                        "Concurrent quantum tasks exceed QPU qubit capacity.",
                        resource=qid,
                        time_ms=time_ms,
                        used_qubits=used,
                        capacity_qubits=capacity,
                        active_tasks=sorted(active),
                    ))
    return violations


def validate_concurrent_physical_qubits(system, workload, sched, assignment,
                                        mapping):
    """Reject overlapping physical-qubit mappings on concurrent tasks."""
    violations = []
    events_by_qpu = {}
    for tid in workload.quantum_tasks:
        if tid not in sched:
            continue
        item = sched[tid]
        start, end = item["start"], item["end"]
        if end <= start + TIME_TOLERANCE_MS:
            continue
        grouped = item.get("physical_qubits_by_qpu")
        if grouped is None:
            task_mapping = mapping.get(tid, {})
            grouped = physical_qubits_by_qpu(
                task_mapping.get("logical_partition", {}),
                task_mapping.get("physical_mapping", {}),
            )
        for qid, physical in grouped.items():
            if qid not in system.qpus or not physical:
                continue
            physical = set(physical)
            events_by_qpu.setdefault(qid, []).extend([
                (start, 1, tid, physical),
                (end, 0, tid, physical),
            ])

    for qid, events in events_by_qpu.items():
        active = {}
        for time_ms, event_type, tid, physical in sorted(
                events, key=lambda event: (event[0], event[1], str(event[2]))):
            if event_type == 0:
                active.pop(tid, None)
                continue
            for conflicting_task, occupied in active.items():
                conflict = sorted(physical.intersection(occupied))
                if not conflict:
                    continue
                violations.append(_violation(
                    "C2_QPU_PHYSICAL_CONFLICT",
                    tid,
                    "Concurrent quantum tasks use the same physical qubit.",
                    resource=qid,
                    time_ms=time_ms,
                    conflicting_task=conflicting_task,
                    physical_qubits=conflict,
                    active_tasks=sorted(
                        list(active) + [tid], key=str
                    ),
                ))
            active[tid] = physical
    return violations


def validate_physical_mapping(system, workload, sched, assignment, mapping):
    """Validate C10: mapped physical qubits are present and connected."""
    violations = []
    for tid in workload.quantum_tasks:
        task_mapping = mapping.get(tid, {})
        logical_partition = task_mapping.get("logical_partition", {})
        physical_mapping = task_mapping.get("physical_mapping", {})
        for qid in set(logical_partition.values()):
            if qid not in system.qpus:
                continue
            topology = system.qpus[qid].get("topology", [])
            adjacency = {}
            for left, right in topology:
                adjacency.setdefault(left, set()).add(right)
                adjacency.setdefault(right, set()).add(left)
            mapped = [physical_mapping[logical]
                      for logical, owner in logical_partition.items()
                      if owner == qid and logical in physical_mapping]
            if not mapped:
                continue
            reachable = {mapped[0]}
            frontier = [mapped[0]]
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency.get(current, ()):
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        frontier.append(neighbor)
            disconnected = [physical for physical in mapped if physical not in reachable]
            if disconnected:
                violations.append(_violation(
                    "C10_PHYSICAL_COUPLING", tid,
                    "Mapped physical qubits are disconnected in the coupling graph.",
                    resource=qid, disconnected_physical_qubits=disconnected,
                ))
    return violations


def validate_physical_routing(system, workload, sched, assignment, mapping):
    """Validate C10 routing metadata for local two-qubit interactions."""
    violations = []
    for tid in workload.quantum_tasks:
        if tid not in sched:
            continue
        routing = mapping.get(tid, {}).get("physical_routing", {})
        routing_mode = routing.get("routing_mode", "static_aggregate")
        if routing_mode not in {"static_aggregate", "sequence_exact"}:
            violations.append(_violation(
                "C10_PHYSICAL_ROUTING", tid,
                "Physical routing metadata contains an unknown effective mode.",
                routing_mode=routing_mode,
            ))
            continue
        for edge in routing.get("unroutable_edges", []):
            violations.append(_violation(
                "C10_PHYSICAL_ROUTING",
                tid,
                "A local two-qubit interaction has no physical coupling path.",
                resource=edge.get("qpu"),
                logical_pair=edge.get("logical_pair"),
                physical_pair=edge.get("physical_pair"),
                two_qubit_gates=edge.get("two_qubit_gates"),
            ))

        if routing_mode == "sequence_exact":
            for operation in routing.get("routed_operations", []):
                qid = operation.get("qpu")
                topology = system.qpus.get(qid, {}).get("topology", [])
                direct_edges = {frozenset(edge) for edge in topology}
                for swap in operation.get("swaps", []):
                    if frozenset(swap) not in direct_edges:
                        violations.append(_violation(
                            "C10_PHYSICAL_ROUTING", tid,
                            "A dynamic SWAP step is not an edge of the physical coupling graph.",
                            resource=qid,
                            operation_index=operation.get("operation_index"),
                            swap=swap,
                        ))
                final_pair = operation.get("physical_pair_after")
                if final_pair and frozenset(final_pair) not in direct_edges:
                    violations.append(_violation(
                        "C10_PHYSICAL_ROUTING", tid,
                        "A dynamically routed two-qubit gate is not adjacent after SWAPs.",
                        resource=qid,
                        operation_index=operation.get("operation_index"),
                        logical_pair=operation.get("logical_pair"),
                        physical_pair_after=final_pair,
                    ))

            final_mapping = routing.get("final_physical_mapping", {})
            stored_mapping = mapping.get(tid, {}).get("physical_mapping", {})
            if final_mapping != stored_mapping:
                violations.append(_violation(
                    "C10_PHYSICAL_ROUTING", tid,
                    "Stored physical mapping does not match the dynamic routing final layout.",
                    final_physical_mapping=final_mapping,
                    stored_physical_mapping=stored_mapping,
                ))
            logical_partition = mapping.get(tid, {}).get("logical_partition", {})
            for qid in set(logical_partition.values()):
                physical = [
                    final_mapping[logical]
                    for logical, owner in logical_partition.items()
                    if owner == qid and logical in final_mapping
                ]
                if len(physical) != len(set(physical)):
                    violations.append(_violation(
                        "C10_PHYSICAL_ROUTING", tid,
                        "Dynamic routing final layout is not one-to-one.",
                        resource=qid,
                        physical_qubits=physical,
                    ))
            continue

        routed_edges = routing.get("routed_edges", [])
        for edge in routed_edges:
            if edge.get("distance", 0) <= 1 or edge.get("swap_gate_count", 0) <= 0:
                violations.append(_violation(
                    "C10_PHYSICAL_ROUTING",
                    tid,
                    "A non-adjacent local interaction lacks positive SWAP routing overhead.",
                    resource=edge.get("qpu"),
                    logical_pair=edge.get("logical_pair"),
                    physical_pair=edge.get("physical_pair"),
                    distance=edge.get("distance"),
                    swap_gate_count=edge.get("swap_gate_count"),
                ))
    return violations


def validate_fidelity_threshold(system, workload, sched, assignment, mapping,
                                params=None):
    """Validate/report C4 using the configured fidelity policy."""
    params = params or {}
    mode = params.get("fidelity_constraint_mode", "report_only")
    if mode not in {"report_only", "paper_strict", "hardware_explicit"}:
        raise ValueError(
            "fidelity_constraint_mode must be report_only, paper_strict, "
            "or hardware_explicit"
        )
    fidelity_params = system.sim_params["fidelity_parameters"]
    if mode == "hardware_explicit":
        threshold = params.get("fidelity_threshold")
        if threshold is None:
            raise ValueError(
                "hardware_explicit fidelity mode requires fidelity_threshold."
            )
    else:
        threshold = (
            fidelity_params.get("F_min_qaoa", fidelity_params["F_min"])
            if "qaoa" in workload.id.lower()
            else fidelity_params["F_min"]
        )
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("fidelity threshold must be in the interval (0, 1].")

    from .fidelity import estimate_task_log_fidelity

    log_threshold = math.log(threshold)
    violations = []
    enforced = mode != "report_only"
    for tid in workload.quantum_tasks:
        if tid not in sched or assignment.get(tid) not in system.qpus:
            continue
        estimate = estimate_task_log_fidelity(
            system,
            workload.tasks[tid],
            system.qpus[assignment[tid]],
            schedule_item=sched[tid],
        )
        if estimate["log_fidelity"] + TIME_TOLERANCE_MS < log_threshold:
            violations.append(_violation(
                "C4_FIDELITY_THRESHOLD",
                tid,
                "Estimated task fidelity is below the configured threshold.",
                resource=assignment.get(tid),
                enforced=enforced,
                estimated_fidelity=estimate["fidelity"],
                log_fidelity=estimate["log_fidelity"],
                threshold=threshold,
                log_threshold=log_threshold,
                mode=mode,
                n_remote_gates=estimate["n_remote_gates"],
                n_swap_gates=sched[tid].get("n_swap_gates", 0),
            ))
    return violations


def validate_classical_bandwidth(system, workload, sched, assignment, mapping,
                                 classical_link_timeline=None):
    """Validate C8 for recorded classical-link reservations."""
    violations = []
    for link_key, reservations in (classical_link_timeline or {}).items():
        capacity = system.net_edge_info.get(link_key, {}).get("bw")
        if capacity is None:
            continue
        events = []
        for index, reservation in enumerate(reservations):
            bandwidth = float(reservation["bandwidth_gbps"])
            events.extend([
                (
                    float(reservation["start"]),
                    1,
                    index,
                    bandwidth,
                    reservation,
                ),
                (
                    float(reservation["end"]),
                    0,
                    index,
                    -bandwidth,
                    reservation,
                ),
            ])
        active = {}
        used_bandwidth = 0.0
        for time_ms, event_type, index, delta, reservation in sorted(
                events, key=lambda item: (item[0], item[1], item[2])):
            if event_type == 0:
                if index in active:
                    used_bandwidth += delta
                    active.pop(index, None)
                continue
            active[index] = reservation
            used_bandwidth += delta
            if used_bandwidth > capacity + 1e-12:
                violations.append(_violation(
                    "C8_CLASSICAL_BANDWIDTH", reservation["task"],
                    "Concurrent reservations exceed classical link capacity.",
                    resource=sorted(link_key, key=str),
                    capacity_gbps=capacity,
                    used_bandwidth_gbps=used_bandwidth,
                    time_ms=time_ms,
                    active_reservations=list(active.values()),
                ))
    return violations


def validate_classical_memory(system, workload, sched, assignment, mapping):
    """Validate C7 for concurrently running classical tasks."""
    violations = []
    reservations = {}
    for tid in workload.classical_tasks:
        if tid not in sched:
            continue
        node_id = assignment.get(tid)
        if node_id not in system.nodes:
            continue  # C1 reports invalid resources.
        start, end = sched[tid]["start"], sched[tid]["end"]
        if end <= start + TIME_TOLERANCE_MS:
            continue
        memory_mb = max(workload.tasks[tid].get("memory_mb", 0), 0)
        reservations.setdefault(node_id, []).extend([
            (start, 1, memory_mb, tid),
            (end, 0, -memory_mb, tid),
        ])

    for node_id, events in reservations.items():
        capacity = system.nodes[node_id]["memory_gb"] * 1024
        active = {}
        used = 0
        reported_at = set()
        for time_ms, event_type, delta, tid in sorted(events):
            used += delta
            if event_type == 0:
                active.pop(tid, None)
            else:
                active[tid] = delta
                if used > capacity and time_ms not in reported_at:
                    reported_at.add(time_ms)
                    violations.append(_violation(
                        "C7_CLASSICAL_MEMORY",
                        tid,
                        "Concurrent classical tasks exceed node memory capacity.",
                        resource=node_id,
                        time_ms=time_ms,
                        used_memory_mb=used,
                        capacity_memory_mb=capacity,
                        active_tasks=sorted(active),
                    ))
    return violations


def validate_dag_precedence(system, workload, sched, assignment, mapping):
    """Validate C3 including the communication term in paper Eq. (14)."""
    violations = []
    for (pred, tid), edge in workload.edge_data.items():
        if pred not in sched or tid not in sched:
            continue  # Missing tasks are reported by C1.
        pred_end = sched[pred]["end"]
        task_start = sched[tid]["start"]
        transfers = [
            transfer
            for transfer in sched[tid].get("dependency_transfers", [])
            if transfer.get("predecessor") == pred
        ]
        if len(transfers) != 1:
            violations.append(_violation(
                "C3_DAG_PRECEDENCE",
                tid,
                "Dependency edge does not have exactly one transfer record.",
                resource=assignment.get(tid),
                predecessor=pred,
                violation_kind=(
                    "missing_transfer" if not transfers
                    else "duplicate_transfer"
                ),
                transfer_count=len(transfers),
            ))
            continue

        transfer = transfers[0]
        pred_type = workload.tasks[pred]["type"]
        task_type = workload.tasks[tid]["type"]
        source_resource = assignment.get(pred)
        target_resource = assignment.get(tid)
        if pred_type == "quantum" and task_type == "quantum":
            expected_type = (
                "quantum-quantum"
                if source_resource != target_resource else None
            )
        elif pred_type == "classical" and task_type == "classical":
            expected_type = (
                "classical-classical"
                if source_resource != target_resource else None
            )
        else:
            expected_type = f"{pred_type}-{task_type}"

        if (transfer.get("type") != expected_type
                or transfer.get("source_resource") != source_resource
                or transfer.get("target_resource") != target_resource):
            violations.append(_violation(
                "C3_DAG_PRECEDENCE",
                tid,
                "Dependency transfer metadata does not match current resources.",
                resource=target_resource,
                predecessor=pred,
                violation_kind="stale_transfer_metadata",
                expected_type=expected_type,
                recorded_type=transfer.get("type"),
                expected_source_resource=source_resource,
                recorded_source_resource=transfer.get("source_resource"),
                expected_target_resource=target_resource,
                recorded_target_resource=transfer.get("target_resource"),
            ))

        transfer_start = transfer.get("start")
        transfer_end = transfer.get("end")
        if (not isinstance(transfer_start, (int, float))
                or not isinstance(transfer_end, (int, float))
                or transfer_end + TIME_TOLERANCE_MS < transfer_start):
            violations.append(_violation(
                "C3_DAG_PRECEDENCE",
                tid,
                "Dependency transfer has an invalid time interval.",
                resource=target_resource,
                predecessor=pred,
                violation_kind="invalid_transfer_interval",
                transfer_start_ms=transfer_start,
                transfer_end_ms=transfer_end,
            ))
            continue

        if transfer_start + TIME_TOLERANCE_MS < pred_end:
            violations.append(_violation(
                "C3_DAG_PRECEDENCE",
                tid,
                "Dependency communication starts before its predecessor finishes.",
                resource=target_resource,
                predecessor=pred,
                violation_kind="transfer_before_predecessor_end",
                predecessor_end_ms=pred_end,
                transfer_start_ms=transfer_start,
                overlap_ms=pred_end - transfer_start,
            ))

        required_ready = max(pred_end, transfer_end)
        if task_start + TIME_TOLERANCE_MS < required_ready:
            violations.append(_violation(
                "C3_DAG_PRECEDENCE",
                tid,
                "Task starts before its dependency communication finishes.",
                resource=target_resource,
                predecessor=pred,
                violation_kind="task_before_transfer_end",
                predecessor_end_ms=pred_end,
                transfer_end_ms=transfer_end,
                task_start_ms=task_start,
                overlap_ms=required_ready - task_start,
            ))
    return violations


def validate_mapping_consistency(system, workload, sched, assignment, mapping):
    """Validate C6 physical-location continuity on shared logical qubits."""
    violations = []
    for (predecessor, tid), edge in workload.edge_data.items():
        if (predecessor not in workload.quantum_tasks
                or tid not in workload.quantum_tasks):
            continue
        shared, inferred = workload.get_shared_logical_qubits(predecessor, tid)
        if not shared:
            continue
        left = mapping.get(predecessor, {})
        right = mapping.get(tid, {})
        left_partition = left.get("logical_partition", {})
        right_partition = right.get("logical_partition", {})
        left_physical = left.get("physical_mapping", {})
        right_physical = right.get("physical_mapping", {})
        for logical in shared:
            if (logical not in left_partition or logical not in right_partition
                    or logical not in left_physical or logical not in right_physical):
                continue  # C2 reports incomplete mappings.
            predecessor_location = (
                left_partition[logical], left_physical[logical]
            )
            successor_location = (
                right_partition[logical], right_physical[logical]
            )
            if predecessor_location != successor_location:
                violations.append(_violation(
                    "C6_MAPPING_CONSISTENCY",
                    tid,
                    "A shared logical qubit changes physical location across a dependency.",
                    resource=assignment.get(tid),
                    predecessor=predecessor,
                    logical_qubit=logical,
                    predecessor_location=predecessor_location,
                    successor_location=successor_location,
                    shared_qubits_inferred=inferred,
                ))
    return violations


def estimate_gate_batch_time_ms(task, qpu):
    """Return the Eq. (36) aggregate gate-batch duration surrogate."""
    depth = max(task.get("depth", 1), 1)
    n_gates = max(task.get("n_gates", 100), 0)
    n_1q = int(n_gates * 0.67)
    n_2q = n_gates - n_1q
    gate_times = qpu.get(
        "gate_times_ns",
        {"1q": 35.6, "2q": 334.0, "measure": 1200.0},
    )
    return (
        n_1q / depth * gate_times["1q"]
        + n_2q / depth * gate_times["2q"]
        + gate_times["measure"]
    ) / 1e6


def validate_coherence_window(system, workload, sched, assignment, mapping,
                              safety_fraction=0.05, event_mode=None,
                              task_window_mode=None,
                              quantum_link_timeline=None):
    """Validate C5 gate batches, EPR holding, and task-span diagnostics."""
    if not 0.0 <= safety_fraction < 1.0:
        raise ValueError("T_safety_frac must be in [0, 1).")
    mode_params = {}
    if event_mode is not None:
        mode_params["coherence_event_mode"] = event_mode
    if task_window_mode is not None:
        mode_params["coherence_task_window_mode"] = task_window_mode
    event_mode = resolve_coherence_event_mode(mode_params)
    violations = []
    for tid in workload.quantum_tasks:
        if tid not in sched:
            continue

        task = workload.tasks[tid]
        qpu_set = (
            sched[tid].get("qpu_set")
            or mapping.get(tid, {}).get("qpu_set")
            or [assignment.get(tid)]
        )
        qpu_set = sorted({qid for qid in qpu_set if qid in system.qpus})
        if not qpu_set:
            continue  # C1 reports invalid or missing assignments.

        batch_times = {
            qid: estimate_gate_batch_time_ms(task, system.qpus[qid])
            for qid in qpu_set
        }
        coherence_times = {
            qid: system.qpus[qid]["coherence_time_us"] / 1000.0
            for qid in qpu_set
        }
        safe_windows = {
            qid: coherence_times[qid] * (1.0 - safety_fraction)
            for qid in qpu_set
        }
        violating_qpus = [
            qid for qid in qpu_set
            if batch_times[qid] > safe_windows[qid] + TIME_TOLERANCE_MS
        ]
        if violating_qpus:
            violations.append(_violation(
                "C5_COHERENCE_WINDOW",
                tid,
                "Estimated gate batch exceeds a used QPU coherence window.",
                resource=violating_qpus,
                qpu_set=qpu_set,
                violating_qpus=violating_qpus,
                batch_time_ms_by_qpu=batch_times,
                coherence_time_ms_by_qpu=coherence_times,
                safety_fraction=safety_fraction,
                safe_window_ms_by_qpu=safe_windows,
            ))

        timeline_records = [
            record
            for reservations in (quantum_link_timeline or {}).values()
            for record in reservations
            if record.get("task") == tid
        ]
        event_schedule_item = sched[tid]
        if timeline_records:
            event_schedule_item = dict(sched[tid])
            event_schedule_item["entanglement_reservations"] = (
                timeline_records
            )
            event_schedule_item["dependency_transfers"] = []
        epr_events = collect_task_epr_events(system, event_schedule_item)
        missing_lifetime_by_link = {}
        for event in epr_events:
            lifetime_ms = event.get("epr_memory_lifetime_ms")
            link = tuple(event.get("link", ()))
            if lifetime_ms is None:
                aggregate = missing_lifetime_by_link.setdefault(link, {
                    "event_count": 0,
                    "epr_count": 0,
                    "max_hold_ms": 0.0,
                })
                aggregate["event_count"] += 1
                aggregate["epr_count"] += event["epr_count"]
                aggregate["max_hold_ms"] = max(
                    aggregate["max_hold_ms"], event["hold_ms"]
                )
                continue
            if event["hold_ms"] > lifetime_ms + TIME_TOLERANCE_MS:
                violations.append(_violation(
                    "C5_EPR_HOLDING_WINDOW",
                    tid,
                    "An EPR batch is held beyond its calibrated quantum-memory lifetime.",
                    resource=list(link),
                    enforced=event_mode == "strict",
                    mode=event_mode,
                    link=list(link),
                    epr_count=event["epr_count"],
                    ready_time_ms=event["ready"],
                    consume_time_ms=event["consume"],
                    hold_time_ms=event["hold_ms"],
                    memory_lifetime_ms=lifetime_ms,
                    memory_lifetime_source=event.get(
                        "memory_lifetime_source"
                    ),
                ))
        for link, aggregate in sorted(missing_lifetime_by_link.items()):
            violations.append(_violation(
                "C5_EPR_MEMORY_LIFETIME_UNSPECIFIED",
                tid,
                "The EPR holding window cannot be certified without a calibrated memory lifetime.",
                resource=list(link),
                enforced=event_mode == "strict",
                mode=event_mode,
                link=list(link),
                **aggregate,
            ))

        first_entanglement_start = min(
            (
                event.get("generation_start", event["ready"])
                for event in epr_events
            ),
            default=sched[tid]["start"],
        )
        exposure_time = max(
            sched[tid]["end"] - first_entanglement_start, 0.0
        )
        limiting_qpu = min(qpu_set, key=lambda qid: coherence_times[qid])
        task_window_limit = coherence_times[limiting_qpu]
        if exposure_time > task_window_limit + TIME_TOLERANCE_MS:
            violations.append(_violation(
                "C5_TASK_WINDOW_DIAGNOSTIC",
                tid,
                "Aggregate task span exceeds the shortest used-QPU coherence time; this is a diagnostic, not a hard state-lifetime constraint.",
                resource=limiting_qpu,
                enforced=False,
                mode="diagnostic",
                qpu_set=qpu_set,
                first_entanglement_start_ms=first_entanglement_start,
                task_end_ms=sched[tid]["end"],
                exposure_time_ms=exposure_time,
                coherence_time_ms_by_qpu=coherence_times,
                limiting_qpu=limiting_qpu,
                coherence_limit_ms=task_window_limit,
            ))
    return violations


def validate_entanglement_connectivity(system, workload, sched, assignment,
                                        mapping, intra_remote_gates=None):
    """Validate the reachability part of C9."""
    violations = []
    intra_remote_gates = intra_remote_gates or {}
    for (pred, tid), edge in workload.edge_data.items():
        if pred not in workload.quantum_tasks or tid not in workload.quantum_tasks:
            continue
        src_qpu = assignment.get(pred)
        dst_qpu = assignment.get(tid)
        if (src_qpu in system.qpus and dst_qpu in system.qpus
                and src_qpu != dst_qpu
                and system.get_quantum_link(src_qpu, dst_qpu) is None):
            violations.append(_violation(
                "C9_ENTANGLEMENT_LINK",
                tid,
                "Cross-QPU quantum dependency has no direct entanglement link.",
                resource=dst_qpu,
                predecessor=pred,
                source_qpu=src_qpu,
                target_qpu=dst_qpu,
            ))

    return violations


def _sweep_epr_capacity(events, reservations, capacity, constraint,
                        message, quantity_name, capacity_name, link_key):
    """Sweep half-open EPR intervals and report capacity crossings."""
    violations = []
    active = {}
    buffered = 0
    maximum = 0
    reported_at = set()
    for time_ms, event_order, index, delta in sorted(events):
        if delta < 0:
            buffered += delta
            active.pop(index, None)
            continue
        buffered += delta
        maximum = max(maximum, buffered)
        active[index] = reservations[index]
        if (capacity is not None and buffered > capacity
                and time_ms not in reported_at):
            reported_at.add(time_ms)
            violations.append(_violation(
                constraint,
                reservations[index].get("task"),
                message,
                resource=sorted(link_key, key=str),
                time_ms=time_ms,
                **{
                    quantity_name: buffered,
                    capacity_name: capacity,
                },
                active_reservations=list(active.values()),
            ))
    return violations, maximum


def validate_entanglement_capacity(system, workload, sched, assignment, mapping,
                                    quantum_link_timeline=None,
                                    memory_capacity_mode=None):
    """Validate C9 generation parallelism, readiness, and EPR storage."""
    violations = []
    memory_capacity_mode = validate_entanglement_memory_capacity_mode(
        memory_capacity_mode
    )
    for link_key, reservations in (quantum_link_timeline or {}).items():
        link = system.quantum_links.get(frozenset(link_key))
        if link is None:
            continue  # Connectivity validation reports unknown links.
        generation_capacities = [
            int(reservation.get(
                "generation_capacity_epr",
                reservation.get(
                    "capacity_epr", link["max_concurrent_epr"]
                ),
            ))
            for reservation in reservations
        ]
        generation_capacity = min(
            generation_capacities,
            default=int(link["max_concurrent_epr"]),
        )
        generation_events = []
        memory_events = []
        memory_capacities = []
        memory_capacity_complete = True
        for index, reservation in enumerate(reservations):
            start = float(reservation.get(
                "generation_start", reservation["start"]
            ))
            ready = float(reservation.get(
                "actual_end",
                reservation.get("ready", reservation["end"]),
            ))
            count = int(reservation["epr_count"])
            deadline = reservation.get("deadline")
            if count <= 0:
                continue
            if ready > start + TIME_TOLERANCE_MS:
                generation_events.append((start, 1, index, count))
                generation_events.append((ready, 0, index, -count))
            if (deadline is not None
                    and ready > float(deadline) + TIME_TOLERANCE_MS):
                violations.append(_violation(
                    "C9_ENTANGLEMENT_READINESS",
                    reservation.get("task"),
                    "The EPR batch is not ready by its scheduled consumption time.",
                    resource=sorted(link_key, key=str),
                    reservation=reservation,
                    ready_time_ms=ready,
                    deadline_ms=deadline,
                ))
            release = reservation.get(
                "actual_consume", reservation.get("consume")
            )
            if release is None:
                release = reservation.get(
                    "actual_discard", reservation.get("discard", deadline)
                )
            if release is not None:
                release = float(release)
                if release > ready + TIME_TOLERANCE_MS:
                    memory_events.append((ready, 1, index, count))
                    memory_events.append((release, 0, index, -count))
            if "memory_capacity_epr" in reservation:
                memory_capacity = reservation["memory_capacity_epr"]
            else:
                memory_capacity = link.get("memory_capacity_epr")
            if memory_capacity is None:
                memory_capacity_complete = False
            else:
                memory_capacities.append(int(memory_capacity))

        generation_violations, _ = _sweep_epr_capacity(
            generation_events,
            reservations,
            generation_capacity,
            "C9_EPR_GENERATION_CAPACITY",
            "Concurrent EPR generation exceeds the selected service parallelism.",
            "generating_epr",
            "generation_capacity_epr",
            link_key,
        )
        violations.extend(generation_violations)

        memory_capacity = (
            min(memory_capacities)
            if memory_capacity_complete and memory_capacities else None
        )
        memory_violations, max_stored = _sweep_epr_capacity(
            memory_events,
            reservations,
            memory_capacity,
            "C9_EPR_MEMORY_CAPACITY",
            "Ready EPR pairs awaiting consumption exceed calibrated quantum-memory capacity.",
            "stored_epr",
            "memory_capacity_epr",
            link_key,
        )
        violations.extend(memory_violations)
        if memory_events and memory_capacity is None:
            violations.append(_violation(
                "C9_EPR_MEMORY_CAPACITY_UNSPECIFIED",
                reservations[memory_events[0][2]].get("task"),
                "EPR storage occupancy cannot be certified without calibrated memory_capacity_epr.",
                resource=sorted(link_key, key=str),
                enforced=memory_capacity_mode == "strict",
                mode=memory_capacity_mode,
                memory_capacity_parameter="memory_capacity_epr",
                maximum_observed_stored_epr=max_stored,
                reservation_count=len({event[2] for event in memory_events}),
            ))
    return violations


def validate_schedule(system, workload, sched, assignment, mapping,
                      params=None, intra_remote_gates=None,
                      classical_link_timeline=None,
                      quantum_link_timeline=None,
                      include_reported_fidelity=True):
    """Run the constraint checks used by the TopoQ feedback loop."""
    params = params or {}
    violations = []
    violations += validate_task_assignment(
        system, workload, sched, assignment, mapping
    )
    violations += validate_qpu_capacity(
        system, workload, sched, assignment, mapping
    )
    violations += validate_concurrent_qpu_capacity(
        system, workload, sched, assignment, mapping
    )
    violations += validate_concurrent_physical_qubits(
        system, workload, sched, assignment, mapping
    )
    violations += validate_dag_precedence(
        system, workload, sched, assignment, mapping
    )
    violations += validate_mapping_consistency(
        system, workload, sched, assignment, mapping
    )
    violations += validate_coherence_window(
        system,
        workload,
        sched,
        assignment,
        mapping,
        safety_fraction=params.get("T_safety_frac", 0.05),
        event_mode=resolve_coherence_event_mode(params),
        quantum_link_timeline=quantum_link_timeline,
    )
    violations += validate_physical_mapping(
        system, workload, sched, assignment, mapping
    )
    violations += validate_physical_routing(
        system, workload, sched, assignment, mapping
    )
    if include_reported_fidelity:
        violations += validate_fidelity_threshold(
            system, workload, sched, assignment, mapping, params=params
        )
    violations += validate_classical_bandwidth(
        system, workload, sched, assignment, mapping,
        classical_link_timeline=classical_link_timeline,
    )
    violations += validate_classical_memory(
        system, workload, sched, assignment, mapping
    )
    violations += validate_entanglement_connectivity(
        system,
        workload,
        sched,
        assignment,
        mapping,
        intra_remote_gates=intra_remote_gates,
    )
    violations += validate_entanglement_capacity(
        system,
        workload,
        sched,
        assignment,
        mapping,
        quantum_link_timeline=quantum_link_timeline,
        memory_capacity_mode=params.get(
            "entanglement_memory_capacity_mode"
        ),
    )
    return violations
