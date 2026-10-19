"""Per-task diagnostic records and critical-path decomposition (C4/E2/E3).

The regular experiment metrics are aggregate; these records preserve enough
per-task timing/fidelity information to plot critical-path breakdowns and to
attribute makespan changes to queueing, EPR wait, communication, or execution.
"""


def _transfer_delay(schedule_item, predecessor):
    delay = 0.0
    for transfer in schedule_item.get("dependency_transfers", []):
        if transfer.get("predecessor") != predecessor:
            continue
        end = float(transfer.get("end", 0.0))
        delay = max(delay, end)
    return delay


def _dependency_ready_ms(workload, sched, task_id):
    ready = 0.0
    for predecessor in workload.predecessors.get(task_id, []):
        if predecessor not in sched:
            continue
        transfer_entry = _transfer_delay(sched[task_id], predecessor)
        finish = float(sched[predecessor]["end"])
        if transfer_entry:
            finish = max(finish, transfer_entry)
        ready = max(ready, finish)
    return ready


def compute_critical_path(workload, sched):
    """Trace the realized critical path through start/end times.

    The path follows the latest-finishing predecessor of the latest-finishing
    task.  Gaps caused by capacity/EPR waits remain visible as ``wait_ms`` in
    the decomposition rather than being hidden in task execution.
    """
    if not sched:
        return {
            "tasks": [],
            "total_path_ms": 0.0,
            "exec_ms": 0.0,
            "quantum_exec_ms": 0.0,
            "classical_exec_ms": 0.0,
            "transfer_ms": 0.0,
            "wait_ms": 0.0,
        }
    current = max(sched, key=lambda tid: float(sched[tid]["end"]))
    path = []
    seen = set()
    while current is not None and current not in seen:
        seen.add(current)
        path.append(current)
        best = None
        best_finish = -1.0
        for predecessor in workload.predecessors.get(current, []):
            if predecessor not in sched:
                continue
            transfer_end = _transfer_delay(sched[current], predecessor)
            finish = float(sched[predecessor]["end"])
            if transfer_end:
                finish = max(finish, transfer_end)
            if finish > best_finish:
                best_finish = finish
                best = predecessor
        current = best
    path.reverse()

    exec_ms = 0.0
    quantum_exec_ms = 0.0
    classical_exec_ms = 0.0
    transfer_ms = 0.0
    wait_ms = 0.0
    previous = None
    for task_id in path:
        item = sched[task_id]
        duration = max(float(item["end"]) - float(item["start"]), 0.0)
        exec_ms += duration
        if workload.tasks[task_id]["type"] == "quantum":
            quantum_exec_ms += duration
        else:
            classical_exec_ms += duration
        if previous is not None:
            transfer_end = _transfer_delay(item, previous)
            if transfer_end:
                transfer_ms += max(transfer_end - float(sched[previous]["end"]), 0.0)
                expected_start = max(float(sched[previous]["end"]), transfer_end)
            else:
                expected_start = float(sched[previous]["end"])
            wait_ms += max(float(item["start"]) - expected_start, 0.0)
        previous = task_id

    total_path_ms = (
        float(sched[path[-1]]["end"]) - float(sched[path[0]]["start"])
    )
    # Attribute any remaining path time to waiting so the decomposition is
    # exact by construction (exec + transfer + wait == total_path_ms).
    wait_ms = max(total_path_ms - exec_ms - transfer_ms, 0.0)
    return {
        "tasks": path,
        "total_path_ms": total_path_ms,
        "exec_ms": exec_ms,
        "quantum_exec_ms": quantum_exec_ms,
        "classical_exec_ms": classical_exec_ms,
        "transfer_ms": transfer_ms,
        "wait_ms": wait_ms,
    }


def build_task_diagnostics(system, workload, sched, assignment, mapping,
                           task_estimates, metrics=None):
    """Build per-task records plus the realized critical path."""
    critical_path = compute_critical_path(workload, sched)
    critical_set = set(critical_path["tasks"])
    tasks = []
    for task_id, item in sched.items():
        task = workload.tasks[task_id]
        record = {
            "task": task_id,
            "type": task["type"],
            "resource": item.get("qpu", item.get("node")),
            "start_ms": float(item["start"]),
            "end_ms": float(item["end"]),
            "duration_ms": max(
                float(item["end"]) - float(item["start"]), 0.0
            ),
            "dependency_ready_ms": _dependency_ready_ms(
                workload, sched, task_id
            ),
            "entanglement_wait_ms": float(
                item.get("entanglement_wait", 0.0)
            ),
            "qpu_capacity_wait_ms": float(
                item.get("qpu_capacity_wait", 0.0)
            ),
            "comm_ms": float(item.get("comm_time", 0.0)),
            "n_remote_gates": int(item.get("n_remote_gates", 0)),
            "n_swap_gates": int(item.get("n_swap_gates", 0)),
            "on_critical_path": task_id in critical_set,
        }
        mapping_item = mapping.get(task_id, {}) if mapping else {}
        partition = mapping_item.get("logical_partition", {})
        if partition:
            qpu_counts = {}
            for qpu in partition.values():
                qpu_counts[qpu] = qpu_counts.get(qpu, 0) + 1
            record["partition_qpu_qubits"] = qpu_counts
        if task["type"] == "quantum":
            estimate = task_estimates.get(task_id, {})
            if estimate:
                record.update({
                    "fidelity": estimate.get("fidelity"),
                    "log_fidelity": estimate.get("log_fidelity"),
                    "log_prep": estimate.get("log_prep"),
                    "log_gate": estimate.get("log_gate"),
                    "log_entanglement": estimate.get("log_entanglement"),
                    "log_epr_generation": estimate.get(
                        "log_epr_generation"
                    ),
                    "log_epr_hold": estimate.get("log_epr_hold"),
                    "log_teleportation": estimate.get("log_teleportation"),
                    "log_memory": estimate.get("log_memory"),
                    "log_epr_wait": estimate.get("log_epr_wait"),
                    "epr_hold_ms": estimate.get("entanglement_hold_ms", 0.0),
                    "epr_pairs": estimate.get("n_remote_epr_pairs", 0),
                })
        tasks.append(record)

    return {
        "tasks": tasks,
        "critical_path": critical_path,
        "summary": {
            "task_count": len(tasks),
            "quantum_tasks": sum(
                item["type"] == "quantum" for item in tasks
            ),
            "classical_tasks": sum(
                item["type"] == "classical" for item in tasks
            ),
            "makespan_ms": (
                max(item["end_ms"] for item in tasks) if tasks else 0.0
            ),
            "critical_path_ms": critical_path["total_path_ms"],
            "critical_path_share": (
                critical_path["total_path_ms"]
                / max(max(item["end_ms"] for item in tasks), 1e-12)
                if tasks else 0.0
            ),
            "comm_total_ms": (metrics or {}).get("comm_total"),
        },
    }
