"""Numerically stable fidelity estimates shared by scheduling and simulation.

Final schedules evaluate paper Eqs. (3)-(6) in the log domain. In particular,
Eq. (5) is evaluated from the recorded generation-ready and consumption times
of every EPR event. Paper Eq. (7) remains a conservative pre-schedule estimate
for candidates that do not yet have reservations. Workload-level reporting
keeps task logs so long circuits do not underflow silently.
"""

import math

from .calibration import resolve_qpu_calibration_profile
from .purification import build_purification_plan
from .timing import resolve_task_gate_profile


LOG_FLOOR = math.log(1e-300)


def resolve_epr_quality(system, link):
    """Resolve one link/channel's EPR fidelity and provenance."""
    configured = link.get("epr_fidelity")
    if configured is None:
        fidelity = min(
            system.sim_params["fidelity_parameters"]["F_EPR_range"]
        )
        source = "global_lower_bound"
    else:
        fidelity = float(configured)
        source = "link_config"
    fidelity = float(fidelity)
    if not 0.0 < fidelity <= 1.0:
        raise ValueError(
            f"EPR fidelity must be in (0, 1]; received {fidelity!r}."
        )
    decoherence = float(link["decoherence_rate_per_ms"])
    if not math.isfinite(decoherence) or decoherence < 0.0:
        raise ValueError(
            "decoherence_rate_per_ms must be a non-negative finite number."
        )
    memory_lifetime = link.get("epr_memory_lifetime_ms")
    if memory_lifetime is not None:
        memory_lifetime = float(memory_lifetime)
        if not math.isfinite(memory_lifetime) or memory_lifetime <= 0.0:
            raise ValueError(
                "epr_memory_lifetime_ms must be a positive finite number."
            )
    return {
        "epr_fidelity": fidelity,
        "fidelity_source": source,
        "decoherence_rate_per_ms": decoherence,
        "epr_memory_lifetime_ms": memory_lifetime,
        "memory_lifetime_source": (
            "link_config" if memory_lifetime is not None else "unspecified"
        ),
    }


def resolve_epr_service_quality(system, channels):
    """Return a conservative quality for an aggregate direct-channel service."""
    qualities = [resolve_epr_quality(system, channel) for channel in channels]
    if not qualities:
        raise ValueError("At least one quantum channel is required.")
    sources = {item["fidelity_source"] for item in qualities}
    if len(qualities) == 1:
        source = qualities[0]["fidelity_source"]
    elif sources == {"link_config"}:
        source = "selected_channels_conservative_min"
    else:
        source = "selected_channels_global_lower_bound"
    memory_lifetimes = [
        item["epr_memory_lifetime_ms"] for item in qualities
        if item["epr_memory_lifetime_ms"] is not None
    ]
    complete_memory_calibration = len(memory_lifetimes) == len(qualities)
    return {
        "epr_fidelity": min(item["epr_fidelity"] for item in qualities),
        "fidelity_source": source,
        "decoherence_rate_per_ms": max(
            item["decoherence_rate_per_ms"] for item in qualities
        ),
        "epr_memory_lifetime_ms": (
            min(memory_lifetimes) if complete_memory_calibration else None
        ),
        "memory_lifetime_source": (
            "selected_channels_conservative_min"
            if complete_memory_calibration else "unspecified_channel"
        ),
    }


def build_epr_event(system, link_key, epr_count, ready, consume,
                    task_id=None, predecessor=None, source="schedule",
                    channels=None):
    """Build one grouped Eq. (5) EPR-consumption event."""
    endpoints = tuple(sorted(link_key))
    if len(endpoints) != 2:
        raise ValueError(f"EPR event requires two endpoints: {endpoints}.")
    if channels is None:
        link = system.get_quantum_link(*endpoints)
        if link is None:
            raise ValueError(f"Missing quantum link {endpoints}.")
        channels = (link,)
    quality = resolve_epr_service_quality(system, tuple(channels))
    count = max(int(epr_count), 0)
    return {
        "task": task_id,
        "predecessor": predecessor,
        "link": endpoints,
        "epr_count": count,
        "ready": float(ready),
        "consume": float(consume),
        "generation_start": float(ready),
        "epr_fidelity": quality["epr_fidelity"],
        "fidelity_source": quality["fidelity_source"],
        "decoherence_rate_per_ms": quality[
            "decoherence_rate_per_ms"
        ],
        "epr_memory_lifetime_ms": quality[
            "epr_memory_lifetime_ms"
        ],
        "memory_lifetime_source": quality["memory_lifetime_source"],
        "event_source": source,
    }


def collect_task_epr_events(system, schedule_item):
    """Collect and normalize intra-task and dependency EPR reservations."""
    records = list(schedule_item.get("entanglement_reservations", []))
    for transfer in schedule_item.get("dependency_transfers", []):
        if transfer.get("type") == "quantum-quantum":
            records.extend(transfer.get("entanglement_reservations", []))
    events = []
    for record in records:
        count = max(int(record.get("epr_count", 0)), 0)
        if count == 0:
            continue
        ready = float(record.get(
            "actual_end",
            record.get("ready", record.get("end", 0.0)),
        ))
        consume = float(record.get(
            "actual_consume",
            record.get("consume", schedule_item.get("start", ready)),
        ))
        generation_start = float(record.get(
            "generation_start", record.get("start", ready)
        ))
        if "epr_fidelity" in record:
            event = {
                "task": record.get("task"),
                "predecessor": record.get("predecessor"),
                "link": tuple(record.get("link", ())),
                "epr_count": count,
                "ready": ready,
                "consume": consume,
                "generation_start": generation_start,
                "epr_fidelity": float(record["epr_fidelity"]),
                "fidelity_source": record.get(
                    "fidelity_source", "reservation_record"
                ),
                "decoherence_rate_per_ms": float(
                    record["decoherence_rate_per_ms"]
                ),
                "epr_memory_lifetime_ms": (
                    float(record["epr_memory_lifetime_ms"])
                    if record.get("epr_memory_lifetime_ms") is not None
                    else None
                ),
                "memory_lifetime_source": record.get(
                    "memory_lifetime_source", "reservation_record"
                ),
                "event_source": record.get(
                    "event_source", "reservation_record"
                ),
            }
        else:
            event = build_epr_event(
                system,
                record.get("link", ()),
                count,
                ready,
                consume,
                task_id=record.get("task"),
                predecessor=record.get("predecessor"),
                source="legacy_reservation_fallback",
            )
            event["generation_start"] = generation_start
        event["hold_ms"] = max(consume - ready, 0.0)
        events.append(event)
    return events


def estimate_coherent_batch_ms(qpu, task, extra_2q_gates=0):
    """Estimate one coherent circuit batch, not aggregate task wall time."""
    n_qubits = max(int(task.get("n_qubits", 1)), 1)
    profile = resolve_task_gate_profile(task)
    depth = profile["depth"]
    n_1q = profile["n_1q_gates"]
    n_2q = profile["n_2q_gates"] + max(int(extra_2q_gates), 0)
    gate_times = qpu.get(
        "gate_times_ns",
        {"1q": 35.6, "2q": 334.0, "measure": 1200.0},
    )

    # This is the same depth-aware batch duration used by the C5 checker.
    batch_ms = (
        n_1q / depth * gate_times["1q"]
        + n_2q / depth * gate_times["2q"]
        + gate_times["measure"]
    ) / 1e6

    # Average active time of one physical qubit. A two-qubit gate occupies two
    # qubits, hence the factor 2. The remaining part of the coherent batch is
    # the task-level approximation of Tmax_idle in paper Eq. (10).
    active_per_qubit_ms = (
        (n_1q * gate_times["1q"] + 2 * n_2q * gate_times["2q"])
        / n_qubits
        + gate_times["measure"]
    ) / 1e6
    idle_per_qubit_ms = max(batch_ms - active_per_qubit_ms, 0.0)
    return {
        "batch_ms": batch_ms,
        "active_per_qubit_ms": active_per_qubit_ms,
        "idle_per_qubit_ms": idle_per_qubit_ms,
    }


def estimate_task_log_fidelity(system, task, qpu, schedule_item=None,
                               estimated_remote_gates=None,
                               estimated_hold_ms=None,
                               estimated_swap_gates=None):
    """Return one task's log-domain Eqs. (3)-(7) fidelity components.

    State-preparation convention (paper definition): ``t_prep`` is the start
    of coherent state preparation (``task.start``).  ``base_ready`` is only a
    resource-availability time and does not start the data-qubit coherence
    clock.  EPR pairs generated before ``t_prep`` contribute only their own
    hold decoherence (``log_epr_hold``); data-qubit idle waiting for EPR
    (``log_epr_wait``) is charged only over ``[t_prep, end]``.
    """
    has_final_schedule = schedule_item is not None
    schedule_item = schedule_item or {}
    fidelity_params = system.sim_params["fidelity_parameters"]
    link_params = system.sim_params["quantum_link_parameters"]

    n_qubits = max(int(task.get("n_qubits", 1)), 1)
    profile = resolve_task_gate_profile(task)
    n_1q = profile["n_1q_gates"]
    if estimated_swap_gates is None:
        estimated_swap_gates = schedule_item.get("n_swap_gates", 0)
    n_swap_gates = max(int(estimated_swap_gates), 0)
    n_2q = profile["n_2q_gates"] + n_swap_gates
    measurement_count = (
        profile["measurement_count"]
        if "paper_measurement_count" in task else 1
    )
    log_prep = math.log(max(fidelity_params["F_prep"], 1e-300))
    epr_wait_mode = fidelity_params.get("epr_wait_mode", "separate")
    teleportation_mode = fidelity_params.get(
        "teleportation_mode", "calibrated"
    )
    tau_epr_wait_ms = 0.0

    scheduled_partitions = schedule_item.get(
        "quantum_timing", {}
    ).get("qpu_partitions", {})
    if scheduled_partitions:
        partitions = [
            (qid, item) for qid, item in sorted(scheduled_partitions.items())
        ]
        calibration_model = "mapped_qpu_partitions"
    else:
        calibration_policy = system.sim_params.get(
            "qpu_calibration_policy",
            {
                "mode": "configured_qpu_average",
                "cx_direction_policy": "conservative_max",
                "missing_calibration_policy": "error",
            },
        )
        calibration = resolve_qpu_calibration_profile(
            qpu, calibration_policy, allow_unmapped=True
        )
        partitions = [(qpu.get("id", "primary_qpu"), {
            "gate_times_ns": calibration["gate_times_ns"],
            "gate_error_rates": calibration["gate_error_rates"],
            "coherence_time_us": calibration["coherence_time_us"],
            "logical_qubit_count": n_qubits,
            "swap_gate_count": n_swap_gates,
            "calibration_provenance": calibration["provenance"],
        })]
        calibration_model = calibration["provenance"]["resolution"]

    total_partition_qubits = sum(
        max(int(item.get("logical_qubit_count", 0)), 0)
        for _, item in partitions
    )
    if total_partition_qubits <= 0:
        total_partition_qubits = n_qubits

    log_gate = 0.0
    log_memory = 0.0
    partition_components = []
    batch_values = []
    active_values = []
    idle_values = []
    depth = profile["depth"]
    for qid, partition in partitions:
        partition_qubits = max(
            int(partition.get("logical_qubit_count", 0)), 0
        )
        if partition_qubits <= 0:
            partition_qubits = n_qubits if len(partitions) == 1 else 1
        weight = partition_qubits / total_partition_qubits
        partition_n_1q = n_1q * weight
        partition_n_2q = profile["n_2q_gates"] * weight + max(
            int(partition.get("swap_gate_count", 0)), 0
        )
        partition_measurements = measurement_count * weight
        errors = partition["gate_error_rates"]
        eps_1q = min(max(float(errors["1q"]), 0.0), 1.0 - 1e-15)
        eps_2q = min(max(float(errors["2q"]), 0.0), 1.0 - 1e-15)
        eps_measure = min(
            max(float(errors["measure"]), 0.0), 1.0 - 1e-15
        )
        partition_log_gate = (
            partition_n_1q * math.log1p(-eps_1q)
            + partition_n_2q * math.log1p(-eps_2q)
            + partition_measurements * math.log1p(-eps_measure)
        )
        log_gate += partition_log_gate

        gate_times = partition["gate_times_ns"]
        measurement_ns = (
            float(gate_times["measure"])
            if partition_measurements > 0.0 else 0.0
        )
        batch_ms = (
            partition_n_1q / depth * float(gate_times["1q"])
            + partition_n_2q / depth * float(gate_times["2q"])
            + measurement_ns
        ) / 1e6
        active_per_qubit_ms = (
            (
                partition_n_1q * float(gate_times["1q"])
                + 2.0 * partition_n_2q * float(gate_times["2q"])
            ) / max(partition_qubits, 1)
            + measurement_ns
        ) / 1e6
        idle_per_qubit_ms = max(batch_ms - active_per_qubit_ms, 0.0)
        coherence_ms = float(partition["coherence_time_us"]) / 1000.0
        gamma = 1.0 / coherence_ms if coherence_ms > 0.0 else math.inf
        partition_log_memory = (
            -gamma * idle_per_qubit_ms * partition_qubits
        )
        log_memory += partition_log_memory
        eps_tel = 1.0 - (
            (1.0 - eps_measure) ** 2 * (1.0 - eps_1q)
        )
        batch_values.append(batch_ms)
        active_values.append(active_per_qubit_ms)
        idle_values.append(idle_per_qubit_ms)
        partition_components.append({
            "qpu": qid,
            "logical_qubit_count": partition_qubits,
            "allocated_1q_gates": partition_n_1q,
            "allocated_2q_gates": partition_n_2q,
            "allocated_measurements": partition_measurements,
            "log_gate": partition_log_gate,
            "log_memory": partition_log_memory,
            "log_epr_wait": 0.0,
            "tau_epr_wait_ms": 0.0,
            "coherence_ms": coherence_ms,
            "epsilon_tel": eps_tel,
            "batch_ms": batch_ms,
            "idle_per_qubit_ms": idle_per_qubit_ms,
            "calibration_provenance": partition.get(
                "calibration_provenance", {}
            ),
        })

    epr_events = collect_task_epr_events(system, schedule_item)
    if epr_events:
        n_remote_epr = sum(item["epr_count"] for item in epr_events)
        log_epr_generation = sum(
            item["epr_count"] * math.log(
                max(item["epr_fidelity"], 1e-300)
            )
            for item in epr_events
        )
        log_epr_hold = sum(
            item["epr_count"] * (
                -item["decoherence_rate_per_ms"] * item["hold_ms"]
            )
            for item in epr_events
        )
        log_entanglement = log_epr_generation + log_epr_hold
        hold_ms = max(
            (item["hold_ms"] for item in epr_events), default=0.0
        )
        fidelity_model = "eq5_per_epr_events"
    elif has_final_schedule:
        n_remote_epr = max(int(schedule_item.get("n_remote_gates", 0)), 0)
        if n_remote_epr:
            raise ValueError(
                "A final scheduled task reports remote gates but has no "
                "per-EPR reservation events for paper Eq. (5)."
            )
        # Eq. (5) over an empty EPR set is the multiplicative identity.
        log_epr_generation = 0.0
        log_epr_hold = 0.0
        log_entanglement = 0.0
        hold_ms = 0.0
        fidelity_model = "eq5_per_epr_events"
    else:
        if estimated_remote_gates is None:
            estimated_remote_gates = schedule_item.get("n_remote_gates", 0)
        n_remote_epr = max(int(estimated_remote_gates), 0)
        if estimated_hold_ms is None:
            estimated_hold_ms = schedule_item.get(
                "entanglement_hold_ms",
                schedule_item.get("entanglement_wait", 0.0),
            )
        hold_ms = max(float(estimated_hold_ms), 0.0)
        f_epr_min = min(fidelity_params["F_EPR_range"])
        eta_max = max(link_params["eta_range_per_ms"])
        log_epr_generation = n_remote_epr * math.log(
            max(f_epr_min, 1e-300)
        )
        log_epr_hold = n_remote_epr * (-eta_max * hold_ms)
        log_entanglement = log_epr_generation + log_epr_hold
        fidelity_model = "eq7_conservative_estimate"

    # Charge EPR wait only over the interval in which the task's data qubits
    # are already prepared and live.  With proactive reservation EPR pairs are
    # ready before the task starts, so this interval is empty; the EPR hold
    # decoherence is already charged by the log_entanglement term.
    if epr_events and has_final_schedule:
        task_start = max(float(schedule_item.get("start", 0.0)), 0.0)
        task_end = max(
            float(schedule_item.get("end", task_start)), task_start
        )
        wait_values = []
        for event in epr_events:
            generation_start = max(
                float(event.get("generation_start", task_start)), 0.0
            )
            ready = max(float(event.get("ready", task_start)), 0.0)
            wait_values.append(max(
                0.0,
                min(ready, task_end) - max(generation_start, task_start),
            ))
        tau_epr_wait_ms = max(wait_values, default=0.0)
    elif has_final_schedule:
        tau_epr_wait_ms = 0.0

    log_epr_wait = 0.0
    for component in partition_components:
        coherence_ms = component["coherence_ms"]
        gamma = 1.0 / coherence_ms if coherence_ms > 0.0 else math.inf
        component_log_epr_wait = (
            -gamma * tau_epr_wait_ms * component["logical_qubit_count"]
        )
        component["log_epr_wait"] = component_log_epr_wait
        component["tau_epr_wait_ms"] = tau_epr_wait_ms
        log_epr_wait += component_log_epr_wait

    coherent = {
        "batch_ms": max(batch_values, default=0.0),
        "active_per_qubit_ms": max(active_values, default=0.0),
        "idle_per_qubit_ms": max(idle_values, default=0.0),
    }

    # Teleportation endpoint error: one Bell measurement (two measurements)
    # plus one single-qubit Pauli correction per remote gate.
    partition_weights = [
        max(int(item["logical_qubit_count"]), 0)
        for item in partition_components
    ]
    total_partition_weight = sum(partition_weights) or 1
    eps_tel_effective = sum(
        item["epsilon_tel"] * weight
        for item, weight in zip(partition_components, partition_weights)
    ) / total_partition_weight
    if teleportation_mode == "calibrated" and n_remote_epr > 0:
        log_teleportation = n_remote_epr * math.log1p(
            -min(max(eps_tel_effective, 0.0), 1.0 - 1e-15)
        )
    else:
        log_teleportation = 0.0

    if epr_wait_mode == "exclude":
        effective_log_epr_wait = 0.0
    elif epr_wait_mode == "merged":
        effective_log_epr_wait = 0.0
        log_memory += log_epr_wait
    else:  # separate (default)
        effective_log_epr_wait = log_epr_wait

    log_total = (
        log_prep + log_gate + log_entanglement + log_teleportation
        + log_memory + effective_log_epr_wait
    )
    fidelity = math.exp(max(log_total, LOG_FLOOR))
    return {
        "fidelity": min(max(fidelity, 0.0), 1.0),
        "log_fidelity": log_total,
        "log_prep": log_prep,
        "log_gate": log_gate,
        "log_entanglement": log_entanglement,
        "log_epr_generation": log_epr_generation,
        "log_epr_hold": log_epr_hold,
        "log_teleportation": log_teleportation,
        "log_memory": log_memory,
        "log_epr_wait": effective_log_epr_wait,
        "epr_wait_mode": epr_wait_mode,
        "teleportation_mode": teleportation_mode,
        "epsilon_tel": eps_tel_effective,
        "tau_epr_wait_ms": tau_epr_wait_ms,
        "n_remote_gates": max(
            int(schedule_item.get("n_remote_gates", n_remote_epr)), 0
        ),
        "n_remote_epr_pairs": n_remote_epr,
        "n_1q_gates": n_1q,
        "n_2q_gates": n_2q,
        "n_swap_gates": n_swap_gates,
        "measurement_count": measurement_count,
        "entanglement_hold_ms": hold_ms,
        "fidelity_model": fidelity_model,
        "calibration_model": calibration_model,
        "partition_fidelity_components": partition_components,
        "epr_events": epr_events,
        **coherent,
    }


def estimate_schedule_task_fidelities(system, workload, sched, assignment,
                                      mapping=None):
    """Return task-level physical fidelity estimates keyed by task id."""
    estimates = {}
    for tid in workload.quantum_tasks:
        if tid not in sched:
            continue
        qid = assignment.get(tid)
        if qid not in system.qpus:
            continue
        estimates[tid] = estimate_task_log_fidelity(
            system,
            workload.tasks[tid],
            system.qpus[qid],
            schedule_item=sched[tid],
        )
    return estimates


def estimate_purification_scenario_fidelities(system, estimates):
    """Return idealized EPR-purification scenario estimates.

    The physical task estimate remains the reference. Only the EPR factor is
    replaced by the separately reported idealized BBPSSW output factor.
    """
    plan = system.sim_params.get("purification_policy", {})
    scenario = {}
    for tid, item in estimates.items():
        events = item.get("epr_events", [])
        delta = 0.0
        raw_pair_cost = 0.0
        purification_attempts = 0.0
        max_rounds = 0
        if events:
            for event in events:
                event_plan = build_purification_plan(
                    event["epr_fidelity"],
                    mode=plan.get("mode", "none"),
                    rounds=(
                        plan.get("rounds")
                        if plan.get("mode") == "bbpssw_fixed_rounds"
                        else None
                    ),
                    target=(
                        plan.get("target_fidelity")
                        if plan.get("mode") == "bbpssw_target"
                        else None
                    ),
                    max_rounds=plan.get("max_rounds", 8),
                )
                count = event["epr_count"]
                delta += count * (
                    math.log(event_plan["output_fidelity"])
                    - math.log(event["epr_fidelity"])
                )
                raw_pair_cost += count * event_plan[
                    "expected_raw_pairs_per_output"
                ]
                purification_attempts += count * event_plan[
                    "expected_purification_attempts_per_output"
                ]
                max_rounds = max(max_rounds, event_plan["rounds"])
        else:
            count = max(int(item.get("n_remote_epr_pairs", 0)), 0)
            raw = float(plan.get("raw_fidelity", min(
                system.sim_params["fidelity_parameters"]["F_EPR_range"]
            )))
            output = float(plan.get("output_fidelity", raw))
            if raw > 0.0 and output > 0.0:
                delta = count * (math.log(output) - math.log(raw))
            raw_pair_cost = count * float(
                plan.get("expected_raw_pairs_per_output", 1.0)
            )
            purification_attempts = count * float(
                plan.get("expected_purification_attempts_per_output", 0.0)
            )
            max_rounds = int(plan.get("rounds", 0))
        log_fidelity = item["log_fidelity"] + delta
        scenario[tid] = {
            **item,
            "fidelity": min(max(math.exp(max(log_fidelity, LOG_FLOOR)), 0.0), 1.0),
            "log_fidelity": log_fidelity,
            "expected_raw_epr_pairs": raw_pair_cost,
            "expected_purification_attempts": purification_attempts,
            "purification_rounds": max_rounds,
        }
    return scenario


def build_epr_quality_diagnostics(sim_params, purification_policy,
                                  threshold):
    """Record the manuscript threshold contradiction as machine-readable data."""
    fidelity_params = sim_params["fidelity_parameters"]
    raw_min = min(fidelity_params["F_EPR_range"])
    raw_max = max(fidelity_params["F_EPR_range"])
    prep = float(fidelity_params["F_prep"])
    threshold = float(threshold)
    purified = float(purification_policy.get("output_fidelity", raw_min))
    raw_min_ceiling = prep * raw_min
    raw_max_ceiling = prep * raw_max
    purified_ceiling = prep * purified
    return {
        "configured_fidelity_threshold": threshold,
        "raw_epr_fidelity_min": raw_min,
        "raw_epr_fidelity_max": raw_max,
        "purified_epr_fidelity": purified,
        "optimistic_raw_min_single_remote_ceiling": raw_min_ceiling,
        "optimistic_raw_max_single_remote_ceiling": raw_max_ceiling,
        "optimistic_purified_single_remote_ceiling": purified_ceiling,
        "raw_min_can_meet_threshold": raw_min_ceiling >= threshold,
        "raw_max_can_meet_threshold": raw_max_ceiling >= threshold,
        "purified_can_meet_threshold": purified_ceiling >= threshold,
        "ceiling_assumptions": "ideal_gate_and_memory_single_remote_epr",
        "threshold_conflict": max(raw_max_ceiling, purified_ceiling) < threshold,
        "threshold_conflict_reason": (
            "Even the optimistic single-remote-EPR ceiling is below the "
            "configured task threshold before gate and memory losses."
            if max(raw_max_ceiling, purified_ceiling) < threshold else None
        ),
    }


def aggregate_task_fidelities(estimates, default_fidelity):
    """Aggregate task-level physical or postselected fidelity values."""
    values = list(estimates.values())
    if values and isinstance(values[0], dict):
        fidelities = [item["fidelity"] for item in values]
        logs = [item["log_fidelity"] for item in values]
    else:
        fidelities = [min(max(float(value), 1e-300), 1.0)
                      for value in values]
        logs = [math.log(value) for value in fidelities]

    if not fidelities:
        fp = min(max(float(default_fidelity), 1e-300), 1.0)
        return {
            "fidelity": fp,
            "fidelity_task_mean": fp,
            "fidelity_total_product": fp,
            "fidelity_min_task": fp,
            "fidelity_geometric_mean": fp,
            "fidelity_log10_product": math.log10(max(fp, 1e-300)),
            "fidelity_task_count": 1,
        }

    log_sum = sum(logs)
    return {
        "fidelity": sum(fidelities) / len(fidelities),
        "fidelity_task_mean": sum(fidelities) / len(fidelities),
        "fidelity_total_product": math.exp(max(log_sum, LOG_FLOOR)),
        "fidelity_min_task": min(fidelities),
        "fidelity_geometric_mean": math.exp(
            max(log_sum / len(fidelities), LOG_FLOOR)
        ),
        "fidelity_log10_product": log_sum / math.log(10.0),
        "fidelity_task_count": len(fidelities),
    }


def estimate_schedule_fidelity(system, workload, sched, assignment):
    """Aggregate stable per-task physical fidelity metrics for a schedule."""
    estimates = estimate_schedule_task_fidelities(
        system, workload, sched, assignment
    )
    default_fidelity = system.sim_params["fidelity_parameters"]["F_prep"]
    return aggregate_task_fidelities(estimates, default_fidelity)
