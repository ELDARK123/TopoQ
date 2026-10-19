"""
Experiment Runner
=================
Functions to run single and repeated experiments, wiring together
the system model, scheduler, and simulator.
"""

import copy
import time

from .config import (
    DEFAULT_PARAMS,
    PHASE5_OBJECTIVE_ZERO_REFERENCE_SCALES,
    PHASE5_OBJECTIVE_ZERO_REFERENCE_UNITS,
    PAPER_CLASSICAL_EFFECTIVE_FACTOR,
    PAPER_ENTANGLEMENT_EFFECTIVE_FACTOR,
    N_REPETITIONS,
    RANDOM_SEED_BASE,
    load_dag,
    load_system_config,
    load_network_topology,
    load_sim_params,
    resolve_coherence_event_mode,
    EXPERIMENT_PROFILES,
    validate_classical_bandwidth_sharing_mode,
    validate_classical_routing_policy,
    validate_communication_overlap_mode,
    validate_error_mitigation_policy,
    validate_entanglement_memory_capacity_mode,
    validate_execution_time_policy,
    validate_qpu_calibration_policy,
    validate_interface_latency_mode,
    validate_quantum_epr_routing_policy,
    validate_quantum_centrality_mode,
    validate_phase5_objective_weights,
    validate_topology_inverse_policy,
    validate_fidelity_policy,
)
from .models import HybridSystem, Workload
from .mapping import (
    BASELINE_FEASIBILITY_BEAM_WIDTH,
    BASELINE_FEASIBILITY_MAX_BEAM_WIDTH,
    PARTITION_FILL_PENALTY,
    PARTITION_MAX_REFINEMENT_PASSES,
    PARTITION_METHOD_NAME,
    PARTITION_NODE_ORDER,
)
from .topoq_scheduler import TopoQScheduler
from .baselines import (
    RandomScheduler,
    GreedyScheduler,
    QOnlyScheduler,
    DQTetrisScheduler,
)
from .simulator import Simulator
from .validators import validate_fidelity_threshold
from .mitigation import zne_sampling_overhead
from .purification import resolve_purification_policy
from .fidelity import build_epr_quality_diagnostics
from .entanglement import resolve_quantum_link_latency_policy
from .provenance import (
    build_result_provenance,
    collect_experiment_input_paths,
)


def _trim_system_config(config, n_qpus):
    """Trim a system config to use only the first n_qpus QPUs.

    Removes excess QPUs and their quantum links, keeping the system
    consistent for scalability experiments.
    """
    trimmed = copy.deepcopy(config)
    trimmed["n_qpus"] = n_qpus
    # Keep only first n_qpus QPU instances
    kept_qpu_ids = set(q["qpu_id"] for q in trimmed["qpu_instances"][:n_qpus])
    trimmed["qpu_instances"] = trimmed["qpu_instances"][:n_qpus]
    # Keep only quantum links between remaining QPUs
    trimmed["quantum_links"] = [
        link for link in trimmed["quantum_links"]
        if link["qpu_i"] in kept_qpu_ids and link["qpu_j"] in kept_qpu_ids
    ]
    # Keep only interface links for remaining QPUs
    trimmed["interface_links"] = [
        link for link in trimmed["interface_links"]
        if link["qpu_id"] in kept_qpu_ids
    ]
    return trimmed


def _quantum_channel_inventory(config, n_qpu_limit=None):
    """Summarize explicitly configured direct channels by QPU pair."""
    qpu_instances = config["qpu_instances"][:n_qpu_limit]
    kept = {item["qpu_id"] for item in qpu_instances}
    counts = {}
    for link in config["quantum_links"]:
        if link["qpu_i"] not in kept or link["qpu_j"] not in kept:
            continue
        pair = "--".join(sorted((link["qpu_i"], link["qpu_j"])))
        counts[pair] = counts.get(pair, 0) + 1
    return {
        "channel_counts_by_pair": dict(sorted(counts.items())),
        "multi_channel_pairs": sorted(
            pair for pair, count in counts.items() if count > 1
        ),
        "maximum_channel_count": max(counts.values(), default=0),
    }


def run_single_experiment(dag_id, config_name, network_name, algorithm,
                          params=None, ablation=None, seed=42, n_qpu_limit=None,
                          record_diagnostics=False):
    """Run a single experiment.

    Args:
        dag_id:        Workload DAG identifier.
        config_name:   System configuration name (small/medium/large).
        network_name:  Classical network topology name.
        algorithm:     Algorithm name (Random/Greedy/Q-Only/DQTetris/TopoQ).
        params:        Optional algorithm parameter overrides.
        ablation:      Optional ablation flags dict.
        seed:          Random seed for reproducibility.
        n_qpu_limit:   If set, only use the first n_qpu_limit QPUs from the
                       system config. Used for scalability experiments.
    """
    dag_data = load_dag(dag_id)
    config = load_system_config(config_name)
    network = load_network_topology(network_name)
    sim_params = load_sim_params()
    purification_policy = resolve_purification_policy(sim_params, params)
    sim_params = copy.deepcopy(sim_params)
    sim_params["purification_policy"] = purification_policy
    calibration_policy = validate_qpu_calibration_policy(
        (params or {}).get("qpu_calibration_mode"),
        (params or {}).get("ibm_cx_direction_policy"),
        (params or {}).get("ibm_missing_calibration_policy"),
    )
    sim_params["qpu_calibration_policy"] = calibration_policy
    quantum_link_latency_policy = resolve_quantum_link_latency_policy(
        params, sim_params
    )
    sim_params["quantum_link_latency_policy"] = (
        quantum_link_latency_policy
    )

    # If limiting QPU count, trim the config
    if n_qpu_limit and n_qpu_limit < config["n_qpus"]:
        config = _trim_system_config(config, n_qpu_limit)

    system = HybridSystem(config, network, sim_params)
    workload = Workload(dag_data)
    mitigation_policy = validate_error_mitigation_policy(
        (params or {}).get("error_mitigation_mode"),
        (params or {}).get("symmetry_error_detection_fraction"),
        (params or {}).get("zne_noise_scales"),
    )
    configured_overlap_mode = validate_communication_overlap_mode(
        (params or {}).get("communication_overlap_mode")
    )
    effective_overlap_mode = (
        configured_overlap_mode
        if algorithm == "TopoQ" and not (ablation or {}).get("no_comm_overlap")
        else "none"
    )
    execution_time_policy = validate_execution_time_policy(
        (params or {}).get("execution_time_mode"),
        (params or {}).get("quantum_shots"),
        (params or {}).get("classical_compute_efficiency"),
    )

    if algorithm == "TopoQ":
        scheduler = TopoQScheduler(system, workload, params, ablation, seed)
    elif algorithm == "Random":
        scheduler = RandomScheduler(system, workload, seed, params=params)
    elif algorithm == "Greedy":
        scheduler = GreedyScheduler(system, workload, seed, params=params)
    elif algorithm == "Q-Only":
        scheduler = QOnlyScheduler(system, workload, seed, params=params)
    elif algorithm == "DQTetris":
        scheduler = DQTetrisScheduler(system, workload, seed, params=params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    sched, assignment, mapping = scheduler.schedule()

    simulator = Simulator(
        system, sim_params, seed, algorithm,
        entanglement_mode=(params or {}).get(
            "entanglement_simulation_mode", "deterministic"
        ),
        error_mitigation_policy=mitigation_policy,
        communication_overlap_mode=effective_overlap_mode,
    )
    metrics = simulator.simulate(sched, assignment, workload, mapping)
    if record_diagnostics:
        from .diagnostics import build_task_diagnostics
        sim_sched = getattr(simulator, "last_sim_sched", sched)
        diagnostics = build_task_diagnostics(
            system,
            workload,
            sim_sched,
            assignment,
            mapping,
            getattr(simulator, "last_task_estimates", {}),
            metrics=metrics,
        )
        metrics["task_diagnostics"] = diagnostics["tasks"]
        metrics["critical_path"] = diagnostics["critical_path"]
        metrics["diagnostics_summary"] = diagnostics["summary"]
    metrics.update({
        "execution_time_mode": execution_time_policy["mode"],
        "quantum_shots": execution_time_policy["quantum_shots"],
        "classical_compute_efficiency": execution_time_policy[
            "classical_compute_efficiency"
        ],
        "qpu_calibration_mode": calibration_policy["mode"],
        "ibm_cx_direction_policy": calibration_policy[
            "cx_direction_policy"
        ],
        "ibm_missing_calibration_policy": calibration_policy[
            "missing_calibration_policy"
        ],
        "c4_task_count": len(workload.quantum_tasks),
    })
    if algorithm != "TopoQ":
        # Every scheduler must be checked against the same C4 threshold once
        # strict fidelity is enabled; earlier only TopoQ ran the validator.
        c4_params = dict(DEFAULT_PARAMS)
        c4_params.update(params or {})
        c4_violations = validate_fidelity_threshold(
            system, workload, sched, assignment, mapping, params=c4_params
        )
        quantum_task_count = len(workload.quantum_tasks)
        metrics["c4_task_count"] = quantum_task_count
        metrics["c4_reported_violations"] = len(c4_violations)
        metrics["c4_enforced_violations"] = sum(
            1 for violation in c4_violations
            if violation.get("enforced", True)
        )
        metrics["c4_violation_fraction"] = (
            len(c4_violations) / quantum_task_count
            if quantum_task_count else 0.0
        )
        if c4_violations:
            metrics["c4_min_violating_fidelity"] = min(
                violation["details"]["estimated_fidelity"]
                for violation in c4_violations
            )
            metrics["c4_max_log_margin_deficit"] = max(
                violation["details"]["log_threshold"]
                - violation["details"]["log_fidelity"]
                for violation in c4_violations
            )
    if algorithm != "TopoQ":
        feasibility = scheduler.baseline_physical_feasibility
        task_projection = feasibility["task_assignment"]
        metrics.update({
            "baseline_task_assignment_projected": int(
                task_projection["projection_applied"]
            ),
            "baseline_quantum_tasks_moved": task_projection[
                "moved_quantum_tasks"
            ],
            "baseline_dependency_violations_before": task_projection[
                "before_violations"
            ],
            "baseline_dependency_violations_after": task_projection[
                "after_violations"
            ],
            "baseline_partition_tasks_projected": feasibility[
                "partition_tasks_projected"
            ],
            "baseline_partition_logical_qubits_moved": feasibility[
                "partition_logical_qubits_moved"
            ],
            "baseline_partition_violating_edges_before": feasibility[
                "partition_before_violating_edges"
            ],
            "baseline_partition_violating_edges_after": feasibility[
                "partition_after_violating_edges"
            ],
            "baseline_partition_violating_gates_before": feasibility[
                "partition_before_violating_gates"
            ],
            "baseline_partition_violating_gates_after": feasibility[
                "partition_after_violating_gates"
            ],
        })
        if algorithm == "DQTetris":
            segmentation = getattr(scheduler, "dqtetris_segmentation", {})
            metrics["dqtetris_segmented_tasks"] = len(segmentation)
            if segmentation:
                metrics["dqtetris_mean_segments"] = (
                    sum(item["segments"] for item in segmentation.values())
                    / len(segmentation)
                )
                metrics["dqtetris_total_teleports"] = sum(
                    item["teleports"] for item in segmentation.values()
                )
                metrics["dqtetris_total_cut_gates"] = sum(
                    item["cut_gates"] for item in segmentation.values()
                )
                metrics["dqtetris_total_modelled_remote_gates"] = sum(
                    item["total"] for item in segmentation.values()
                )
    if algorithm == "TopoQ":
        epr_records = [
            record
            for reservations in getattr(
                scheduler, "quantum_link_timeline", {}
            ).values()
            for record in reservations
        ]
        metrics["quantum_epr_reservation_batches"] = len(epr_records)
        metrics["quantum_epr_backup_reservation_batches"] = sum(
            int(record.get("backup_channel_count", 0) > 0)
            for record in epr_records
        )
        metrics["quantum_epr_backup_channel_uses"] = sum(
            int(record.get("backup_channel_count", 0))
            for record in epr_records
        )
        metrics["quantum_epr_max_selected_channels"] = max(
            (
                int(record.get("quantum_channel_count", 1))
                for record in epr_records
            ),
            default=0,
        )
        reservation_policy = getattr(
            scheduler, "entanglement_reservation_policy", {}
        )
        metrics["entanglement_reservation_mode"] = reservation_policy.get(
            "mode", "proactive"
        )
        for mode in ("proactive", "on_demand", "just_in_time"):
            metrics[f"epr_{mode}_reservation_batches"] = sum(
                int(record.get("reservation_mode") == mode)
                for record in epr_records
            )
        hold_times = [
            max(
                float(record.get("consume", record.get("ready", 0.0)))
                - float(record.get("ready", 0.0)),
                0.0,
            )
            for record in epr_records
        ]
        metrics["epr_mean_hold_ms"] = (
            sum(hold_times) / len(hold_times) if hold_times else 0.0
        )
        metrics["epr_max_hold_ms"] = max(hold_times, default=0.0)
        metrics["swap_gates_total"] = sum(
            max(int(item.get("n_swap_gates", 0)), 0)
            for tid, item in sched.items()
            if tid in workload.quantum_tasks
        )
        metrics["effective_two_qubit_gates_max"] = max(
            (
                int(item.get("effective_two_qubit_gates", 0))
                for tid, item in sched.items()
                if tid in workload.quantum_tasks
            ),
            default=0,
        )
        metrics["c4_reported_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_constraint_report", [])
            if violation.get("constraint") == "C4_FIDELITY_THRESHOLD"
        ])
        metrics["c4_enforced_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_enforced_violations", [])
            if violation.get("constraint") == "C4_FIDELITY_THRESHOLD"
        ])
        metrics["c4_violation_fraction"] = (
            metrics["c4_reported_violations"] / metrics["c4_task_count"]
            if metrics["c4_task_count"] else 0.0
        )
        metrics["c5_batch_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_enforced_violations", [])
            if violation.get("constraint") == "C5_COHERENCE_WINDOW"
        ])
        metrics["c5_epr_holding_reported_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_constraint_report", [])
            if (violation.get("constraint") == "C5_EPR_HOLDING_WINDOW"
                and not violation.get("enforced", True))
        ])
        metrics["c5_epr_holding_enforced_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_enforced_violations", [])
            if violation.get("constraint") == "C5_EPR_HOLDING_WINDOW"
        ])
        metrics["c5_epr_memory_unspecified_reported_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_constraint_report", [])
            if (violation.get("constraint") ==
                "C5_EPR_MEMORY_LIFETIME_UNSPECIFIED"
                and not violation.get("enforced", True))
        ])
        metrics["c5_epr_memory_unspecified_enforced_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_enforced_violations", [])
            if (violation.get("constraint") ==
                "C5_EPR_MEMORY_LIFETIME_UNSPECIFIED")
        ])
        metrics["c5_task_diagnostic_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_constraint_report", [])
            if violation.get("constraint") == "C5_TASK_WINDOW_DIAGNOSTIC"
        ])
        metrics["c5_task_reported_violations"] = metrics[
            "c5_task_diagnostic_violations"
        ]
        metrics["c5_task_enforced_violations"] = 0
        metrics["c9_generation_capacity_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_enforced_violations", [])
            if violation.get("constraint") ==
            "C9_EPR_GENERATION_CAPACITY"
        ])
        metrics["c9_readiness_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_enforced_violations", [])
            if violation.get("constraint") ==
            "C9_ENTANGLEMENT_READINESS"
        ])
        metrics["c9_memory_capacity_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_enforced_violations", [])
            if violation.get("constraint") == "C9_EPR_MEMORY_CAPACITY"
        ])
        metrics["c9_memory_capacity_unspecified_reported_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_constraint_report", [])
            if (violation.get("constraint") ==
                "C9_EPR_MEMORY_CAPACITY_UNSPECIFIED"
                and not violation.get("enforced", True))
        ])
        metrics["c9_memory_capacity_unspecified_enforced_violations"] = len([
            violation for violation in
            getattr(scheduler, "last_enforced_violations", [])
            if violation.get("constraint") ==
            "C9_EPR_MEMORY_CAPACITY_UNSPECIFIED"
        ])
        sa_diagnostics = getattr(scheduler, "last_sa_objective", None)
        if sa_diagnostics:
            metrics["objective_mode"] = sa_diagnostics.get(
                "objective_mode",
                getattr(scheduler, "objective_mode", "joint"),
            )
            metrics["sa_initial_objective"] = sa_diagnostics[
                "initial"
            ]["total"]
            metrics["sa_best_objective"] = sa_diagnostics["best"]["total"]
            metrics["sa_best_makespan"] = sa_diagnostics["best_makespan"]
            metrics["sa_initial_temperature"] = sa_diagnostics[
                "initial_temperature"
            ]
            metrics["sa_final_temperature"] = sa_diagnostics[
                "final_temperature"
            ]

    return metrics


def run_with_repetitions(dag_id, config_name, network_name, algorithm,
                         params=None, ablation=None, n_rep=N_REPETITIONS, n_qpu_limit=None):
    """Run repeated experiments and report mean, spread, and wall time.

    Numeric metrics are averaged. Configuration/metric strings are passed
    through unchanged when all repetitions agree; if they disagree (which
    would indicate a protocol inconsistency) the raw list is kept instead of
    crashing with ``sum(int, str)``.
    """
    results = []
    for i in range(n_rep):
        seed = RANDOM_SEED_BASE + i
        started_at = time.perf_counter()
        metrics = dict(run_single_experiment(
            dag_id, config_name, network_name, algorithm,
            params, ablation, seed, n_qpu_limit=n_qpu_limit,
        ))
        metrics["runtime_seconds"] = time.perf_counter() - started_at
        results.append(metrics)

    if not results:
        raise ValueError("run_with_repetitions requires at least one repetition")

    # Average numeric fields; keep identical non-numeric fields verbatim.
    avg = {}
    for key in results[0]:
        vals = [r.get(key) for r in results]
        numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in vals
        )
        if numeric:
            mean = sum(vals) / len(vals)
            avg[key] = mean
            avg[f"{key}_std"] = (
                sum((v - mean) ** 2 for v in vals) / len(vals)
            ) ** 0.5
            continue
        rendered = {repr(v) for v in vals}
        if len(rendered) == 1:
            avg[key] = vals[0]
        else:
            # Keep all variants instead of raising; callers can audit them.
            avg[key] = vals
    return avg


def check_experiment_feasibility(dag_id, config_name, n_qpu_limit=None):
    dag = load_dag(dag_id)
    config = load_system_config(config_name)
    qpus = config["qpu_instances"][:n_qpu_limit] if n_qpu_limit else config["qpu_instances"]
    available = sum(qpu["n_qubits"] for qpu in qpus)
    quantum_tasks = [task for task in dag["tasks"] if task["type"] == "quantum"]
    largest = max(quantum_tasks, key=lambda task: task.get("n_qubits", 0), default=None)
    required = largest.get("n_qubits", 0) if largest else 0
    return {
        "feasible": required <= available,
        "required_qubits": required,
        "available_qubits": available,
        "largest_task": largest.get("id") if largest else None,
        "reason": None if required <= available else
                  f"Largest task requires {required} qubits; selected system provides {available}.",
    }


def run_experiment_spec(spec, profile="paper", repetitions=None,
                        algorithms=None, on_infeasible="error", params=None):
    feasibility = check_experiment_feasibility(
        spec.dag_id, spec.system_config, spec.qpu_limit
    )
    n_rep = (
        repetitions or spec.repetitions
        or EXPERIMENT_PROFILES[profile]["repetitions"]
    )
    selected = tuple(algorithms or spec.algorithms)
    merged_params = dict(spec.params)
    merged_params.update(params or {})
    merged_params.setdefault(
        "entanglement_simulation_mode",
        "poisson" if profile in {"paper", "large"} else "deterministic",
    )
    execution_time_policy = validate_execution_time_policy(
        merged_params.get("execution_time_mode"),
        merged_params.get("quantum_shots"),
        merged_params.get("classical_compute_efficiency"),
    )
    merged_params.update({
        "execution_time_mode": execution_time_policy["mode"],
        "quantum_shots": execution_time_policy["quantum_shots"],
        "classical_compute_efficiency": execution_time_policy[
            "classical_compute_efficiency"
        ],
    })
    effective_parameters = dict(DEFAULT_PARAMS)
    effective_parameters.update(merged_params)
    metadata = {
        "experiment_id": spec.experiment_id, "profile": profile,
        "dag_id": spec.dag_id, "system_config": spec.system_config,
        "network_topology": spec.network_topology,
        "qpu_limit": spec.qpu_limit, "feasibility": feasibility,
        "random_seed_base": RANDOM_SEED_BASE,
        "repetitions": n_rep,
        "algorithms": selected,
        "execution_time_model": {
            **execution_time_policy,
            "quantum_equation": (
                "Tq=N_shots*(depth*tau_bar+tau_measure)"
            ),
            "classical_equation": "Tc=w/(kappa*p)",
            "communication_and_epr_separate": True,
            "routing_overhead_added_once": True,
            "applies_to": list(selected),
        },
        "provenance": build_result_provenance(
            protocol={
                "kind": "single_experiment",
                "experiment_id": spec.experiment_id,
                "profile": profile,
                "dag_id": spec.dag_id,
                "system_config": spec.system_config,
                "network_topology": spec.network_topology,
                "qpu_limit": spec.qpu_limit,
                "algorithms": list(selected),
                "repetitions": n_rep,
                "random_seed_base": RANDOM_SEED_BASE,
                "random_seeds": [
                    RANDOM_SEED_BASE + index for index in range(n_rep)
                ],
                "effective_parameters": effective_parameters,
                "ablation": dict(spec.ablation),
            },
            input_paths=collect_experiment_input_paths(
                spec.dag_id,
                spec.system_config,
                spec.network_topology,
            ),
        ),
    }
    if not feasibility["feasible"]:
        if on_infeasible == "error":
            raise ValueError(feasibility["reason"])
        return {"metadata": metadata, "status": "skipped", "results": {}}
    phase5_objective_weights = validate_phase5_objective_weights(
        merged_params
    )
    merged_params.update(phase5_objective_weights)
    purification_policy = resolve_purification_policy(
        load_sim_params(), merged_params
    )
    quantum_centrality_mode = validate_quantum_centrality_mode(
        merged_params.get("quantum_centrality_mode")
    )
    interface_latency_mode = validate_interface_latency_mode(
        merged_params.get("interface_latency_mode")
    )
    interface_signal_speed = merged_params.get(
        "interface_signal_speed_m_s"
    )
    if interface_signal_speed is None:
        interface_signal_speed = load_sim_params().get(
            "interface_link_parameters", {}
        ).get("v_signal_m_s")
    if interface_signal_speed is not None:
        interface_signal_speed = float(interface_signal_speed)
        if interface_signal_speed <= 0:
            raise ValueError(
                "interface_signal_speed_m_s must be positive."
            )
    if (interface_latency_mode == "paper_eq24"
            and interface_signal_speed is None):
        raise ValueError(
            "paper_eq24 requires interface_signal_speed_m_s."
        )
    inverse_policy = validate_topology_inverse_policy(
        merged_params.get("topology_inverse_mode"),
        merged_params.get("topology_reciprocal_epsilon"),
    )
    classical_routing_policy = validate_classical_routing_policy(
        merged_params.get("classical_routing_mode"),
        merged_params.get("classical_multipath_k"),
    )
    classical_bandwidth_sharing_mode = (
        validate_classical_bandwidth_sharing_mode(
            merged_params.get("classical_bandwidth_sharing_mode")
        )
    )
    communication_overlap_mode = validate_communication_overlap_mode(
        merged_params.get("communication_overlap_mode")
    )
    quantum_epr_routing_policy = validate_quantum_epr_routing_policy(
        merged_params.get("quantum_epr_routing_mode"),
        merged_params.get("quantum_epr_reliability_window_ms"),
    )
    quantum_epr_routing_mode = quantum_epr_routing_policy["mode"]
    quantum_link_latency_policy = resolve_quantum_link_latency_policy(
        merged_params, load_sim_params()
    )
    mitigation_policy = validate_error_mitigation_policy(
        merged_params.get("error_mitigation_mode"),
        merged_params.get("symmetry_error_detection_fraction"),
        merged_params.get("zne_noise_scales"),
    )
    if mitigation_policy["zne_noise_scales"] is None:
        zne_coefficients = None
        zne_overhead = 1.0
    else:
        zne_coefficients, zne_overhead = zne_sampling_overhead(
            mitigation_policy["zne_noise_scales"]
        )
    channel_inventory = _quantum_channel_inventory(
        load_system_config(spec.system_config), spec.qpu_limit
    )
    multipath_disabled = bool(spec.ablation.get("no_multipath"))
    if quantum_epr_routing_mode == "deadline_direct_backup":
        effective_epr_service = "reservation_dependent_direct_channels"
    elif (quantum_epr_routing_mode == "parallel_channels"
            and not multipath_disabled
            and channel_inventory["maximum_channel_count"] > 1):
        effective_epr_service = "parallel_channels"
    else:
        effective_epr_service = "single_channel"
    if (quantum_epr_routing_mode == "paper_fixed_25"
            and not multipath_disabled):
        paper_duration_scale = 0.75
    elif (quantum_epr_routing_mode == "paper_algorithm_145"
          and multipath_disabled):
        paper_duration_scale = 1.45
    else:
        paper_duration_scale = 1.0
    policy = validate_fidelity_policy(
        merged_params.get("fidelity_constraint_mode", "report_only"),
        merged_params.get("fidelity_threshold"),
    )
    fidelity_params = load_sim_params()["fidelity_parameters"]
    if policy["mode"] == "hardware_explicit":
        effective_threshold = policy["threshold"]
        threshold_source = "user"
    else:
        effective_threshold = (
            fidelity_params.get("F_min_qaoa", fidelity_params["F_min"])
            if "qaoa" in spec.dag_id.lower()
            else fidelity_params["F_min"]
        )
        threshold_source = "dataset"
    metadata.update({
        "runtime_measurement": {
            "metric": "wall_clock_elapsed_seconds",
            "timer": "time.perf_counter",
            "scope": (
                "per_algorithm_repetition_including_setup_scheduler_"
                "and_simulator"
            ),
            "aggregation": "mean_and_population_standard_deviation",
            "included_in_objective": False,
            "result_fields": [
                "runtime_seconds", "runtime_seconds_std",
            ],
            "applies_to": list(selected),
        },
        "phase2_partitioning": {
            "method": PARTITION_METHOD_NAME,
            "multilevel": False,
            "coarsening": False,
            "uncoarsening": False,
            "pair_swaps": False,
            "node_order": PARTITION_NODE_ORDER,
            "greedy_incremental_cost": (
                "sum_new_cut(w_int*quantum_link_cost)"
                "+fill_penalty*assigned_qubits"
            ),
            "fill_penalty": PARTITION_FILL_PENALTY,
            "refinement_move": "single_vertex",
            "refinement_acceptance": "strict_objective_decrease",
            "maximum_refinement_passes": (
                PARTITION_MAX_REFINEMENT_PASSES
            ),
            "applies_to": [
                algorithm for algorithm in selected if algorithm == "TopoQ"
            ],
        },
        "baseline_physical_feasibility": {
            "method": "minimum_hamming_direct_link_projection",
            "violation_formula": (
                "V=sum_task_edges(g_uv*I[partition_cut_without_direct_link])"
                "+sum_QQ_dependencies(I[assigned_pair_without_direct_link])"
            ),
            "optimization_order": (
                "first_require_V_equal_0_then_minimize_Hamming_distance_"
                "from_original_baseline_result"
            ),
            "already_feasible_result": "preserved_exactly",
            "failure_policy": "raise_explicit_error_without_link_fabrication",
            "partition_capacity_enforced": True,
            "direct_link_required": True,
            "multi_hop_epr_routing": False,
            "bounded_beam_widths": [
                BASELINE_FEASIBILITY_BEAM_WIDTH,
                BASELINE_FEASIBILITY_MAX_BEAM_WIDTH,
            ],
            "topoq_constraints_or_objective_reused": False,
            "applies_to": [
                algorithm for algorithm in selected if algorithm != "TopoQ"
            ],
        },
        "phase5_objective": {
            "objective_mode": merged_params.get(
                "objective_mode", DEFAULT_PARAMS["objective_mode"]
            ),
            "formula": (
                "J=Cmax (pure makespan)"
                if merged_params.get(
                    "objective_mode", DEFAULT_PARAMS["objective_mode"]
                ) in {"makespan", "makespan_penalty"}
                else "J=w_makespan*Cmax_hat+w_quantum*CQ_hat"
                "+w_classical*CC_hat+w_fidelity*LF_hat"
            ),
            "quantum_communication_formula": (
                "CQ_hat=0.70*entanglement_wait_hat"
                "+0.30*remote_gate_count_hat"
            ),
            "fidelity_loss_formula": "LF=sum_t(-ln(F_hat_t))",
            "weights": phase5_objective_weights,
            "normalization_reference": "initial_sa_state",
            "zero_reference_scales": {
                name: {
                    "value": value,
                    "unit": PHASE5_OBJECTIVE_ZERO_REFERENCE_UNITS[name],
                }
                for name, value
                in PHASE5_OBJECTIVE_ZERO_REFERENCE_SCALES.items()
            },
            "initial_temperature_fraction": merged_params.get(
                "T_SA_init_frac", DEFAULT_PARAMS["T_SA_init_frac"]
            ),
            "initial_temperature_formula": (
                "T0=fraction*Cmax0"
                if merged_params.get(
                    "objective_mode", DEFAULT_PARAMS["objective_mode"]
                ) in {"makespan", "makespan_penalty"}
                else "T0=fraction*J0"
            ),
            "acceptance_probability": "exp(-(J_new-J_current)/T_SA)",
            "gamma_modulates_acceptance": False,
            "cooling": merged_params.get(
                "SA_cooling", DEFAULT_PARAMS["SA_cooling"]
            ),
            "max_non_improving_iterations": merged_params.get(
                "SA_max_non_improve",
                DEFAULT_PARAMS["SA_max_non_improve"],
            ),
            "maximum_iterations": 2000,
            "applies_to": [
                algorithm for algorithm in selected if algorithm == "TopoQ"
            ],
        },
        "entanglement_simulation": {
            "mode": merged_params["entanglement_simulation_mode"],
            "random_seed_base": RANDOM_SEED_BASE,
            "distribution": (
                "exponential" if merged_params["entanglement_simulation_mode"] == "poisson"
                else "scheduled_threshold_window"
            ),
            "sample_formula": (
                "T_ms=-ln(1-U)/lambda_hz*1000" if
                merged_params["entanglement_simulation_mode"] == "poisson"
                else None
            ),
        },
        "quantum_link_latency": {
            **quantum_link_latency_policy,
            "rate_semantics": (
                "end_to_end_ready_pair_service_rate"
                if quantum_link_latency_policy["mode"]
                == "delivered_pair_rate"
                else "generation_rate_before_explicit_fixed_stages"
            ),
            "deterministic_equation": (
                "-ln(1-p_threshold)/lambda_eff * duration_scale "
                "+ stage_overhead"
            ),
            "stage_equation": (
                "T_BSM + T_switch + signal_traversals*distance/signal_speed"
            ),
            "multi_channel_stage_aggregation": "conservative_max",
            "applies_to": list(selected),
        },
        "fidelity_constraint": {
            "mode": policy["mode"],
            "threshold": effective_threshold,
            "threshold_source": threshold_source,
            "enforced": policy["mode"] != "report_only",
            "solution_class": (
                "strict_tahs_candidate"
                if policy["mode"] != "report_only"
                else "scheduling_result_with_fidelity_diagnostics"
            ),
            "strict_tahs_feasible": (
                True if policy["mode"] != "report_only" else None
            ),
            "c4_certification_scope": (
                "enforced"
                if policy["mode"] != "report_only"
                else "not_certified"
            ),
            "applies_to": [algorithm for algorithm in selected if algorithm == "TopoQ"],
        },
        "error_mitigation": {
            "mode": mitigation_policy["mode"],
            "physical_fidelity_unchanged_by_zne": True,
            "symmetry_error_detection_fraction": mitigation_policy[
                "symmetry_error_detection_fraction"
            ],
            "zne_noise_scales": mitigation_policy["zne_noise_scales"],
            "zne_richardson_coefficients": zne_coefficients,
            "zne_sampling_overhead": zne_overhead,
            "work_metric": "expected_qpu_execution_work_ms_not_makespan",
            "applies_to": list(selected),
        },
        "epr_purification": {
            **purification_policy,
            "physical_fidelity_unchanged": True,
            "scenario_metric": "purification_adjusted_fidelity",
            "cost_metric": (
                "expected_raw_epr_pairs_and_purification_attempts"
            ),
        },
        "epr_quality_diagnostics": build_epr_quality_diagnostics(
            load_sim_params(), purification_policy, effective_threshold
        ),
        "coherence_constraint": {
            "batch_equation": "T_batch <= 0.95 * T_coh",
            "batch_safety_fraction": merged_params.get(
                "T_safety_frac", DEFAULT_PARAMS["T_safety_frac"]
            ),
            "batch_enforced": True,
            "epr_holding_equation": (
                "max(T_consume - T_ready, 0) <= L_mem_ij"
            ),
            "epr_memory_lifetime_parameter": "epr_memory_lifetime_ms",
            "event_mode": resolve_coherence_event_mode(merged_params),
            "task_window_equation": (
                "task_end - first_entanglement_start"
            ),
            "task_window_mode": "diagnostic_only",
            "applies_to": [
                algorithm for algorithm in selected if algorithm == "TopoQ"
            ],
        },
        "entanglement_capacity_constraint": {
            "generation_equation": (
                "sum(n_k * I(T_gen_start_k <= t < T_ready_k)) <= G_ij"
            ),
            "generation_capacity_parameter": "max_concurrent_epr",
            "generation_enforced": True,
            "memory_equation": (
                "sum(n_k * I(T_ready_k <= t < T_release_k)) <= M_ij"
            ),
            "memory_capacity_parameter": "memory_capacity_epr",
            "missing_memory_capacity_mode": (
                validate_entanglement_memory_capacity_mode(
                    merged_params.get(
                        "entanglement_memory_capacity_mode"
                    )
                )
            ),
            "applies_to": [
                algorithm for algorithm in selected if algorithm == "TopoQ"
            ],
        },
        "topology_characterization": {
            "quantum_centrality_mode": quantum_centrality_mode,
            "interface_latency_mode": interface_latency_mode,
            "interface_signal_speed_m_s": (
                interface_signal_speed
                if interface_latency_mode == "paper_eq24" else None
            ),
            "topology_inverse_mode": inverse_policy["mode"],
            "topology_reciprocal_epsilon": inverse_policy["epsilon"],
            "applies_to": [
                algorithm for algorithm in selected if algorithm == "TopoQ"
            ],
        },
        "classical_routing": {
            "mode": (
                "single_widest"
                if spec.ablation.get("no_multipath")
                else classical_routing_policy["mode"]
            ),
            "configured_mode": classical_routing_policy["mode"],
            "multipath_k": (
                classical_routing_policy["multipath_k"]
                if (not spec.ablation.get("no_multipath")
                    and classical_routing_policy["mode"]
                    == "edge_disjoint_split") else None
            ),
            "paper_fixed_reduction_fraction": (
                0.15
                if (not spec.ablation.get("no_multipath")
                    and classical_routing_policy["mode"]
                    == "paper_fixed_15") else None
            ),
            "applies_to": [
                algorithm for algorithm in selected if algorithm == "TopoQ"
            ],
        },
        "classical_bandwidth_sharing": {
            "mode": classical_bandwidth_sharing_mode,
            "rate_allocation": "fixed_path_bottleneck",
            "shared_edge_policy": "serialize_conflicting_intervals",
            "edge_disjoint_subflows_can_overlap": True,
            "capacity_constraint": "sum_active_rates<=link_capacity",
            "applies_to": list(selected),
        },
        "communication_overlap": {
            "configured_mode": communication_overlap_mode,
            "effective_mode": (
                "none" if spec.ablation.get("no_comm_overlap")
                else communication_overlap_mode
            ),
            "reporting_only": True,
            "physical_schedule_unchanged": True,
            "paper_entanglement_effective_factor": (
                PAPER_ENTANGLEMENT_EFFECTIVE_FACTOR
            ),
            "paper_classical_effective_factor": (
                PAPER_CLASSICAL_EFFECTIVE_FACTOR
            ),
            "paper_entanglement_credit_fraction": (
                1.0 - PAPER_ENTANGLEMENT_EFFECTIVE_FACTOR
            ),
            "paper_classical_credit_fraction": (
                1.0 - PAPER_CLASSICAL_EFFECTIVE_FACTOR
            ),
            "applies_to": [
                algorithm for algorithm in selected if algorithm == "TopoQ"
            ],
        },
        "quantum_epr_routing": {
            "configured_mode": quantum_epr_routing_mode,
            "effective_physical_service": effective_epr_service,
            "multipath_disabled": multipath_disabled,
            "reliability_window_ms": quantum_epr_routing_policy[
                "reliability_window_ms"
            ],
            "target_success_probability": merged_params.get(
                "p_threshold", DEFAULT_PARAMS["p_threshold"]
            ),
            "backup_selection": (
                "primary_then_descending_rate_until_threshold"
                if quantum_epr_routing_mode == "deadline_direct_backup"
                else None
            ),
            "paper_duration_scale": paper_duration_scale,
            "rate_equation": "lambda_eff=sum(channel.lambda_hz)",
            "capacity_equation": "b_eff=sum(channel.max_concurrent_epr)",
            "wait_equation": (
                "T_ms=-ln(1-p_threshold)/lambda_eff_hz*1000"
            ),
            **channel_inventory,
            "applies_to": [
                algorithm for algorithm in selected if algorithm == "TopoQ"
            ],
        },
    })
    results = {
        algorithm: run_with_repetitions(
            spec.dag_id, spec.system_config, spec.network_topology, algorithm,
            params=(merged_params or None), ablation=(spec.ablation or None),
            n_rep=n_rep, n_qpu_limit=spec.qpu_limit,
        )
        for algorithm in selected
    }
    return {"metadata": metadata, "status": "completed", "results": results}


def run_experiment_batch(specs, profile="paper", repetitions=None,
                         algorithms=None, on_infeasible="skip", params=None):
    return {
        spec.experiment_id: run_experiment_spec(
            spec, profile, repetitions, algorithms, on_infeasible, params
        )
        for spec in specs
    }
