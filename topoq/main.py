"""
Main Entry Point
================
Run all experiments and save results.
"""

import json
import os
import argparse
from datetime import datetime

from .config import (
    CLASSICAL_BANDWIDTH_SHARING_MODES,
    CLASSICAL_ROUTING_MODES,
    COMMUNICATION_OVERLAP_MODES,
    COHERENCE_EVENT_MODES,
    COHERENCE_TASK_WINDOW_MODES,
    DEFAULT_PARAMS,
    PHASE5_OBJECTIVE_ZERO_REFERENCE_SCALES,
    PHASE5_OBJECTIVE_ZERO_REFERENCE_UNITS,
    ENTANGLEMENT_MEMORY_CAPACITY_MODES,
    ENTANGLEMENT_SIMULATION_MODES,
    ERROR_MITIGATION_MODES,
    FIDELITY_CONSTRAINT_MODES,
    EPR_PURIFICATION_MODES,
    EXPERIMENT_PROFILES,
    INTERFACE_LATENCY_MODES,
    QUANTUM_EPR_ROUTING_MODES,
    QUANTUM_LINK_LATENCY_MODES,
    QUANTUM_CENTRALITY_MODES,
    TOPOLOGY_INVERSE_MODES,
    RANDOM_SEED_BASE,
    RESULTS_DIR,
    get_experiment,
    iter_experiments,
    load_sim_params,
    resolve_coherence_event_mode,
    validate_entanglement_memory_capacity_mode,
    validate_classical_bandwidth_sharing_mode,
    validate_fidelity_policy,
    validate_communication_overlap_mode,
    validate_error_mitigation_policy,
    validate_quantum_epr_routing_policy,
    validate_purification_policy,
    validate_phase5_objective_weights,
    validate_objective_mode,
    OBJECTIVE_MODES,
)
from .runner import check_experiment_feasibility, run_experiment_spec
from .fidelity import build_epr_quality_diagnostics
from .entanglement import resolve_quantum_link_latency_policy
from .experiments import (
    experiment_makespan_comparison,
    experiment_communication_overhead,
    experiment_fidelity,
    experiment_scalability,
    experiment_ablation,
    experiment_parameter_sensitivity,
)
from .reporting import make_json_safe, print_summary
from .mitigation import zne_sampling_overhead
from .purification import resolve_purification_policy
from .provenance import (
    build_result_provenance,
    collect_dataset_json_paths,
)


def _build_run_context(now=None):
    """Return stable run metadata and a filesystem-safe local run ID."""
    started_at = now or datetime.now().astimezone()
    return {
        "run_id": started_at.strftime("%Y%m%d-%H%M%S-%f%z"),
        "started_at": started_at.isoformat(),
    }


def _default_output_path(filename, run_id, results_dir=RESULTS_DIR):
    """Place default outputs in a unique, traceable run directory."""
    return os.path.abspath(os.path.join(
        results_dir, "runs", run_id, filename
    ))


def _resolve_output_path(requested_output, filename, run_id,
                         results_dir=RESULTS_DIR):
    """Resolve an explicit output or the protected timestamped default."""
    if requested_output:
        return os.path.abspath(os.path.expanduser(requested_output))
    return _default_output_path(filename, run_id, results_dir=results_dir)


def _preflight_output_path(path, overwrite=False):
    """Reject unsafe output targets before a potentially long experiment."""
    if os.path.isdir(path):
        raise IsADirectoryError(f"Output path is a directory: {path}")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite with an "
            "explicit --output path to replace it."
        )


def _write_json_result(path, result, overwrite=False):
    """Write JSON with exclusive creation unless overwrite was explicit."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    mode = "w" if overwrite else "x"
    with open(path, mode, encoding="utf-8") as stream:
        json.dump(make_json_safe(result), stream, indent=2,
                  ensure_ascii=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="TopoQ experiment suite")
    parser.add_argument("--profile", choices=("quick", "paper", "large"), default="paper")
    parser.add_argument("--experiment")
    parser.add_argument("--algorithm", action="append", dest="algorithms")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--on-infeasible", choices=("error", "skip", "record"), default="error")
    parser.add_argument("--fidelity-mode", choices=FIDELITY_CONSTRAINT_MODES)
    parser.add_argument("--fidelity-threshold", type=float)
    parser.add_argument(
        "--error-mitigation-mode", choices=ERROR_MITIGATION_MODES
    )
    parser.add_argument(
        "--symmetry-error-detection-fraction", type=float
    )
    parser.add_argument(
        "--zne-noise-scales", type=float, nargs="+"
    )
    parser.add_argument(
        "--entanglement-mode", choices=ENTANGLEMENT_SIMULATION_MODES
    )
    parser.add_argument(
        "--epr-purification-mode", choices=EPR_PURIFICATION_MODES,
        help="optional idealized EPR purification research report",
    )
    parser.add_argument("--epr-purification-rounds", type=int)
    parser.add_argument("--epr-purification-target", type=float)
    parser.add_argument("--epr-purification-max-rounds", type=int)
    parser.add_argument(
        "--quantum-epr-routing-mode",
        choices=QUANTUM_EPR_ROUTING_MODES,
        help="direct-channel EPR service policy used by TopoQ",
    )
    parser.add_argument(
        "--quantum-epr-reliability-window-ms",
        type=float,
        help="required reliability window for deadline_direct_backup",
    )
    parser.add_argument(
        "--quantum-link-latency-mode",
        choices=QUANTUM_LINK_LATENCY_MODES,
        help="delivered-rate or explicit physical-stage EPR timing",
    )
    parser.add_argument("--quantum-link-bsm-latency-ms", type=float)
    parser.add_argument("--quantum-link-switch-latency-ms", type=float)
    parser.add_argument("--quantum-link-signal-speed-m-s", type=float)
    parser.add_argument("--quantum-link-signal-traversals", type=int)
    parser.add_argument(
        "--coherence-event-mode",
        choices=COHERENCE_EVENT_MODES,
        help="C5 EPR holding policy: report or strict; gate batches remain enforced",
    )
    parser.add_argument(
        "--coherence-task-window-mode",
        choices=COHERENCE_TASK_WINDOW_MODES,
        help="deprecated alias for --coherence-event-mode",
    )
    parser.add_argument(
        "--epr-memory-capacity-mode",
        choices=ENTANGLEMENT_MEMORY_CAPACITY_MODES,
        help="C9 behavior when memory_capacity_epr is unspecified",
    )
    parser.add_argument(
        "--quantum-centrality-mode",
        choices=QUANTUM_CENTRALITY_MODES,
        help="Phase-1 quantum centrality policy",
    )
    parser.add_argument(
        "--interface-latency-mode",
        choices=INTERFACE_LATENCY_MODES,
        help="Phase-1 interface latency policy",
    )
    parser.add_argument(
        "--interface-signal-speed-m-s",
        type=float,
        help="signal speed required by paper_eq24",
    )
    parser.add_argument(
        "--topology-inverse-mode",
        choices=TOPOLOGY_INVERSE_MODES,
        help="Eq. (25) inverse-cost policy",
    )
    parser.add_argument(
        "--topology-reciprocal-epsilon",
        type=float,
        help="positive epsilon for normalized_reciprocal",
    )
    parser.add_argument(
        "--classical-routing-mode",
        choices=CLASSICAL_ROUTING_MODES,
        help="Phase-4 classical network routing policy",
    )
    parser.add_argument(
        "--classical-multipath-k",
        type=int,
        help="maximum paths for edge_disjoint_split",
    )
    parser.add_argument(
        "--classical-bandwidth-sharing-mode",
        choices=CLASSICAL_BANDWIDTH_SHARING_MODES,
        help="classical-link bandwidth allocation policy",
    )
    parser.add_argument(
        "--communication-overlap-mode",
        choices=COMMUNICATION_OVERLAP_MODES,
        help="paper Eq. (40) reporting proxy; never shortens physical tasks",
    )
    parser.add_argument(
        "--objective-mode", choices=OBJECTIVE_MODES,
        help="Phase-5 objective: joint (default), pure makespan, or makespan+penalty",
    )
    parser.add_argument("--objective-hard-penalty", type=float)
    parser.add_argument("--objective-w-makespan", type=float)
    parser.add_argument("--objective-w-quantum-comm", type=float)
    parser.add_argument("--objective-w-classical-comm", type=float)
    parser.add_argument("--objective-w-fidelity", type=float)
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--output")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing explicit --output file",
    )
    args = parser.parse_args(argv)

    if args.overwrite and not args.output:
        parser.error("--overwrite requires an explicit --output path")
    if (args.coherence_event_mode is not None
            and args.coherence_task_window_mode is not None
            and args.coherence_event_mode
            != args.coherence_task_window_mode):
        parser.error(
            "--coherence-event-mode and its deprecated alias "
            "--coherence-task-window-mode must match when both are supplied"
        )

    includes_topoq = args.algorithms is None or "TopoQ" in args.algorithms
    if (not args.list_experiments
            and args.profile in {"paper", "large"}
            and includes_topoq):
        if args.fidelity_mode is None:
            parser.error(
                "paper/large runs containing TopoQ require an explicit "
                "--fidelity-mode"
            )
    try:
        configured_purification = load_sim_params().get(
            "purification_parameters", {}
        )
        purification_mode = (
            args.epr_purification_mode
            or configured_purification.get(
                "default_mode", DEFAULT_PARAMS["epr_purification_mode"]
            )
        )
        purification_max_rounds = (
            args.epr_purification_max_rounds
            if args.epr_purification_max_rounds is not None
            else configured_purification.get(
                "max_rounds",
                DEFAULT_PARAMS["epr_purification_max_rounds"],
            )
        )
        fidelity_policy = validate_fidelity_policy(
            args.fidelity_mode or "report_only", args.fidelity_threshold
        )
        quantum_epr_policy = validate_quantum_epr_routing_policy(
            args.quantum_epr_routing_mode,
            args.quantum_epr_reliability_window_ms,
        )
        mitigation_policy = validate_error_mitigation_policy(
            args.error_mitigation_mode,
            args.symmetry_error_detection_fraction,
            args.zne_noise_scales,
        )
        purification_policy = validate_purification_policy(
            purification_mode,
            args.epr_purification_rounds,
            args.epr_purification_target,
            purification_max_rounds,
        )
        communication_overlap_mode = validate_communication_overlap_mode(
            args.communication_overlap_mode
        )
        phase5_objective_weights = validate_phase5_objective_weights({
            key: value for key, value in {
                "objective_w_makespan": args.objective_w_makespan,
                "objective_w_quantum_comm": args.objective_w_quantum_comm,
                "objective_w_classical_comm": (
                    args.objective_w_classical_comm
                ),
                "objective_w_fidelity": args.objective_w_fidelity,
            }.items() if value is not None
        })
        objective_mode = validate_objective_mode(args.objective_mode)
        coherence_event_mode = resolve_coherence_event_mode({
            "coherence_event_mode": args.coherence_event_mode,
            "coherence_task_window_mode": args.coherence_task_window_mode,
        })
        entanglement_memory_capacity_mode = (
            validate_entanglement_memory_capacity_mode(
                args.epr_memory_capacity_mode
            )
        )
        quantum_link_latency_policy = resolve_quantum_link_latency_policy({
            "quantum_link_latency_mode": args.quantum_link_latency_mode,
            "quantum_link_bsm_latency_ms": (
                args.quantum_link_bsm_latency_ms
            ),
            "quantum_link_switch_latency_ms": (
                args.quantum_link_switch_latency_ms
            ),
            "quantum_link_signal_speed_m_s": (
                args.quantum_link_signal_speed_m_s
            ),
            "quantum_link_signal_traversals": (
                args.quantum_link_signal_traversals
            ),
        }, load_sim_params())
    except ValueError as exc:
        parser.error(str(exc))
    experiment_params = {}
    if args.fidelity_mode is not None:
        experiment_params["fidelity_constraint_mode"] = fidelity_policy["mode"]
    if fidelity_policy["threshold"] is not None:
        experiment_params["fidelity_threshold"] = fidelity_policy["threshold"]
    if args.error_mitigation_mode is not None:
        experiment_params["error_mitigation_mode"] = (
            args.error_mitigation_mode
        )
    if args.symmetry_error_detection_fraction is not None:
        experiment_params["symmetry_error_detection_fraction"] = (
            args.symmetry_error_detection_fraction
        )
    if args.zne_noise_scales is not None:
        experiment_params["zne_noise_scales"] = args.zne_noise_scales
    if args.entanglement_mode is not None:
        experiment_params["entanglement_simulation_mode"] = args.entanglement_mode
    if args.epr_purification_mode is not None:
        experiment_params["epr_purification_mode"] = purification_policy[
            "mode"
        ]
    if args.epr_purification_rounds is not None:
        experiment_params["epr_purification_rounds"] = purification_policy[
            "rounds"
        ]
    if args.epr_purification_target is not None:
        experiment_params["epr_purification_target"] = purification_policy[
            "target_fidelity"
        ]
    if args.epr_purification_max_rounds is not None:
        experiment_params["epr_purification_max_rounds"] = purification_policy[
            "max_rounds"
        ]
    if args.quantum_epr_routing_mode is not None:
        experiment_params["quantum_epr_routing_mode"] = (
            args.quantum_epr_routing_mode
        )
    if args.quantum_epr_reliability_window_ms is not None:
        experiment_params["quantum_epr_reliability_window_ms"] = (
            args.quantum_epr_reliability_window_ms
        )
    quantum_latency_args = {
        "quantum_link_latency_mode": args.quantum_link_latency_mode,
        "quantum_link_bsm_latency_ms": args.quantum_link_bsm_latency_ms,
        "quantum_link_switch_latency_ms": (
            args.quantum_link_switch_latency_ms
        ),
        "quantum_link_signal_speed_m_s": (
            args.quantum_link_signal_speed_m_s
        ),
        "quantum_link_signal_traversals": (
            args.quantum_link_signal_traversals
        ),
    }
    if any(value is not None for value in quantum_latency_args.values()):
        experiment_params.update({
            key: value for key, value in quantum_latency_args.items()
            if value is not None
        })
    if (args.coherence_event_mode is not None
            or args.coherence_task_window_mode is not None):
        experiment_params["coherence_event_mode"] = coherence_event_mode
    if args.epr_memory_capacity_mode is not None:
        experiment_params["entanglement_memory_capacity_mode"] = (
            entanglement_memory_capacity_mode
        )
    if args.quantum_centrality_mode is not None:
        experiment_params["quantum_centrality_mode"] = (
            args.quantum_centrality_mode
        )
    if args.interface_latency_mode is not None:
        experiment_params["interface_latency_mode"] = (
            args.interface_latency_mode
        )
    if args.interface_signal_speed_m_s is not None:
        if args.interface_signal_speed_m_s <= 0:
            parser.error("--interface-signal-speed-m-s must be positive")
        experiment_params["interface_signal_speed_m_s"] = (
            args.interface_signal_speed_m_s
        )
    if args.topology_inverse_mode is not None:
        experiment_params["topology_inverse_mode"] = (
            args.topology_inverse_mode
        )
    if args.topology_reciprocal_epsilon is not None:
        if args.topology_reciprocal_epsilon <= 0:
            parser.error("--topology-reciprocal-epsilon must be positive")
        experiment_params["topology_reciprocal_epsilon"] = (
            args.topology_reciprocal_epsilon
        )
    if args.classical_routing_mode is not None:
        experiment_params["classical_routing_mode"] = (
            args.classical_routing_mode
        )
    if args.classical_multipath_k is not None:
        if args.classical_multipath_k < 1:
            parser.error("--classical-multipath-k must be positive")
        experiment_params["classical_multipath_k"] = (
            args.classical_multipath_k
        )
    if args.classical_bandwidth_sharing_mode is not None:
        experiment_params["classical_bandwidth_sharing_mode"] = (
            validate_classical_bandwidth_sharing_mode(
                args.classical_bandwidth_sharing_mode
            )
        )
    if args.communication_overlap_mode is not None:
        experiment_params["communication_overlap_mode"] = (
            communication_overlap_mode
        )
    objective_cli_values = {
        "objective_w_makespan": args.objective_w_makespan,
        "objective_w_quantum_comm": args.objective_w_quantum_comm,
        "objective_w_classical_comm": args.objective_w_classical_comm,
        "objective_w_fidelity": args.objective_w_fidelity,
    }
    experiment_params.update({
        key: value for key, value in objective_cli_values.items()
        if value is not None
    })
    if args.objective_mode is not None:
        experiment_params["objective_mode"] = validate_objective_mode(
            args.objective_mode
        )
    if args.objective_hard_penalty is not None:
        experiment_params["objective_hard_penalty"] = float(
            args.objective_hard_penalty
        )

    entanglement_mode = args.entanglement_mode or (
        "poisson" if args.profile in {"paper", "large"} else "deterministic"
    )

    if args.list_experiments:
        for spec in iter_experiments(enabled_only=False, include_legacy=True):
            feasibility = check_experiment_feasibility(
                spec.dag_id, spec.system_config, spec.qpu_limit
            )
            print(f"{spec.experiment_id:22s} feasible={str(feasibility['feasible']).lower():5s} "
                  f"tags={','.join(spec.tags)} {spec.note}")
        return {}

    run_context = _build_run_context()
    output_filename = (
        f"{args.experiment}.json" if args.experiment else "all_results.json"
    )
    output_path = _resolve_output_path(
        args.output, output_filename, run_context["run_id"]
    )
    try:
        _preflight_output_path(output_path, overwrite=args.overwrite)
    except (FileExistsError, IsADirectoryError) as exc:
        parser.error(str(exc))
    run_metadata = {
        **run_context,
        "output_path": output_path,
        "overwrite_enabled": bool(args.overwrite),
    }

    if args.experiment:
        result = run_experiment_spec(
            get_experiment(args.experiment), profile=args.profile,
            repetitions=args.repetitions, algorithms=args.algorithms,
            on_infeasible=args.on_infeasible,
            params=experiment_params,
        )
        result.setdefault("metadata", {})["run"] = run_metadata
        _write_json_result(
            output_path, result, overwrite=args.overwrite
        )
        print(f"Result saved to: {output_path}")
        return result

    print("=" * 60)
    print("TopoQ Experiment Suite")
    print("All data computed from dataset -- no hardcoded numbers")
    print("=" * 60)

    all_results = {}
    suite_repetitions = (
        args.repetitions
        or EXPERIMENT_PROFILES[args.profile]["repetitions"]
    )
    suite_algorithms = (
        list(args.algorithms)
        if args.algorithms is not None
        else ["Random", "Greedy", "Q-Only", "DQTetris", "TopoQ"]
    )
    suite_effective_parameters = dict(DEFAULT_PARAMS)
    suite_effective_parameters.update(experiment_params)
    suite_effective_parameters[
        "entanglement_simulation_mode"
    ] = entanglement_mode
    common_options = {
        "profile": args.profile,
        "repetitions": args.repetitions,
        "params": experiment_params,
    }
    comparison_options = {
        **common_options,
        "algorithms": args.algorithms,
    }
    all_results["metadata"] = {
        "profile": args.profile,
        "repetitions": args.repetitions,
        "algorithms": args.algorithms,
        "provenance": build_result_provenance(
            protocol={
                "kind": "full_experiment_suite",
                "profile": args.profile,
                "experiments": [
                    spec.experiment_id for spec in iter_experiments()
                ],
                "algorithms": suite_algorithms,
                "repetitions": suite_repetitions,
                "random_seed_base": RANDOM_SEED_BASE,
                "random_seeds": [
                    RANDOM_SEED_BASE + index
                    for index in range(suite_repetitions)
                ],
                "effective_parameters": suite_effective_parameters,
            },
            input_paths=collect_dataset_json_paths(),
            input_scope="complete_dataset_json_snapshot",
        ),
        "phase5_objective": {
            "formula": (
                "J=w_makespan*Cmax_hat+w_quantum*CQ_hat"
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
            "initial_temperature_fraction": DEFAULT_PARAMS[
                "T_SA_init_frac"
            ],
            "initial_temperature_formula": "T0=fraction*J0",
            "acceptance_probability": "exp(-(J_new-J_current)/T_SA)",
            "gamma_modulates_acceptance": False,
            "applies_to": ["TopoQ"] if includes_topoq else [],
        },
        "entanglement_simulation": {
            "mode": entanglement_mode,
            "random_seed_base": RANDOM_SEED_BASE,
            "distribution": (
                "exponential" if entanglement_mode == "poisson"
                else "scheduled_threshold_window"
            ),
            "sample_formula": (
                "T_ms=-ln(1-U)/lambda_hz*1000"
                if entanglement_mode == "poisson" else None
            ),
        },
        "quantum_epr_routing": {
            "configured_mode": (
                quantum_epr_policy["mode"]
            ),
            "reliability_window_ms": quantum_epr_policy[
                "reliability_window_ms"
            ],
            "rate_equation": "lambda_eff=sum(channel.lambda_hz)",
            "capacity_equation": "b_eff=sum(channel.max_concurrent_epr)",
            "wait_equation": (
                "T_ms=-ln(1-p_threshold)/lambda_eff_hz*1000"
            ),
            "applies_to": ["TopoQ"] if includes_topoq else [],
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
            "applies_to": (
                list(args.algorithms)
                if args.algorithms is not None
                else ["Random", "Greedy", "Q-Only", "DQTetris", "TopoQ"]
            ),
        },
        "fidelity_constraint": {
            "mode": fidelity_policy["mode"],
            "threshold": fidelity_policy["threshold"],
            "threshold_source": (
                "user" if fidelity_policy["mode"] == "hardware_explicit"
                else "dataset_by_workload"
            ),
            "enforced": fidelity_policy["mode"] != "report_only",
            "solution_class": (
                "strict_tahs_candidate"
                if fidelity_policy["mode"] != "report_only"
                else "scheduling_result_with_fidelity_diagnostics"
            ),
            "strict_tahs_feasible": (
                True if fidelity_policy["mode"] != "report_only" else None
            ),
            "c4_certification_scope": (
                "enforced"
                if fidelity_policy["mode"] != "report_only"
                else "not_certified"
            ),
            "applies_to": ["TopoQ"] if includes_topoq else [],
        },
        "error_mitigation": {
            "mode": mitigation_policy["mode"],
            "physical_fidelity_unchanged_by_zne": True,
            "symmetry_error_detection_fraction": mitigation_policy[
                "symmetry_error_detection_fraction"
            ],
            "zne_noise_scales": mitigation_policy["zne_noise_scales"],
            "zne_richardson_coefficients": (
                zne_sampling_overhead(
                    mitigation_policy["zne_noise_scales"]
                )[0]
                if mitigation_policy["zne_noise_scales"] is not None
                else None
            ),
            "zne_sampling_overhead": (
                zne_sampling_overhead(
                    mitigation_policy["zne_noise_scales"]
                )[1]
                if mitigation_policy["zne_noise_scales"] is not None
                else 1.0
            ),
            "work_metric": "expected_qpu_execution_work_ms_not_makespan",
            "applies_to": (
                list(args.algorithms)
                if args.algorithms is not None
                else ["Random", "Greedy", "Q-Only", "DQTetris", "TopoQ"]
            ),
        },
        "epr_purification": {
            **resolve_purification_policy(
                load_sim_params(),
                experiment_params,
            ),
            "physical_fidelity_unchanged": True,
            "scenario_metric": "purification_adjusted_fidelity",
            "cost_metric": (
                "expected_raw_epr_pairs_and_purification_attempts"
            ),
        },
        "epr_quality_diagnostics": {
            "default_threshold": build_epr_quality_diagnostics(
                load_sim_params(),
                resolve_purification_policy(
                    load_sim_params(), experiment_params
                ),
                load_sim_params()["fidelity_parameters"]["F_min"],
            ),
            "qaoa_threshold": build_epr_quality_diagnostics(
                load_sim_params(),
                resolve_purification_policy(
                    load_sim_params(), experiment_params
                ),
                load_sim_params()["fidelity_parameters"].get(
                    "F_min_qaoa",
                    load_sim_params()["fidelity_parameters"]["F_min"],
                ),
            ),
        },
        "coherence_constraint": {
            "batch_equation": "T_batch <= 0.95 * T_coh",
            "batch_safety_fraction": 0.05,
            "batch_enforced": True,
            "epr_holding_equation": (
                "max(T_consume - T_ready, 0) <= L_mem_ij"
            ),
            "epr_memory_lifetime_parameter": "epr_memory_lifetime_ms",
            "event_mode": coherence_event_mode,
            "task_window_equation": (
                "task_end - first_entanglement_start"
            ),
            "task_window_mode": "diagnostic_only",
            "applies_to": ["TopoQ"] if includes_topoq else [],
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
                entanglement_memory_capacity_mode
            ),
            "applies_to": ["TopoQ"] if includes_topoq else [],
        },
        "topology_characterization": {
            "quantum_centrality_mode": (
                args.quantum_centrality_mode
                or DEFAULT_PARAMS["quantum_centrality_mode"]
            ),
            "interface_latency_mode": (
                args.interface_latency_mode
                or DEFAULT_PARAMS["interface_latency_mode"]
            ),
            "interface_signal_speed_m_s": (
                args.interface_signal_speed_m_s
            ),
            "topology_inverse_mode": (
                args.topology_inverse_mode
                or DEFAULT_PARAMS["topology_inverse_mode"]
            ),
            "topology_reciprocal_epsilon": (
                (
                    args.topology_reciprocal_epsilon
                    if args.topology_reciprocal_epsilon is not None
                    else DEFAULT_PARAMS["topology_reciprocal_epsilon"]
                )
                if (args.topology_inverse_mode
                    or DEFAULT_PARAMS["topology_inverse_mode"])
                == "normalized_reciprocal" else None
            ),
            "applies_to": ["TopoQ"] if includes_topoq else [],
        },
        "classical_routing": {
            "mode": (
                args.classical_routing_mode
                or DEFAULT_PARAMS["classical_routing_mode"]
            ),
            "multipath_k": (
                (
                    args.classical_multipath_k
                    if args.classical_multipath_k is not None
                    else DEFAULT_PARAMS["classical_multipath_k"]
                )
                if (args.classical_routing_mode
                    or DEFAULT_PARAMS["classical_routing_mode"])
                == "edge_disjoint_split" else None
            ),
            "applies_to": ["TopoQ"] if includes_topoq else [],
        },
        "classical_bandwidth_sharing": {
            "mode": (
                args.classical_bandwidth_sharing_mode
                or DEFAULT_PARAMS["classical_bandwidth_sharing_mode"]
            ),
            "rate_allocation": "fixed_path_bottleneck",
            "shared_edge_policy": "serialize_conflicting_intervals",
            "edge_disjoint_subflows_can_overlap": True,
            "capacity_constraint": "sum_active_rates<=link_capacity",
            "applies_to": (
                list(args.algorithms)
                if args.algorithms is not None
                else ["Random", "Greedy", "Q-Only", "DQTetris", "TopoQ"]
            ),
        },
        "run": run_metadata,
    }

    # 1. Makespan comparison
    all_results["makespan"] = experiment_makespan_comparison(**comparison_options)

    # 2. Communication overhead
    all_results["communication"] = experiment_communication_overhead(**comparison_options)

    # 3. Fidelity
    all_results["fidelity"] = experiment_fidelity(**comparison_options)

    # 4. Scalability
    all_results["scalability"], all_results["scalability_efficiency"] = (
        experiment_scalability(**comparison_options)
    )

    run_topoq_specific = args.algorithms is None or "TopoQ" in args.algorithms
    if run_topoq_specific:
        # 5. Ablation
        all_results["ablation"] = experiment_ablation(**common_options)

        # 6. Parameter sensitivity
        all_results["parameter_sensitivity"] = (
            experiment_parameter_sensitivity(**common_options)
        )
    else:
        all_results["ablation"] = {}
        all_results["parameter_sensitivity"] = {}

    # Save results
    _write_json_result(
        output_path, all_results, overwrite=args.overwrite
    )

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary tables
    print_summary(all_results)

    return all_results
