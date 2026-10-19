"""
Experiment Functions
====================
Generate all data for the paper's tables and figures:

  Experiment 1 — Makespan Comparison (Table III)
  Experiment 2 — Communication Overhead Breakdown (Table IV)
  Experiment 3 — Fidelity Comparison (Table V)
  Experiment 4 — Scalability Analysis (Figure 4)
  Experiment 5 — Ablation Study (Table VI)
  Experiment 6 — Parameter Sensitivity Analysis
"""

import time as _time

from .config import (
    DEFAULT_ALGORITHMS,
    EXPERIMENT_PROFILES,
    FIDELITY_EXPERIMENT_IDS,
    REFERENCE_EXPERIMENT_ID,
    SCALABILITY_ALGORITHMS,
    SCALABILITY_CONFIGS,
    ABLATION_CONFIGS,
    get_experiment,
    iter_experiments,
    validate_phase5_objective_weights,
)
from .runner import run_with_repetitions


DEFAULT_WEIGHT_CONFIGS = (
    {"w1": 0.35, "w2": 0.25, "w3": 0.20, "w4": 0.20},
    {"w1": 0.50, "w2": 0.20, "w3": 0.15, "w4": 0.15},
    {"w1": 0.20, "w2": 0.35, "w3": 0.25, "w4": 0.20},
    {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25},
    {"w1": 0.40, "w2": 0.30, "w3": 0.15, "w4": 0.15},
)
DEFAULT_OBJECTIVE_WEIGHT_CONFIGS = (
    {
        "objective_w_makespan": 0.55,
        "objective_w_quantum_comm": 0.15,
        "objective_w_classical_comm": 0.10,
        "objective_w_fidelity": 0.20,
    },
    {
        "objective_w_makespan": 0.70,
        "objective_w_quantum_comm": 0.10,
        "objective_w_classical_comm": 0.05,
        "objective_w_fidelity": 0.15,
    },
    {
        "objective_w_makespan": 0.45,
        "objective_w_quantum_comm": 0.30,
        "objective_w_classical_comm": 0.10,
        "objective_w_fidelity": 0.15,
    },
    {
        "objective_w_makespan": 0.45,
        "objective_w_quantum_comm": 0.10,
        "objective_w_classical_comm": 0.30,
        "objective_w_fidelity": 0.15,
    },
    {
        "objective_w_makespan": 0.40,
        "objective_w_quantum_comm": 0.10,
        "objective_w_classical_comm": 0.10,
        "objective_w_fidelity": 0.40,
    },
    {
        "objective_w_makespan": 0.25,
        "objective_w_quantum_comm": 0.25,
        "objective_w_classical_comm": 0.25,
        "objective_w_fidelity": 0.25,
    },
)
DEFAULT_BETA_VALUES = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_GAMMA_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_SA_TEMPERATURE_VALUES = (0.05, 0.10, 0.15, 0.20)


def _simulation_params(profile, params=None):
    resolved = dict(params or {})
    resolved.setdefault(
        "entanglement_simulation_mode",
        "poisson" if profile in {"paper", "large"} else "deterministic",
    )
    return resolved


def _resolve_repetitions(profile, repetitions):
    if repetitions is not None:
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        return repetitions
    try:
        return EXPERIMENT_PROFILES[profile]["repetitions"]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENT_PROFILES))
        raise ValueError(f"Unknown experiment profile {profile!r}; choose from {choices}") from exc


def _resolve_spec(value):
    return get_experiment(value) if isinstance(value, str) else value


def _resolve_labeled_experiments(experiments=None, default=None, tags=None):
    source = default if experiments is None else experiments
    if source is None:
        source = iter_experiments(tags=tags)
    if isinstance(source, dict):
        return [(label, _resolve_spec(value)) for label, value in source.items()]
    resolved = []
    for value in source:
        spec = _resolve_spec(value)
        resolved.append((spec.label, spec))
    return resolved


def _run_spec(spec, algorithm, repetitions, params=None, ablation=None):
    merged_params = dict(spec.params)
    merged_params.update(params or {})
    merged_ablation = dict(spec.ablation)
    merged_ablation.update(ablation or {})
    return run_with_repetitions(
        spec.dag_id, spec.system_config, spec.network_topology, algorithm,
        params=merged_params or None, ablation=merged_ablation or None,
        n_rep=repetitions, n_qpu_limit=spec.qpu_limit,
    )


def experiment_makespan_comparison(experiments=None, algorithms=None,
                                   profile="paper", repetitions=None,
                                   params=None):
    """Table III: Makespan comparison across workloads and systems."""
    print("\n" + "=" * 60)
    print("Experiment 1: Makespan Comparison (Table III)")
    print("=" * 60)

    results = {}
    n_rep = _resolve_repetitions(profile, repetitions)
    params = _simulation_params(profile, params)

    for label, spec in _resolve_labeled_experiments(experiments, tags=("makespan",)):
        print(f"\n  Running {label}...")
        results[label] = {}
        for algo in tuple(algorithms or spec.algorithms or DEFAULT_ALGORITHMS):
            t0 = _time.time()
            metrics = _run_spec(spec, algo, n_rep, params=params)
            elapsed = _time.time() - t0
            results[label][algo] = metrics
            print(f"    {algo:12s}: makespan={metrics['makespan']:.1f} ms  ({elapsed:.1f}s)")

    return results


def experiment_communication_overhead(experiment=None, algorithms=None,
                                      profile="paper", repetitions=None,
                                      params=None):
    """Table IV: Communication overhead breakdown for VQE-200."""
    print("\n" + "=" * 60)
    print("Experiment 2: Communication Overhead Breakdown (Table IV)")
    print("=" * 60)

    spec = _resolve_spec(experiment or REFERENCE_EXPERIMENT_ID)
    selected_algorithms = tuple(algorithms or spec.algorithms or DEFAULT_ALGORITHMS)
    n_rep = _resolve_repetitions(profile, repetitions)
    params = _simulation_params(profile, params)
    results = {}

    for algo in selected_algorithms:
        print(f"  Running {algo}...")
        metrics = _run_spec(spec, algo, n_rep, params=params)
        results[algo] = metrics
        print(f"    Entanglement: {metrics['comm_entanglement']:.1f}, "
              f"QC: {metrics['comm_qc_transfer']:.1f}, "
              f"Classical: {metrics['comm_classical_net']:.1f}, "
              f"Total: {metrics['comm_total']:.1f}")

    return results


def experiment_fidelity(experiments=None, algorithms=None,
                        profile="paper", repetitions=None, params=None):
    """Table V: Fidelity comparison."""
    print("\n" + "=" * 60)
    print("Experiment 3: Fidelity Comparison (Table V)")
    print("=" * 60)

    results = {}
    n_rep = _resolve_repetitions(profile, repetitions)
    params = _simulation_params(profile, params)

    selected = _resolve_labeled_experiments(
        experiments, default=FIDELITY_EXPERIMENT_IDS
    )
    for label, spec in selected:
        print(f"\n  Running {label}...")
        results[label] = {}
        for algo in tuple(algorithms or spec.algorithms or DEFAULT_ALGORITHMS):
            metrics = _run_spec(spec, algo, n_rep, params=params)
            results[label][algo] = metrics["fidelity"] * 100
            print(f"    {algo:12s}: fidelity={metrics['fidelity']*100:.1f}%")

    return results


def experiment_scalability(configurations=None, algorithms=None,
                           profile="paper", repetitions=None, params=None):
    """Figure 4: Scalability analysis."""
    print("\n" + "=" * 60)
    print("Experiment 4: Scalability Analysis (Figure 4)")
    print("=" * 60)

    selected_algorithms = tuple(algorithms or SCALABILITY_ALGORITHMS)
    selected_configs = configurations or SCALABILITY_CONFIGS
    n_rep = _resolve_repetitions(profile, repetitions)
    params = _simulation_params(profile, params)
    results = {}

    for n_qpus, (dag_id, config, network) in selected_configs.items():
        print(f"\n  Running with {n_qpus} QPUs...")
        results[n_qpus] = {}

        # Limit QPU count for scalability testing
        # For n_qpus >= system's native count, no trimming needed
        for algo in selected_algorithms:
            merged_params = dict(params or {})
            metrics = run_with_repetitions(
                dag_id, config, network, algo,
                params=merged_params or None, n_rep=n_rep,
                n_qpu_limit=n_qpus,
            )
            results[n_qpus][algo] = metrics["makespan"]
            print(f"    {algo:12s}: makespan={metrics['makespan']:.1f} ms")

    # Compute scaling efficiency
    print("\n  Scaling Efficiency:")
    efficiency = {}
    baseline_n = min(results)
    for algo in selected_algorithms:
        t1 = results[baseline_n][algo]
        efficiency[algo] = {}
        for n in sorted(results.keys()):
            tn = results[n][algo]
            relative_scale = n / baseline_n
            eff = (t1 / (relative_scale * tn)) * 100 if tn > 0 else 0
            efficiency[algo][n] = eff
            print(f"    {algo:12s} @ {n:2d} QPUs: {eff:.1f}%")

    return results, efficiency


def experiment_ablation(experiment=None, ablation_configs=None,
                        profile="paper", repetitions=None, params=None):
    """Table VI: Ablation study for VQE-200."""
    print("\n" + "=" * 60)
    print("Experiment 5: Ablation Study (Table VI)")
    print("=" * 60)

    spec = _resolve_spec(experiment or REFERENCE_EXPERIMENT_ID)
    selected_ablations = ablation_configs or ABLATION_CONFIGS
    n_rep = _resolve_repetitions(profile, repetitions)
    params = _simulation_params(profile, params)
    results = {}

    for ablation_name, ablation_config in selected_ablations.items():
        print(f"  Running {ablation_name}...")
        metrics = _run_spec(
            spec, "TopoQ", n_rep, params=params, ablation=ablation_config
        )
        results[ablation_name] = metrics
        print(f"    Makespan={metrics['makespan']:.1f}, Comm={metrics['comm_total']:.1f}, Fidelity={metrics['fidelity']*100:.1f}%")

    return results


def experiment_parameter_sensitivity(experiment=None, weight_configs=None,
                                     objective_weight_configs=None,
                                     beta_values=None, gamma_values=None,
                                     sa_temperature_values=None,
                                     profile="paper", repetitions=None,
                                     params=None):
    """Parameter sensitivity analysis."""
    print("\n" + "=" * 60)
    print("Experiment 6: Parameter Sensitivity Analysis")
    print("=" * 60)

    spec = _resolve_spec(experiment or REFERENCE_EXPERIMENT_ID)
    n_rep = _resolve_repetitions(profile, repetitions)
    params = _simulation_params(profile, params)
    results = {}

    # Topology score weights
    selected_weight_configs = weight_configs or DEFAULT_WEIGHT_CONFIGS
    results["weight_configs"] = []
    for i, wc in enumerate(selected_weight_configs):
        local_params = dict(params or {})
        local_params.update(wc)
        metrics = _run_spec(spec, "TopoQ", n_rep, params=local_params)
        results["weight_configs"].append({"config": wc, "makespan": metrics["makespan"]})
        print(f"  Weight config {i+1}: makespan={metrics['makespan']:.1f}")

    # Phase-5 joint-objective weights. This is separate from topology w1-w4.
    selected_objective_configs = (
        DEFAULT_OBJECTIVE_WEIGHT_CONFIGS
        if objective_weight_configs is None
        else objective_weight_configs
    )
    results["objective_weight_configs"] = []
    for i, config in enumerate(selected_objective_configs):
        weights = validate_phase5_objective_weights(config)
        local_params = dict(params or {})
        local_params.update(weights)
        metrics = _run_spec(spec, "TopoQ", n_rep, params=local_params)
        summary = {
            "makespan": metrics["makespan"],
            "comm_total": metrics.get("comm_total"),
            "comm_entanglement": metrics.get("comm_entanglement"),
            "comm_classical_net": metrics.get("comm_classical_net"),
            "fidelity": metrics.get("fidelity"),
            "sa_best_objective": metrics.get("sa_best_objective"),
        }
        results["objective_weight_configs"].append({
            "config": weights,
            "metrics": summary,
        })
        print(
            f"  Objective config {i+1}: "
            f"J={summary['sa_best_objective']}, "
            f"makespan={summary['makespan']:.1f}, "
            f"comm={summary['comm_total']}, "
            f"fidelity={summary['fidelity']}"
        )

    # Beta
    results["beta"] = []
    for beta in beta_values or DEFAULT_BETA_VALUES:
        local_params = dict(params or {})
        local_params["beta"] = beta
        metrics = _run_spec(spec, "TopoQ", n_rep, params=local_params)
        results["beta"].append({"beta": beta, "makespan": metrics["makespan"]})
        print(f"  beta={beta}: makespan={metrics['makespan']:.1f}")

    # Gamma
    results["gamma"] = []
    for gamma in gamma_values or DEFAULT_GAMMA_VALUES:
        local_params = dict(params or {})
        local_params["gamma"] = gamma
        metrics = _run_spec(spec, "TopoQ", n_rep, params=local_params)
        results["gamma"].append({"gamma": gamma, "makespan": metrics["makespan"]})
        print(f"  gamma={gamma}: makespan={metrics['makespan']:.1f}")

    # SA temperature
    results["sa_temp"] = []
    for t_frac in sa_temperature_values or DEFAULT_SA_TEMPERATURE_VALUES:
        local_params = dict(params or {})
        local_params["T_SA_init_frac"] = t_frac
        metrics = _run_spec(spec, "TopoQ", n_rep, params=local_params)
        results["sa_temp"].append({"t_frac": t_frac, "makespan": metrics["makespan"]})
        print(f"  T_SA_frac={t_frac}: makespan={metrics['makespan']:.1f}")

    return results
