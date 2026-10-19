"""
Configuration & Data Loading
============================
Algorithm parameters, dataset paths, data loaders, and experiment configurations.
"""

import json
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

# ------------------------------------------------------------------
# Paths — resolve relative to the project root (parent of this package)
# ------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(_BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(_BASE_DIR, "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_DAG_DIR_CONTEXT = ContextVar("topoq_dag_dir", default=None)
_CIRCUIT_DIRS_CONTEXT = ContextVar("topoq_circuit_dirs", default=())
_SYSTEM_DATA_ROOT_CONTEXT = ContextVar(
    "topoq_system_data_root", default=None
)
EXTERNAL_DAG_DIR_ENV = "TOPOQ_WORKLOAD_DAG_DIR"
EXTERNAL_CIRCUIT_DIRS_ENV = "TOPOQ_QUANTUM_CIRCUIT_DIRS"
EXTERNAL_SYSTEM_DATA_ROOT_ENV = "TOPOQ_SYSTEM_DATA_ROOT"

# ------------------------------------------------------------------
# Default algorithm parameters (from paper Section 4)
# ------------------------------------------------------------------
DEFAULT_PARAMS = {
    "w1": 0.35, "w2": 0.25, "w3": 0.20, "w4": 0.20,  # topology score weights
    # Phase-1 quantum centrality: paper_eq22 | effective_service.
    "quantum_centrality_mode": "paper_eq22",
    # Phase-1 interface latency: readout_proxy | paper_eq24.
    "interface_latency_mode": "readout_proxy",
    # Required by paper_eq24 unless supplied by interface sim parameters.
    "interface_signal_speed_m_s": None,
    # Eq. (25) inverse-cost policy.
    "topology_inverse_mode": "bounded_complement",
    "topology_reciprocal_epsilon": 1.0,
    # Phase-4 classical routing policy.
    "classical_routing_mode": "single_widest",
    "classical_multipath_k": 2,
    # Classical-link capacity allocation is separate from path selection.
    # The validated model reserves a fixed bottleneck rate and serializes
    # transfers whose paths share at least one classical link.
    "classical_bandwidth_sharing_mode": "exclusive_reservation",
    # Paper Eq. (40) overlap is reporting-only because the workload data does
    # not expose communication/computation sub-intervals that can be safely
    # overlapped on the physical resource timeline.
    "communication_overlap_mode": "paper_fixed_report",
    # Task execution-time semantics. legacy_proxy preserves the historical
    # per-task estimated_time_ms values. paper_equation evaluates the revised
    # manuscript model on the actually assigned resources.
    "execution_time_mode": "legacy_proxy",
    # QPU calibration source used by timing, routing, and fidelity.
    "qpu_calibration_mode": "configured_qpu_average",
    "ibm_cx_direction_policy": "conservative_max",
    "ibm_missing_calibration_policy": "error",
    # Repetitions used only by quantum tasks explicitly marked shot_scaled.
    "quantum_shots": 1000,
    # Effective fraction of a classical node's configured peak throughput.
    # The revised-paper default is the idealized kappa=1 case in w/(kappa p).
    "classical_compute_efficiency": 1.0,
    "alpha": 0.7,         # interaction graph edge weight
    "beta": 0.5,          # partitioning link cost weight
    "gamma": 0.3,         # decoherence penalty
    "mu1": 0.5, "mu2": 0.3, "mu3": 0.2,  # gate scheduling priority weights
    "p_threshold": 0.95,  # entanglement success threshold
    "T_safety_frac": 0.05, # Eq. (36): T_batch <= 0.95 * T_coh
    "T_SA_init_frac": 0.15,# SA initial temperature fraction of objective J0
    "SA_cooling": 0.97,   # SA cooling rate (slower = more exploration)
    "SA_max_non_improve": 300,  # SA max non-improving iterations
    "max_iterations": 3,  # feedback loop iterations
    # Implementation-level Phase-2 assignment heuristic.  Components are
    # normalized across candidates before these weights are applied.
    "assignment_w_execution": 0.20,
    "assignment_w_communication": 0.25,
    "assignment_w_fidelity": 0.25,
    "assignment_w_load": 0.15,
    "assignment_w_topology": 0.10,
    "assignment_w_capacity": 0.05,
    # Implementation-level search controls for capacity-feasible QPU sets.
    # These are not paper-model weights; they bound deterministic fallback
    # search when the lowest-cost connected set cannot be partitioned.
    "qpu_candidate_beam_width": 32,
    "qpu_partition_max_attempts": 64,
    "qpu_partition_extra_qpus": 1,
    # C10 routing policy: auto | static_aggregate | sequence_exact.
    # auto uses exact routing only when an ordered two-qubit gate sequence exists.
    "physical_routing_mode": "auto",
    # deterministic keeps the scheduled threshold window; poisson replays
    # exponential EPR generation for paper-style stochastic simulation.
    "entanglement_simulation_mode": "deterministic",
    # Quantum EPR service policy: physical modes use configured direct
    # channels; paper modes preserve the two inconsistent manuscript proxies.
    "quantum_epr_routing_mode": "single_channel",
    # EPR reservation timing. ``proactive`` preserves the historical 90%
    # lead-time heuristic; the other modes are explicit diagnostic policies.
    "entanglement_reservation_mode": "proactive",
    "epr_proactive_lead_fraction": 0.90,
    "epr_jit_memory_fraction": 0.25,
    "epr_adaptive_2q_gate_threshold": 5000,
    "epr_adaptive_high_gate_mode": "just_in_time",
    # Required only by deadline_direct_backup. The scheduler selects the
    # smallest direct-channel set whose Poisson success probability reaches
    # p_threshold within this explicit reliability window.
    "quantum_epr_reliability_window_ms": None,
    # Quantum-link timing semantics. delivered_pair_rate treats rate_hz as an
    # end-to-end ready-pair service rate. physical_stage adds explicit stages.
    "quantum_link_latency_mode": "delivered_pair_rate",
    "quantum_link_bsm_latency_ms": None,
    "quantum_link_switch_latency_ms": None,
    "quantum_link_signal_speed_m_s": None,
    "quantum_link_signal_traversals": None,
    # C4 policy: report_only | paper_strict | hardware_explicit.
    # hardware_explicit requires a user-provided fidelity_threshold.
    "fidelity_constraint_mode": "report_only",
    "fidelity_threshold": None,
    # Error mitigation reporting is algorithm-independent. ZNE affects
    # estimator sampling cost, not the physical state-fidelity constraint.
    "error_mitigation_mode": "none",
    "symmetry_error_detection_fraction": None,
    "zne_noise_scales": None,
    # Optional idealized BBPSSW research reporting. This does not change
    # physical fidelity unless a later experiment explicitly consumes the
    # separately reported scenario metric.
    "epr_purification_mode": "none",
    "epr_purification_rounds": None,
    "epr_purification_target": None,
    "epr_purification_max_rounds": 8,
    # C5 event policy: report | strict. Gate-batch coherence remains enforced
    # in both modes. The legacy task-window name is retained as an alias.
    "coherence_event_mode": "report",
    "coherence_task_window_mode": "report",
    # C9 memory-capacity certification policy: report | strict. Generation
    # parallelism remains enforced because max_concurrent_epr is required.
    "entanglement_memory_capacity_mode": "report",
    # Phase-5 objective mode and weights (normalized components).
    # joint: original weighted surrogate; makespan: pure Cmax (E13);
    # makespan_penalty: pure Cmax plus a hard-feasibility penalty.
    "objective_mode": "joint",
    "objective_hard_penalty": 1000.0,
    "objective_w_makespan": 0.55,
    "objective_w_quantum_comm": 0.15,
    "objective_w_classical_comm": 0.10,
    "objective_w_fidelity": 0.20,
}

QUICK_REPETITIONS = 1
PAPER_REPETITIONS = 3  # paper methodology: three independent random seeds
LARGE_REPETITIONS = 30
N_REPETITIONS = PAPER_REPETITIONS
RANDOM_SEED_BASE = 42

EXPERIMENT_PROFILES = {
    "quick": {"repetitions": QUICK_REPETITIONS, "description": "Fast workflow check"},
    "paper": {"repetitions": PAPER_REPETITIONS, "description": "Paper methodology"},
    "large": {"repetitions": LARGE_REPETITIONS, "description": "Large-sample statistics"},
}

FIDELITY_CONSTRAINT_MODES = (
    "report_only", "paper_strict", "hardware_explicit"
)
ERROR_MITIGATION_MODES = (
    "none",
    "symmetry_verification",
    "zne",
    "zne_symmetry",
)
EPR_PURIFICATION_MODES = (
    "none",
    "bbpssw_fixed_rounds",
    "bbpssw_target",
)
PHYSICAL_ROUTING_MODES = ("auto", "static_aggregate", "sequence_exact")
ENTANGLEMENT_SIMULATION_MODES = ("deterministic", "poisson")
QUANTUM_EPR_ROUTING_MODES = (
    "single_channel",
    "parallel_channels",
    "deadline_direct_backup",
    "paper_fixed_25",
    "paper_algorithm_145",
)
ENTANGLEMENT_RESERVATION_MODES = (
    "proactive",
    "on_demand",
    "just_in_time",
    "adaptive",
)
EPR_ADAPTIVE_HIGH_GATE_MODES = ("on_demand", "just_in_time")
QUANTUM_LINK_LATENCY_MODES = (
    "delivered_pair_rate",
    "physical_stage",
)
COHERENCE_EVENT_MODES = ("report", "strict")
# Backward-compatible public alias for existing experiment scripts.
COHERENCE_TASK_WINDOW_MODES = COHERENCE_EVENT_MODES
ENTANGLEMENT_MEMORY_CAPACITY_MODES = ("report", "strict")
QUANTUM_CENTRALITY_MODES = ("paper_eq22", "effective_service")
INTERFACE_LATENCY_MODES = ("readout_proxy", "paper_eq24")
TOPOLOGY_INVERSE_MODES = (
    "bounded_complement",
    "normalized_reciprocal",
    "paper_raw_strict",
)
CLASSICAL_ROUTING_MODES = (
    "single_widest",
    "paper_fixed_15",
    "edge_disjoint_split",
)
CLASSICAL_BANDWIDTH_SHARING_MODES = (
    "exclusive_reservation",
)
COMMUNICATION_OVERLAP_MODES = (
    "none",
    "paper_fixed_report",
)
EXECUTION_TIME_MODES = ("legacy_proxy", "paper_equation")
QPU_CALIBRATION_MODES = (
    "configured_qpu_average",
    "ibm_chip_average",
    "ibm_physical_qubit",
)
IBM_CX_DIRECTION_POLICIES = ("conservative_max", "mean", "best")
IBM_MISSING_CALIBRATION_POLICIES = ("error", "configured_fallback")
PAPER_ENTANGLEMENT_EFFECTIVE_FACTOR = 0.70
PAPER_CLASSICAL_EFFECTIVE_FACTOR = 0.75
PHASE5_OBJECTIVE_WEIGHT_KEYS = (
    "objective_w_makespan",
    "objective_w_quantum_comm",
    "objective_w_classical_comm",
    "objective_w_fidelity",
)
OBJECTIVE_MODES = ("joint", "makespan", "makespan_penalty")


def validate_objective_mode(mode=None):
    """Validate the Phase-5 objective mode."""
    mode = mode or DEFAULT_PARAMS["objective_mode"]
    if mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"Unknown objective_mode {mode!r}; choose from {OBJECTIVE_MODES}"
        )
    return mode
# A zero initial component has no relative denominator. These explicit native-
# unit scales preserve the existing numerical behavior while keeping every
# normalized Phase-5 component dimensionless.
PHASE5_OBJECTIVE_ZERO_REFERENCE_SCALES = {
    "makespan": 1.0,          # milliseconds
    "entanglement_wait": 1.0, # milliseconds
    "remote_gates": 1.0,      # gate count
    "classical_comm": 1.0,    # milliseconds
    "fidelity_loss": 1.0,     # dimensionless -log(F)
}
PHASE5_OBJECTIVE_ZERO_REFERENCE_UNITS = {
    "makespan": "ms",
    "entanglement_wait": "ms",
    "remote_gates": "gate",
    "classical_comm": "ms",
    "fidelity_loss": "dimensionless_log_loss",
}


def validate_phase5_objective_weights(params=None):
    """Return finite, non-negative Phase-5 weights summing to one."""
    params = params or {}
    weights = {
        key: params.get(key, DEFAULT_PARAMS[key])
        for key in PHASE5_OBJECTIVE_WEIGHT_KEYS
    }
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite non-negative number.")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be a finite non-negative number.")
        weights[key] = value
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "Phase-5 objective weights must sum to 1.0; "
            f"received {total:.12g}."
        )
    return weights


def validate_execution_time_policy(mode=None, quantum_shots=None,
                                   classical_compute_efficiency=None):
    """Validate the shared task execution-time model and its parameters."""
    mode = mode or DEFAULT_PARAMS["execution_time_mode"]
    if mode not in EXECUTION_TIME_MODES:
        choices = ", ".join(EXECUTION_TIME_MODES)
        raise ValueError(
            f"Unknown execution time mode {mode!r}; choose from {choices}"
        )

    if quantum_shots is None:
        quantum_shots = DEFAULT_PARAMS["quantum_shots"]
    if (isinstance(quantum_shots, bool)
            or not isinstance(quantum_shots, (int, float))):
        raise ValueError("quantum_shots must be a positive integer.")
    quantum_shots = float(quantum_shots)
    if (not math.isfinite(quantum_shots) or quantum_shots <= 0.0
            or not quantum_shots.is_integer()):
        raise ValueError("quantum_shots must be a positive integer.")

    if classical_compute_efficiency is None:
        classical_compute_efficiency = DEFAULT_PARAMS[
            "classical_compute_efficiency"
        ]
    if (isinstance(classical_compute_efficiency, bool)
            or not isinstance(classical_compute_efficiency, (int, float))):
        raise ValueError(
            "classical_compute_efficiency must be in the interval (0, 1]."
        )
    classical_compute_efficiency = float(classical_compute_efficiency)
    if (not math.isfinite(classical_compute_efficiency)
            or not 0.0 < classical_compute_efficiency <= 1.0):
        raise ValueError(
            "classical_compute_efficiency must be in the interval (0, 1]."
        )

    return {
        "mode": mode,
        "quantum_shots": int(quantum_shots),
        "classical_compute_efficiency": classical_compute_efficiency,
    }


def validate_qpu_calibration_policy(mode=None, cx_direction_policy=None,
                                    missing_calibration_policy=None):
    """Validate IBM calibration resolution and explicit fallback policy."""
    mode = mode or DEFAULT_PARAMS["qpu_calibration_mode"]
    cx_direction_policy = (
        cx_direction_policy or DEFAULT_PARAMS["ibm_cx_direction_policy"]
    )
    missing_calibration_policy = (
        missing_calibration_policy
        or DEFAULT_PARAMS["ibm_missing_calibration_policy"]
    )
    if mode not in QPU_CALIBRATION_MODES:
        raise ValueError(
            f"Unknown QPU calibration mode {mode!r}; choose from "
            + ", ".join(QPU_CALIBRATION_MODES)
        )
    if cx_direction_policy not in IBM_CX_DIRECTION_POLICIES:
        raise ValueError(
            f"Unknown IBM CX direction policy {cx_direction_policy!r}; "
            "choose from " + ", ".join(IBM_CX_DIRECTION_POLICIES)
        )
    if missing_calibration_policy not in IBM_MISSING_CALIBRATION_POLICIES:
        raise ValueError(
            "Unknown IBM missing-calibration policy "
            f"{missing_calibration_policy!r}; choose from "
            + ", ".join(IBM_MISSING_CALIBRATION_POLICIES)
        )
    return {
        "mode": mode,
        "cx_direction_policy": cx_direction_policy,
        "missing_calibration_policy": missing_calibration_policy,
    }


def validate_coherence_event_mode(mode=None):
    """Validate the C5 EPR-holding enforcement policy."""
    mode = mode or DEFAULT_PARAMS["coherence_event_mode"]
    if mode not in COHERENCE_EVENT_MODES:
        choices = ", ".join(COHERENCE_EVENT_MODES)
        raise ValueError(
            f"Unknown coherence event mode {mode!r}; choose from {choices}"
        )
    return mode


def resolve_coherence_event_mode(params=None):
    """Resolve the canonical C5 mode and its deprecated task-window alias."""
    params = params or {}
    canonical = params.get("coherence_event_mode")
    legacy = params.get("coherence_task_window_mode")
    default = DEFAULT_PARAMS["coherence_event_mode"]
    if canonical is not None and legacy is not None and canonical != legacy:
        # DEFAULT_PARAMS contains both names for compatibility. Treat changing
        # only one of them from the default as an intentional override.
        if canonical == default:
            canonical = legacy
        elif legacy != default:
            raise ValueError(
                "coherence_event_mode and coherence_task_window_mode "
                "must match when both are supplied."
            )
    return validate_coherence_event_mode(
        canonical if canonical is not None else legacy
    )


def validate_entanglement_memory_capacity_mode(mode=None):
    """Validate the C9 EPR memory-capacity certification policy."""
    mode = mode or DEFAULT_PARAMS["entanglement_memory_capacity_mode"]
    if mode not in ENTANGLEMENT_MEMORY_CAPACITY_MODES:
        choices = ", ".join(ENTANGLEMENT_MEMORY_CAPACITY_MODES)
        raise ValueError(
            f"Unknown entanglement memory capacity mode {mode!r}; "
            f"choose from {choices}"
        )
    return mode


def validate_quantum_link_latency_policy(
        mode=None, bsm_latency_ms=None, switch_latency_ms=None,
        signal_speed_m_s=None, signal_traversals=None):
    """Validate explicit quantum-link timing semantics and stage inputs."""
    mode = mode or DEFAULT_PARAMS["quantum_link_latency_mode"]
    if mode not in QUANTUM_LINK_LATENCY_MODES:
        choices = ", ".join(QUANTUM_LINK_LATENCY_MODES)
        raise ValueError(
            f"Unknown quantum link latency mode {mode!r}; "
            f"choose from {choices}"
        )
    raw_values = {
        "bsm_latency_ms": bsm_latency_ms,
        "switch_latency_ms": switch_latency_ms,
        "signal_speed_m_s": signal_speed_m_s,
        "signal_traversals": signal_traversals,
    }
    if mode == "delivered_pair_rate" and any(
            value is not None for value in raw_values.values()):
        raise ValueError(
            "Explicit quantum-link physical-stage parameters require "
            "quantum_link_latency_mode='physical_stage'."
        )

    def optional_finite(name, value, positive=False):
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric.") from exc
        if (not math.isfinite(parsed)
                or (parsed <= 0.0 if positive else parsed < 0.0)):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be a finite {qualifier} value.")
        return parsed

    bsm_latency_ms = optional_finite(
        "quantum_link_bsm_latency_ms", bsm_latency_ms
    )
    switch_latency_ms = optional_finite(
        "quantum_link_switch_latency_ms", switch_latency_ms
    )
    signal_speed_m_s = optional_finite(
        "quantum_link_signal_speed_m_s", signal_speed_m_s, positive=True
    )
    if signal_traversals is not None:
        parsed_traversals = optional_finite(
            "quantum_link_signal_traversals",
            signal_traversals,
            positive=True,
        )
        if not parsed_traversals.is_integer():
            raise ValueError(
                "quantum_link_signal_traversals must be a positive integer."
            )
        signal_traversals = int(parsed_traversals)
    return {
        "mode": mode,
        "bsm_latency_ms": bsm_latency_ms,
        "switch_latency_ms": switch_latency_ms,
        "signal_speed_m_s": signal_speed_m_s,
        "signal_traversals": signal_traversals,
    }


def validate_quantum_epr_routing_mode(mode=None):
    """Validate the public quantum EPR direct-channel service policy."""
    mode = mode or DEFAULT_PARAMS["quantum_epr_routing_mode"]
    if mode not in QUANTUM_EPR_ROUTING_MODES:
        choices = ", ".join(QUANTUM_EPR_ROUTING_MODES)
        raise ValueError(
            f"Unknown quantum EPR routing mode {mode!r}; "
            f"choose from {choices}"
        )
    return mode


def validate_quantum_epr_routing_policy(mode=None, reliability_window_ms=None):
    """Validate the EPR service mode and its optional reliability window."""
    mode = validate_quantum_epr_routing_mode(mode)
    if mode == "deadline_direct_backup":
        if reliability_window_ms is None:
            raise ValueError(
                "deadline_direct_backup requires "
                "quantum_epr_reliability_window_ms."
            )
        try:
            reliability_window_ms = float(reliability_window_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "quantum_epr_reliability_window_ms must be a positive "
                "finite number."
            ) from exc
        if (not math.isfinite(reliability_window_ms)
                or reliability_window_ms <= 0):
            raise ValueError(
                "quantum_epr_reliability_window_ms must be a positive "
                "finite number."
            )
    elif reliability_window_ms is not None:
        raise ValueError(
            "quantum_epr_reliability_window_ms may only be supplied with "
            "deadline_direct_backup."
        )
    return {
        "mode": mode,
        "reliability_window_ms": reliability_window_ms,
    }


def validate_entanglement_reservation_policy(
        mode=None, proactive_lead_fraction=None, jit_memory_fraction=None,
        adaptive_2q_gate_threshold=None, adaptive_high_gate_mode=None):
    """Validate task-level EPR reservation timing policies.

    The policy changes only when EPR generation is reserved relative to a
    known task boundary; it does not alter the Poisson generation model.
    """
    mode = mode or DEFAULT_PARAMS["entanglement_reservation_mode"]
    if mode not in ENTANGLEMENT_RESERVATION_MODES:
        choices = ", ".join(ENTANGLEMENT_RESERVATION_MODES)
        raise ValueError(
            f"Unknown entanglement reservation mode {mode!r}; "
            f"choose from {choices}"
        )

    lead = DEFAULT_PARAMS["epr_proactive_lead_fraction"] \
        if proactive_lead_fraction is None else proactive_lead_fraction
    jit_fraction = DEFAULT_PARAMS["epr_jit_memory_fraction"] \
        if jit_memory_fraction is None else jit_memory_fraction
    threshold = DEFAULT_PARAMS["epr_adaptive_2q_gate_threshold"] \
        if adaptive_2q_gate_threshold is None else adaptive_2q_gate_threshold
    high_gate_mode = DEFAULT_PARAMS["epr_adaptive_high_gate_mode"] \
        if adaptive_high_gate_mode is None else adaptive_high_gate_mode

    try:
        lead = float(lead)
        jit_fraction = float(jit_fraction)
        threshold = int(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "EPR reservation fractions must be numeric and the adaptive "
            "gate threshold must be an integer."
        ) from exc
    if not math.isfinite(lead) or not 0.0 <= lead <= 1.0:
        raise ValueError("epr_proactive_lead_fraction must be in [0, 1].")
    if not math.isfinite(jit_fraction) or not 0.0 < jit_fraction <= 1.0:
        raise ValueError("epr_jit_memory_fraction must be in (0, 1].")
    if threshold < 0:
        raise ValueError(
            "epr_adaptive_2q_gate_threshold must be non-negative."
        )
    if high_gate_mode not in EPR_ADAPTIVE_HIGH_GATE_MODES:
        choices = ", ".join(EPR_ADAPTIVE_HIGH_GATE_MODES)
        raise ValueError(
            f"Unknown adaptive high-gate mode {high_gate_mode!r}; "
            f"choose from {choices}"
        )
    return {
        "mode": mode,
        "proactive_lead_fraction": lead,
        "jit_memory_fraction": jit_fraction,
        "adaptive_2q_gate_threshold": threshold,
        "adaptive_high_gate_mode": high_gate_mode,
    }


def validate_error_mitigation_policy(mode=None, symmetry_fraction=None,
                                     zne_noise_scales=None):
    """Validate the algorithm-independent mitigation reporting policy."""
    mode = mode or DEFAULT_PARAMS["error_mitigation_mode"]
    if mode not in ERROR_MITIGATION_MODES:
        choices = ", ".join(ERROR_MITIGATION_MODES)
        raise ValueError(
            f"Unknown error mitigation mode {mode!r}; choose from {choices}"
        )

    uses_symmetry = mode in {"symmetry_verification", "zne_symmetry"}
    if uses_symmetry:
        if symmetry_fraction is None:
            raise ValueError(
                f"{mode} requires symmetry_error_detection_fraction."
            )
        try:
            symmetry_fraction = float(symmetry_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "symmetry_error_detection_fraction must be in [0, 1]."
            ) from exc
        if (not math.isfinite(symmetry_fraction)
                or not 0.0 <= symmetry_fraction <= 1.0):
            raise ValueError(
                "symmetry_error_detection_fraction must be in [0, 1]."
            )
    elif symmetry_fraction is not None:
        raise ValueError(
            "symmetry_error_detection_fraction may only be supplied with "
            "symmetry_verification or zne_symmetry."
        )

    uses_zne = mode in {"zne", "zne_symmetry"}
    if uses_zne:
        if (isinstance(zne_noise_scales, (str, bytes))
                or zne_noise_scales is None):
            raise ValueError(
                f"{mode} requires at least two zne_noise_scales."
            )
        try:
            normalized_scales = tuple(float(value)
                                      for value in zne_noise_scales)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "zne_noise_scales must be finite positive numbers."
            ) from exc
        if len(normalized_scales) < 2:
            raise ValueError(
                f"{mode} requires at least two zne_noise_scales."
            )
        if any(not math.isfinite(value) or value <= 0
               for value in normalized_scales):
            raise ValueError(
                "zne_noise_scales must be finite positive numbers."
            )
        if any(right <= left for left, right in zip(
                normalized_scales, normalized_scales[1:])):
            raise ValueError(
                "zne_noise_scales must be strictly increasing and distinct."
            )
        if not math.isclose(
                normalized_scales[0], 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("zne_noise_scales must start with 1.0.")
        zne_noise_scales = normalized_scales
    elif zne_noise_scales is not None:
        raise ValueError(
            "zne_noise_scales may only be supplied with zne or zne_symmetry."
        )

    return {
        "mode": mode,
        "symmetry_error_detection_fraction": symmetry_fraction,
        "zne_noise_scales": zne_noise_scales,
    }


def validate_purification_policy(mode=None, rounds=None, target=None,
                                 max_rounds=None):
    """Validate the explicit EPR purification research policy."""
    mode = mode or DEFAULT_PARAMS["epr_purification_mode"]
    if mode not in EPR_PURIFICATION_MODES:
        choices = ", ".join(EPR_PURIFICATION_MODES)
        raise ValueError(
            f"Unknown EPR purification mode {mode!r}; choose from {choices}"
        )
    def positive_integer(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a positive integer."
            ) from exc
        if (not math.isfinite(numeric) or not numeric.is_integer()
                or numeric < 1):
            raise ValueError(f"{name} must be a positive integer.")
        return int(numeric)

    if rounds is not None:
        rounds = positive_integer(
            rounds, "epr_purification_rounds"
        )
    if target is not None:
        if isinstance(target, bool):
            raise ValueError(
                "epr_purification_target must be a finite number in (0.5, 1]."
            )
        try:
            target = float(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "epr_purification_target must be a finite number in (0.5, 1]."
            ) from exc
        if not math.isfinite(target) or not 0.5 < target <= 1.0:
            raise ValueError(
                "epr_purification_target must be a finite number in (0.5, 1]."
            )
    if max_rounds is None:
        max_rounds = DEFAULT_PARAMS["epr_purification_max_rounds"]
    max_rounds = positive_integer(
        max_rounds, "epr_purification_max_rounds"
    )
    if mode == "none":
        if rounds is not None or target is not None:
            raise ValueError(
                "purification rounds/target require an explicit BBPSSW mode."
            )
    elif mode == "bbpssw_fixed_rounds":
        if rounds is None:
            raise ValueError(
                "bbpssw_fixed_rounds requires epr_purification_rounds."
            )
        if target is not None:
            raise ValueError(
                "bbpssw_fixed_rounds does not accept epr_purification_target."
            )
        if rounds > max_rounds:
            raise ValueError(
                "epr_purification_rounds cannot exceed max_rounds."
            )
    else:
        if target is None:
            raise ValueError(
                "bbpssw_target requires epr_purification_target."
            )
        if rounds is not None:
            raise ValueError(
                "bbpssw_target does not accept epr_purification_rounds."
            )
    return {
        "mode": mode,
        "rounds": rounds,
        "target_fidelity": target,
        "max_rounds": max_rounds,
    }


def validate_quantum_centrality_mode(mode=None):
    """Validate the public Phase-1 quantum-centrality policy."""
    mode = mode or DEFAULT_PARAMS["quantum_centrality_mode"]
    if mode not in QUANTUM_CENTRALITY_MODES:
        choices = ", ".join(QUANTUM_CENTRALITY_MODES)
        raise ValueError(
            f"Unknown quantum centrality mode {mode!r}; "
            f"choose from {choices}"
        )
    return mode


def validate_interface_latency_mode(mode=None):
    """Validate the public Phase-1 interface-latency policy."""
    mode = mode or DEFAULT_PARAMS["interface_latency_mode"]
    if mode not in INTERFACE_LATENCY_MODES:
        choices = ", ".join(INTERFACE_LATENCY_MODES)
        raise ValueError(
            f"Unknown interface latency mode {mode!r}; "
            f"choose from {choices}"
        )
    return mode


def validate_topology_inverse_policy(mode=None, epsilon=None):
    """Validate the public Eq. (25) inverse-cost policy."""
    mode = mode or DEFAULT_PARAMS["topology_inverse_mode"]
    if mode not in TOPOLOGY_INVERSE_MODES:
        choices = ", ".join(TOPOLOGY_INVERSE_MODES)
        raise ValueError(
            f"Unknown topology inverse mode {mode!r}; "
            f"choose from {choices}"
        )
    if epsilon is None:
        epsilon = DEFAULT_PARAMS["topology_reciprocal_epsilon"]
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "topology_reciprocal_epsilon must be a positive finite number."
        ) from exc
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(
            "topology_reciprocal_epsilon must be a positive finite number."
        )
    return {
        "mode": mode,
        "epsilon": epsilon if mode == "normalized_reciprocal" else None,
    }


def validate_classical_routing_policy(mode=None, multipath_k=None):
    """Validate the public Phase-4 classical-routing policy."""
    mode = mode or DEFAULT_PARAMS["classical_routing_mode"]
    if mode not in CLASSICAL_ROUTING_MODES:
        choices = ", ".join(CLASSICAL_ROUTING_MODES)
        raise ValueError(
            f"Unknown classical routing mode {mode!r}; "
            f"choose from {choices}"
        )
    if multipath_k is None:
        multipath_k = DEFAULT_PARAMS["classical_multipath_k"]
    if isinstance(multipath_k, bool):
        raise ValueError("classical_multipath_k must be a positive integer.")
    try:
        normalized_k = int(multipath_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "classical_multipath_k must be a positive integer."
        ) from exc
    if normalized_k < 1 or normalized_k != float(multipath_k):
        raise ValueError("classical_multipath_k must be a positive integer.")
    return {
        "mode": mode,
        "multipath_k": (
            normalized_k if mode == "edge_disjoint_split" else None
        ),
    }


def validate_classical_bandwidth_sharing_mode(mode=None):
    """Validate the classical-link bandwidth allocation policy."""
    mode = mode or DEFAULT_PARAMS["classical_bandwidth_sharing_mode"]
    if mode not in CLASSICAL_BANDWIDTH_SHARING_MODES:
        choices = ", ".join(CLASSICAL_BANDWIDTH_SHARING_MODES)
        raise ValueError(
            f"Unknown classical bandwidth sharing mode {mode!r}; "
            f"choose from {choices}"
        )
    return mode


def validate_communication_overlap_mode(mode=None):
    """Validate the reporting-only paper Eq. (40) overlap policy."""
    mode = mode or DEFAULT_PARAMS["communication_overlap_mode"]
    if mode not in COMMUNICATION_OVERLAP_MODES:
        choices = ", ".join(COMMUNICATION_OVERLAP_MODES)
        raise ValueError(
            f"Unknown communication overlap mode {mode!r}; "
            f"choose from {choices}"
        )
    return mode


def validate_fidelity_policy(mode=None, threshold=None):
    """Validate and normalize the public C4 policy arguments."""
    mode = mode or "report_only"
    if mode not in FIDELITY_CONSTRAINT_MODES:
        choices = ", ".join(FIDELITY_CONSTRAINT_MODES)
        raise ValueError(f"Unknown fidelity constraint mode {mode!r}; choose from {choices}")
    if mode == "hardware_explicit":
        if threshold is None:
            raise ValueError(
                "hardware_explicit fidelity mode requires fidelity_threshold."
            )
        threshold = float(threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("fidelity threshold must be in the interval (0, 1].")
    elif threshold is not None:
        raise ValueError(
            "fidelity_threshold may only be supplied with hardware_explicit mode."
        )
    return {"mode": mode, "threshold": threshold}

DEFAULT_ALGORITHMS = ("Random", "Greedy", "Q-Only", "DQTetris", "TopoQ")
SCALABILITY_ALGORITHMS = ("Greedy", "DQTetris", "TopoQ")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    label: str
    dag_id: str
    system_config: str
    network_topology: str
    algorithms: tuple = DEFAULT_ALGORITHMS
    repetitions: int = None
    qpu_limit: int = None
    params: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)
    tags: tuple = ()
    enabled: bool = True
    expected_feasible: bool = True
    note: str = ""


# ------------------------------------------------------------------
# Data loading utilities
# ------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_path_list(value):
    if not value:
        return []
    return [item for item in value.split(os.pathsep) if item]


def _find_json_by_id(item_id, roots, recursive=False):
    """Return the unique JSON path for an external dataset id."""
    matches = []
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(os.path.expanduser(os.fspath(root)))
        direct = os.path.join(root, f"{item_id}.json")
        if os.path.isfile(direct):
            matches.append(direct)
        if recursive and os.path.isdir(root):
            for current_root, _, files in os.walk(root):
                filename = f"{item_id}.json"
                if filename in files:
                    candidate = os.path.join(current_root, filename)
                    if candidate not in matches:
                        matches.append(candidate)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Dataset id {item_id!r} is ambiguous across external roots: "
            + ", ".join(matches)
        )
    return matches[0]


def _active_dag_roots(explicit=None):
    if explicit:
        return [explicit]
    context_root = _DAG_DIR_CONTEXT.get()
    if context_root:
        return [context_root]
    return _split_path_list(os.environ.get(EXTERNAL_DAG_DIR_ENV))


def _active_circuit_roots(explicit=None):
    if explicit:
        if isinstance(explicit, (list, tuple)):
            return list(explicit)
        return [explicit]
    context_roots = _CIRCUIT_DIRS_CONTEXT.get()
    if context_roots:
        return list(context_roots)
    return _split_path_list(os.environ.get(EXTERNAL_CIRCUIT_DIRS_ENV))


@contextmanager
def dataset_source_context(
        dag_dir=None, circuit_dirs=None, system_data_root=None):
    """Temporarily route DAG/circuit loads to external public-benchmark roots."""
    dag_token = _DAG_DIR_CONTEXT.set(
        os.path.abspath(os.path.expanduser(os.fspath(dag_dir)))
        if dag_dir else None
    )
    if circuit_dirs is None:
        normalized_circuit_dirs = ()
    elif isinstance(circuit_dirs, (list, tuple)):
        normalized_circuit_dirs = tuple(
            os.path.abspath(os.path.expanduser(os.fspath(path)))
            for path in circuit_dirs
        )
    else:
        normalized_circuit_dirs = (
            os.path.abspath(os.path.expanduser(os.fspath(circuit_dirs))),
        )
    circuit_token = _CIRCUIT_DIRS_CONTEXT.set(normalized_circuit_dirs)
    system_token = _SYSTEM_DATA_ROOT_CONTEXT.set(
        os.path.abspath(os.path.expanduser(os.fspath(system_data_root)))
        if system_data_root else None
    )
    try:
        yield
    finally:
        _SYSTEM_DATA_ROOT_CONTEXT.reset(system_token)
        _DAG_DIR_CONTEXT.reset(dag_token)
        _CIRCUIT_DIRS_CONTEXT.reset(circuit_token)


def _active_system_data_root():
    context_root = _SYSTEM_DATA_ROOT_CONTEXT.get()
    if context_root:
        return context_root
    environment_root = os.environ.get(EXTERNAL_SYSTEM_DATA_ROOT_ENV)
    if environment_root:
        return os.path.abspath(os.path.expanduser(environment_root))
    return DATASET_DIR


def load_system_config(name):
    """Load system configuration (Small/Medium/Large)."""
    return load_json(os.path.join(
        _active_system_data_root(),
        "system_configs",
        f"{name.lower()}.json",
    ))


def load_network_topology(name):
    """Load classical network topology."""
    return load_json(os.path.join(
        _active_system_data_root(),
        "network_topologies",
        f"{name}.json",
    ))


def resolve_dag_path(dag_id, dag_dir=None):
    """Resolve a workload DAG path from an external root or the bundled dataset."""
    external = _find_json_by_id(dag_id, _active_dag_roots(dag_dir), recursive=False)
    if external:
        return external
    return os.path.join(DATASET_DIR, "workload_dags", f"{dag_id}.json")


def load_dag(dag_id, dag_dir=None):
    """Load a workload DAG."""
    return load_json(resolve_dag_path(dag_id, dag_dir=dag_dir))


def load_sim_params():
    """Load simulation parameters."""
    return load_json(os.path.join(
        _active_system_data_root(),
        "sim_params",
        "sim_params.json",
    ))


def resolve_circuit_path(circuit_id, circuit_dirs=None):
    """Resolve a circuit metadata path from public benchmark roots or dataset."""
    external = _find_json_by_id(
        circuit_id, _active_circuit_roots(circuit_dirs), recursive=True
    )
    if external:
        return external
    return os.path.join(DATASET_DIR, "quantum_circuits", f"{circuit_id}.json")


def load_circuit(circuit_id, circuit_dirs=None):
    """Load a quantum circuit specification."""
    return load_json(resolve_circuit_path(circuit_id, circuit_dirs=circuit_dirs))


# ------------------------------------------------------------------
# Experiment configurations (mapping paper experiments to dataset)
# ------------------------------------------------------------------

# (workload_label, dag_id, system_config, network_topology)
EXPERIMENT_CONFIGS = {
    "VQE-50_Small":   ("vqe_dag_50q_10iter",   "small",  "fattree_k4"),
    "VQE-100_Medium": ("vqe_dag_100q_10iter",  "medium", "dragonfly_4x4x2"),
    "VQE-200_Medium": ("vqe_dag_200q_10iter",  "medium", "dragonfly_4x4x2"),
    "QAOA-200_Large": ("qaoa_dag_200q_p1_10iter", "large", "dragonflyplus_8x8x4"),
    "QML-150_Large":  ("qml_dag_150q_10iter",  "large",  "dragonflyplus_8x8x4"),
}

FIDELITY_EXPERIMENT_IDS = {
    "VQE-100": "VQE-100_Medium",
    "VQE-200": "VQE-200_Medium",
    "QAOA-200": "QAOA-200_Large",
    "QML-150": "QML-150_Large",
}
REFERENCE_EXPERIMENT_ID = "VQE-200_Medium"

EXPERIMENT_REGISTRY = {
    label: ExperimentSpec(label, label, dag, config, network,
                          tags=("executable", "makespan") +
                               (("fidelity",) if label in FIDELITY_EXPERIMENT_IDS.values() else ()))
    for label, (dag, config, network) in EXPERIMENT_CONFIGS.items()
}

LEGACY_PAPER_EXPERIMENTS = {
    "VQE-100_Small": ExperimentSpec(
        "VQE-100_Small", "VQE-100 Small (paper assumption)",
        "vqe_dag_100q_10iter", "small", "fattree_k4",
        tags=("legacy-paper",), enabled=False, expected_feasible=False,
        note="Paper assumes 50 qubits/QPU; physical Guadalupe provides 16.",
    )
}


def register_experiment(spec):
    if spec.experiment_id in EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment already registered: {spec.experiment_id}")
    EXPERIMENT_REGISTRY[spec.experiment_id] = spec


def get_experiment(experiment_id):
    if experiment_id in EXPERIMENT_REGISTRY:
        return EXPERIMENT_REGISTRY[experiment_id]
    if experiment_id in LEGACY_PAPER_EXPERIMENTS:
        return LEGACY_PAPER_EXPERIMENTS[experiment_id]
    raise KeyError(f"Unknown experiment: {experiment_id}")


def iter_experiments(tags=None, enabled_only=True, include_legacy=False):
    specs = list(EXPERIMENT_REGISTRY.values())
    if include_legacy:
        specs += list(LEGACY_PAPER_EXPERIMENTS.values())
    tags = set(tags or ())
    return [spec for spec in specs
            if (not enabled_only or spec.enabled)
            and (not tags or tags.issubset(set(spec.tags)))]

# Scalability configs (varying QPU count on VQE-100)
# All use the small system (4 QPUs) as base, trimmed to desired QPU count
SCALABILITY_CONFIGS = {
    4:  ("vqe_dag_50q_10iter", "small", "fattree_k4"),
    8:  ("vqe_dag_50q_10iter", "medium", "dragonfly_4x4x2"),
    16: ("vqe_dag_50q_10iter", "large", "dragonflyplus_8x8x4"),
}

ABLATION_CONFIGS = {
    "full": {},
    "no_topology_score": {"no_topology_score": True},
    "no_entanglement_reserve": {"no_entanglement_reserve": True},
    "no_sim_annealing": {"no_sim_annealing": True},
    "no_comm_overlap": {"no_comm_overlap": True},
    "no_multipath": {"no_multipath": True},
    "no_feedback_loop": {"no_feedback_loop": True},
}
