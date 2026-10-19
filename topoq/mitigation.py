"""Algorithm-independent error-mitigation reporting models."""

import math

from .fidelity import LOG_FLOOR, aggregate_task_fidelities


def richardson_coefficients(noise_scales):
    """Return zero-noise Lagrange extrapolation coefficients at scale zero."""
    coefficients = []
    for index, scale in enumerate(noise_scales):
        coefficient = 1.0
        for other_index, other_scale in enumerate(noise_scales):
            if other_index == index:
                continue
            coefficient *= other_scale / (other_scale - scale)
        coefficients.append(coefficient)
    return tuple(coefficients)


def zne_sampling_overhead(noise_scales):
    """Return optimal-allocation sampling overhead for Richardson ZNE."""
    coefficients = richardson_coefficients(noise_scales)
    l1_norm = sum(abs(value) for value in coefficients)
    return coefficients, l1_norm * l1_norm


def estimate_error_mitigation_metrics(task_estimates, sched, policy,
                                      default_fidelity=1.0):
    """Report postselection quality and sampling work without changing F_phys."""
    mode = policy["mode"]
    uses_symmetry = mode in {"symmetry_verification", "zne_symmetry"}
    uses_zne = mode in {"zne", "zne_symmetry"}
    detection_fraction = policy.get("symmetry_error_detection_fraction")

    if uses_zne:
        coefficients, zne_overhead = zne_sampling_overhead(
            policy["zne_noise_scales"]
        )
    else:
        coefficients, zne_overhead = (1.0,), 1.0

    postselected = {}
    acceptances = []
    shot_multipliers = []
    raw_quantum_work_ms = 0.0
    adjusted_quantum_work_ms = 0.0

    for tid, estimate in task_estimates.items():
        physical_fidelity = min(
            max(float(estimate["fidelity"]), math.exp(LOG_FLOOR)),
            1.0,
        )
        if uses_symmetry:
            acceptance = 1.0 - detection_fraction * (
                1.0 - physical_fidelity
            )
            acceptance = min(max(acceptance, math.exp(LOG_FLOOR)), 1.0)
            conditional_fidelity = min(
                physical_fidelity / acceptance,
                1.0,
            )
            symmetry_overhead = 1.0 / acceptance
        else:
            acceptance = 1.0
            conditional_fidelity = physical_fidelity
            symmetry_overhead = 1.0

        duration_ms = max(
            float(sched[tid]["end"]) - float(sched[tid]["start"]),
            0.0,
        )
        total_multiplier = zne_overhead * symmetry_overhead
        raw_quantum_work_ms += duration_ms
        adjusted_quantum_work_ms += duration_ms * total_multiplier
        acceptances.append(acceptance)
        shot_multipliers.append(total_multiplier)
        # Preserve the log-domain estimate when postselection is disabled.
        # Recomputing log(fidelity) from the 1e-300 display floor would lose
        # information for long circuits and make mode=none differ from F_phys.
        if uses_symmetry:
            conditional_log_fidelity = (
                float(estimate["log_fidelity"]) - math.log(acceptance)
            )
        else:
            conditional_log_fidelity = float(estimate["log_fidelity"])
        postselected[tid] = {
            "fidelity": conditional_fidelity,
            "log_fidelity": conditional_log_fidelity,
        }

    aggregate = aggregate_task_fidelities(
        postselected, default_fidelity
    )
    return {
        "error_mitigation_mode": mode,
        "postselected_fidelity": aggregate["fidelity"],
        "postselected_fidelity_min_task": aggregate["fidelity_min_task"],
        "postselected_fidelity_geometric_mean": aggregate[
            "fidelity_geometric_mean"
        ],
        "postselected_fidelity_log10_product": aggregate[
            "fidelity_log10_product"
        ],
        "symmetry_acceptance_probability": (
            sum(acceptances) / len(acceptances) if acceptances else 1.0
        ),
        "symmetry_min_acceptance_probability": (
            min(acceptances) if acceptances else 1.0
        ),
        "zne_sampling_overhead": zne_overhead,
        "mitigation_mean_shot_multiplier": (
            sum(shot_multipliers) / len(shot_multipliers)
            if shot_multipliers else 1.0
        ),
        "mitigation_max_shot_multiplier": (
            max(shot_multipliers) if shot_multipliers else 1.0
        ),
        "raw_quantum_work_ms": raw_quantum_work_ms,
        "mitigation_adjusted_quantum_work_ms": adjusted_quantum_work_ms,
        "mitigation_extra_quantum_work_ms": (
            adjusted_quantum_work_ms - raw_quantum_work_ms
        ),
    }
