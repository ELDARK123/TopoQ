"""Optional EPR purification models used for research reporting.

The BBPSSW equations here assume Werner-state inputs and ideal local
operations.  The model is deliberately kept separate from physical fidelity:
it reports an idealized purification scenario and its expected resource cost.
"""

from .config import DEFAULT_PARAMS, validate_purification_policy


def bbpssw_round(fidelity):
    """Return one ideal BBPSSW round as ``(output_fidelity, p_success)``."""
    fidelity = float(fidelity)
    if not 0.5 < fidelity <= 1.0:
        raise ValueError(
            "BBPSSW requires an input fidelity strictly greater than 0.5."
        )
    one_minus = 1.0 - fidelity
    success = (
        fidelity * fidelity
        + 2.0 * fidelity * one_minus / 3.0
        + 5.0 * one_minus * one_minus / 9.0
    )
    output = (
        fidelity * fidelity + one_minus * one_minus / 9.0
    ) / success
    return output, success


def build_purification_plan(raw_fidelity, mode="none", rounds=None,
                            target=None, max_rounds=None):
    """Build a deterministic BBPSSW cost/fidelity plan.

    ``expected_raw_pairs_per_output`` and
    ``expected_purification_attempts_per_output`` are expectation values, not
    guarantees for an individual stochastic run.  Physical fidelity is never
    changed by this plan; callers may use ``output_fidelity`` only for the
    separately labelled idealized scenario metric.
    """
    policy = validate_purification_policy(
        mode, rounds, target, max_rounds
    )
    raw_fidelity = float(raw_fidelity)
    if not 0.0 < raw_fidelity <= 1.0:
        raise ValueError("raw EPR fidelity must be in (0, 1].")

    plan = {
        "mode": policy["mode"],
        "raw_fidelity": raw_fidelity,
        "output_fidelity": raw_fidelity,
        "rounds": 0,
        "target_fidelity": policy["target_fidelity"],
        "target_met": None,
        "max_rounds": policy["max_rounds"],
        "round_success_probability_product": 1.0,
        "expected_raw_pairs_per_output": 1.0,
        "expected_purification_attempts_per_output": 0.0,
        "rounds_detail": [],
        "formula": "BBPSSW_Werner_ideal",
        "local_operations_assumption": "ideal",
        "physical_fidelity_unchanged": True,
    }
    if policy["mode"] == "none":
        return plan
    if raw_fidelity <= 0.5:
        raise ValueError(
            "BBPSSW modes require min(F_EPR_range) > 0.5; "
            f"received {raw_fidelity:.6g}."
        )

    requested_rounds = policy["rounds"]
    target = policy["target_fidelity"]
    max_rounds = policy["max_rounds"]
    current = raw_fidelity
    raw_pairs = 1.0
    attempts = 0.0
    success_product = 1.0

    while True:
        if policy["mode"] == "bbpssw_fixed_rounds":
            if len(plan["rounds_detail"]) >= requested_rounds:
                break
        elif current >= target:
            break
        elif len(plan["rounds_detail"]) >= max_rounds:
            raise ValueError(
                "BBPSSW target fidelity is not reached within "
                f"max_rounds={max_rounds}; output={current:.9f}, "
                f"target={target:.9f}."
            )

        output, success = bbpssw_round(current)
        previous_attempts = attempts
        attempts = (1.0 + 2.0 * previous_attempts) / success
        raw_pairs = 2.0 * raw_pairs / success
        success_product *= success
        plan["rounds_detail"].append({
            "round": len(plan["rounds_detail"]) + 1,
            "input_fidelity": current,
            "output_fidelity": output,
            "success_probability": success,
            "expected_raw_pairs_per_output": raw_pairs,
            "expected_purification_attempts_per_output": attempts,
        })
        current = output

    plan.update({
        "output_fidelity": current,
        "rounds": len(plan["rounds_detail"]),
        "target_met": (
            current >= target if target is not None else None
        ),
        "round_success_probability_product": success_product,
        "expected_raw_pairs_per_output": raw_pairs,
        "expected_purification_attempts_per_output": attempts,
    })
    return plan


def resolve_purification_policy(sim_params, params=None):
    """Resolve CLI/experiment parameters against the configured EPR range."""
    params = params or {}
    fidelity_params = sim_params.get("fidelity_parameters", {})
    epr_range = fidelity_params.get("F_EPR_range", ())
    if not epr_range:
        raise ValueError("fidelity_parameters.F_EPR_range is required.")
    raw_fidelity = min(epr_range)
    purification_params = sim_params.get("purification_parameters", {})
    configured_mode = purification_params.get(
        "default_mode", DEFAULT_PARAMS["epr_purification_mode"]
    )
    configured_max_rounds = purification_params.get(
        "max_rounds", DEFAULT_PARAMS["epr_purification_max_rounds"]
    )
    return build_purification_plan(
        raw_fidelity,
        mode=params.get(
            "epr_purification_mode",
            configured_mode,
        ),
        rounds=params.get("epr_purification_rounds"),
        target=params.get("epr_purification_target"),
        max_rounds=params.get(
            "epr_purification_max_rounds",
            configured_max_rounds,
        ),
    )
