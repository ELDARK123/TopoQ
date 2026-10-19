"""Shared EPR service-time semantics for schedulers and simulation."""

import math

from .config import validate_quantum_link_latency_policy


def resolve_quantum_link_latency_policy(params=None, sim_params=None):
    """Resolve global timing inputs without inventing physical-stage values."""
    params = params or {}
    sim_params = sim_params or {}
    policy = validate_quantum_link_latency_policy(
        params.get("quantum_link_latency_mode"),
        params.get("quantum_link_bsm_latency_ms"),
        params.get("quantum_link_switch_latency_ms"),
        params.get("quantum_link_signal_speed_m_s"),
        params.get("quantum_link_signal_traversals"),
    )
    signal_source = "parameter" if policy["signal_speed_m_s"] is not None else None
    if policy["mode"] == "physical_stage" and policy["signal_speed_m_s"] is None:
        configured = sim_params.get("quantum_link_parameters", {})
        fallback = configured.get(
            "signal_speed_m_s", configured.get("v_signal_ms")
        )
        if fallback is not None:
            policy = validate_quantum_link_latency_policy(
                policy["mode"],
                policy["bsm_latency_ms"],
                policy["switch_latency_ms"],
                fallback,
                policy["signal_traversals"],
            )
            signal_source = (
                "sim_params.signal_speed_m_s"
                if "signal_speed_m_s" in configured
                else "sim_params.v_signal_ms_legacy_name"
            )
    policy["signal_speed_source"] = signal_source
    return policy


def resolve_channel_latency_components(channel, policy, sim_params=None):
    """Resolve one channel's fixed physical stages or delivered-rate zeros."""
    if policy["mode"] == "delivered_pair_rate":
        return {
            "mode": "delivered_pair_rate",
            "bsm_latency_ms": 0.0,
            "switch_latency_ms": 0.0,
            "propagation_latency_ms": 0.0,
            "signal_speed_m_s": None,
            "signal_traversals": None,
            "stage_overhead_ms": 0.0,
            "parameter_source": "included_in_delivered_pair_rate",
        }

    sim_link = (sim_params or {}).get("quantum_link_parameters", {})

    def choose(channel_name, policy_name, sim_name=None):
        if channel.get(channel_name) is not None:
            return channel[channel_name], f"channel.{channel_name}"
        if policy.get(policy_name) is not None:
            return policy[policy_name], f"parameter.{policy_name}"
        sim_name = sim_name or channel_name
        if sim_link.get(sim_name) is not None:
            return sim_link[sim_name], f"sim_params.{sim_name}"
        return None, None

    bsm, bsm_source = choose("bsm_latency_ms", "bsm_latency_ms")
    switch, switch_source = choose(
        "switch_latency_ms", "switch_latency_ms"
    )
    traversals, traversals_source = choose(
        "signal_traversals", "signal_traversals"
    )
    speed, speed_source = choose(
        "signal_speed_m_s", "signal_speed_m_s"
    )
    if speed is None:
        speed = sim_link.get("v_signal_ms")
        if speed is not None:
            speed_source = "sim_params.v_signal_ms_legacy_name"

    missing = [
        name for name, value in (
            ("bsm_latency_ms", bsm),
            ("switch_latency_ms", switch),
            ("signal_speed_m_s", speed),
            ("signal_traversals", traversals),
        )
        if value is None
    ]
    channel_id = channel.get("channel_id", "unknown")
    if missing:
        raise ValueError(
            "physical_stage requires calibrated "
            f"{', '.join(missing)} for quantum channel {channel_id!r}."
        )
    resolved = validate_quantum_link_latency_policy(
        "physical_stage", bsm, switch, speed, traversals
    )
    distance_km = float(channel["distance_km"])
    if not math.isfinite(distance_km) or distance_km < 0.0:
        raise ValueError(
            f"distance_km must be non-negative for channel {channel_id!r}."
        )
    propagation_ms = (
        resolved["signal_traversals"] * distance_km * 1e6
        / resolved["signal_speed_m_s"]
    )
    stage_overhead_ms = (
        resolved["bsm_latency_ms"]
        + resolved["switch_latency_ms"]
        + propagation_ms
    )
    return {
        "mode": "physical_stage",
        "bsm_latency_ms": resolved["bsm_latency_ms"],
        "switch_latency_ms": resolved["switch_latency_ms"],
        "propagation_latency_ms": propagation_ms,
        "signal_speed_m_s": resolved["signal_speed_m_s"],
        "signal_traversals": resolved["signal_traversals"],
        "stage_overhead_ms": stage_overhead_ms,
        "parameter_source": {
            "bsm_latency_ms": bsm_source,
            "switch_latency_ms": switch_source,
            "signal_speed_m_s": speed_source,
            "signal_traversals": traversals_source,
        },
    }


def resolve_service_latency(selected_channels, policy, sim_params=None):
    """Return conservative fixed-stage timing for a selected channel set."""
    components = [
        resolve_channel_latency_components(channel, policy, sim_params)
        for channel in selected_channels
    ]
    maximum = max(
        components,
        key=lambda item: item["stage_overhead_ms"],
        default={"stage_overhead_ms": 0.0},
    )
    return {
        "quantum_link_latency_mode": policy["mode"],
        "stage_overhead_ms": maximum["stage_overhead_ms"],
        "selected_channel_stage_components": components,
        "multi_channel_stage_aggregation": (
            "conservative_max" if len(components) > 1 else "single_channel"
        ),
    }


def _erlang_cdf(unit_rate_time, epr_count):
    """CDF of an integer-shape, unit-rate Erlang distribution."""
    term = 1.0
    series = 1.0
    for order in range(1, epr_count):
        term *= unit_rate_time / order
        series += term
    survival = math.exp(-unit_rate_time) * series
    return min(max(1.0 - survival, 0.0), 1.0)


def erlang_quantile(p_threshold, epr_count):
    """Return the unit-rate Erlang quantile by deterministic bisection."""
    epr_count = int(epr_count)
    if epr_count <= 0:
        raise ValueError("epr_count must be a positive integer")
    if not 0.0 < p_threshold < 1.0:
        raise ValueError("p_threshold must be strictly between 0 and 1")
    lower = 0.0
    upper = max(float(epr_count), 1.0)
    while _erlang_cdf(upper, epr_count) < p_threshold:
        upper *= 2.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _erlang_cdf(midpoint, epr_count) < p_threshold:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def deterministic_epr_service_window_ms(rate_hz, p_threshold,
                                        duration_scale=1.0,
                                        stage_overhead_ms=0.0,
                                        epr_count=1):
    """Return an Erlang threshold window plus fixed physical stages."""
    rate_hz = float(rate_hz)
    duration_scale = float(duration_scale)
    stage_overhead_ms = float(stage_overhead_ms)
    if duration_scale <= 0.0:
        raise ValueError("EPR duration scale must be positive.")
    if stage_overhead_ms < 0.0 or not math.isfinite(stage_overhead_ms):
        raise ValueError("EPR stage overhead must be non-negative and finite.")
    epr_count = int(epr_count)
    if epr_count <= 0:
        raise ValueError("epr_count must be a positive integer")
    if rate_hz <= 0.0:
        raise ValueError("EPR pair generation rate must be positive.")
    generation_ms = max(
        erlang_quantile(p_threshold, epr_count) / rate_hz * 1000.0,
        0.05,
    ) * duration_scale
    return generation_ms + stage_overhead_ms
