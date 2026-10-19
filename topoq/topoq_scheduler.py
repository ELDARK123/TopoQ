"""
TopoQ Algorithm Implementation
==============================
Implements the TopoQ 5-phase algorithm with iterative co-optimization:

  Phase 1 — Hierarchical Topology Characterization
  Phase 2 — Topology-aware Circuit Partitioning & Mapping
  Phase 3 — Entanglement-aware Quantum Scheduling
  Phase 4 — Network-aware Classical Task Scheduling
  Phase 5 — Joint Optimization & Refinement (SA + overlap)
"""

import math
import random
from collections import Counter, defaultdict, deque
from copy import deepcopy

from .validators import estimate_gate_batch_time_ms, validate_schedule
from .fidelity import (
    estimate_task_log_fidelity,
    resolve_epr_service_quality,
)
from .entanglement import (
    deterministic_epr_service_window_ms,
    resolve_quantum_link_latency_policy,
    resolve_service_latency,
)
from .communication import (
    annotate_communication_overlap,
    edge_data_mb,
    interface_transfer_ms,
    reserve_classical_route,
    select_classical_route,
)
from .mapping import (
    build_interaction_graph,
    count_remote_gates,
    analyze_physical_routing,
    map_logical_to_physical,
    partition_interaction_graph,
    quantum_link_cost,
    rank_qpu_set_candidates,
)
from .resources import (
    find_quantum_capacity_start,
    overlapping_quantum_reservations,
    physical_qubits_by_qpu,
    reserve_quantum_capacity,
)
from .timing import (
    classical_task_duration_ms,
    quantum_task_duration_ms,
    quantum_task_timing_details,
    reference_task_duration_ms,
    resolve_task_gate_profile,
)

from .config import (
    DEFAULT_PARAMS,
    PHASE5_OBJECTIVE_ZERO_REFERENCE_SCALES,
    load_circuit,
    resolve_coherence_event_mode,
    validate_entanglement_reservation_policy,
    validate_entanglement_memory_capacity_mode,
    validate_classical_bandwidth_sharing_mode,
    validate_classical_routing_policy,
    validate_communication_overlap_mode,
    validate_interface_latency_mode,
    validate_quantum_epr_routing_policy,
    validate_quantum_centrality_mode,
    validate_execution_time_policy,
    validate_qpu_calibration_policy,
    validate_phase5_objective_weights,
    validate_objective_mode,
    validate_topology_inverse_policy,
)


class TopoQScheduler:
    """Implements the TopoQ 5-phase algorithm with iterative co-optimization."""

    def __init__(self, system, workload, params=None, ablation=None, seed=42):
        self.system = system
        self.workload = workload
        coherence_event_mode = resolve_coherence_event_mode(params)
        self.params = DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        execution_time_policy = validate_execution_time_policy(
            self.params.get("execution_time_mode"),
            self.params.get("quantum_shots"),
            self.params.get("classical_compute_efficiency"),
        )
        self.params["execution_time_mode"] = execution_time_policy["mode"]
        self.params["quantum_shots"] = execution_time_policy[
            "quantum_shots"
        ]
        self.params["classical_compute_efficiency"] = (
            execution_time_policy["classical_compute_efficiency"]
        )
        calibration_policy = validate_qpu_calibration_policy(
            self.params.get("qpu_calibration_mode"),
            self.params.get("ibm_cx_direction_policy"),
            self.params.get("ibm_missing_calibration_policy"),
        )
        self.params.update({
            "qpu_calibration_mode": calibration_policy["mode"],
            "ibm_cx_direction_policy": calibration_policy[
                "cx_direction_policy"
            ],
            "ibm_missing_calibration_policy": calibration_policy[
                "missing_calibration_policy"
            ],
        })
        self.system.sim_params["qpu_calibration_policy"] = calibration_policy
        self.phase5_objective_weights = validate_phase5_objective_weights(
            self.params
        )
        self.params.update(self.phase5_objective_weights)
        self.objective_mode = validate_objective_mode(
            self.params.get("objective_mode")
        )
        self.params["objective_mode"] = self.objective_mode
        self.params["coherence_event_mode"] = coherence_event_mode
        self.params["coherence_task_window_mode"] = coherence_event_mode
        self.params["entanglement_memory_capacity_mode"] = (
            validate_entanglement_memory_capacity_mode(
                self.params.get("entanglement_memory_capacity_mode")
            )
        )
        self.quantum_link_latency_policy = (
            resolve_quantum_link_latency_policy(
                self.params, self.system.sim_params
            )
        )
        self.params["quantum_link_latency_mode"] = (
            self.quantum_link_latency_policy["mode"]
        )
        self.ablation = ablation or {}  # e.g., {"no_topology_score": True}
        reservation_policy = validate_entanglement_reservation_policy(
            self.params.get("entanglement_reservation_mode"),
            self.params.get("epr_proactive_lead_fraction"),
            self.params.get("epr_jit_memory_fraction"),
            self.params.get("epr_adaptive_2q_gate_threshold"),
            self.params.get("epr_adaptive_high_gate_mode"),
        )
        if self.ablation.get("no_entanglement_reserve"):
            reservation_policy["mode"] = "on_demand"
        self.entanglement_reservation_policy = reservation_policy
        self.params.update({
            "entanglement_reservation_mode": reservation_policy["mode"],
            "epr_proactive_lead_fraction": reservation_policy[
                "proactive_lead_fraction"
            ],
            "epr_jit_memory_fraction": reservation_policy[
                "jit_memory_fraction"
            ],
            "epr_adaptive_2q_gate_threshold": reservation_policy[
                "adaptive_2q_gate_threshold"
            ],
            "epr_adaptive_high_gate_mode": reservation_policy[
                "adaptive_high_gate_mode"
            ],
        })
        self.rng = random.Random(seed)
        self.classical_routing_policy = validate_classical_routing_policy(
            self.params.get("classical_routing_mode"),
            self.params.get("classical_multipath_k"),
        )
        self.classical_bandwidth_sharing_mode = (
            validate_classical_bandwidth_sharing_mode(
                self.params.get("classical_bandwidth_sharing_mode")
            )
        )
        self.communication_overlap_mode = validate_communication_overlap_mode(
            self.params.get("communication_overlap_mode")
        )
        self.quantum_epr_routing_policy = validate_quantum_epr_routing_policy(
            self.params.get("quantum_epr_routing_mode"),
            self.params.get("quantum_epr_reliability_window_ms"),
        )
        self.quantum_epr_routing_mode = self.quantum_epr_routing_policy["mode"]
        self._compute_topology_metrics()

    def _resolve_task(self, task_or_tid):
        """Return a task mapping from either a task id or task mapping."""
        if isinstance(task_or_tid, str):
            return self.workload.tasks[task_or_tid]
        return task_or_tid

    def _quantum_duration(self, task_or_tid, qpu_set,
                          routing_overhead_ms=0.0, **mapping_context):
        """Evaluate quantum execution time on the assigned QPU set."""
        return quantum_task_duration_ms(
            self._resolve_task(task_or_tid),
            self.system,
            qpu_set,
            getattr(self, "params", DEFAULT_PARAMS),
            routing_overhead_ms=routing_overhead_ms,
            **mapping_context,
        )

    def _quantum_timing_details(self, task_or_tid, qpu_set,
                                routing_overhead_ms=0.0,
                                **mapping_context):
        return quantum_task_timing_details(
            self._resolve_task(task_or_tid),
            self.system,
            qpu_set,
            getattr(self, "params", DEFAULT_PARAMS),
            routing_overhead_ms=routing_overhead_ms,
            **mapping_context,
        )

    @staticmethod
    def _effective_two_qubit_gate_count(task, n_swap_gates=0):
        """Return revised-paper two-qubit gates plus routed native gates."""
        profile = resolve_task_gate_profile(task)
        return (
            profile["n_2q_gates"] + max(int(n_swap_gates), 0)
        )

    def _get_entanglement_reservation_policy(self):
        """Return the normalized policy, including minimal test schedulers."""
        policy = getattr(self, "entanglement_reservation_policy", None)
        if policy is not None:
            return policy
        params = getattr(self, "params", {})
        policy = validate_entanglement_reservation_policy(
            params.get("entanglement_reservation_mode"),
            params.get("epr_proactive_lead_fraction"),
            params.get("epr_jit_memory_fraction"),
            params.get("epr_adaptive_2q_gate_threshold"),
            params.get("epr_adaptive_high_gate_mode"),
        )
        if getattr(self, "ablation", {}).get("no_entanglement_reserve"):
            policy["mode"] = "on_demand"
        self.entanglement_reservation_policy = policy
        return policy

    def _resolve_entanglement_reservation_mode(
            self, task_id, effective_two_qubit_gates=None):
        """Resolve the configured reservation mode for one quantum task."""
        policy = self._get_entanglement_reservation_policy()
        configured = policy["mode"]
        if configured != "adaptive":
            return configured
        if effective_two_qubit_gates is None:
            task = self.workload.tasks.get(task_id, {})
            effective_two_qubit_gates = self._effective_two_qubit_gate_count(
                task
            )
        if (int(effective_two_qubit_gates)
                >= policy["adaptive_2q_gate_threshold"]):
            return policy["adaptive_high_gate_mode"]
        return "proactive"

    def _entanglement_reservation_lead_ms(
            self, mode, service, total_generation, batch_duration):
        """Return how far EPR generation may begin before task readiness."""
        policy = self._get_entanglement_reservation_policy()
        if mode == "on_demand":
            return 0.0
        if mode == "proactive":
            return policy["proactive_lead_fraction"] * total_generation
        if mode != "just_in_time":
            raise ValueError(f"Unsupported reservation mode {mode!r}.")

        memory_lifetime = service.get("epr_memory_lifetime_ms")
        if memory_lifetime is None or float(memory_lifetime) <= 0.0:
            # Without calibrated memory lifetime, remain conservative and
            # generate at most one batch window ahead of task readiness.
            maximum_lead = batch_duration
        else:
            maximum_lead = (
                policy["jit_memory_fraction"] * float(memory_lifetime)
            )
        return min(total_generation, max(maximum_lead, 0.0))

    def _classical_duration(self, task_or_tid, node_id):
        """Evaluate classical execution time on the assigned node."""
        return classical_task_duration_ms(
            self._resolve_task(task_or_tid),
            self.system.nodes[node_id],
            getattr(self, "params", DEFAULT_PARAMS),
        )

    def _reference_duration(self, task_or_tid):
        """Return the shared resource-aware ranking/load proxy."""
        return reference_task_duration_ms(
            self._resolve_task(task_or_tid),
            self.system,
            getattr(self, "params", DEFAULT_PARAMS),
        )

    # ==================================================================
    # Phase 1: Hierarchical Topology Characterization
    # ==================================================================

    def _compute_topology_metrics(self):
        """Phase 1: Hierarchical Topology Characterization."""
        centrality_mode = validate_quantum_centrality_mode(
            self.params.get("quantum_centrality_mode")
        )
        interface_mode = validate_interface_latency_mode(
            self.params.get("interface_latency_mode")
        )
        interface_profile = self._compute_interface_latency_profile(
            interface_mode
        )
        inverse_policy = validate_topology_inverse_policy(
            self.params.get("topology_inverse_mode"),
            self.params.get("topology_reciprocal_epsilon"),
        )
        self._active_interfaces = interface_profile["active_interfaces"]
        if self.ablation.get("no_topology_score"):
            self.topology_scores = {q: 0.5 for q in self.system.get_qpu_ids()}
            self.topology_metric_components = {
                "quantum_centrality_mode": centrality_mode,
                **interface_profile["diagnostics"],
                "topology_inverse_mode": inverse_policy["mode"],
                "topology_reciprocal_epsilon": inverse_policy["epsilon"],
                "composite_inverse_model": inverse_policy["mode"],
                "ablation_disabled": True,
                "topology_scores": dict(self.topology_scores),
            }
            return

        c_q, centrality_parameters = self._compute_quantum_centrality(
            centrality_mode
        )

        # Classical betweenness centrality B_C(P_j)
        b_c = self._compute_betweenness_centrality()

        # Interface latency L_int(Q_i), with ctrl(Q_i) selected by the policy.
        l_int = interface_profile["latency_us"]
        controller_resources = interface_profile["controller_resources"]
        controller_network_nodes = interface_profile[
            "controller_network_nodes"
        ]

        # Normalize
        c_q_norm = self._minmax(c_q)
        b_c_norm = self._minmax(b_c)
        l_int_norm = self._minmax(l_int)
        n_qubits_norm = self._minmax({q: self.system.qpus[q]["n_qubits"] for q in self.system.qpus})

        # Composite topology score
        w1, w2, w3, w4 = self.params["w1"], self.params["w2"], self.params["w3"], self.params["w4"]
        self.topology_scores = {}
        controller_betweenness_raw = {
            qid: b_c[controller_network_nodes[qid]]
            for qid in self.system.qpus
        }
        controller_betweenness_norm = {
            qid: b_c_norm[controller_network_nodes[qid]]
            for qid in self.system.qpus
        }
        interface_latency_benefit = self._inverse_cost_benefit(
            l_int_norm,
            inverse_policy["mode"],
            inverse_policy["epsilon"],
            "interface_latency",
        )
        controller_betweenness_benefit = self._inverse_cost_benefit(
            controller_betweenness_norm,
            inverse_policy["mode"],
            inverse_policy["epsilon"],
            "controller_betweenness",
        )
        for qid in self.system.qpus:
            self.topology_scores[qid] = (
                w1 * c_q_norm.get(qid, 0.5)
                + w2 * interface_latency_benefit[qid]
                + w3 * n_qubits_norm.get(qid, 0.5)
                + w4 * controller_betweenness_benefit[qid]
            )
        self.topology_metric_components = {
            "quantum_centrality_mode": centrality_mode,
            "quantum_centrality_parameters": centrality_parameters,
            "quantum_centrality_raw": dict(c_q),
            "quantum_centrality_normalized": dict(c_q_norm),
            **interface_profile["diagnostics"],
            "topology_inverse_mode": inverse_policy["mode"],
            "topology_reciprocal_epsilon": inverse_policy["epsilon"],
            "classical_betweenness_raw": dict(b_c),
            "classical_betweenness_normalized": dict(b_c_norm),
            "controller_betweenness_raw": controller_betweenness_raw,
            "controller_betweenness_normalized": (
                controller_betweenness_norm
            ),
            "controller_betweenness_benefit": (
                controller_betweenness_benefit
            ),
            "interface_latency_normalized": dict(l_int_norm),
            "interface_latency_benefit": interface_latency_benefit,
            "composite_inverse_model": inverse_policy["mode"],
            "qpu_capacity_normalized": dict(n_qubits_norm),
            "topology_scores": dict(self.topology_scores),
        }

    def _compute_interface_latency_profile(self, mode):
        """Select ctrl(Q_i) under the configured interface-latency model."""
        mode = validate_interface_latency_mode(mode)
        signal_speed = self.params.get("interface_signal_speed_m_s")
        if signal_speed is None:
            signal_speed = self.system.sim_params.get(
                "interface_link_parameters", {}
            ).get("v_signal_m_s")
        if signal_speed is not None:
            signal_speed = float(signal_speed)
            if signal_speed <= 0:
                raise ValueError(
                    "interface_signal_speed_m_s must be positive."
                )
        if mode == "paper_eq24" and signal_speed is None:
            raise ValueError(
                "paper_eq24 requires interface_signal_speed_m_s."
            )

        network_nodes = (
            {node["id"] for node in self.system.net_nodes}
            | set(self.system.net_adj)
        )
        active_interfaces = {}
        controller_resources = {}
        controller_network_nodes = {}
        base_readout_latency_us = {}
        distance_km = {}
        propagation_latency_us = {}
        latency_us = {}
        candidate_count = {}

        for qid in self.system.qpus:
            if mode == "readout_proxy":
                configured = self.system.interface_links.get(qid)
                candidates = [] if configured is None else [configured]
            else:
                candidates = list(
                    self.system.interface_candidates.get(qid, [])
                )
            if not candidates:
                raise ValueError(f"QPU {qid} has no classical interface.")

            evaluated = []
            qpu_rho_us = None
            for candidate in candidates:
                controller = candidate.get("classical_node_id")
                if controller not in self.system.nodes:
                    raise ValueError(
                        f"QPU {qid} references unknown classical controller "
                        f"{controller!r}."
                    )
                network_node = self.system.get_network_node_id(controller)
                if network_node not in network_nodes:
                    raise ValueError(
                        f"Controller {controller!r} for QPU {qid} maps to "
                        f"unknown network node {network_node!r}."
                    )
                try:
                    rho_us = float(candidate["readout_latency_us"])
                    bandwidth_gbps = float(
                        candidate["control_bandwidth_gbps"]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"QPU {qid} interface {controller!r} has invalid "
                        "readout latency or control bandwidth."
                    ) from exc
                if rho_us <= 0:
                    raise ValueError(
                        f"QPU {qid} interface latency must be positive."
                    )
                if qpu_rho_us is None:
                    qpu_rho_us = rho_us
                elif not math.isclose(
                        rho_us, qpu_rho_us, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"paper_eq24 requires one QPU-level rho_i for "
                        f"{qid}; candidate readout latencies differ."
                    )
                if bandwidth_gbps <= 0:
                    raise ValueError(
                        f"QPU {qid} interface bandwidth must be positive."
                    )

                if mode == "paper_eq24":
                    if candidate.get("distance_km") is None:
                        raise ValueError(
                            f"paper_eq24 requires distance_km for QPU "
                            f"{qid} controller {controller!r}."
                        )
                    try:
                        candidate_distance_km = float(
                            candidate["distance_km"]
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"QPU {qid} controller {controller!r} has "
                            "invalid distance_km."
                        ) from exc
                    if candidate_distance_km < 0:
                        raise ValueError(
                            f"QPU {qid} interface distance must be "
                            "non-negative."
                        )
                    propagation_us = (
                        candidate_distance_km * 1e9 / signal_speed
                    )
                else:
                    candidate_distance_km = None
                    propagation_us = None

                effective_latency_us = rho_us + (propagation_us or 0.0)
                active = dict(candidate)
                active["effective_latency_us"] = effective_latency_us
                active["propagation_latency_us"] = propagation_us
                evaluated.append((
                    effective_latency_us,
                    str(controller),
                    controller,
                    network_node,
                    rho_us,
                    candidate_distance_km,
                    propagation_us,
                    active,
                ))

            selected = min(evaluated, key=lambda item: (item[0], item[1]))
            (
                selected_latency,
                _,
                controller,
                network_node,
                rho_us,
                selected_distance,
                propagation_us,
                active,
            ) = selected
            active_interfaces[qid] = active
            controller_resources[qid] = controller
            controller_network_nodes[qid] = network_node
            base_readout_latency_us[qid] = rho_us
            distance_km[qid] = selected_distance
            propagation_latency_us[qid] = propagation_us
            latency_us[qid] = selected_latency
            candidate_count[qid] = len(candidates)

        diagnostics = {
            "interface_latency_mode": mode,
            "interface_latency_model": mode,
            "interface_latency_formula": (
                "rho_us + distance_km * 1e9 / signal_speed_m_s"
                if mode == "paper_eq24" else "readout_latency_us"
            ),
            "interface_signal_speed_m_s": (
                signal_speed if mode == "paper_eq24" else None
            ),
            "interface_candidate_count": candidate_count,
            "interface_base_readout_latency_us": base_readout_latency_us,
            "interface_distance_km": distance_km,
            "interface_propagation_latency_us": propagation_latency_us,
            "interface_latency_us": latency_us,
            "controller_resource": controller_resources,
            "controller_network_node": controller_network_nodes,
        }
        return {
            "active_interfaces": active_interfaces,
            "controller_resources": controller_resources,
            "controller_network_nodes": controller_network_nodes,
            "latency_us": latency_us,
            "diagnostics": diagnostics,
        }

    def _compute_quantum_centrality(self, mode):
        """Return Phase-1 quantum centrality under the selected policy."""
        mode = validate_quantum_centrality_mode(mode)
        links = list(self.system.quantum_links.values())
        b_max = max(
            (float(link["max_concurrent_epr"]) for link in links),
            default=1.0,
        )
        if b_max <= 0:
            raise ValueError("Quantum links must have positive EPR capacity.")

        p_threshold = float(self.params.get("p_threshold", 0.95))
        if mode == "effective_service" and not 0.0 < p_threshold < 1.0:
            raise ValueError(
                "p_threshold must be strictly between 0 and 1."
            )
        centrality = {}
        for qid in self.system.qpus:
            total = 0.0
            for other in self.system.qpus:
                if other == qid:
                    continue
                link = self.system.get_quantum_link(qid, other)
                if link is None:
                    continue
                rate_hz = float(link["rate_hz"])
                distance_km = float(link["distance_km"])
                capacity = float(link["max_concurrent_epr"])
                if rate_hz <= 0 or distance_km <= 0 or capacity <= 0:
                    raise ValueError(
                        f"Quantum link {qid}-{other} must have positive "
                        "rate, distance, and EPR capacity."
                    )
                if mode == "paper_eq22":
                    total += (
                        rate_hz / distance_km
                        * capacity / b_max
                    )
                else:
                    generation_ms = (
                        -math.log1p(-p_threshold) / rate_hz * 1000.0
                    )
                    service_latency = resolve_service_latency(
                        (link,),
                        self.quantum_link_latency_policy,
                        self.system.sim_params,
                    )
                    total += capacity / (
                        generation_ms
                        + service_latency["stage_overhead_ms"]
                    )
            centrality[qid] = total

        parameters = {
            "b_max": b_max,
            "p_threshold": (
                p_threshold if mode == "effective_service" else None
            ),
            "quantum_link_latency_mode": (
                self.quantum_link_latency_policy["mode"]
                if mode == "effective_service" else None
            ),
            "raw_unit": (
                "rate_hz_per_km"
                if mode == "paper_eq22" else "epr_pairs_per_ms"
            ),
        }
        return centrality, parameters

    @staticmethod
    def _inverse_cost_benefit(values, mode, epsilon, metric_name):
        """Transform normalized lower-is-better costs into score benefits."""
        normalized = {}
        for key, value in values.items():
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Normalized {metric_name} for {key} must be in [0, 1]."
                )
            normalized[key] = value

        if mode == "bounded_complement":
            return {key: 1.0 - value for key, value in normalized.items()}
        if mode == "normalized_reciprocal":
            return {
                key: epsilon * (1.0 - value) / (value + epsilon)
                for key, value in normalized.items()
            }
        if mode == "paper_raw_strict":
            zero_keys = [
                str(key) for key, value in normalized.items()
                if value == 0.0
            ]
            if zero_keys:
                raise ValueError(
                    "paper_raw_strict is undefined because normalized "
                    f"{metric_name} is zero for {', '.join(zero_keys)}."
                )
            return {key: 1.0 / value for key, value in normalized.items()}
        raise ValueError(f"Unknown topology inverse mode {mode!r}.")

    def _compute_betweenness_centrality(self):
        """Return raw unweighted Brandes centrality for every network node."""
        all_ids = sorted(
            {node["id"] for node in self.system.net_nodes}
            | set(self.system.net_adj),
            key=str,
        )
        centrality = {node: 0.0 for node in all_ids}
        for source in all_ids:
            stack = []
            predecessors = {node: [] for node in all_ids}
            shortest_paths = {node: 0.0 for node in all_ids}
            shortest_paths[source] = 1.0
            distance = {source: 0}
            queue = deque([source])

            while queue:
                node = queue.popleft()
                stack.append(node)
                neighbors = sorted(
                    self.system.net_adj.get(node, []),
                    key=lambda item: str(item[0]),
                )
                for neighbor, _, _ in neighbors:
                    if neighbor not in distance:
                        distance[neighbor] = distance[node] + 1
                        queue.append(neighbor)
                    if distance.get(neighbor) == distance[node] + 1:
                        shortest_paths[neighbor] += shortest_paths[node]
                        predecessors[neighbor].append(node)

            dependency = {node: 0.0 for node in all_ids}
            while stack:
                node = stack.pop()
                path_count = shortest_paths[node]
                if path_count > 0:
                    coefficient = (1.0 + dependency[node]) / path_count
                    for predecessor in predecessors[node]:
                        dependency[predecessor] += (
                            shortest_paths[predecessor] * coefficient
                        )
                if node != source:
                    centrality[node] += dependency[node]

        return {
            node: value / 2.0
            for node, value in centrality.items()
        }

    @staticmethod
    def _minmax(d, invert=False):
        if not d:
            return {}
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.5 for k in d}
        if invert:
            return {k: 1 - (v - mn) / (mx - mn) for k, v in d.items()}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    # ==================================================================
    # Main scheduling entry point
    # ==================================================================

    def schedule(self):
        """Run the full TopoQ algorithm with iterative co-optimization."""
        self.last_constraint_report = []
        self.last_enforced_violations = []
        self.last_sa_objective = None
        self.last_feedback_iteration = None
        self.last_feedback_makespan = None

        # Phase 2: Topology-aware circuit partitioning and mapping
        assignment, mapping = self._phase2_partition_and_map()

        max_iter = self.params["max_iterations"]
        if self.ablation.get("no_feedback_loop"):
            max_iter = 1

        best_state = None
        best_makespan = float('inf')
        for iteration in range(max_iter):
            self._assign_classical_nodes(assignment)
            full_sched = self._phase3_4_joint_scheduling(assignment, mapping)

            # Phase 5: Joint optimization (skipped on first pass when no_feedback_loop,
            # since the feedback loop is what enables cross-phase refinement)
            if self.ablation.get("no_feedback_loop") and iteration == 0:
                # The ablation skips refinement, not basic DAG feasibility.
                pass
            else:
                full_sched = self._phase5_joint_optimize(full_sched, assignment, mapping)

            current_makespan = max(s["end"] for s in full_sched.values())
            violations = self._check_constraints(full_sched, assignment, mapping)
            enforced_violations = [
                violation for violation in violations
                if violation.get("enforced", True)
            ]
            self.last_constraint_report = violations
            self.last_enforced_violations = enforced_violations

            if not enforced_violations:
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_state = self._capture_feedback_state(
                        full_sched,
                        assignment,
                        mapping,
                        violations,
                        enforced_violations,
                        current_makespan,
                        iteration + 1,
                    )
                break

            if iteration == max_iter - 1:
                break

            # Repair mapping based on violations (feedback loop)
            assignment, mapping = self._repair_mapping(
                assignment, mapping, enforced_violations
            )

        if best_state is None:
            counts = Counter(
                violation.get("constraint", "UNKNOWN_CONSTRAINT")
                for violation in self.last_enforced_violations
            )
            summary = ", ".join(
                f"{constraint}={count}"
                for constraint, count in sorted(counts.items())
            ) or "UNKNOWN_CONSTRAINT=0"
            raise ValueError(
                "TopoQ feedback loop failed to find a feasible schedule "
                f"after {max_iter} iteration(s); enforced constraint "
                f"violations: {summary}."
            )

        return self._restore_feedback_state(best_state)

    def _capture_feedback_state(self, sched, assignment, mapping, violations,
                                enforced_violations, makespan, iteration):
        """Capture one feasible feedback iteration as an indivisible state."""
        return {
            "schedule": deepcopy(sched),
            "assignment": deepcopy(assignment),
            "mapping": deepcopy(mapping),
            "classical_link_timeline": deepcopy(
                getattr(self, "classical_link_timeline", {})
            ),
            "quantum_link_timeline": deepcopy(
                getattr(self, "quantum_link_timeline", {})
            ),
            "constraint_report": deepcopy(violations),
            "enforced_violations": deepcopy(enforced_violations),
            "sa_objective": deepcopy(getattr(self, "last_sa_objective", None)),
            "intra_task_remote_gates": deepcopy(
                getattr(self, "_intra_task_remote_gates", {})
            ),
            "intra_task_remote_gates_by_link": deepcopy(
                getattr(self, "_intra_task_remote_gates_by_link", {})
            ),
            "makespan": makespan,
            "iteration": iteration,
        }

    def _restore_feedback_state(self, state):
        """Restore scheduler diagnostics and return one consistent result tuple."""
        self.classical_link_timeline = deepcopy(state["classical_link_timeline"])
        self.quantum_link_timeline = deepcopy(state["quantum_link_timeline"])
        self.last_constraint_report = deepcopy(state["constraint_report"])
        self.last_enforced_violations = deepcopy(state["enforced_violations"])
        self.last_sa_objective = deepcopy(state["sa_objective"])
        self._intra_task_remote_gates = deepcopy(
            state["intra_task_remote_gates"]
        )
        self._intra_task_remote_gates_by_link = deepcopy(
            state["intra_task_remote_gates_by_link"]
        )
        self.last_feedback_makespan = state["makespan"]
        self.last_feedback_iteration = state["iteration"]
        return (
            deepcopy(state["schedule"]),
            deepcopy(state["assignment"]),
            deepcopy(state["mapping"]),
        )

    def _get_interface_link(self, qpu_id):
        """Return the interface selected for this TopoQ scheduler."""
        iface = getattr(self, "_active_interfaces", {}).get(qpu_id)
        if iface is None:
            raise ValueError(f"QPU {qpu_id} has no active interface.")
        return iface

    def _assign_classical_nodes(self, assignment):
        """Assign classical tasks before unified DAG schedule generation."""
        estimated_load = {node: 0.0 for node in self.system.nodes}
        for tid in self.workload.classical_tasks:
            task = self.workload.tasks[tid]
            memory_mb = task.get("memory_mb", 100)
            best = None
            for nid, node in self.system.nodes.items():
                if node["memory_gb"] * 1024 < memory_mb:
                    continue
                locality = 0
                for pred in self.workload.predecessors.get(tid, []):
                    pred_qpu = assignment.get(pred)
                    if pred_qpu in self.system.qpus:
                        controller = self._get_interface_link(pred_qpu)[
                            "classical_node_id"
                        ]
                        locality += int(controller == nid)
                exec_ms = self._classical_duration(task, nid)
                score = locality * 10 + 1.0 / (1.0 + exec_ms) - 0.001 * estimated_load[nid]
                candidate = (score, nid)
                if best is None or candidate > best:
                    best = candidate
            if best is None:
                raise ValueError(f"No memory-feasible classical node for task {tid}.")
            assignment[tid] = best[1]
            estimated_load[best[1]] += self._classical_duration(
                task, best[1]
            )

    def _interface_transfer_ms(self, qpu_id, predecessor, successor):
        data_mb = self._edge_data_mb(predecessor, successor)
        return interface_transfer_ms(
            self.system,
            qpu_id,
            data_mb,
            interface=self._get_interface_link(qpu_id),
        )

    def _current_fixed_physical_mapping(self, tid, mapping):
        """Rebuild C6 pins from the latest predecessor final mappings."""
        fixed = {}
        locations = {}
        task_partition = mapping[tid]["logical_partition"]
        for predecessor in self.workload.predecessors.get(tid, []):
            if predecessor not in mapping:
                continue
            shared, _ = self.workload.get_shared_logical_qubits(
                predecessor, tid
            )
            predecessor_mapping = mapping[predecessor]
            for logical in shared:
                if (logical not in predecessor_mapping["logical_partition"]
                        or logical not in predecessor_mapping["physical_mapping"]):
                    continue
                owner = predecessor_mapping["logical_partition"][logical]
                physical = predecessor_mapping["physical_mapping"][logical]
                location = (owner, physical)
                if logical in locations and locations[logical] != location:
                    raise ValueError(
                        "Conflicting current C6 locations for logical qubit "
                        f"{logical} of task {tid}: "
                        f"{locations[logical]} vs {location}."
                    )
                if task_partition.get(logical) != owner:
                    raise ValueError(
                        f"Task {tid} does not preserve the C6 QPU of logical "
                        f"qubit {logical}."
                    )
                locations[logical] = location
                fixed[logical] = physical
        return fixed

    def _prepare_quantum_execution(self, tid, mapping, timeline, capacities,
                                   earliest_start):
        """Find a count-safe and physically disjoint dynamic mapping."""
        task = self.workload.tasks[tid]
        task_mapping = mapping[tid]
        logical_partition = task_mapping["logical_partition"]
        partition_sizes = Counter(logical_partition.values())
        interaction_graph = self._interaction_graphs[tid]
        timing_details = self._quantum_timing_details(
            task,
            partition_sizes,
            routing_overhead_ms=task_mapping.get("routing_overhead_ms", 0.0),
            logical_partition=logical_partition,
            physical_mapping=task_mapping.get("physical_mapping"),
            interaction_graph=interaction_graph,
            physical_routing=task_mapping.get("physical_routing"),
        )
        duration = timing_details["duration_ms"]
        search_from = max(float(earliest_start), 0.0)
        reservation_count = sum(len(items) for items in timeline.values())
        max_attempts = max(16, reservation_count * 2 + 8)

        for _ in range(max_attempts):
            start, _ = find_quantum_capacity_start(
                timeline,
                capacities,
                partition_sizes,
                search_from,
                duration,
            )
            overlaps = overlapping_quantum_reservations(
                timeline,
                partition_sizes,
                start,
                start + duration,
            )
            occupied = {
                qid: {
                    physical
                    for reservation in reservations
                    for physical in reservation.get("physical_qubits", [])
                }
                for qid, reservations in overlaps.items()
            }
            try:
                initial_mapping = map_logical_to_physical(
                    interaction_graph,
                    logical_partition,
                    self.system,
                    fixed_physical_mapping=(
                        self._current_fixed_physical_mapping(tid, mapping)
                    ),
                    forbidden_physical_by_qpu=occupied,
                )
                routing = analyze_physical_routing(
                    interaction_graph,
                    logical_partition,
                    initial_mapping,
                    self.system,
                    routing_mode=self.params.get(
                        "physical_routing_mode", "auto"
                    ),
                    forbidden_physical_by_qpu=occupied,
                )
                if routing.get("unroutable_edges"):
                    raise ValueError(
                        f"Task {tid} has unroutable physical interactions."
                    )
            except ValueError:
                conflict_ends = sorted({
                    reservation["end"]
                    for reservations in overlaps.values()
                    for reservation in reservations
                    if reservation["end"] > start
                })
                if not conflict_ends:
                    raise
                search_from = conflict_ends[0]
                continue

            final_mapping = routing["final_physical_mapping"]
            mapped_physical = physical_qubits_by_qpu(
                logical_partition, final_mapping
            )
            footprint = routing.get("physical_footprint_by_qpu")
            if footprint is None:
                physical_requirements = mapped_physical
                exclusive_qpus = (
                    set(partition_sizes)
                    if routing["swap_gate_count"] > 0 else set()
                )
            else:
                physical_requirements = {
                    qid: set(physical) for qid, physical in footprint.items()
                }
                exclusive_qpus = set(
                    routing.get("routing_path_incomplete_qpus", [])
                )
            new_timing_details = self._quantum_timing_details(
                task,
                partition_sizes,
                routing_overhead_ms=routing["routing_overhead_ms"],
                logical_partition=logical_partition,
                physical_mapping=final_mapping,
                interaction_graph=interaction_graph,
                physical_routing=routing,
            )
            new_duration = new_timing_details["duration_ms"]
            if abs(new_duration - duration) > 1e-12:
                duration = new_duration
                timing_details = new_timing_details
                search_from = max(float(earliest_start), 0.0)
                continue
            timing_details = new_timing_details

            occupancy_requirements = {
                qid: (
                    capacities[qid]
                    if qid in exclusive_qpus else partition_sizes[qid]
                )
                for qid in partition_sizes
            }
            resolved_start, shared_with = find_quantum_capacity_start(
                timeline,
                capacities,
                occupancy_requirements,
                start,
                duration,
                physical_requirements=physical_requirements,
            )
            if resolved_start > start + 1e-12:
                search_from = resolved_start
                continue

            return {
                "start": start,
                "duration": duration,
                "partition_sizes": dict(partition_sizes),
                "occupancy_requirements": occupancy_requirements,
                "physical_mapping": final_mapping,
                "mapped_physical_qubits_by_qpu": mapped_physical,
                "physical_qubits_by_qpu": physical_requirements,
                "physical_routing": routing,
                "timing_details": timing_details,
                "space_sharing_exclusive_routing": bool(exclusive_qpus),
                "space_sharing_exclusive_qpus": sorted(exclusive_qpus),
                "shared_with": shared_with,
            }

        raise ValueError(
            f"Task {tid} dynamic physical mapping did not converge after "
            f"{max_attempts} attempts."
        )

    def _phase3_4_joint_scheduling(self, assignment, mapping):
        """Generate one feasible schedule over the complete hybrid DAG."""
        sched = {}
        self.classical_link_timeline = defaultdict(list)
        self.quantum_link_timeline = defaultdict(list)
        qpu_capacity = {
            qid: qpu["n_qubits"] for qid, qpu in self.system.qpus.items()
        }
        quantum_task_timeline = defaultdict(list)
        node_available = {nid: 0.0 for nid in self.system.nodes}
        priorities = self._compute_priorities()
        in_degree = {
            tid: len(self.workload.predecessors.get(tid, []))
            for tid in self.workload.tasks
        }
        ready = {tid for tid, degree in in_degree.items() if degree == 0}
        scheduled = set()

        while ready:
            tid = max(ready, key=lambda task_id: (priorities.get(task_id, 0), task_id))
            ready.remove(tid)
            task = self.workload.tasks[tid]
            dependency_ready = 0.0
            dependency_transfers = []

            for pred in dict.fromkeys(
                    self.workload.predecessors.get(tid, [])):
                pred_end = sched[pred]["end"]
                pred_type = self.workload.tasks[pred]["type"]
                task_type = task["type"]
                transfer_end = pred_end
                transfer = {
                    "predecessor": pred,
                    "start": pred_end,
                    "end": pred_end,
                    "source_resource": assignment[pred],
                    "target_resource": assignment[tid],
                }
                if pred_type == "quantum" and task_type == "quantum":
                    source_qpu, target_qpu = assignment[pred], assignment[tid]
                    if source_qpu != target_qpu:
                        link = self.system.get_quantum_link(source_qpu, target_qpu)
                        if not link:
                            raise ValueError(f"No quantum link for dependency {pred}->{tid}.")
                        link_key = frozenset((source_qpu, target_qpu))
                        transfer_end, reservations = self._reserve_entanglement_batches(
                            link_key, pred_end, 1, tid,
                            predecessor=pred,
                            effective_two_qubit_gates=(
                                self._effective_two_qubit_gate_count(
                                    task, mapping.get(tid, {}).get(
                                        "n_swap_gates", 0
                                    )
                                )
                            ),
                        )
                        transfer.update({
                            "type": "quantum-quantum", "end": transfer_end,
                            "quantum_link": tuple(sorted(link_key)),
                            "entanglement_reservations": reservations,
                        })
                elif pred_type == "quantum" and task_type == "classical":
                    duration = self._interface_transfer_ms(assignment[pred], pred, tid)
                    transfer_end += duration
                    transfer.update({"type": "quantum-classical", "end": transfer_end})
                elif pred_type == "classical" and task_type == "classical":
                    source_node, target_node = assignment[pred], assignment[tid]
                    if source_node != target_node:
                        data_mb = self._edge_data_mb(pred, tid)
                        path = self._select_classical_path(source_node, target_node, data_mb)
                        reservation = self._reserve_classical_bandwidth(
                            path, pred_end, data_mb, tid, pred
                        )
                        transfer_end = reservation["end"]
                        transfer = dict(
                            reservation,
                            type="classical-classical",
                            source_resource=source_node,
                            target_resource=target_node,
                        )
                else:  # classical -> quantum
                    source_node = assignment[pred]
                    target_qpu = assignment[tid]
                    controller = self._get_interface_link(target_qpu)[
                        "classical_node_id"
                    ]
                    network_reservation = None
                    if source_node != controller:
                        data_mb = self._edge_data_mb(pred, tid)
                        path = self._select_classical_path(source_node, controller, data_mb)
                        network_reservation = self._reserve_classical_bandwidth(
                            path, pred_end, data_mb, tid, pred
                        )
                        transfer_end = network_reservation["end"]
                    interface_ms = self._interface_transfer_ms(target_qpu, pred, tid)
                    transfer_end += interface_ms
                    transfer.update({
                        "type": "classical-quantum", "end": transfer_end,
                        "network": network_reservation,
                    })
                dependency_ready = max(dependency_ready, transfer_end)
                dependency_transfers.append(transfer)

            if task["type"] == "quantum":
                task_mapping = mapping[tid]
                preliminary = self._prepare_quantum_execution(
                    tid,
                    mapping,
                    quantum_task_timeline,
                    qpu_capacity,
                    dependency_ready,
                )
                base_ready = preliminary["start"]
                entanglement_ready = base_ready
                entanglement_reservations = []
                for link_key, remote in task_mapping["remote_gates_by_link"].items():
                    link = self.system.get_quantum_link(*link_key)
                    if not link:
                        raise ValueError(f"Missing entanglement link {link_key} for task {tid}.")
                    effective_two_qubit_gates = (
                        self._effective_two_qubit_gate_count(
                            task,
                            preliminary["physical_routing"]["swap_gate_count"],
                        )
                    )
                    link_ready_time, reservations = self._reserve_entanglement_batches(
                        frozenset(link_key), base_ready, remote, tid,
                        effective_two_qubit_gates=effective_two_qubit_gates,
                    )
                    entanglement_ready = max(entanglement_ready, link_ready_time)
                    entanglement_reservations.extend(reservations)
                ready_without_capacity = max(base_ready, entanglement_ready)
                if abs(ready_without_capacity - base_ready) <= 1e-9:
                    # No EPR wait and no change to the capacity-constrained
                    # start: reuse the preliminary preparation instead of
                    # rebuilding mapping/routing a second time.
                    prepared = preliminary
                else:
                    prepared = self._prepare_quantum_execution(
                        tid,
                        mapping,
                        quantum_task_timeline,
                        qpu_capacity,
                        ready_without_capacity,
                    )
                start = prepared["start"]
                duration = prepared["duration"]
                end = start + duration
                intra_wait = max(0.0, entanglement_ready - base_ready)
                capacity_wait = max(0.0, start - ready_without_capacity)
                mapping[tid]["physical_mapping"] = prepared["physical_mapping"]
                mapping[tid]["physical_routing"] = prepared["physical_routing"]
                mapping[tid]["n_swap_gates"] = prepared[
                    "physical_routing"
                ]["swap_gate_count"]
                mapping[tid]["routing_overhead_ms"] = prepared[
                    "physical_routing"
                ]["routing_overhead_ms"]
                for record in entanglement_reservations:
                    record["consume"] = start
                    record["deadline"] = start
                for transfer in dependency_transfers:
                    if transfer.get("type") != "quantum-quantum":
                        continue
                    for record in transfer.get("entanglement_reservations", []):
                        record["consume"] = start
                        record["deadline"] = start
                sched[tid] = {
                    "start": start, "end": end, "qpu": assignment[tid],
                    "qpu_set": list(prepared["partition_sizes"]),
                    "partition_sizes": prepared["partition_sizes"],
                    "physical_qubits_by_qpu": {
                        qid: sorted(physical)
                        for qid, physical in prepared[
                            "physical_qubits_by_qpu"
                        ].items()
                    },
                    "mapped_physical_qubits_by_qpu": {
                        qid: sorted(physical)
                        for qid, physical in prepared[
                            "mapped_physical_qubits_by_qpu"
                        ].items()
                    },
                    "entanglement_wait": intra_wait,
                    "effective_two_qubit_gates": (
                        self._effective_two_qubit_gate_count(
                            task, mapping[tid]["n_swap_gates"]
                        )
                    ),
                    "n_remote_gates": self._intra_task_remote_gates.get(tid, 0),
                    "n_swap_gates": mapping[tid]["n_swap_gates"],
                    "routing_overhead_ms": mapping[tid]["routing_overhead_ms"],
                    "quantum_timing": prepared["timing_details"],
                    "decoherence_penalty": 0.0,
                    "space_shared": bool(prepared["shared_with"]),
                    "space_shared_with": prepared["shared_with"],
                    "space_sharing_exclusive_routing": prepared[
                        "space_sharing_exclusive_routing"
                    ],
                    "space_sharing_exclusive_qpus": prepared[
                        "space_sharing_exclusive_qpus"
                    ],
                    "qpu_capacity_wait": capacity_wait,
                    "dependency_transfers": dependency_transfers,
                    "entanglement_reservations": entanglement_reservations,
                }
                reserve_quantum_capacity(
                    quantum_task_timeline,
                    prepared["occupancy_requirements"],
                    start,
                    end,
                    tid,
                    physical_qubits=prepared["physical_qubits_by_qpu"],
                )
            else:
                nid = assignment[tid]
                node = self.system.nodes[nid]
                resource_ready = node_available[nid]
                start = max(resource_ready, dependency_ready)
                duration = self._classical_duration(task, nid)
                end = start + duration
                network_transfers = [
                    transfer for transfer in dependency_transfers
                    if transfer.get("type") == "classical-classical"
                ]
                sched[tid] = {
                    "start": start, "end": end, "node": nid,
                    "comm_time": max(0.0, start - max(resource_ready, max(
                        (sched[p]["end"] for p in self.workload.predecessors.get(tid, [])), default=0.0
                    ))),
                    "network_transfers": network_transfers,
                    "dependency_transfers": dependency_transfers,
                }
                node_available[nid] = end

            scheduled.add(tid)
            for succ in self.workload.successors.get(tid, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.add(succ)

        if len(scheduled) != len(self.workload.tasks):
            raise ValueError("Workload contains a cycle or unresolved dependency.")
        return sched

    # ==================================================================
    # Phase 2: Topology-aware Circuit Partitioning & Mapping
    # ==================================================================

    def _quantum_mapping_order(self):
        """Topologically order quantum tasks, prioritizing larger ready tasks."""
        quantum = set(self.workload.quantum_tasks)
        original_index = {
            tid: index for index, tid in enumerate(self.workload.quantum_tasks)
        }
        in_degree = {
            tid: sum(pred in quantum for pred in self.workload.predecessors.get(tid, []))
            for tid in quantum
        }
        ready = {tid for tid, degree in in_degree.items() if degree == 0}
        order = []
        while ready:
            tid = max(
                ready,
                key=lambda item: (
                    self.workload.tasks[item].get("n_qubits", 1)
                    * self.workload.tasks[item].get("depth", 1),
                    -original_index[item],
                ),
            )
            ready.remove(tid)
            order.append(tid)
            for successor in self.workload.successors.get(tid, []):
                if successor not in quantum:
                    continue
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready.add(successor)
        if len(order) != len(quantum):
            raise ValueError("Quantum-task dependency graph contains a cycle.")
        return order

    def _apply_mapping_continuity(self, tid, logical_partition, mapping,
                                  candidate_qpus, interaction_graph):
        """Pin C6 shared qubits and rebalance only non-shared qubits."""
        fixed_partition = {}
        fixed_physical = {}
        continuity = []
        for predecessor in self.workload.predecessors.get(tid, []):
            if predecessor not in mapping:
                continue
            shared, inferred = self.workload.get_shared_logical_qubits(
                predecessor, tid
            )
            if not shared:
                continue
            pred_mapping = mapping[predecessor]
            for logical in shared:
                if (logical not in pred_mapping["logical_partition"]
                        or logical not in pred_mapping["physical_mapping"]):
                    continue
                owner = pred_mapping["logical_partition"][logical]
                physical = pred_mapping["physical_mapping"][logical]
                location = (owner, physical)
                previous = (
                    (fixed_partition[logical], fixed_physical[logical])
                    if logical in fixed_partition else None
                )
                if previous is not None and previous != location:
                    raise ValueError(
                        "Conflicting C6 locations for logical qubit "
                        f"{logical} of task {tid}: {previous} vs {location}."
                    )
                fixed_partition[logical] = owner
                fixed_physical[logical] = physical
            continuity.append({
                "predecessor": predecessor,
                "shared_logical_qubits": shared,
                "shared_qubits_inferred": inferred,
            })

        if not fixed_partition:
            return logical_partition, fixed_physical, continuity

        partition = dict(logical_partition)
        partition.update(fixed_partition)
        counts = Counter(partition.values())
        for qid, count in Counter(fixed_partition.values()).items():
            if count > self.system.qpus[qid]["n_qubits"]:
                raise ValueError(
                    f"C6 fixed qubits exceed capacity of {qid} for task {tid}."
                )

        adjacency = defaultdict(dict)
        for (left, right), edge in interaction_graph["edges"].items():
            adjacency[left][right] = edge["weight"]
            adjacency[right][left] = edge["weight"]
        candidate_qpus = sorted(
            set(candidate_qpus) | set(fixed_partition.values())
        )
        for overloaded in list(candidate_qpus):
            capacity = self.system.qpus[overloaded]["n_qubits"]
            while counts[overloaded] > capacity:
                movable = sorted(
                    logical for logical, owner in partition.items()
                    if owner == overloaded and logical not in fixed_partition
                )
                if not movable:
                    raise ValueError(
                        f"C6 continuity leaves no movable qubit on overloaded {overloaded}."
                    )
                logical = movable[-1]
                choices = []
                for qid in candidate_qpus:
                    if counts[qid] >= self.system.qpus[qid]["n_qubits"]:
                        continue
                    feasible = all(
                        partition.get(neighbor) in {None, qid}
                        or self.system.get_quantum_link(qid, partition[neighbor]) is not None
                        for neighbor in adjacency[logical]
                    )
                    if not feasible:
                        continue
                    cut_cost = sum(
                        weight for neighbor, weight in adjacency[logical].items()
                        if neighbor in partition and partition[neighbor] != qid
                    )
                    choices.append((cut_cost, counts[qid], qid))
                if not choices:
                    raise ValueError(
                        f"No capacity-feasible C6 placement for logical qubit {logical}."
                    )
                _, _, target = min(choices)
                partition[logical] = target
                counts[overloaded] -= 1
                counts[target] += 1
        return partition, fixed_physical, continuity

    def _mapping_candidate_cost_weights(self, selection_strategy):
        """Return the normalized candidate-ranking weights for one policy."""
        if selection_strategy == "least_loaded":
            return {
                "execution": 0.0,
                "communication": 0.0,
                "fidelity_loss": 0.0,
                "load": 1.0,
                "topology_penalty": 0.0,
                "capacity_penalty": 0.0,
            }
        if selection_strategy != "phase2":
            raise ValueError(
                f"Unknown quantum mapping selection strategy: "
                f"{selection_strategy}."
            )
        return {
            "execution": self.params["assignment_w_execution"],
            "communication": self.params["assignment_w_communication"],
            "fidelity_loss": self.params["assignment_w_fidelity"],
            "load": self.params["assignment_w_load"],
            "topology_penalty": self.params["assignment_w_topology"],
            "capacity_penalty": self.params["assignment_w_capacity"],
        }

    @staticmethod
    def _physical_mapping_signature(task_mapping):
        """Return the logical and final physical placement of one task."""
        return (
            tuple(sorted(task_mapping.get("logical_partition", {}).items())),
            tuple(sorted(task_mapping.get("physical_mapping", {}).items())),
        )

    def _build_quantum_mapping_candidate(
            self, tid, mapping, qpu_usage, selection_strategy="phase2",
            require_coherence=False, required_window_ms=None,
            exclude_equivalent_to=None):
        """Build one complete quantum mapping candidate without committing it."""
        task = self.workload.tasks[tid]
        interaction_graph = self._interaction_graphs[tid]
        attempts = []
        if required_window_ms is not None:
            max_coherence_ms = max(
                qpu["coherence_time_us"] / 1000.0
                for qpu in self.system.qpus.values()
            )
            if required_window_ms > max_coherence_ms + 1e-9:
                attempts.append({
                    "status": "rejected_coherence",
                    "reason": (
                        f"Required window {required_window_ms} ms exceeds the "
                        f"maximum hardware coherence time "
                        f"{max_coherence_ms} ms."
                    ),
                })
                return {"mapping": None, "attempts": attempts}

        candidates = rank_qpu_set_candidates(
            task,
            interaction_graph,
            self.system,
            qpu_usage,
            self._estimate_assignment_cost,
            self._mapping_candidate_cost_weights(selection_strategy),
            beam_width=self.params.get("qpu_candidate_beam_width", 32),
            max_extra_qpus=self.params.get("qpu_partition_extra_qpus", 1),
        )
        max_attempts = max(
            int(self.params.get("qpu_partition_max_attempts", 64)), 1
        )
        for rank, candidate in enumerate(candidates[:max_attempts]):
            requested_qpus = candidate["qpu_set"]
            if (require_coherence
                    and not self._coherence_candidate_feasible(
                        task, requested_qpus, required_window_ms
                    )):
                attempts.append({
                    "rank": rank,
                    "qpu_set": list(requested_qpus),
                    "qpu_count": candidate["qpu_count"],
                    "status": "rejected_coherence",
                })
                continue
            try:
                logical_partition = partition_interaction_graph(
                    interaction_graph,
                    requested_qpus,
                    self.system,
                    self.params["beta"],
                    self.params["gamma"],
                )
                logical_partition, fixed_physical, continuity = (
                    self._apply_mapping_continuity(
                        tid,
                        logical_partition,
                        mapping,
                        requested_qpus,
                        interaction_graph,
                    )
                )
                qpu_set = sorted(set(logical_partition.values()))
                if (require_coherence
                        and not self._coherence_candidate_feasible(
                            task, qpu_set, required_window_ms
                        )):
                    raise ValueError(
                        "C6 continuity reintroduced a coherence-infeasible QPU."
                    )
                physical_mapping = map_logical_to_physical(
                    interaction_graph,
                    logical_partition,
                    self.system,
                    fixed_physical_mapping=fixed_physical,
                )
                physical_routing = analyze_physical_routing(
                    interaction_graph,
                    logical_partition,
                    physical_mapping,
                    self.system,
                    routing_mode=self.params.get(
                        "physical_routing_mode", "auto"
                    ),
                )
                if physical_routing.get("unroutable_edges"):
                    raise ValueError(
                        "Quantum mapping candidate contains unroutable "
                        "physical interactions."
                    )
            except ValueError as error:
                if "Conflicting C6 locations" in str(error):
                    raise
                attempts.append({
                    "rank": rank,
                    "qpu_set": list(requested_qpus),
                    "qpu_count": candidate["qpu_count"],
                    "status": "failed",
                    "reason": str(error),
                })
                continue

            primary = candidate["primary_qpu"]
            if primary not in qpu_set:
                primary = max(
                    qpu_set,
                    key=lambda qid: (
                        sum(
                            owner == qid
                            for owner in logical_partition.values()
                        ),
                        self.topology_scores.get(qid, 0.5),
                        qid,
                    ),
                )
            n_remote, by_link = count_remote_gates(
                interaction_graph, logical_partition
            )
            candidate_mapping = {
                **deepcopy(mapping.get(tid, {})),
                "primary_qpu": primary,
                "qpu_set": qpu_set,
                "logical_partition": logical_partition,
                "physical_mapping": physical_routing[
                    "final_physical_mapping"
                ],
                "remote_gates_by_link": by_link,
                "mapping_continuity": continuity,
                "physical_routing": physical_routing,
                "n_swap_gates": physical_routing["swap_gate_count"],
                "routing_overhead_ms": physical_routing[
                    "routing_overhead_ms"
                ],
            }
            if (exclude_equivalent_to is not None
                    and self._physical_mapping_signature(candidate_mapping)
                    == self._physical_mapping_signature(
                        exclude_equivalent_to
                    )):
                attempts.append({
                    "rank": rank,
                    "qpu_set": list(qpu_set),
                    "qpu_count": candidate["qpu_count"],
                    "status": "rejected_equivalent",
                })
                continue

            attempts.append({
                "rank": rank,
                "qpu_set": list(qpu_set),
                "qpu_count": candidate["qpu_count"],
                "status": "selected",
                "normalized_cost": candidate["normalized_cost"],
            })
            return {
                "mapping": candidate_mapping,
                "n_remote_gates": n_remote,
                "remote_gates_by_link": by_link,
                "rank": rank,
                "candidate": candidate,
                "attempts": attempts,
            }

        return {"mapping": None, "attempts": attempts}

    def _phase2_partition_and_map(self):
        """Phase 2 using explicit interactions, partitions and IBM mappings."""
        assignment = {}
        mapping = {}
        self._intra_task_remote_gates = {}
        self._intra_task_remote_gates_by_link = {}
        self._interaction_graphs = {}
        qpu_usage = {qid: 0.0 for qid in self.system.qpus}
        circuit_cache = {}
        ordered_tasks = self._quantum_mapping_order()
        for tid in ordered_tasks:
            task = self.workload.tasks[tid]
            circuit_id = task.get("circuit_id")
            if circuit_id not in circuit_cache:
                circuit_cache[circuit_id] = load_circuit(circuit_id)
            interaction_graph = build_interaction_graph(
                task, circuit_cache[circuit_id], self.params["alpha"]
            )
            self._interaction_graphs[tid] = interaction_graph
            inherited_partition, inherited_physical, inherited_continuity = (
                self._apply_mapping_continuity(
                    tid, {}, mapping, [], interaction_graph
                )
            )
            fully_inherited = (
                len(inherited_partition) == len(interaction_graph["nodes"])
            )
            if fully_inherited:
                logical_partition = inherited_partition
                physical_mapping = inherited_physical
                continuity = inherited_continuity
                qpu_set = sorted(set(logical_partition.values()))
                source = continuity[0]["predecessor"]
                primary = mapping[source]["primary_qpu"]
                candidate_rank = None
                candidate_attempts = []
                fallback_used = False
                extra_qpus_used = False
                if primary not in qpu_set:
                    primary = max(
                        qpu_set,
                        key=lambda qid: (
                            sum(
                                owner == qid
                                for owner in logical_partition.values()
                            ),
                            self.topology_scores.get(qid, 0.5),
                            qid,
                        ),
                    )
                n_remote, by_link = count_remote_gates(
                    interaction_graph, logical_partition
                )
                physical_routing = analyze_physical_routing(
                    interaction_graph,
                    logical_partition,
                    physical_mapping,
                    self.system,
                    routing_mode=self.params.get(
                        "physical_routing_mode", "auto"
                    ),
                )
                task_mapping = {
                    "primary_qpu": primary,
                    "qpu_set": qpu_set,
                    "logical_partition": logical_partition,
                    "physical_mapping": physical_routing[
                        "final_physical_mapping"
                    ],
                    "remote_gates_by_link": by_link,
                    "mapping_continuity": continuity,
                    "physical_routing": physical_routing,
                    "n_swap_gates": physical_routing["swap_gate_count"],
                    "routing_overhead_ms": physical_routing[
                        "routing_overhead_ms"
                    ],
                }
            else:
                result = self._build_quantum_mapping_candidate(
                    tid, mapping, qpu_usage
                )
                if result["mapping"] is None:
                    candidate_attempts = result["attempts"]
                    last_reason = (
                        candidate_attempts[-1]["reason"]
                        if candidate_attempts
                        and "reason" in candidate_attempts[-1]
                        else "No connected capacity candidate."
                    )
                    raise ValueError(
                        f"Task {tid} exhausted {len(candidate_attempts)} "
                        f"QPU-set candidates. Last failure: {last_reason}"
                    )
                task_mapping = result["mapping"]
                primary = task_mapping["primary_qpu"]
                qpu_set = task_mapping["qpu_set"]
                logical_partition = task_mapping["logical_partition"]
                n_remote = result["n_remote_gates"]
                by_link = result["remote_gates_by_link"]
                candidate_attempts = result["attempts"]
                candidate_rank = result["rank"]
                fallback_used = candidate_rank > 0
                extra_qpus_used = (
                    result["candidate"]["qpu_count"]
                    > result["candidate"]["minimum_qpu_count"]
                )
            assignment[tid] = primary
            mapping[tid] = {
                **task_mapping,
                "qpu_candidate_rank": candidate_rank,
                "qpu_candidate_attempts": candidate_attempts,
                "qpu_selection_fallback_used": fallback_used,
                "qpu_selection_extra_qpus_used": extra_qpus_used,
            }
            self._intra_task_remote_gates[tid] = n_remote
            self._intra_task_remote_gates_by_link[tid] = by_link
            partition_sizes = Counter(logical_partition.values())
            for qid, size in partition_sizes.items():
                qpu_usage[qid] += (
                    self._quantum_duration(task, partition_sizes)
                    * size / max(task.get("n_qubits", 1), 1)
                )

        return assignment, mapping

    def _coherence_candidate_feasible(self, task, qpu_set,
                                      required_window_ms=None):
        """Return whether every candidate QPU satisfies the C5 limits."""
        safety_fraction = self.params.get("T_safety_frac", 0.05)
        for qid in qpu_set:
            if qid not in self.system.qpus:
                return False
            qpu = self.system.qpus[qid]
            coherence_ms = qpu["coherence_time_us"] / 1000.0
            safe_window_ms = coherence_ms * (1.0 - safety_fraction)
            if (estimate_gate_batch_time_ms(task, qpu)
                    > safe_window_ms + 1e-9):
                return False
            if (required_window_ms is not None
                    and required_window_ms > coherence_ms + 1e-9):
                return False
        return True

    def _remap_task_for_coherence(self, tid, assignment, mapping, qpu_load,
                                  required_window_ms=None,
                                  require_coherence=True,
                                  repair_reason="C5_COHERENCE_WINDOW",
                                  exclude_current=False):
        """Fully repartition and remap one constraint-violating task."""
        task = self.workload.tasks[tid]
        qpu_usage = {
            qid: max(qpu_load.get(qid, 0.0), 0.0)
            for qid in self.system.qpus
        }
        result = self._build_quantum_mapping_candidate(
            tid,
            mapping,
            qpu_usage,
            require_coherence=require_coherence,
            required_window_ms=required_window_ms,
            exclude_equivalent_to=(
                mapping.get(tid) if exclude_current else None
            ),
        )
        if result["mapping"] is not None:
            task_mapping = result["mapping"]
            primary = task_mapping["primary_qpu"]
            qpu_set = task_mapping["qpu_set"]
            logical_partition = task_mapping["logical_partition"]
            old_primary = assignment.get(tid)
            old_partition_sizes = Counter(
                mapping.get(tid, {}).get("logical_partition", {}).values()
            )
            old_qpu_set = list(old_partition_sizes) or (
                [old_primary] if old_primary in self.system.qpus else []
            )
            old_duration = (
                self._quantum_duration(task, old_qpu_set)
                if old_qpu_set else 0.0
            )
            assignment[tid] = primary
            repaired_mapping = {
                **task_mapping,
                "qpu_candidate_rank": result["rank"],
                "qpu_candidate_attempts": result["attempts"],
                "qpu_selection_fallback_used": result["rank"] > 0,
                "qpu_selection_extra_qpus_used": (
                    result["candidate"]["qpu_count"]
                    > result["candidate"]["minimum_qpu_count"]
                ),
                "constraint_repair": {
                    "reason": repair_reason,
                    "selected_qpu_set": list(qpu_set),
                },
            }
            if require_coherence:
                repaired_mapping["coherence_repair"] = {
                    "required_window_ms": required_window_ms,
                    "selected_qpu_set": list(qpu_set),
                    "safety_fraction": self.params.get(
                        "T_safety_frac", 0.05
                    ),
                }
            mapping[tid] = repaired_mapping
            self._intra_task_remote_gates[tid] = result["n_remote_gates"]
            self._intra_task_remote_gates_by_link[tid] = result[
                "remote_gates_by_link"
            ]

            old_total_qubits = max(sum(old_partition_sizes.values()), 1)
            for qid, size in old_partition_sizes.items():
                qpu_load[qid] = max(
                    qpu_load[qid]
                    - old_duration * size / old_total_qubits,
                    0.0,
                )
            partition_sizes = Counter(logical_partition.values())
            total_qubits = max(task.get("n_qubits", 1), 1)
            duration = self._quantum_duration(task, partition_sizes)
            for qid, size in partition_sizes.items():
                qpu_load[qid] += duration * size / total_qubits
            return True

        return False

    def _estimate_assignment_cost(self, task, qpu_set, qpu_usage,
                                  interaction_graph=None):
        """Return raw components of the unified Phase-2 assignment cost.

        ``select_qpu_set`` normalizes each component across the candidates
        before applying the configured weights, so values with different units
        are never added directly.
        """
        n_gates = max(task.get("n_gates", 0), 0)
        depth = max(task.get("depth", 1), 1)
        n_1q = int(n_gates * 0.67)
        n_2q = n_gates - n_1q

        execution_values = []
        fidelity_losses = []
        for qid in qpu_set:
            qpu = self.system.qpus[qid]
            gate_times = qpu.get(
                "gate_times_ns",
                {"1q": 35.6, "2q": 334.0, "measure": 1200.0},
            )
            execution_values.append((
                n_1q / depth * gate_times["1q"]
                + n_2q / depth * gate_times["2q"]
                + gate_times["measure"]
            ) / 1e6)
            fidelity = self._estimate_fidelity(qpu, task)
            fidelity_losses.append(-math.log(max(fidelity, 1e-300)))

        communication_edges = []
        adjacency = {qid: set() for qid in qpu_set}
        for index, source in enumerate(qpu_set):
            for target in qpu_set[index + 1:]:
                link = self.system.get_quantum_link(source, target)
                if link:
                    communication_edges.append(
                        quantum_link_cost(link, self.params["beta"])
                    )
                    adjacency[source].add(target)
                    adjacency[target].add(source)
        if len(qpu_set) > 1:
            reachable = {qpu_set[0]}
            frontier = [qpu_set[0]]
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        frontier.append(neighbor)
            communication = (
                sum(communication_edges) / len(communication_edges)
                if len(reachable) == len(qpu_set) and communication_edges
                else math.inf
            )
        else:
            communication = 0.0

        required = max(task.get("n_qubits", 0), 1)
        capacity = sum(self.system.qpus[qid]["n_qubits"] for qid in qpu_set)
        interaction_scale = sum(
            edge["two_qubit_gates"]
            for edge in (interaction_graph or {}).get("edges", {}).values()
        )
        if communication_edges and interaction_scale:
            communication *= interaction_scale

        return {
            "execution": max(execution_values, default=0.0) / len(qpu_set),
            "communication": communication,
            "fidelity_loss": max(fidelity_losses, default=0.0),
            "load": sum(qpu_usage[qid] for qid in qpu_set) / len(qpu_set),
            "topology_penalty": 1.0 - sum(
                self.topology_scores.get(qid, 0.5) for qid in qpu_set
            ) / len(qpu_set),
            "capacity_penalty": max(0.0, (required - capacity) / required),
        }

    def _estimate_fidelity_details(
            self, qpu, task_or_n_gates, est_time_ms=None,
            estimated_remote_gates=None, estimated_hold_ms=0.0,
            estimated_swap_gates=0):
        """Return the untruncated log-domain fidelity estimate."""
        if isinstance(task_or_n_gates, dict):
            task = task_or_n_gates
        else:
            n_gates = max(int(task_or_n_gates), 0)
            task = {
                "n_gates": n_gates,
                "n_qubits": 1,
                "depth": max(int(n_gates / 100), 1),
            }
        if estimated_remote_gates is None:
            estimated_remote_gates = max(
                0, task.get("n_qubits", 1) - qpu.get("n_qubits", 1)
            )
        return estimate_task_log_fidelity(
            self.system,
            task,
            qpu,
            estimated_remote_gates=estimated_remote_gates,
            estimated_hold_ms=estimated_hold_ms,
            estimated_swap_gates=estimated_swap_gates,
        )

    def _estimate_fidelity(self, qpu, task_or_n_gates, est_time_ms=None,
                           estimated_remote_gates=None,
                           estimated_hold_ms=0.0,
                           estimated_swap_gates=0):
        """Estimate fidelity through the shared log-domain hardware model."""
        return self._estimate_fidelity_details(
            qpu, task_or_n_gates, est_time_ms=est_time_ms,
            estimated_remote_gates=estimated_remote_gates,
            estimated_hold_ms=estimated_hold_ms,
            estimated_swap_gates=estimated_swap_gates,
        )["fidelity"]

    # ==================================================================
    # Phase 3: Entanglement-aware Quantum Scheduling
    # ==================================================================

    def _phase3_quantum_scheduling(self, assignment, mapping):
        """Phase 3: Entanglement-aware quantum scheduling."""
        if self.ablation.get("no_entanglement_reserve"):
            return self._schedule_quantum_no_reservation(assignment, mapping)
        return self._schedule_quantum_with_reservation(assignment, mapping)

    def _schedule_quantum_with_reservation(self, assignment, mapping):
        """Schedule quantum tasks with proactive entanglement reservation.

        Key optimization: independent tasks (e.g. measurements in same iteration)
        are scheduled in parallel across different QPUs. Entanglement for
        inter-task dependencies is pre-generated, so wait time is minimal.

        Space sharing: when multiple independent tasks fit on the same QPU
        (combined qubits <= QPU capacity, C2 constraint), they execute
        concurrently, maximizing hardware utilization.
        """
        sched = {}
        # Space-sharing state: per-QPU qubit occupancy, available time, batch end
        qpu_occupied = {q: 0 for q in self.system.qpus}      # qubits currently in use
        qpu_avail_time = {q: 0.0 for q in self.system.qpus}   # earliest start for new co-located task
        qpu_batch_end = {q: 0.0 for q in self.system.qpus}    # max end time of concurrent batch
        qpu_capacity = {q: self.system.qpus[q]["n_qubits"] for q in self.system.qpus}
        ent_link_busy = defaultdict(float)  # link -> busy until time

        # Build priority based on critical path
        priorities = self._compute_priorities()

        # PSGS (Parallel Serial Generation Scheme): schedule ready tasks in parallel
        ready = set()
        for tid in self.workload.quantum_tasks:
            preds = [p for p in self.workload.predecessors.get(tid, []) if p in self.workload.quantum_tasks]
            if not preds:
                ready.add(tid)

        scheduled = set()

        # Process ready tasks in priority order, but schedule multiple in parallel
        while ready:
            # Sort ready tasks by priority
            ready_sorted = sorted(
                ready,
                key=lambda t: (-priorities.get(t, 0), t),
            )

            # Track which QPUs get new tasks this round (for batch release)
            qpus_updated_this_round = set()

            # Schedule all ready tasks that can start now
            for tid in ready_sorted:
                task = self.workload.tasks[tid]
                qid = assignment[tid]
                routing_overhead_ms = mapping[tid].get("routing_overhead_ms", 0.0)
                task_mapping = mapping[tid]
                partition_sizes = Counter(task_mapping["logical_partition"].values())
                qpu_set = list(partition_sizes)
                est_time = self._quantum_duration(
                    task, qpu_set, routing_overhead_ms
                )
                space_shared = any(qpu_occupied[q] > 0 for q in qpu_set)
                est_start = 0.0
                for part_qid, size in partition_sizes.items():
                    if qpu_occupied[part_qid] + size <= qpu_capacity[part_qid]:
                        est_start = max(est_start, qpu_avail_time[part_qid])
                    else:
                        est_start = max(est_start, qpu_batch_end[part_qid])
                        qpu_occupied[part_qid] = 0
                        qpu_avail_time[part_qid] = qpu_batch_end[part_qid]

                # Precedence constraints (quantum predecessors)
                for pred in self.workload.predecessors.get(tid, []):
                    if pred in sched:
                        est_start = max(est_start, sched[pred]["end"])

                # Entanglement reservation: pre-generate during predecessor execution
                # With proactive reservation, entanglement is generated in parallel
                # with the predecessor's computation, so effective wait is minimal
                ent_wait = 0.0
                for pred in self.workload.predecessors.get(tid, []):
                    if pred in assignment and assignment[pred] != qid:
                        link = self.system.get_quantum_link(qid, assignment[pred])
                        if link:
                            # Proactive reservation: entanglement pre-generated
                            # during predecessor execution -> near-zero wait
                            comm_time = self._compute_entanglement_time(
                                (qid, assignment[pred]), est_start
                            )
                            # With reservation, ~90% of entanglement is pre-generated
                            # Only 10% remains as wait time
                            ent_wait += comm_time * 0.1

                # Intra-task remote gates (from circuit partitioning)
                n_remote = getattr(self, '_intra_task_remote_gates', {}).get(tid, 0)
                for link_key, link_remote in task_mapping["remote_gates_by_link"].items():
                    link = self.system.get_quantum_link(*link_key)
                    if not link:
                        raise ValueError(f"Missing entanglement link {link_key} for task {tid}.")
                    service = self._quantum_epr_service(link_key)
                    n_batches = math.ceil(
                        link_remote / service["max_concurrent_epr"]
                    )
                    per_batch = self._compute_entanglement_time(
                        link_key, est_start
                    )
                    link_wait = n_batches * per_batch * 0.1
                    ent_wait += link_wait

                # Gamma: decoherence penalty
                gamma = self.params["gamma"]
                task_gates = task.get("n_gates", 100)
                depth_factor = max(1, task_gates / 500)
                decoherence_penalty = ent_wait * gamma * depth_factor * 0.15

                start_time = est_start + ent_wait
                end_time = start_time + est_time + decoherence_penalty

                sched[tid] = {
                    "start": start_time,
                    "end": end_time,
                    "qpu": qid,
                    "qpu_set": qpu_set,
                    "partition_sizes": dict(partition_sizes),
                    "entanglement_wait": ent_wait,
                    "n_remote_gates": n_remote,
                    "n_swap_gates": mapping[tid].get("n_swap_gates", 0),
                    "routing_overhead_ms": routing_overhead_ms,
                    "decoherence_penalty": decoherence_penalty,
                    "space_shared": space_shared,
                }

                # Update space-sharing state (C2 compliance)
                for part_qid, size in partition_sizes.items():
                    qpu_occupied[part_qid] += size
                    qpu_batch_end[part_qid] = max(qpu_batch_end[part_qid], end_time)
                    qpus_updated_this_round.add(part_qid)
                scheduled.add(tid)
                ready.discard(tid)

            # Release qubits for QPUs whose concurrent batch has completed
            # A batch completes when no more ready tasks will be added to it this round
            for qid in qpus_updated_this_round:
                # If the QPU is at capacity or no more tasks to co-schedule,
                # advance avail_time to batch_end and release occupancy
                qpu_avail_time[qid] = qpu_batch_end[qid]
                qpu_occupied[qid] = 0

            # Update ready set: add tasks whose all quantum predecessors are scheduled
            for tid in self.workload.quantum_tasks:
                if tid in scheduled or tid in ready:
                    continue
                preds = [p for p in self.workload.predecessors.get(tid, []) if p in self.workload.quantum_tasks]
                if all(p in scheduled for p in preds):
                    ready.add(tid)

        return sched

    def _schedule_quantum_no_reservation(self, assignment, mapping):
        """Schedule quantum tasks without proactive entanglement reservation (ablation).

        Also uses space sharing for C2 compliance, same as the main scheduler.
        """
        sched = {}
        # Space-sharing state (same as main scheduler)
        qpu_occupied = {q: 0 for q in self.system.qpus}
        qpu_avail_time = {q: 0.0 for q in self.system.qpus}
        qpu_batch_end = {q: 0.0 for q in self.system.qpus}
        qpu_capacity = {q: self.system.qpus[q]["n_qubits"] for q in self.system.qpus}
        priorities = self._compute_priorities()

        ready = set()
        for tid in self.workload.quantum_tasks:
            preds = [p for p in self.workload.predecessors.get(tid, []) if p in self.workload.quantum_tasks]
            if not preds:
                ready.add(tid)

        scheduled = set()
        while ready:
            tid = min(
                ready,
                key=lambda t: (-priorities.get(t, 0), t),
            )
            ready.remove(tid)
            task = self.workload.tasks[tid]
            qid = assignment[tid]
            routing_overhead_ms = mapping[tid].get("routing_overhead_ms", 0.0)
            task_mapping = mapping[tid]
            partition_sizes = Counter(task_mapping["logical_partition"].values())
            qpu_set = list(partition_sizes)
            est_time = self._quantum_duration(
                task, qpu_set, routing_overhead_ms
            )
            space_shared = any(qpu_occupied[q] > 0 for q in qpu_set)
            est_start = 0.0
            for part_qid, size in partition_sizes.items():
                if qpu_occupied[part_qid] + size <= qpu_capacity[part_qid]:
                    est_start = max(est_start, qpu_avail_time[part_qid])
                else:
                    est_start = max(est_start, qpu_batch_end[part_qid])
                    qpu_occupied[part_qid] = 0
                    qpu_avail_time[part_qid] = qpu_batch_end[part_qid]

            ent_wait = 0.0
            for pred in self.workload.predecessors.get(tid, []):
                if pred in sched:
                    est_start = max(est_start, sched[pred]["end"])
                    pred_qid = assignment.get(pred)
                    if pred_qid and pred_qid != qid:
                        link = self.system.get_quantum_link(qid, pred_qid)
                        if link:
                            # On-demand generation uses the full Poisson window.
                            comm_time = self._compute_entanglement_time(
                                (qid, pred_qid), est_start
                            )
                            ent_wait += comm_time

            # Intra-task remote gates (no reservation = longer wait)
            n_remote = getattr(self, '_intra_task_remote_gates', {}).get(tid, 0)
            for link_key, link_remote in task_mapping["remote_gates_by_link"].items():
                link = self.system.get_quantum_link(*link_key)
                if not link:
                    raise ValueError(f"Missing entanglement link {link_key} for task {tid}.")
                service = self._quantum_epr_service(link_key)
                n_batches = math.ceil(
                    link_remote / service["max_concurrent_epr"]
                )
                per_batch = self._compute_entanglement_time(
                    link_key, est_start
                )
                ent_wait += n_batches * per_batch

            start_time = est_start + ent_wait
            end_time = start_time + est_time
            sched[tid] = {
                "start": start_time,
                "end": end_time,
                "qpu": qid,
                "qpu_set": qpu_set,
                "partition_sizes": dict(partition_sizes),
                "entanglement_wait": ent_wait,
                "n_remote_gates": n_remote,
                "n_swap_gates": mapping[tid].get("n_swap_gates", 0),
                "routing_overhead_ms": routing_overhead_ms,
                "space_shared": space_shared,
            }

            # Update space-sharing state
            for part_qid, size in partition_sizes.items():
                qpu_occupied[part_qid] += size
                qpu_batch_end[part_qid] = max(qpu_batch_end[part_qid], end_time)
                qpu_avail_time[part_qid] = qpu_batch_end[part_qid]
                qpu_occupied[part_qid] = 0

            scheduled.add(tid)

            for succ in self.workload.successors.get(tid, []):
                if succ in self.workload.quantum_tasks and succ not in scheduled:
                    preds = [p for p in self.workload.predecessors.get(succ, []) if p in self.workload.quantum_tasks]
                    if all(p in scheduled for p in preds):
                        ready.add(succ)

        return sched

    def _quantum_epr_service(self, link_key):
        """Resolve the direct-channel service used by one EPR reservation."""
        link_key = frozenset(link_key)
        endpoints = tuple(sorted(link_key))
        if len(endpoints) != 2:
            raise ValueError(
                f"Quantum EPR service requires two QPU endpoints: {endpoints}."
            )
        primary = self.system.get_quantum_link(*endpoints)
        if primary is None:
            raise ValueError(f"Missing entanglement link {endpoints}.")
        get_channels = getattr(self.system, "get_quantum_link_channels", None)
        channels = tuple(get_channels(*endpoints)) if get_channels else ()
        if not channels:
            channels = (primary,)

        configured_mode = self.quantum_epr_routing_mode
        multipath_disabled = bool(self.ablation.get("no_multipath"))
        selected_channels = (primary,)
        service_mode = "single_channel"
        duration_scale = 1.0
        reliability_window_ms = None
        planned_success_probability = None
        backup_channel_count = 0

        if configured_mode == "parallel_channels" and not multipath_disabled:
            selected_channels = channels
            if len(selected_channels) > 1:
                service_mode = "parallel_channels"
        elif configured_mode == "deadline_direct_backup":
            reliability_window_ms = self.quantum_epr_routing_policy[
                "reliability_window_ms"
            ]
            p_threshold = float(self.params["p_threshold"])
            if not 0.0 < p_threshold < 1.0:
                raise ValueError("p_threshold must be strictly between 0 and 1")

            primary_id = str(primary.get("channel_id", "primary"))
            backup_candidates = sorted(
                (
                    channel for channel in channels
                    if channel is not primary
                    and str(channel.get("channel_id")) != primary_id
                ),
                key=lambda channel: (
                    -float(channel["rate_hz"]),
                    str(channel.get("channel_id", "")),
                ),
            )
            selected = [primary]

            def success_probability(items):
                rate = sum(float(item["rate_hz"]) for item in items)
                if rate <= 0:
                    raise ValueError(
                        f"Quantum link {endpoints} must have a positive "
                        "effective EPR generation rate."
                    )
                return -math.expm1(-rate * reliability_window_ms / 1000.0)

            planned_success_probability = success_probability(selected)
            if not multipath_disabled:
                for backup in backup_candidates:
                    if planned_success_probability >= p_threshold:
                        break
                    selected.append(backup)
                    planned_success_probability = success_probability(selected)

            if planned_success_probability < p_threshold:
                reason = (
                    " because no_multipath disables direct backup channels"
                    if multipath_disabled else " with all available direct channels"
                )
                raise ValueError(
                    "deadline_direct_backup cannot reach "
                    f"p_threshold={p_threshold} for {endpoints} within "
                    f"{reliability_window_ms} ms{reason}; achieved "
                    f"P_succ={planned_success_probability}."
                )
            selected_channels = tuple(selected)
            backup_channel_count = len(selected_channels) - 1
            if backup_channel_count:
                service_mode = "deadline_direct_backup"
        elif configured_mode == "paper_fixed_25" and not multipath_disabled:
            duration_scale = 0.75
        elif configured_mode == "paper_algorithm_145" and multipath_disabled:
            duration_scale = 1.45

        rate_hz = sum(float(channel["rate_hz"])
                      for channel in selected_channels)
        capacity_epr = sum(max(int(channel["max_concurrent_epr"]), 1)
                           for channel in selected_channels)
        memory_capacities = [
            int(channel["memory_capacity_epr"])
            for channel in selected_channels
            if channel.get("memory_capacity_epr") is not None
        ]
        complete_memory_capacity = (
            len(memory_capacities) == len(selected_channels)
        )
        memory_capacity_epr = (
            sum(memory_capacities) if complete_memory_capacity else None
        )
        memory_capacity_source = (
            "selected_channels_sum"
            if complete_memory_capacity else "unspecified_channel"
        )
        channel_ids = tuple(
            str(channel.get("channel_id", f"channel_{index}"))
            for index, channel in enumerate(selected_channels)
        )
        quality = resolve_epr_service_quality(
            self.system, selected_channels
        )
        latency_policy = getattr(
            self, "quantum_link_latency_policy", None
        )
        if latency_policy is None:
            latency_policy = resolve_quantum_link_latency_policy(
                self.params, self.system.sim_params
            )
        service_latency = resolve_service_latency(
            selected_channels,
            latency_policy,
            self.system.sim_params,
        )
        return {
            "link": endpoints,
            "configured_mode": configured_mode,
            "service_mode": service_mode,
            "multipath_disabled": multipath_disabled,
            "channels": selected_channels,
            "channel_ids": channel_ids,
            "channel_count": len(selected_channels),
            "available_channel_count": len(channels),
            "rate_hz": rate_hz,
            "max_concurrent_epr": capacity_epr,
            "memory_capacity_epr": memory_capacity_epr,
            "memory_capacity_source": memory_capacity_source,
            "duration_scale": duration_scale,
            "reliability_window_ms": reliability_window_ms,
            "target_success_probability": (
                float(self.params["p_threshold"])
                if configured_mode == "deadline_direct_backup" else None
            ),
            "planned_success_probability": planned_success_probability,
            "backup_channel_count": backup_channel_count,
            **service_latency,
            **quality,
        }

    def _entanglement_generation_window_ms(self, service, epr_count=1):
        """Return generation threshold time plus explicit fixed stages."""
        return deterministic_epr_service_window_ms(
            service["rate_hz"],
            self.params["p_threshold"],
            service.get("duration_scale", 1.0),
            service.get("stage_overhead_ms", 0.0),
            epr_count=epr_count,
        )

    @staticmethod
    def _quantum_interval_fits(reservations, start, end, epr_count, capacity):
        """Check a half-open candidate interval against existing EPR usage."""
        events = [(start, 1, epr_count), (end, 0, -epr_count)]
        for item in reservations:
            if item["end"] <= start or item["start"] >= end:
                continue
            events.append((max(start, item["start"]), 1, item["epr_count"]))
            events.append((min(end, item["end"]), 0, -item["epr_count"]))
        active = 0
        for _, _, delta in sorted(events):
            active += delta
            if active > capacity:
                return False
        return True

    def _reserve_entanglement_batches(
            self, link_key, base_ready, demand, task_id, predecessor=None,
            timeline=None, deadline=None, effective_two_qubit_gates=None):
        """Reserve capacity-feasible EPR generation batches on one link."""
        timeline = timeline if timeline is not None else self.quantum_link_timeline
        link_key = frozenset(link_key)
        service = self._quantum_epr_service(link_key)
        demand = max(int(demand), 0)
        if demand == 0:
            return base_ready, []

        capacity = service["max_concurrent_epr"]
        batch_counts = []
        remaining = demand
        while remaining:
            batch_counts.append(min(capacity, remaining))
            remaining -= batch_counts[-1]
        batch_durations = [
            self._entanglement_generation_window_ms(service, count)
            for count in batch_counts
        ]
        n_batches = len(batch_counts)
        total_generation = sum(batch_durations)
        batch_duration = max(batch_durations)
        reservation_mode = self._resolve_entanglement_reservation_mode(
            task_id, effective_two_qubit_gates
        )
        lead_ms = self._entanglement_reservation_lead_ms(
            reservation_mode, service, total_generation, batch_duration
        )
        if deadline is None:
            candidate_start = max(
                0.0, base_ready - lead_ms
            )
        else:
            # Candidate reconstruction works backwards from the already chosen
            # consumption time. Capacity conflicts may still push readiness
            # past this deadline, which C9 reports explicitly.
            candidate_start = max(0.0, deadline - total_generation)
        records = []
        for batch_index, (epr_count, current_batch_duration) in enumerate(
                zip(batch_counts, batch_durations)):
            while True:
                candidate_end = candidate_start + current_batch_duration
                if self._quantum_interval_fits(
                        timeline[link_key], candidate_start, candidate_end,
                        epr_count, capacity):
                    break
                overlapping_ends = [
                    item["end"] for item in timeline[link_key]
                    if item["end"] > candidate_start
                    and item["start"] < candidate_end
                ]
                candidate_start = (
                    min(overlapping_ends) if overlapping_ends
                    else candidate_end
                )

            record = {
                "task": task_id,
                "predecessor": predecessor,
                "link": tuple(sorted(link_key)),
                "batch_index": batch_index,
                "start": candidate_start,
                "generation_start": candidate_start,
                "end": candidate_end,
                "ready": candidate_end,
                "consume": (
                    deadline if deadline is not None else candidate_end
                ),
                "deadline": deadline,
                "epr_count": epr_count,
                "capacity_epr": capacity,
                "generation_capacity_epr": capacity,
                "memory_capacity_epr": service[
                    "memory_capacity_epr"
                ],
                "memory_capacity_source": service[
                    "memory_capacity_source"
                ],
                "quantum_epr_routing_mode": service["configured_mode"],
                "quantum_epr_service_mode": service["service_mode"],
                "quantum_channel_ids": service["channel_ids"],
                "quantum_channel_count": service["channel_count"],
                "available_quantum_channel_count": service[
                    "available_channel_count"
                ],
                "effective_rate_hz": service["rate_hz"],
                "duration_scale": service["duration_scale"],
                "generation_distribution": (
                    "erlang_poisson_success_count"
                ),
                "quantum_link_latency_mode": service[
                    "quantum_link_latency_mode"
                ],
                "stage_overhead_ms": service["stage_overhead_ms"],
                "selected_channel_stage_components": service[
                    "selected_channel_stage_components"
                ],
                "multi_channel_stage_aggregation": service[
                    "multi_channel_stage_aggregation"
                ],
                "reliability_window_ms": service[
                    "reliability_window_ms"
                ],
                "target_success_probability": service[
                    "target_success_probability"
                ],
                "planned_success_probability": service[
                    "planned_success_probability"
                ],
                "backup_channel_count": service[
                    "backup_channel_count"
                ],
                "epr_fidelity": service["epr_fidelity"],
                "fidelity_source": service["fidelity_source"],
                "decoherence_rate_per_ms": service[
                    "decoherence_rate_per_ms"
                ],
                "epr_memory_lifetime_ms": service[
                    "epr_memory_lifetime_ms"
                ],
                "memory_lifetime_source": service[
                    "memory_lifetime_source"
                ],
                "event_source": "topoq_reservation",
                "reservation_mode": reservation_mode,
                "reservation_lead_ms": lead_ms,
                "effective_two_qubit_gates": (
                    effective_two_qubit_gates
                    if effective_two_qubit_gates is not None else 0
                ),
            }
            timeline[link_key].append(record)
            records.append(record)
            candidate_start = candidate_end

        ready_time = records[-1]["end"]
        if deadline is None:
            for record in records:
                record["deadline"] = ready_time
        return ready_time, records

    def _compute_entanglement_time(self, link_key, start_time):
        """Compute the configured EPR service window.

        Paper Eq. (33) gives ``P_succ(T) = 1 - exp(-lambda * T)``.
        The selected direct-channel service supplies the effective rate,
        capacity, timing semantics, and any explicitly requested paper proxy.
        The returned value is in ms.
        """
        service = self._quantum_epr_service(link_key)
        return self._entanglement_generation_window_ms(service)

    def _compute_priorities(self):
        """Compute task priorities based on critical path length."""
        task_ids = sorted(self.workload.tasks, key=str)
        indegree = {
            tid: len(self.workload.predecessors.get(tid, ()))
            for tid in task_ids
        }
        ready = [tid for tid in task_ids if indegree[tid] == 0]
        topological_order = []

        while ready:
            tid = ready.pop(0)
            topological_order.append(tid)
            for successor in sorted(
                    self.workload.successors.get(tid, ()), key=str):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=str)

        if len(topological_order) != len(task_ids):
            cyclic = sorted(
                (tid for tid in task_ids if indegree[tid] > 0), key=str
            )
            raise ValueError(
                "Workload dependencies must form a DAG; cycle involves: "
                + ", ".join(map(str, cyclic))
            )

        priorities = {}
        for tid in reversed(topological_order):
            successor_priority = max(
                (
                    priorities[successor]
                    for successor in self.workload.successors.get(tid, ())
                ),
                default=0.0,
            )
            priorities[tid] = (
                self._reference_duration(tid) + successor_priority
            )
        return priorities

    # ==================================================================
    # Phase 4: Network-aware Classical Task Scheduling
    # ==================================================================

    def _phase4_classical_scheduling(self, assignment, sched_q,
                                     preserve_assignment=False):
        """Phase 4: Network-aware classical task scheduling.

        Key: classical tasks are co-scheduled with quantum tasks using
        computation-communication overlap. Data-locality-aware placement
        minimizes quantum-classical transfer overhead.
        """
        sched = {}
        node_busy_until = {n: 0.0 for n in self.system.nodes}
        self.classical_link_timeline = defaultdict(list)

        # Data-locality-aware assignment.  SA migration can request a rebuild
        # that preserves its candidate node choices.
        if not preserve_assignment:
            for tid in self.workload.classical_tasks:
                task = self.workload.tasks[tid]
                mem_mb = task.get("memory_mb", 100)

                # Find best node (data locality)
                best_node = None
                best_score = -1
                for nid, node in self.system.nodes.items():
                    if node["memory_gb"] * 1024 < mem_mb:
                        continue
                    locality_score = 0
                    for pred in self.workload.predecessors.get(tid, []):
                        if pred in assignment and assignment[pred] in self.system.qpus:
                            pred_qid = assignment[pred]
                            iface = self._get_interface_link(pred_qid)
                            if iface["classical_node_id"] == nid:
                                locality_score += 1
                    est_time = self._classical_duration(task, nid)
                    load_penalty = node_busy_until[nid] * 0.001
                    score = locality_score * 10 + 1.0 / (1 + est_time) - load_penalty
                    if score > best_score:
                        best_score = score
                        best_node = nid

                if best_node is None:
                    best_node = self.system.get_node_ids()[0]
                assignment[tid] = best_node

        # Schedule with critical path
        priorities = self._compute_priorities()
        ready = set()
        for tid in self.workload.classical_tasks:
            preds = [p for p in self.workload.predecessors.get(tid, []) if p in self.workload.classical_tasks]
            if not preds:
                ready.add(tid)

        scheduled = set()
        while ready:
            # Schedule all ready classical tasks in parallel (different nodes)
            ready_sorted = sorted(
                ready,
                key=lambda t: (-priorities.get(t, 0), t),
            )
            for tid in ready_sorted:
                task = self.workload.tasks[tid]
                nid = assignment[tid]
                node = self.system.nodes[nid]
                est_time = self._classical_duration(task, nid)

                resource_ready = node_busy_until[nid]
                predecessor_ready = 0.0
                transfer_ready = 0.0
                network_transfers = []
                dependency_transfers = []
                for pred in dict.fromkeys(
                        self.workload.predecessors.get(tid, [])):
                    if pred in sched_q:
                        pred_end = sched_q[pred]["end"]
                        predecessor_ready = max(predecessor_ready, pred_end)
                        pred_qid = assignment.get(pred)
                        transfer = {
                            "predecessor": pred,
                            "start": pred_end,
                            "end": pred_end,
                            "type": "quantum-classical",
                            "source_resource": pred_qid,
                            "target_resource": nid,
                        }
                        # Quantum-classical interface transfer
                        # With co-scheduling, QC transfer overlaps with quantum execution
                        if pred_qid and pred_qid in self.system.qpus:
                            interface_ms = self._interface_transfer_ms(
                                pred_qid, pred, tid
                            )
                            transfer["end"] = pred_end + interface_ms
                            transfer_ready = max(
                                transfer_ready, transfer["end"]
                            )
                        dependency_transfers.append(transfer)
                    elif pred in sched:
                        pred_end = sched[pred]["end"]
                        predecessor_ready = max(predecessor_ready, pred_end)
                        # Classical-classical network transfer
                        pred_nid = assignment.get(pred)
                        transfer = {
                            "predecessor": pred,
                            "start": pred_end,
                            "end": pred_end,
                            "source_resource": pred_nid,
                            "target_resource": nid,
                        }
                        if pred_nid and pred_nid != nid:
                            data_mb = self._edge_data_mb(pred, tid)
                            path_info = self._select_classical_path(pred_nid, nid, data_mb)
                            reservation = self._reserve_classical_bandwidth(
                                path_info, sched[pred]["end"], data_mb, tid, pred
                            )
                            network_transfers.append(reservation)
                            transfer_ready = max(transfer_ready, reservation["end"])
                            transfer = dict(
                                reservation,
                                type="classical-classical",
                                source_resource=pred_nid,
                                target_resource=nid,
                            )
                        dependency_transfers.append(transfer)

                base_ready = max(resource_ready, predecessor_ready)
                start_time = max(base_ready, transfer_ready)
                comm_time = max(0.0, start_time - base_ready)
                end_time = start_time + est_time
                sched[tid] = {
                    "start": start_time,
                    "end": end_time,
                    "node": nid,
                    "comm_time": comm_time,
                    "network_transfers": network_transfers,
                    "dependency_transfers": dependency_transfers,
                }
                node_busy_until[nid] = end_time
                scheduled.add(tid)
                ready.discard(tid)

            # Update ready set
            for tid in self.workload.classical_tasks:
                if tid in scheduled or tid in ready:
                    continue
                preds = [p for p in self.workload.predecessors.get(tid, []) if p in self.workload.classical_tasks]
                if all(p in scheduled for p in preds):
                    ready.add(tid)

        return sched

    def _rebuild_classical_candidate(self, sched, assignment):
        """Rebuild Phase 4 and its link timeline for an SA candidate."""
        quantum_sched = {
            tid: deepcopy(item) for tid, item in sched.items()
            if tid in self.workload.quantum_tasks
        }
        previous_timeline = getattr(self, "classical_link_timeline", defaultdict(list))
        classical_sched = self._phase4_classical_scheduling(
            assignment, quantum_sched, preserve_assignment=True
        )
        candidate_timeline = deepcopy(self.classical_link_timeline)
        self.classical_link_timeline = previous_timeline

        # Refresh reporting-only Eq. (40) fields for rebuilt classical tasks.
        if not self.ablation.get("no_comm_overlap"):
            self._apply_overlap(classical_sched, assignment)

        rebuilt = dict(quantum_sched)
        rebuilt.update(classical_sched)
        return self._fix_dag_precedence(rebuilt), candidate_timeline

    def _rebuild_quantum_candidate_timeline(self, sched, mapping):
        """Recreate candidate EPR reservations without mutating scheduler state."""
        timeline = defaultdict(list)
        quantum_tasks = sorted(
            (tid for tid in self.workload.quantum_tasks if tid in sched),
            key=lambda tid: (sched[tid]["start"], tid),
        )
        for tid in quantum_tasks:
            item = sched[tid]
            task_start = item["start"]
            base_ready = max(0.0, task_start - item.get("entanglement_wait", 0.0))
            for link_key, remote in mapping.get(tid, {}).get(
                    "remote_gates_by_link", {}).items():
                self._reserve_entanglement_batches(
                    frozenset(link_key), base_ready, remote, tid,
                    timeline=timeline, deadline=task_start,
                    effective_two_qubit_gates=item.get(
                        "effective_two_qubit_gates"
                    ),
                )

            for transfer in item.get("dependency_transfers", []):
                if transfer.get("type") != "quantum-quantum":
                    continue
                link_key = transfer.get("quantum_link")
                if not link_key:
                    continue
                transfer_deadline = transfer.get("end", task_start)
                transfer_base = transfer.get("start", transfer_deadline)
                self._reserve_entanglement_batches(
                    frozenset(link_key), transfer_base, 1, tid,
                    predecessor=transfer.get("predecessor"),
                    timeline=timeline, deadline=transfer_deadline,
                    effective_two_qubit_gates=item.get(
                        "effective_two_qubit_gates"
                    ),
                )
        return timeline

    def _edge_data_mb(self, predecessor, task_id):
        return edge_data_mb(self.workload, predecessor, task_id)

    def _select_classical_path(self, src_node, tgt_node, data_mb):
        mode = self.classical_routing_policy["mode"]
        if self.ablation.get("no_multipath"):
            mode = "single_widest"
        return select_classical_route(
            self.system,
            src_node,
            tgt_node,
            mode,
            multipath_k=self.classical_routing_policy["multipath_k"],
        )

    def _reserve_classical_bandwidth(self, path_info, earliest_start, data_mb,
                                     task_id, predecessor_id):
        return reserve_classical_route(
            self.system, self.classical_link_timeline, path_info,
            earliest_start, data_mb, task_id, predecessor_id,
            sharing_mode=self.classical_bandwidth_sharing_mode,
        )

    # ==================================================================
    # Phase 5: Joint Optimization & Refinement
    # ==================================================================

    def _phase5_joint_optimize(self, full_sched, assignment, mapping):
        """Phase 5: Joint optimization and refinement.

        Flow: unified hybrid-DAG schedule → reporting proxy → SA → DAG repair.
        The input already satisfies cross-domain precedence because Phase 3/4
        uses one ready queue and reserves each dependency transfer before its
        successor starts.  The final repair is defensive: it fixes violations
        that an SA neighborhood move may introduce.
        """
        full_sched = deepcopy(full_sched)

        # Communication-computation overlap
        if not self.ablation.get("no_comm_overlap"):
            full_sched = self._apply_overlap(full_sched, assignment)

        # Simulated annealing local search
        if not self.ablation.get("no_sim_annealing"):
            (full_sched, optimized_assignment, optimized_mapping,
             optimized_classical_timeline,
             optimized_quantum_timeline,
             optimized_remote_gates,
             optimized_remote_gates_by_link) = self._simulated_annealing(
                full_sched, assignment, mapping
            )
            assignment.clear()
            assignment.update(optimized_assignment)
            mapping.clear()
            mapping.update(optimized_mapping)
            self.classical_link_timeline = optimized_classical_timeline
            self.quantum_link_timeline = optimized_quantum_timeline
            self._intra_task_remote_gates = optimized_remote_gates
            self._intra_task_remote_gates_by_link = (
                optimized_remote_gates_by_link
            )

        # DAG precedence repair (post-SA): fix violations introduced
        # by critical_path_opt's blind propagation to successors.
        full_sched = self._fix_dag_precedence(full_sched)

        return full_sched

    def _fix_dag_precedence(self, sched):
        """Resolve DAG precedence violations (C3) by shifting tasks later.

        Unified schedule generation already enforces predecessor and dependency-
        transfer completion.  This method processes tasks in topological order
        as a defensive post-SA repair and pushes any task whose start precedes
        either boundary.

        Complexity: O(|V| + |E|) via Kahn's algorithm.
        """
        if not sched:
            return sched

        # 1. Topological sort (Kahn's algorithm) over scheduled tasks
        in_degree = {}
        for tid in sched:
            in_degree[tid] = 0
        for tid in sched:
            for pred in dict.fromkeys(
                    self.workload.predecessors.get(tid, [])):
                if pred in sched:
                    in_degree[tid] += 1

        queue = deque([tid for tid, d in in_degree.items() if d == 0])
        topo_order = []
        while queue:
            tid = queue.popleft()
            topo_order.append(tid)
            for succ in dict.fromkeys(
                    self.workload.successors.get(tid, [])):
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        # 2. Fix in topological order: wait for both predecessors and transfers.
        for tid in topo_order:
            if tid not in sched:
                continue
            earliest = 0.0
            for pred in dict.fromkeys(
                    self.workload.predecessors.get(tid, [])):
                if pred in sched:
                    earliest = max(earliest, sched[pred]["end"])
                    transfers = [
                        transfer
                        for transfer in sched[tid].get(
                            "dependency_transfers", []
                        )
                        if transfer.get("predecessor") == pred
                    ]
                    if len(transfers) == 1:
                        earliest = max(
                            earliest,
                            transfers[0].get("end", sched[pred]["end"]),
                        )
            current_start = sched[tid]["start"]
            if current_start < earliest:
                shift = earliest - current_start
                sched[tid]["start"] += shift
                sched[tid]["end"] += shift

        return sched

    def _apply_overlap(self, sched, assignment):
        """Report paper Eq. (40) credits without changing physical timing."""
        mode = (
            "none" if self.ablation.get("no_comm_overlap")
            else getattr(
                self,
                "communication_overlap_mode",
                DEFAULT_PARAMS["communication_overlap_mode"],
            )
        )
        return annotate_communication_overlap(sched, mode)

    @staticmethod
    def _normalize_objective_component(value, baseline,
                                       zero_reference_scale):
        """Normalize one cost against the initial SA state."""
        eps = 1e-12
        value = max(float(value), 0.0)
        baseline = max(float(baseline), 0.0)
        if baseline > eps:
            return value / baseline
        if value <= eps:
            return 0.0
        return value / zero_reference_scale

    def _objective_raw_components(self, sched, assignment):
        """Measure the unnormalised terms used by the Phase-5 objective."""
        makespan = max((item["end"] for item in sched.values()), default=0.0)
        if getattr(self, "objective_mode", "joint") in {"makespan", "makespan_penalty"}:
            # Pure-Cmax mode: the communication/fidelity terms are not part of
            # the search objective, so skip their (expensive) evaluation.
            return {
                "makespan": makespan,
                "entanglement_wait": 0.0,
                "remote_gates": 0.0,
                "classical_comm": 0.0,
                "fidelity_loss": 0.0,
            }
        entanglement_wait = sum(
            max(item.get("entanglement_wait", 0.0), 0.0)
            for item in sched.values()
        )
        remote_gates = sum(
            max(item.get("n_remote_gates", 0), 0)
            for tid, item in sched.items()
            if tid in self.workload.quantum_tasks
        )
        classical_comm = sum(
            max(item.get("comm_time", 0.0), 0.0)
            for item in sched.values()
        )

        # Products of many gate fidelities underflow easily.  The equivalent
        # additive loss -log(F) is stable and preserves fidelity ordering.
        fidelity_loss = 0.0
        for tid in self.workload.quantum_tasks:
            if tid not in sched:
                continue
            qid = assignment.get(tid)
            qpu = self.system.qpus.get(qid)
            if qpu is None:
                fidelity_loss += 745.0
                continue
            fidelity_details = self._estimate_fidelity_details(
                qpu,
                self.workload.tasks[tid],
                estimated_remote_gates=sched[tid].get("n_remote_gates", 0),
                estimated_hold_ms=sched[tid].get("entanglement_wait", 0.0),
                estimated_swap_gates=sched[tid].get("n_swap_gates", 0),
            )
            # Keep the true log value below the reporting floor. Otherwise
            # long QAOA candidates all collapse to the same 1e-300 score.
            fidelity_loss += max(-float(fidelity_details["log_fidelity"]), 0.0)

        return {
            "makespan": makespan,
            "entanglement_wait": entanglement_wait,
            "remote_gates": float(remote_gates),
            "classical_comm": classical_comm,
            "fidelity_loss": fidelity_loss,
        }

    def _build_objective_reference(self, sched, assignment):
        """Freeze the initial SA solution as the dimensionless scale."""
        return self._objective_raw_components(sched, assignment)

    def _compute_schedule_objective(self, sched, assignment, mapping,
                                    classical_link_timeline,
                                    quantum_link_timeline, reference,
                                    intra_remote_gates=None):
        """Evaluate the weighted joint objective and candidate constraints.

        Each physical quantity is divided by its value in the initial SA
        solution. A zero initial value uses an explicit native-unit reference
        scale. Quantum communication combines entanglement waiting (70%) and
        the exact number of inter-QPU gates (30%).
        """
        raw = self._objective_raw_components(sched, assignment)

        violations = validate_schedule(
            self.system,
            self.workload,
            sched,
            assignment,
            mapping,
            params=self.params,
            intra_remote_gates=(
                getattr(self, "_intra_task_remote_gates", {})
                if intra_remote_gates is None else intra_remote_gates
            ),
            classical_link_timeline=classical_link_timeline,
            quantum_link_timeline=quantum_link_timeline,
            include_reported_fidelity=not (
                getattr(self, "_fast_validation", False)
                and str(self.params.get(
                    "fidelity_constraint_mode", "report_only"
                )) == "report_only"
            ),
        )
        enforced_violations = [
            violation for violation in violations
            if violation.get("enforced", True)
        ]

        def ratio(name):
            return self._normalize_objective_component(
                raw.get(name, 0.0),
                reference.get(name, 0.0),
                PHASE5_OBJECTIVE_ZERO_REFERENCE_SCALES[name],
            )

        normalized = {
            "makespan": ratio("makespan"),
            "entanglement_wait": ratio("entanglement_wait"),
            "remote_gates": ratio("remote_gates"),
            "classical_comm": ratio("classical_comm"),
            "fidelity_loss": ratio("fidelity_loss"),
        }
        normalized["quantum_comm"] = (
            0.70 * normalized["entanglement_wait"]
            + 0.30 * normalized["remote_gates"]
        )

        weights = {
            "makespan": self.phase5_objective_weights[
                "objective_w_makespan"
            ],
            "quantum_comm": self.phase5_objective_weights[
                "objective_w_quantum_comm"
            ],
            "classical_comm": self.phase5_objective_weights[
                "objective_w_classical_comm"
            ],
            "fidelity_loss": self.phase5_objective_weights[
                "objective_w_fidelity"
            ],
        }
        if getattr(self, "objective_mode", "joint") in {"makespan", "makespan_penalty"}:
            # Pure makespan uses the native millisecond value so that the SA
            # temperature keeps a physical meaning (normalized ratios are ~1).
            total = raw["makespan"]
            if getattr(self, "objective_mode", "joint") == "makespan_penalty":
                total += (
                    float(self.params.get("objective_hard_penalty", 1000.0))
                    * len(enforced_violations)
                )
            weights = {
                "makespan": 1.0,
                "quantum_comm": 0.0,
                "classical_comm": 0.0,
                "fidelity_loss": 0.0,
            }
        else:
            total = (
                weights["makespan"] * normalized["makespan"]
                + weights["quantum_comm"] * normalized["quantum_comm"]
                + weights["classical_comm"] * normalized["classical_comm"]
                + weights["fidelity_loss"] * normalized["fidelity_loss"]
            )
        return {
            "total": total,
            "raw": raw,
            "normalized": normalized,
            "weights": weights,
            "objective_mode": getattr(self, "objective_mode", "joint"),
            "zero_reference_scales": dict(
                PHASE5_OBJECTIVE_ZERO_REFERENCE_SCALES
            ),
            "violations": violations,
            "enforced_violations": enforced_violations,
            "reported_violations": [
                violation for violation in violations
                if not violation.get("enforced", True)
            ],
        }

    def _quantum_qpu_load(self, sched):
        """Return duration-weighted physical QPU load for SA remapping."""
        qpu_load = {qid: 0.0 for qid in self.system.get_qpu_ids()}
        for tid in self.workload.quantum_tasks:
            if tid not in sched:
                continue
            item = sched[tid]
            duration = max(item["end"] - item["start"], 0.0)
            partition_sizes = item.get("partition_sizes")
            if partition_sizes:
                total_qubits = max(sum(partition_sizes.values()), 1)
                for qid, size in partition_sizes.items():
                    qpu_load[qid] += duration * size / total_qubits
                continue
            qid = item.get("qpu")
            if qid in qpu_load:
                qpu_load[qid] += duration
        return qpu_load

    def _build_sa_physical_remap_state(
            self, tid, sched, assignment, mapping, remote_gates,
            remote_gates_by_link):
        """Build and fully reschedule one transactional SA remap state."""
        qpu_load = self._quantum_qpu_load(sched)
        result = self._build_quantum_mapping_candidate(
            tid,
            mapping,
            qpu_load,
            selection_strategy="least_loaded",
            require_coherence=True,
            exclude_equivalent_to=mapping.get(tid, {}),
        )
        if result["mapping"] is None:
            return None

        candidate_assignment = deepcopy(assignment)
        candidate_mapping = deepcopy(mapping)
        previous_mapping = deepcopy(mapping[tid])
        candidate_mapping[tid] = result["mapping"]
        candidate_mapping[tid]["sa_remap"] = {
            "selection_strategy": "least_loaded",
            "candidate_rank": result["rank"],
            "candidate_attempts": result["attempts"],
            "previous_qpu_set": list(
                previous_mapping.get("qpu_set", [])
            ),
            "selected_qpu_set": list(
                candidate_mapping[tid]["qpu_set"]
            ),
            "qpu_load_ms": dict(qpu_load),
        }
        candidate_assignment[tid] = candidate_mapping[tid]["primary_qpu"]
        candidate_remote_gates = deepcopy(remote_gates)
        candidate_remote_gates[tid] = result["n_remote_gates"]
        candidate_remote_gates_by_link = deepcopy(remote_gates_by_link)
        candidate_remote_gates_by_link[tid] = result[
            "remote_gates_by_link"
        ]

        saved_state = {
            "classical_link_timeline": deepcopy(
                getattr(self, "classical_link_timeline", {})
            ),
            "quantum_link_timeline": deepcopy(
                getattr(self, "quantum_link_timeline", {})
            ),
            "remote_gates": deepcopy(
                getattr(self, "_intra_task_remote_gates", {})
            ),
            "remote_gates_by_link": deepcopy(
                getattr(self, "_intra_task_remote_gates_by_link", {})
            ),
        }
        try:
            self._intra_task_remote_gates = deepcopy(
                candidate_remote_gates
            )
            self._intra_task_remote_gates_by_link = deepcopy(
                candidate_remote_gates_by_link
            )
            candidate_sched = self._phase3_4_joint_scheduling(
                candidate_assignment, candidate_mapping
            )
            if not self.ablation.get("no_comm_overlap"):
                candidate_sched = self._apply_overlap(
                    candidate_sched, candidate_assignment
                )
            candidate_classical_timeline = deepcopy(
                self.classical_link_timeline
            )
            candidate_quantum_timeline = deepcopy(
                self.quantum_link_timeline
            )
        except ValueError:
            return None
        finally:
            self.classical_link_timeline = saved_state[
                "classical_link_timeline"
            ]
            self.quantum_link_timeline = saved_state[
                "quantum_link_timeline"
            ]
            self._intra_task_remote_gates = saved_state["remote_gates"]
            self._intra_task_remote_gates_by_link = saved_state[
                "remote_gates_by_link"
            ]

        if (self._physical_mapping_signature(candidate_mapping[tid])
                == self._physical_mapping_signature(mapping[tid])):
            return None
        return {
            "schedule": candidate_sched,
            "assignment": candidate_assignment,
            "mapping": candidate_mapping,
            "classical_link_timeline": candidate_classical_timeline,
            "quantum_link_timeline": candidate_quantum_timeline,
            "remote_gates": candidate_remote_gates,
            "remote_gates_by_link": candidate_remote_gates_by_link,
        }

    @staticmethod
    def _sa_acceptance_probability(delta_objective, temperature):
        """Return the standard Metropolis probability for a worse move."""
        if delta_objective <= 0.0:
            return 1.0
        if temperature <= 0.0:
            return 0.0
        return math.exp(-delta_objective / max(temperature, 1e-12))

    def _simulated_annealing(self, sched, assignment, mapping):
        """Simulated annealing local search (Step 5.3).

        Tries small perturbations and accepts improvements (or worse solutions
        with probability exp(-delta/T)). Only keeps the best solution.

        Key moves:
        - reorder: shift tasks earlier when dependencies allow
        - migration: move classical tasks to less loaded nodes
        - remap: reassign quantum tasks to less loaded QPUs
        """
        current_sched = deepcopy(sched)
        current_assignment = deepcopy(assignment)
        current_mapping = deepcopy(mapping)
        current_classical_timeline = deepcopy(
            getattr(self, "classical_link_timeline", {})
        )
        current_quantum_timeline = deepcopy(
            getattr(self, "quantum_link_timeline", {})
        )
        current_remote_gates = deepcopy(
            getattr(self, "_intra_task_remote_gates", {})
        )
        current_remote_gates_by_link = deepcopy(
            getattr(self, "_intra_task_remote_gates_by_link", {})
        )
        initial_classical_timeline = deepcopy(current_classical_timeline)
        initial_quantum_timeline = deepcopy(current_quantum_timeline)
        current_makespan = max(s["end"] for s in current_sched.values())

        reference = self._build_objective_reference(
            current_sched, current_assignment
        )
        self._fast_validation = True
        current_objective = self._compute_schedule_objective(
            current_sched, current_assignment, current_mapping,
            initial_classical_timeline, initial_quantum_timeline, reference,
            intra_remote_gates=current_remote_gates,
        )
        initial_objective = deepcopy(current_objective)

        initial_temperature = (
            self.params["T_SA_init_frac"] * current_objective["total"]
        )
        T_SA = initial_temperature
        cooling = self.params["SA_cooling"]
        max_non_improve = self.params["SA_max_non_improve"]
        non_improve_count = 0
        best_sched = deepcopy(current_sched)
        best_assignment = deepcopy(current_assignment)
        best_mapping = deepcopy(current_mapping)
        best_classical_timeline = deepcopy(current_classical_timeline)
        best_quantum_timeline = deepcopy(current_quantum_timeline)
        best_remote_gates = deepcopy(current_remote_gates)
        best_remote_gates_by_link = deepcopy(
            current_remote_gates_by_link
        )
        best_makespan = current_makespan
        best_objective = deepcopy(current_objective)

        # Active moves must represent real scheduling or resource decisions.
        # EPR-policy and critical-path extensions must rebuild physical state
        # before they can be re-enabled in this list.
        moves = ["reorder", "reorder", "migration", "remap"]
        move_stats = {
            move_name: {
                "attempted": 0,
                "feasible": 0,
                "accepted": 0,
                "best": 0,
            }
            for move_name in dict.fromkeys(moves)
        }
        qpu_ids = self.system.get_qpu_ids()
        node_ids = self.system.get_node_ids()

        for _ in range(2000):
            if non_improve_count >= max_non_improve:
                break
            if T_SA < 1e-4:
                break

            move = self.rng.choice(moves)
            move_stats[move]["attempted"] += 1
            new_sched = deepcopy(current_sched)
            new_assignment = deepcopy(current_assignment)
            new_mapping = deepcopy(current_mapping)
            new_classical_timeline = deepcopy(current_classical_timeline)
            new_quantum_timeline = deepcopy(current_quantum_timeline)
            new_remote_gates = deepcopy(current_remote_gates)
            new_remote_gates_by_link = deepcopy(
                current_remote_gates_by_link
            )
            candidate_changed = False

            if move == "reorder":
                # Try shifting a task earlier if possible (gap filling)
                reorder_candidates = [
                    tid for tid in self.workload.quantum_tasks if tid in new_sched
                ]
                tid = self.rng.choice(reorder_candidates) if reorder_candidates else None
                if tid in new_sched:
                    # Find earliest possible start based on predecessors
                    earliest = 0.0
                    for pred in self.workload.predecessors.get(tid, []):
                        if pred in new_sched:
                            earliest = max(earliest, new_sched[pred]["end"])

                    current_start = new_sched[tid]["start"]
                    if current_start > earliest + 0.1:
                        # Shift earlier by a meaningful amount
                        shift_amount = self.rng.uniform(0.3, 1.0) * (current_start - earliest)
                        new_start = max(earliest, current_start - shift_amount)
                        duration = new_sched[tid]["end"] - new_sched[tid]["start"]
                        new_sched[tid]["start"] = new_start
                        new_sched[tid]["end"] = new_start + duration
                        candidate_changed = True

            elif move == "migration" and len(node_ids) >= 2:
                c_tasks = [t for t in self.workload.classical_tasks if t in new_sched]
                if c_tasks:
                    tid = self.rng.choice(c_tasks)
                    old_node = new_assignment.get(tid)
                    # Try migrating to a less loaded node
                    memory_mb = self.workload.tasks[tid].get("memory_mb", 0)
                    feasible_nodes = [
                        node for node in node_ids
                        if self.system.nodes[node]["memory_gb"] * 1024 >= memory_mb
                    ]
                    new_node = min(feasible_nodes, key=lambda n: sum(
                        s["end"] - s["start"] for t, s in new_sched.items()
                        if new_assignment.get(t) == n and "node" in s
                    ))
                    if new_node != old_node:
                        new_assignment[tid] = new_node
                        candidate_changed = True

            elif move == "remap" and len(qpu_ids) >= 2:
                q_tasks = [t for t in self.workload.quantum_tasks if t in new_sched]
                if q_tasks:
                    tid = self.rng.choice(q_tasks)
                    try:
                        remap_state = self._build_sa_physical_remap_state(
                            tid,
                            current_sched,
                            current_assignment,
                            current_mapping,
                            current_remote_gates,
                            current_remote_gates_by_link,
                        )
                    except ValueError:
                        remap_state = None
                    if remap_state is not None:
                        new_sched = remap_state["schedule"]
                        new_assignment = remap_state["assignment"]
                        new_mapping = remap_state["mapping"]
                        new_classical_timeline = remap_state[
                            "classical_link_timeline"
                        ]
                        new_quantum_timeline = remap_state[
                            "quantum_link_timeline"
                        ]
                        new_remote_gates = remap_state["remote_gates"]
                        new_remote_gates_by_link = remap_state[
                            "remote_gates_by_link"
                        ]
                        candidate_changed = True

            elif move == "compress":
                # Deprecated — kept for reference, not in moves list
                pass

            elif move == "entanglement_opt":
                # SA optimizes entanglement routing and reduces decoherence penalty.
                # Find quantum tasks with decoherence_penalty and reduce them
                q_tasks = [t for t in self.workload.quantum_tasks if t in new_sched
                          and new_sched[t].get("decoherence_penalty", 0) > 0.01]
                if q_tasks:
                    tid = self.rng.choice(q_tasks)
                    # SA can find 10-30% reduction in decoherence penalty
                    reduction = self.rng.uniform(0.10, 0.30)
                    saved = new_sched[tid].get("decoherence_penalty", 0) * reduction
                    new_sched[tid]["decoherence_penalty"] = new_sched[tid].get("decoherence_penalty", 0) - saved
                    # Clamp to prevent end < start (Eq.~\ref{eq:36} clamping)
                    min_end = new_sched[tid]["start"] + 0.1
                    new_sched[tid]["end"] = max(new_sched[tid]["end"] - saved, min_end)
                    candidate_changed = True
                else:
                    # No decoherence tasks — try reducing execution time via better compilation
                    q_tasks = [t for t in self.workload.quantum_tasks if t in new_sched]
                    if q_tasks:
                        tid = self.rng.choice(q_tasks)
                        duration = new_sched[tid]["end"] - new_sched[tid]["start"]
                        if duration > 1.0:
                            reduction = self.rng.uniform(0.02, 0.08)
                            saved = duration * reduction
                            # Clamp to prevent end < start (Eq.~\ref{eq:36} clamping)
                            min_end = new_sched[tid]["start"] + 0.1
                            new_sched[tid]["end"] = max(new_sched[tid]["end"] - saved, min_end)
                            candidate_changed = True

            elif move == "critical_path_opt":
                # SA optimizes the critical path by reducing execution time
                # of a task on the critical path and propagating the reduction
                # to all successors.
                quantum_items = [t for t in self.workload.quantum_tasks if t in new_sched]
                cp_tid = max(quantum_items, key=lambda t: new_sched[t]["end"]) if quantum_items else None
                # Walk up the predecessor chain to find a task with real duration
                current = cp_tid
                target_tid = None
                for _ in range(15):
                    if current is None or current not in new_sched:
                        break
                    dur = new_sched[current]["end"] - new_sched[current]["start"]
                    if dur > 2.0:
                        target_tid = current
                        break
                    preds = self.workload.predecessors.get(current, [])
                    if not preds:
                        break
                    current = preds[0] if preds[0] in new_sched else current

                if target_tid is not None:
                    dur = new_sched[target_tid]["end"] - new_sched[target_tid]["start"]
                    reduction = self.rng.uniform(0.03, 0.08)
                    saved = dur * reduction
                    # Clamp to prevent end < start (Eq.~\ref{eq:36} clamping)
                    min_end = new_sched[target_tid]["start"] + 0.1
                    new_sched[target_tid]["end"] = max(new_sched[target_tid]["end"] - saved, min_end)
                    candidate_changed = True
                    # Propagate reduction to all successors
                    to_update = [target_tid]
                    visited_set = set()
                    while to_update:
                        tid = to_update.pop(0)
                        if tid in visited_set:
                            continue
                        visited_set.add(tid)
                        for succ in self.workload.successors.get(tid, []):
                            if succ in new_sched:
                                new_sched[succ]["start"] -= saved
                                new_sched[succ]["end"] -= saved
                                to_update.append(succ)

            if not candidate_changed:
                non_improve_count += 1
                T_SA *= cooling
                continue

            # Classical timing and paths are derived state.  Rebuild them for
            # every candidate after its quantum timing or node assignment move.
            try:
                new_sched, new_classical_timeline = (
                    self._rebuild_classical_candidate(
                        new_sched, new_assignment
                    )
                )
                new_quantum_timeline = (
                    self._rebuild_quantum_candidate_timeline(
                        new_sched, new_mapping
                    )
                )
                new_makespan = max(
                    item["end"] for item in new_sched.values()
                )
                new_objective = self._compute_schedule_objective(
                    new_sched,
                    new_assignment,
                    new_mapping,
                    new_classical_timeline,
                    new_quantum_timeline,
                    reference,
                    intra_remote_gates=new_remote_gates,
                )
            except ValueError:
                non_improve_count += 1
                T_SA *= cooling
                continue

            if new_objective["enforced_violations"]:
                non_improve_count += 1
                T_SA *= cooling
                continue

            move_stats[move]["feasible"] += 1

            if new_objective["total"] <= current_objective["total"]:
                current_sched = new_sched
                current_assignment = new_assignment
                current_mapping = new_mapping
                current_classical_timeline = new_classical_timeline
                current_quantum_timeline = new_quantum_timeline
                current_remote_gates = new_remote_gates
                current_remote_gates_by_link = new_remote_gates_by_link
                current_makespan = new_makespan
                current_objective = new_objective
                move_stats[move]["accepted"] += 1
                non_improve_count = 0
                if new_objective["total"] < best_objective["total"]:
                    best_sched = deepcopy(new_sched)
                    best_assignment = deepcopy(new_assignment)
                    best_mapping = deepcopy(new_mapping)
                    best_classical_timeline = deepcopy(new_classical_timeline)
                    best_quantum_timeline = deepcopy(new_quantum_timeline)
                    best_remote_gates = deepcopy(new_remote_gates)
                    best_remote_gates_by_link = deepcopy(
                        new_remote_gates_by_link
                    )
                    best_makespan = new_makespan
                    best_objective = deepcopy(new_objective)
                    move_stats[move]["best"] += 1
            else:
                delta = new_objective["total"] - current_objective["total"]
                acceptance_probability = self._sa_acceptance_probability(
                    delta, T_SA
                )
                if self.rng.random() < acceptance_probability:
                    current_sched = new_sched
                    current_assignment = new_assignment
                    current_mapping = new_mapping
                    current_classical_timeline = new_classical_timeline
                    current_quantum_timeline = new_quantum_timeline
                    current_remote_gates = new_remote_gates
                    current_remote_gates_by_link = (
                        new_remote_gates_by_link
                    )
                    current_makespan = new_makespan
                    current_objective = new_objective
                    move_stats[move]["accepted"] += 1
                non_improve_count += 1

            T_SA *= cooling

        self.last_sa_objective = {
            "initial": initial_objective,
            "best": best_objective,
            "best_makespan": best_makespan,
            "objective_mode": getattr(self, "objective_mode", "joint"),
            "initial_temperature": initial_temperature,
            "final_temperature": T_SA,
            "acceptance_rule": "exp(-delta_objective/temperature)",
            "move_stats": move_stats,
        }
        self._fast_validation = False
        return (
            best_sched,
            best_assignment,
            best_mapping,
            best_classical_timeline,
            best_quantum_timeline,
            best_remote_gates,
            best_remote_gates_by_link,
        )

    # ==================================================================
    # Constraint Checking & Repair (Feedback Loop)
    # ==================================================================

    def _check_constraints(self, sched, assignment, mapping):
        """Return structured violations for the TopoQ feedback loop."""
        return validate_schedule(
            self.system,
            self.workload,
            sched,
            assignment,
            mapping,
            params=self.params,
            intra_remote_gates=getattr(self, "_intra_task_remote_gates", {}),
            classical_link_timeline=getattr(self, "classical_link_timeline", {}),
            quantum_link_timeline=getattr(self, "quantum_link_timeline", {}),
        )

    def _repair_mapping(self, assignment, mapping, violations):
        """Repair mapping based on constraint violations.

        Key: repairs should consider load balancing, not just QPU connectivity.
        All tasks should not be moved to the same QPU.
        """
        # Count current QPU assignments for load-aware repair
        qpu_load = defaultdict(float)
        c6_repaired = set()
        coherence_repaired = set()
        fidelity_repaired = set()
        entanglement_repaired = set()
        for tid in self.workload.quantum_tasks:
            task_mapping = mapping.get(tid, {})
            partition_sizes = Counter(
                task_mapping.get("logical_partition", {}).values()
            )
            if not partition_sizes:
                qid = assignment.get(tid)
                if qid in self.system.qpus:
                    partition_sizes[qid] = self.workload.tasks[tid].get(
                        "n_qubits", 1
                    )
            if not partition_sizes:
                continue
            duration = self._quantum_duration(tid, partition_sizes)
            total_qubits = max(sum(partition_sizes.values()), 1)
            for qid, size in partition_sizes.items():
                qpu_load[qid] += duration * size / total_qubits

        for violation in violations:
            vtype = violation["constraint"]
            tid = violation["task"]
            qid = violation.get("resource")
            if vtype == "C6_MAPPING_CONSISTENCY" and tid not in c6_repaired:
                predecessor = violation.get("details", {}).get("predecessor")
                if predecessor in mapping:
                    shared, _ = self.workload.get_shared_logical_qubits(
                        predecessor, tid
                    )
                    required = self.workload.tasks[tid].get("n_qubits", 0)
                    if set(shared) == set(range(required)):
                        source = mapping[predecessor]
                        mapping[tid]["logical_partition"] = deepcopy(
                            source["logical_partition"]
                        )
                        mapping[tid]["physical_mapping"] = deepcopy(
                            source["physical_mapping"]
                        )
                        mapping[tid]["qpu_set"] = list(source["qpu_set"])
                        mapping[tid]["primary_qpu"] = source["primary_qpu"]
                        assignment[tid] = source["primary_qpu"]
                        circuit = load_circuit(
                            self.workload.tasks[tid].get("circuit_id")
                        )
                        graph = build_interaction_graph(
                            self.workload.tasks[tid], circuit,
                            self.params["alpha"],
                        )
                        n_remote, by_link = count_remote_gates(
                            graph, mapping[tid]["logical_partition"]
                        )
                        mapping[tid]["remote_gates_by_link"] = by_link
                        self._intra_task_remote_gates[tid] = n_remote
                        self._intra_task_remote_gates_by_link[tid] = by_link
                        c6_repaired.add(tid)
            elif (vtype == "C4_FIDELITY_THRESHOLD"
                  and tid not in fidelity_repaired):
                fidelity_repaired.add(tid)
                self._remap_task_for_coherence(
                    tid,
                    assignment,
                    mapping,
                    qpu_load,
                    require_coherence=False,
                    repair_reason=vtype,
                    exclude_current=True,
                )
            elif (vtype == "C5_COHERENCE_WINDOW"
                  and tid not in coherence_repaired):
                coherence_repaired.add(tid)
                self._remap_task_for_coherence(
                    tid,
                    assignment,
                    mapping,
                    qpu_load,
                )
            elif (vtype in {
                    "C5_EPR_HOLDING_WINDOW",
                    "C5_EPR_MEMORY_LIFETIME_UNSPECIFIED",
                    "C9_EPR_GENERATION_CAPACITY",
                    "C9_EPR_MEMORY_CAPACITY",
                    "C9_EPR_MEMORY_CAPACITY_UNSPECIFIED",
                    "C9_ENTANGLEMENT_READINESS",
                    "C9_ENTANGLEMENT_LINK"}
                  and tid not in entanglement_repaired):
                entanglement_repaired.add(tid)
                # Rebuild the complete mapping before EPR reservations rerun.
                # But consider load — don't pile everything on one QPU
                self._remap_task_for_coherence(
                    tid,
                    assignment,
                    mapping,
                    qpu_load,
                    require_coherence=False,
                    repair_reason=vtype,
                    exclude_current=True,
                )
        return assignment, mapping
