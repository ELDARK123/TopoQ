"""Interaction-graph partitioning and physical-qubit mapping for TopoQ."""

import math
import random
from collections import Counter, defaultdict, deque

from .calibration import resolve_cx_calibration


SIGNAL_SPEED_KM_S = 200000.0
PARTITION_METHOD_NAME = (
    "capacity_topology_greedy_single_vertex_refinement"
)
PARTITION_NODE_ORDER = "descending_weighted_degree_then_node_id"
PARTITION_FILL_PENALTY = 1e-6
PARTITION_MAX_REFINEMENT_PASSES = 5
BASELINE_FEASIBILITY_BEAM_WIDTH = 256
BASELINE_FEASIBILITY_MAX_BEAM_WIDTH = 2048

# Physical-distance caches for the unconstrained (no forbidden physical
# qubits) case, which is the dominant path during partitioning/mapping.
_PHYSICAL_DISTANCE_CACHE = {}
_PHYSICAL_DEGREE_CACHE = {}


def _extract_two_qubit_operations(circuit, n_qubits):
    """Return an ordered two-qubit sequence, or None when none is provided."""
    if "operations" not in circuit:
        return None
    operations = circuit["operations"]
    if not isinstance(operations, list):
        raise ValueError("Circuit operations must be a list.")
    result = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise ValueError(f"Circuit operation {index} must be an object.")
        qubits = operation.get("qubits")
        if qubits is None and {"source", "target"}.issubset(operation):
            qubits = [operation["source"], operation["target"]]
        if qubits is None or len(qubits) != 2:
            continue
        left, right = (int(qubits[0]), int(qubits[1]))
        if left == right or not (0 <= left < n_qubits and 0 <= right < n_qubits):
            raise ValueError(
                f"Invalid two-qubit operation {index}: logical qubits {left}, {right}."
            )
        result.append({
            "index": index,
            "gate": operation.get("gate", operation.get("name", operation.get("type", "2q"))),
            "logical_pair": (left, right),
        })
    return result


def _allocate_counts(weighted_items, total):
    """Allocate an integer total proportionally with deterministic rounding."""
    if total <= 0 or not weighted_items:
        return {}
    weight_sum = sum(weight for _, weight in weighted_items)
    if weight_sum <= 0:
        weighted_items = [(item, 1.0) for item, _ in weighted_items]
        weight_sum = len(weighted_items)
    raw = [(item, total * weight / weight_sum) for item, weight in weighted_items]
    result = {item: int(value) for item, value in raw}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda pair: (-(pair[1] - int(pair[1])), pair[0]))
    for item, _ in order[:remainder]:
        result[item] += 1
    return {item: count for item, count in result.items() if count > 0}


def build_interaction_graph(task, circuit, alpha=0.7):
    """Build the weighted interaction graph from explicit circuit data."""
    n_qubits = task.get("n_qubits", circuit.get("n_qubits", 0))
    if "measure" in task.get("name", "").lower():
        target_2q = 0
    else:
        circuit_gates = max(circuit.get("n_gates", 0), 1)
        circuit_2q = circuit.get("gate_counts", {}).get("2q", 0)
        target_2q = round(task.get("n_gates", circuit_gates) * circuit_2q / circuit_gates)

    source_edges = []
    for edge in circuit.get("interactions", []):
        source, target = sorted((edge["source"], edge["target"]))
        if source < n_qubits and target < n_qubits and source != target:
            source_edges.append(((source, target), edge["two_qubit_gates"]))
    counts = _allocate_counts(source_edges, target_2q)
    total = sum(counts.values())
    edges = {}
    for pair, count in counts.items():
        frequency = count / total if total else 0.0
        edges[pair] = {
            "two_qubit_gates": count,
            "frequency": frequency,
            "weight": alpha * count + (1.0 - alpha) * frequency,
        }
    return {
        "nodes": list(range(n_qubits)),
        "edges": edges,
        "two_qubit_operations": _extract_two_qubit_operations(circuit, n_qubits),
    }


def quantum_link_cost(link, beta):
    """Return the composite topology-ranking proxy for a quantum link."""
    if not link or link["rate_hz"] <= 0 or link["max_concurrent_epr"] <= 0:
        return math.inf
    generation_s = 1.0 / link["rate_hz"]
    propagation_s = link["distance_km"] / SIGNAL_SPEED_KM_S
    capacity_penalty = beta / link["max_concurrent_epr"]
    return generation_s + propagation_s + capacity_penalty


def _rank_by_cost(candidate_sets, cost_estimator, cost_weights):
    """Normalize heterogeneous components and rank feasible candidates."""
    raw = {tuple(candidate): cost_estimator(candidate) for candidate in candidate_sets}
    feasible = {
        candidate: costs for candidate, costs in raw.items()
        if all(math.isfinite(value) for value in costs.values())
    }
    if not feasible:
        return []
    components = tuple(cost_weights)
    normalized = {candidate: {} for candidate in feasible}
    for component in components:
        values = [costs[component] for costs in feasible.values()]
        minimum, maximum = min(values), max(values)
        for candidate, costs in feasible.items():
            normalized[candidate][component] = (
                0.0 if maximum == minimum
                else (costs[component] - minimum) / (maximum - minimum)
            )
    totals = {
        candidate: sum(cost_weights[name] * values[name]
                       for name in components)
        for candidate, values in normalized.items()
    }
    return [
        {
            "candidate": candidate,
            "raw_cost": feasible[candidate],
            "normalized_cost": totals[candidate],
        }
        for candidate in sorted(
            totals, key=lambda candidate: (totals[candidate], candidate)
        )
    ]


def _choose_min_cost(candidate_sets, cost_estimator, cost_weights):
    """Normalize heterogeneous cost components and choose the minimum."""
    ranked = _rank_by_cost(candidate_sets, cost_estimator, cost_weights)
    return ranked[0]["candidate"] if ranked else None


def rank_qpu_set_candidates(task, interaction_graph, system, qpu_usage,
                            assignment_cost_estimator, cost_weights,
                            beam_width=32, max_extra_qpus=1):
    """Return deterministic connected QPU-set candidates in search order.

    Beam search avoids exhaustive ``2**M`` subset enumeration. Candidates with
    minimum cardinality are returned first; overprovisioned sets are considered
    only after all retained minimum-size candidates.
    """
    required = task.get("n_qubits", 0)
    qpu_ids = system.get_qpu_ids()
    total_capacity = sum(system.qpus[qid]["n_qubits"] for qid in qpu_ids)
    if required > total_capacity:
        raise ValueError(
            f"Task {task.get('id', task.get('name', '<unknown>'))} requires "
            f"{required} logical qubits, but the system provides only "
            f"{total_capacity} physical qubits."
        )
    beam_width = max(int(beam_width), 1)
    max_extra_qpus = max(int(max_extra_qpus), 0)
    capacities = sorted(
        (system.qpus[qid]["n_qubits"] for qid in qpu_ids), reverse=True
    )
    minimum_size = 0
    accumulated = 0
    for capacity in capacities:
        accumulated += capacity
        minimum_size += 1
        if accumulated >= required:
            break
    maximum_size = min(len(qpu_ids), minimum_size + max_extra_qpus)

    estimator = lambda candidate: assignment_cost_estimator(
        task, list(candidate), qpu_usage, interaction_graph
    )
    frontier = [tuple([qid]) for qid in qpu_ids]
    result = []
    for size in range(1, maximum_size + 1):
        ranked_frontier = _rank_by_cost(frontier, estimator, cost_weights)
        retained = ranked_frontier[:beam_width]
        if size >= minimum_size:
            feasible_capacity = [
                item for item in retained
                if sum(system.qpus[qid]["n_qubits"]
                       for qid in item["candidate"]) >= required
            ]
            for item in feasible_capacity:
                candidate = item["candidate"]
                primary = _choose_min_cost(
                    [[qid] for qid in candidate], estimator, cost_weights
                )[0]
                result.append({
                    "primary_qpu": primary,
                    "qpu_set": list(candidate),
                    "raw_cost": item["raw_cost"],
                    "normalized_cost": item["normalized_cost"],
                    "qpu_count": size,
                    "minimum_qpu_count": minimum_size,
                })
        if size == maximum_size:
            break

        expansions = set()
        for item in retained:
            candidate = item["candidate"]
            candidate_set = set(candidate)
            for qid in qpu_ids:
                if qid in candidate_set:
                    continue
                if not any(system.get_quantum_link(qid, member)
                           for member in candidate):
                    continue
                expansions.add(tuple(sorted(candidate_set | {qid})))
        frontier = sorted(expansions)
        if not frontier:
            break
    return result


def select_qpu_set(task, interaction_graph, system, qpu_usage,
                   assignment_cost_estimator, cost_weights):
    """Compatibility interface returning the first ranked QPU-set candidate."""
    candidates = rank_qpu_set_candidates(
        task, interaction_graph, system, qpu_usage,
        assignment_cost_estimator, cost_weights,
    )
    if not candidates:
        raise ValueError(
            f"Task {task.get('id', task.get('name', '<unknown>'))} cannot "
            "obtain a connected QPU set with sufficient capacity."
        )
    first = candidates[0]
    return first["primary_qpu"], first["qpu_set"]


def _partition_objective(interaction_graph, partition, system, beta, gamma):
    total = 0.0
    for (source, target), edge in interaction_graph["edges"].items():
        qpu_source, qpu_target = partition[source], partition[target]
        if qpu_source == qpu_target:
            continue
        link = system.get_quantum_link(qpu_source, qpu_target)
        if not link:
            return math.inf
        adjusted_weight = edge["weight"] * (
            1.0 + beta / link["rate_hz"]
            + gamma * link["decoherence_rate_per_ms"]
        )
        total += adjusted_weight * quantum_link_cost(link, beta)
    return total


def partition_interaction_graph(interaction_graph, qpu_set, system, beta=0.5, gamma=0.3):
    """Greedy capacity/topology partition plus single-vertex refinement.

    This is a deterministic two-stage heuristic, not a multilevel KL
    implementation: it has no coarsening, uncoarsening, or pair swaps.
    """
    adjacency = defaultdict(dict)
    for (source, target), edge in interaction_graph["edges"].items():
        adjacency[source][target] = edge["weight"]
        adjacency[target][source] = edge["weight"]
    ordered_nodes = sorted(
        interaction_graph["nodes"],
        key=lambda node: (-sum(adjacency[node].values()), node),
    )
    remaining = {qid: system.qpus[qid]["n_qubits"] for qid in qpu_set}
    partition = {}
    for node in ordered_nodes:
        choices = []
        for qid in qpu_set:
            if remaining[qid] <= 0:
                continue
            incremental = 0.0
            feasible = True
            for neighbor, weight in adjacency[node].items():
                if neighbor not in partition or partition[neighbor] == qid:
                    continue
                link = system.get_quantum_link(qid, partition[neighbor])
                if not link:
                    feasible = False
                    break
                incremental += weight * quantum_link_cost(link, beta)
            if feasible:
                fill_penalty = PARTITION_FILL_PENALTY * (
                    system.qpus[qid]["n_qubits"] - remaining[qid]
                )
                choices.append((incremental + fill_penalty, qid))
        if not choices:
            raise ValueError(f"No capacity-feasible QPU placement for logical qubit {node}.")
        _, chosen = min(choices)
        partition[node] = chosen
        remaining[chosen] -= 1

    current = _partition_objective(interaction_graph, partition, system, beta, gamma)
    for _ in range(PARTITION_MAX_REFINEMENT_PASSES):
        improved = False
        counts = Counter(partition.values())
        for node in ordered_nodes:
            source_qpu = partition[node]
            best_qpu, best_value = source_qpu, current
            for target_qpu in qpu_set:
                if target_qpu == source_qpu or counts[target_qpu] >= system.qpus[target_qpu]["n_qubits"]:
                    continue
                trial = dict(partition)
                trial[node] = target_qpu
                value = _partition_objective(interaction_graph, trial, system, beta, gamma)
                if value + 1e-12 < best_value:
                    best_qpu, best_value = target_qpu, value
            if best_qpu != source_qpu:
                partition[node] = best_qpu
                counts[source_qpu] -= 1
                counts[best_qpu] += 1
                current = best_value
                improved = True
        if not improved:
            break
    return partition


def count_remote_gates(interaction_graph, logical_partition):
    """Count exact cut gates and aggregate them by QPU link."""
    by_link = Counter()
    for (source, target), edge in interaction_graph["edges"].items():
        qpu_source = logical_partition[source]
        qpu_target = logical_partition[target]
        if qpu_source != qpu_target:
            by_link[tuple(sorted((qpu_source, qpu_target)))] += edge["two_qubit_gates"]
    return sum(by_link.values()), dict(by_link)


def random_capacity_partition(interaction_graph, qpu_set, system, rng):
    """Random baseline partition that enforces only physical capacity."""
    nodes = list(interaction_graph["nodes"])
    rng.shuffle(nodes)
    remaining = {qid: system.qpus[qid]["n_qubits"] for qid in qpu_set}
    partition = {}
    for node in nodes:
        choices = [qid for qid in qpu_set if remaining[qid] > 0]
        if not choices:
            raise ValueError("Random partition exhausted QPU capacity.")
        chosen = rng.choice(choices)
        partition[node] = chosen
        remaining[chosen] -= 1
    return partition


def balanced_capacity_partition(interaction_graph, qpu_set, system):
    """Greedy baseline partition that balances capacity usage only."""
    counts = Counter()
    partition = {}
    for node in interaction_graph["nodes"]:
        choices = [qid for qid in qpu_set
                   if counts[qid] < system.qpus[qid]["n_qubits"]]
        chosen = min(
            choices,
            key=lambda qid: (
                counts[qid] / system.qpus[qid]["n_qubits"], qid
            ),
        )
        partition[node] = chosen
        counts[chosen] += 1
    return partition


def interaction_only_partition(interaction_graph, qpu_set, system):
    """Minimize interaction cut weight without quantum-link awareness."""
    adjacency = defaultdict(dict)
    for (source, target), edge in interaction_graph["edges"].items():
        adjacency[source][target] = edge["weight"]
        adjacency[target][source] = edge["weight"]
    nodes = sorted(
        interaction_graph["nodes"],
        key=lambda node: (-sum(adjacency[node].values()), node),
    )
    counts = Counter()
    partition = {}
    for node in nodes:
        choices = []
        for qid in qpu_set:
            if counts[qid] >= system.qpus[qid]["n_qubits"]:
                continue
            cut = sum(weight for neighbor, weight in adjacency[node].items()
                      if neighbor in partition and partition[neighbor] != qid)
            choices.append((cut, counts[qid] / system.qpus[qid]["n_qubits"], qid))
        _, _, chosen = min(choices)
        partition[node] = chosen
        counts[chosen] += 1
    return partition


def dqtetris_partition(interaction_graph, qpu_set, system, rng=None,
                       restarts=8, max_passes=12):
    """DQTetris-inspired capacity-constrained cut-minimizing partition.

    This is a partition-level port of the DQTetris idea (Xu et al., SC 2025):
    keep strongly interacting logical qubits on one module, respect module
    capacity, and minimise inter-module communication.  It does not reproduce
    DQTetris's gate-level circuit slicing / gate teleportation; the scheduler
    models cross-module interactions as remote gates in the shared timing and
    fidelity model.  The multi-start + single-vertex refinement makes it a
    strictly stronger partitioner than the greedy interaction-only baseline.
    """
    nodes = sorted(interaction_graph["nodes"])
    capacities = {
        qid: int(system.qpus[qid]["n_qubits"]) for qid in qpu_set
    }
    if sum(capacities.values()) < len(nodes):
        raise ValueError("QPU set cannot hold all logical qubits.")
    adjacency = defaultdict(dict)
    edge_list = []
    for (source, target), edge in interaction_graph["edges"].items():
        weight = float(edge["weight"])
        adjacency[source][target] = weight
        adjacency[target][source] = weight
        edge_list.append((source, target, weight))
    rng = rng or random.Random(42)

    def edge_cost(partition):
        total = 0.0
        for source, target, weight in edge_list:
            left, right = partition[source], partition[target]
            if left == right:
                continue
            # Penalise cuts that cannot use a direct quantum link; the
            # projection stage repairs those, so prefer direct-link cuts.
            link = system.get_quantum_link(left, right)
            total += weight * (1.0 if link else 1000.0)
        return total

    def incremental_cost(node, qid, partition):
        cost = 0.0
        for neighbor, weight in adjacency[node].items():
            other = partition.get(neighbor)
            if other is None or other == qid:
                continue
            link = system.get_quantum_link(qid, other)
            cost += weight * (1.0 if link else 1000.0)
        return cost

    def greedy(order):
        counts = Counter()
        partition = {}
        for node in order:
            candidates = [
                qid for qid in qpu_set
                if counts[qid] < capacities[qid]
            ]
            if not candidates:
                raise ValueError("DQTetris partition exhausted QPU capacity.")
            chosen = min(
                candidates,
                key=lambda qid: (
                    incremental_cost(node, qid, partition),
                    counts[qid] / capacities[qid],
                    qid,
                ),
            )
            partition[node] = chosen
            counts[chosen] += 1
        return partition

    def refine(partition):
        current = dict(partition)
        counts = Counter(current.values())
        for _ in range(max_passes):
            improved = False
            order = list(nodes)
            rng.shuffle(order)
            for node in order:
                current_qpu = current[node]
                best_qpu = current_qpu
                best_delta = 0.0
                for qid in qpu_set:
                    if qid == current_qpu:
                        continue
                    if counts[qid] >= capacities[qid]:
                        continue
                    delta = (
                        incremental_cost(node, qid, current)
                        - incremental_cost(node, current_qpu, current)
                    )
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_qpu = qid
                if best_qpu != current_qpu:
                    current[node] = best_qpu
                    counts[current_qpu] -= 1
                    counts[best_qpu] += 1
                    improved = True
            if not improved:
                break
        return current

    degree_order = sorted(
        nodes, key=lambda node: (-sum(adjacency[node].values()), node)
    )
    candidates = [greedy(degree_order)]
    for _ in range(max(0, restarts - 1)):
        order = list(nodes)
        rng.shuffle(order)
        candidates.append(greedy(order))
    best = min(candidates, key=edge_cost)
    return refine(best)


def dqtetris_segmented_remote_count(circuit, interaction_graph, qpu_set,
                                    system, rng=None, max_segments=4):
    """Task-level DQTetris-inspired segmentation / teleportation proxy.

    The evaluated workload DAG exposes a quantum task as a single circuit;
    the original DQTetris splits a circuit into consecutive subcircuits and
    re-assigns qubits between segments (paying gate teleportation).  This
    helper approximates that behaviour at task level: the interaction list is
    split into up to ``max_segments`` contiguous chunks, each chunk is
    partitioned with :func:`dqtetris_partition`, and every logical qubit whose
    QPU changes between consecutive chunks is charged as one teleportation
    event.  The returned ``total`` is the number of modelled inter-module
    operations (cut gates + teleports) and is used as ``n_remote_gates``.
    """
    rng = rng or random.Random(42)
    sequence = (
        circuit.get("two_qubit_sequence") if isinstance(circuit, dict) else None
    )
    interactions = (
        circuit.get("interactions") if isinstance(circuit, dict) else None
    )
    if sequence:
        # Gate-level proxy: use the real 2q-gate order parsed from QASM3.
        edge_items = [
            (int(pair[0]), int(pair[1]), 1) for pair in sequence
        ]
        sequence_source = "qasm_gate_order"
    elif interactions:
        edge_items = []
        for item in interactions:
            source = int(item["source"])
            target = int(item["target"])
            count = int(item.get("two_qubit_gates", 0))
            if count <= 0:
                count = 1
            edge_items.append((source, target, count))
        sequence_source = "interaction_list_order"
    else:
        partition = dqtetris_partition(
            interaction_graph, qpu_set, system, rng=rng
        )
        remote, by_link = count_remote_gates(interaction_graph, partition)
        return {
            "segments": 1,
            "cut_gates": int(remote),
            "teleports": 0,
            "total": int(remote),
            "remote_by_link": dict(by_link),
            "per_segment": [],
            "model": "whole_interaction_graph",
        }

    n_qubits = int(
        circuit.get("n_qubits", max(
            max((item[0] for item in edge_items), default=-1),
            max((item[1] for item in edge_items), default=-1),
        ) + 1)
    )
    if n_qubits <= 0:
        n_qubits = len(interaction_graph.get("nodes", []))
    segments_max = min(max_segments, len(edge_items))
    best = None
    for segment_count in range(1, segments_max + 1):
        chunk_size = math.ceil(len(edge_items) / segment_count)
        chunks = [
            edge_items[index:index + chunk_size]
            for index in range(0, len(edge_items), chunk_size)
        ]
        previous_assignment = {}
        cut_gates = 0
        teleports = 0
        remote_by_link = Counter()
        per_segment = []
        for chunk in chunks:
            chunk_graph = {
                "nodes": list(range(n_qubits)),
                "edges": {},
            }
            for source, target, count in chunk:
                key = (source, target)
                if key not in chunk_graph["edges"]:
                    chunk_graph["edges"][key] = {
                        "weight": 0.0,
                        "two_qubit_gates": 0,
                        "frequency": 0.0,
                    }
                chunk_graph["edges"][key]["weight"] += float(count)
                chunk_graph["edges"][key]["two_qubit_gates"] += count
            partition = dqtetris_partition(
                chunk_graph, qpu_set, system, rng=rng, restarts=2,
                max_passes=4,
            )
            chunk_cut = 0
            for source, target, count in chunk:
                if partition[source] == partition[target]:
                    continue
                chunk_cut += count
                remote_by_link[
                    tuple(sorted((partition[source], partition[target])))
                ] += count
            cut_gates += chunk_cut
            touched = sorted({
                node for source, target, _ in chunk
                for node in (source, target)
            })
            chunk_teleports = 0
            for node in touched:
                if (node in previous_assignment
                        and previous_assignment[node] != partition[node]):
                    chunk_teleports += 1
                    remote_by_link[
                        tuple(sorted((
                            previous_assignment[node], partition[node]
                        )))
                    ] += 1
                previous_assignment[node] = partition[node]
            teleports += chunk_teleports
            per_segment.append({
                "cut_gates": chunk_cut,
                "teleports": chunk_teleports,
                "qpu_load": dict(Counter(partition.values())),
            })
        total = cut_gates + teleports
        fallback_link = None
        ordered_qpus = sorted(qpu_set)
        for left_index, left in enumerate(ordered_qpus):
            for right in ordered_qpus[left_index + 1:]:
                if system.get_quantum_link(left, right):
                    fallback_link = (left, right)
                    break
            if fallback_link:
                break
        sanitized_by_link = Counter()
        for link_key, count in remote_by_link.items():
            if system.get_quantum_link(*link_key):
                sanitized_by_link[tuple(sorted(link_key))] += count
            elif fallback_link is not None:
                # Task-level model only supports direct EPR links; charge
                # non-direct segmentation traffic to a feasible direct link.
                sanitized_by_link[fallback_link] += count
            else:
                sanitized_by_link[tuple(sorted(link_key))] += count
        candidate = {
            "segments": len(chunks),
            "cut_gates": int(cut_gates),
            "teleports": int(teleports),
            "total": int(total),
            "remote_by_link": dict(sanitized_by_link),
            "per_segment": per_segment,
            "model": "task_level_segmentation_teleport_proxy",
            "sequence_source": sequence_source,
        }
        if best is None or candidate["total"] < best["total"]:
            best = candidate
    return best


def _has_direct_quantum_connection(system, left, right):
    """Return whether two labels can communicate without EPR routing."""
    return left == right or system.get_quantum_link(left, right) is not None


def count_direct_link_partition_violations(interaction_graph, partition,
                                           system):
    """Count cut interactions assigned to QPU pairs without a direct link."""
    violating_edges = 0
    violating_gates = 0
    for (source, target), edge in interaction_graph["edges"].items():
        left = partition[source]
        right = partition[target]
        if _has_direct_quantum_connection(system, left, right):
            continue
        violating_edges += 1
        violating_gates += int(edge.get("two_qubit_gates", 1))
    return {
        "violating_edges": violating_edges,
        "violating_gates": violating_gates,
    }


def count_quantum_dependency_link_violations(workload, assignment, system):
    """Count Q-Q DAG dependencies placed without a direct quantum link."""
    violating = 0
    for edge in workload.edges:
        source, target = edge["source"], edge["target"]
        if (workload.tasks[source]["type"] != "quantum"
                or workload.tasks[target]["type"] != "quantum"):
            continue
        if not _has_direct_quantum_connection(
                system, assignment[source], assignment[target]):
            violating += 1
    return violating


def _beam_project_labels(nodes, edges, original, labels, compatible,
                         capacities=None, beam_width=256):
    """Project labels into a pairwise-feasible domain with fewest moves.

    This is an algorithm-neutral bounded search. It never evaluates TopoQ's
    objective and uses the original label only through the Hamming distance.
    """
    labels = tuple(sorted(labels))
    label_index = {label: index for index, label in enumerate(labels)}
    adjacency = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    ordered = sorted(nodes, key=lambda node: (-len(adjacency[node]), str(node)))
    order_index = {node: index for index, node in enumerate(ordered)}
    capacity_tuple = (
        tuple(int(capacities[label]) for label in labels)
        if capacities is not None else None
    )

    # State: (number_of_moves, per_label_counts, assigned_label_indices).
    states = [(0, (0,) * len(labels), ())]
    for position, node in enumerate(ordered):
        expanded = []
        preferred = original[node]
        candidates = sorted(
            labels, key=lambda label: (label != preferred, label_index[label])
        )
        assigned_neighbors = [
            neighbor for neighbor in adjacency[node]
            if order_index[neighbor] < position
        ]
        for moves, counts, values in states:
            for label in candidates:
                index = label_index[label]
                if (capacity_tuple is not None
                        and counts[index] >= capacity_tuple[index]):
                    continue
                if any(not compatible(
                        label, labels[values[order_index[neighbor]]]
                ) for neighbor in assigned_neighbors):
                    continue
                next_counts = list(counts)
                next_counts[index] += 1
                expanded.append((
                    moves + int(label != preferred),
                    tuple(next_counts),
                    values + (index,),
                ))
        if not expanded:
            return None
        expanded.sort(key=lambda state: (
            state[0],
            max(state[1]) - min(state[1]),
            state[2],
        ))
        states = expanded[:max(int(beam_width), 1)]

    moves, _, values = min(states, key=lambda state: (state[0], state[2]))
    return {
        "labels": {
            node: labels[values[position]]
            for position, node in enumerate(ordered)
        },
        "moved": moves,
        "beam_width": max(int(beam_width), 1),
    }


def _project_with_bounded_beam(nodes, edges, original, labels, compatible,
                               capacities=None):
    widths = (
        BASELINE_FEASIBILITY_BEAM_WIDTH,
        BASELINE_FEASIBILITY_MAX_BEAM_WIDTH,
    )
    for width in dict.fromkeys(widths):
        projected = _beam_project_labels(
            nodes, edges, original, labels, compatible, capacities, width
        )
        if projected is not None:
            return projected
    return None


def project_partition_to_direct_links(interaction_graph, original_partition,
                                      system):
    """Preserve a baseline partition or project it into the physical domain."""
    nodes = list(interaction_graph["nodes"])
    if set(original_partition) != set(nodes):
        raise ValueError("Baseline partition does not cover every logical qubit.")
    unknown = sorted(set(original_partition.values()) - set(system.qpus))
    if unknown:
        raise ValueError(f"Baseline partition references unknown QPUs: {unknown}.")
    counts = Counter(original_partition.values())
    overloaded = {
        qid: count for qid, count in counts.items()
        if count > system.qpus[qid]["n_qubits"]
    }
    before = count_direct_link_partition_violations(
        interaction_graph, original_partition, system
    )
    if not overloaded and before["violating_gates"] == 0:
        return dict(original_partition), {
            "projection_applied": False,
            "moved_logical_qubits": 0,
            "before": before,
            "after": dict(before),
            "beam_width": 0,
        }

    projected = _project_with_bounded_beam(
        nodes, interaction_graph["edges"], original_partition,
        system.get_qpu_ids(),
        lambda left, right: _has_direct_quantum_connection(
            system, left, right
        ),
        capacities={
            qid: system.qpus[qid]["n_qubits"]
            for qid in system.get_qpu_ids()
        },
    )
    if projected is None:
        raise ValueError(
            "No capacity- and direct-link-feasible projection exists for "
            "the baseline logical partition."
        )
    result = projected["labels"]
    after = count_direct_link_partition_violations(
        interaction_graph, result, system
    )
    if after["violating_gates"]:
        raise AssertionError("Baseline partition projection retained link violations.")
    return result, {
        "projection_applied": True,
        "moved_logical_qubits": projected["moved"],
        "before": before,
        "after": after,
        "beam_width": projected["beam_width"],
    }


def project_quantum_task_assignments(workload, original_assignment, system):
    """Project Q-Q task assignments onto equal or directly linked QPUs."""
    quantum_tasks = list(workload.quantum_tasks)
    qq_edges = [
        (edge["source"], edge["target"]) for edge in workload.edges
        if (workload.tasks[edge["source"]]["type"] == "quantum"
            and workload.tasks[edge["target"]]["type"] == "quantum")
    ]
    before = count_quantum_dependency_link_violations(
        workload, original_assignment, system
    )
    if before == 0:
        return dict(original_assignment), {
            "projection_applied": False,
            "moved_quantum_tasks": 0,
            "before_violations": 0,
            "after_violations": 0,
            "beam_width": 0,
        }
    projected = _project_with_bounded_beam(
        quantum_tasks, qq_edges, original_assignment, system.get_qpu_ids(),
        lambda left, right: _has_direct_quantum_connection(
            system, left, right
        ),
    )
    if projected is None:
        raise ValueError(
            "No direct-link-feasible projection exists for baseline Q-Q "
            "task dependencies."
        )
    result = dict(original_assignment)
    result.update(projected["labels"])
    after = count_quantum_dependency_link_violations(workload, result, system)
    if after:
        raise AssertionError("Baseline task projection retained link violations.")
    return result, {
        "projection_applied": True,
        "moved_quantum_tasks": projected["moved"],
        "before_violations": before,
        "after_violations": after,
        "beam_width": projected["beam_width"],
    }


def _physical_distances(topology, source):
    adjacency = defaultdict(list)
    for left, right in topology:
        adjacency[left].append(right)
        adjacency[right].append(left)
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _cached_qpu_distance_table(system, qid, topology):
    """Cache full-coupling distance/degree tables per QPU.

    The common mapping path has no forbidden physical qubits, so the BFS
    result is invariant across tasks in one experiment.  Keeping it on the
    system object avoids repeated O(|V|^2) distance computation.
    """
    cache = system.__dict__.setdefault("_qpu_distance_table", {})
    table = cache.get(qid)
    if table is None:
        degree = Counter(node for edge in topology for node in edge)
        distances = {
            node: _physical_distances(topology, node)
            for node in {item for edge in topology for item in edge}
        }
        table = (degree, distances)
        cache[qid] = table
    return table


def map_logical_to_physical(interaction_graph, logical_partition, system,
                            fixed_physical_mapping=None,
                            forbidden_physical_by_qpu=None):
    """Greedily minimize weighted IBM-coupling distances (paper Eq. 31)."""
    fixed_physical_mapping = dict(fixed_physical_mapping or {})
    forbidden_physical_by_qpu = {
        qid: set(physical)
        for qid, physical in (forbidden_physical_by_qpu or {}).items()
    }
    physical_mapping = dict(fixed_physical_mapping)
    logical_by_qpu = defaultdict(list)
    for logical, qid in logical_partition.items():
        logical_by_qpu[qid].append(logical)
    logical_adjacency = defaultdict(dict)
    for (source, target), edge in interaction_graph["edges"].items():
        logical_adjacency[source][target] = edge["weight"]
        logical_adjacency[target][source] = edge["weight"]

    for qid, logical_nodes in logical_by_qpu.items():
        qpu = system.qpus[qid]
        topology = qpu["topology"]
        physical_nodes = sorted({node for edge in topology for node in edge})
        forbidden = forbidden_physical_by_qpu.get(qid, set())
        available_nodes = set(physical_nodes) - forbidden
        if len(logical_nodes) > len(available_nodes):
            raise ValueError(f"QPU {qid} coupling graph cannot map {len(logical_nodes)} logical qubits.")
        physical_degree = Counter(node for edge in topology for node in edge)
        available_topology = [
            edge for edge in topology
            if edge[0] in available_nodes and edge[1] in available_nodes
        ]
        if forbidden:
            distances = {
                node: _physical_distances(available_topology, node)
                for node in available_nodes
            }
        else:
            cached_degree, cached_distances = _cached_qpu_distance_table(
                system, qid, topology
            )
            physical_degree = cached_degree
            distances = {
                node: cached_distances[node] for node in available_nodes
            }
        ordered = sorted(
            logical_nodes,
            key=lambda node: (-sum(logical_adjacency[node].values()), node),
        )
        used = set(forbidden)
        for logical in logical_nodes:
            if logical not in fixed_physical_mapping:
                continue
            physical = fixed_physical_mapping[logical]
            if physical in forbidden:
                raise ValueError(
                    f"Fixed physical qubit {physical} on {qid} is occupied."
                )
            if physical not in physical_nodes:
                raise ValueError(
                    f"Fixed physical qubit {physical} is invalid for {qid}."
                )
            if physical in used:
                raise ValueError(
                    f"Fixed mapping reuses physical qubit {physical} on {qid}."
                )
            used.add(physical)
        for logical in ordered:
            if logical in fixed_physical_mapping:
                continue
            candidates = []
            for physical in sorted(available_nodes):
                if physical in used:
                    continue
                cost = 0.0
                feasible = True
                for neighbor, weight in logical_adjacency[logical].items():
                    if neighbor not in physical_mapping or logical_partition[neighbor] != qid:
                        continue
                    distance = distances[physical].get(physical_mapping[neighbor])
                    if distance is None:
                        feasible = False
                        break
                    cost += weight * distance
                if feasible:
                    candidates.append((cost, -physical_degree[physical], physical))
            if not candidates:
                raise ValueError(f"No connected physical placement for logical qubit {logical} on {qid}.")
            _, _, chosen = min(candidates)
            physical_mapping[logical] = chosen
            used.add(chosen)
    return physical_mapping


def _physical_shortest_path(topology, source, target):
    adjacency = defaultdict(list)
    for left, right in topology:
        adjacency[left].append(right)
        adjacency[right].append(left)
    predecessor = {source: None}
    queue = deque([source])
    while queue and target not in predecessor:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in predecessor:
                predecessor[neighbor] = current
                queue.append(neighbor)
    if target not in predecessor:
        return None
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = predecessor[current]
    return list(reversed(path))


def _analyze_static_physical_routing(interaction_graph, logical_partition,
                                     physical_mapping, system,
                                     forbidden_physical_by_qpu=None):
    """Estimate static SWAP routing overhead for non-adjacent local gates.

    A shortest path of distance ``d`` requires ``max(d-1, 0)`` SWAPs for a
    two-qubit interaction. Each SWAP is conservatively counted as three
    native two-qubit gates, matching the standard decomposition used by the
    simulator-level error model.
    """
    routed_edges = []
    total_swap_gates = 0
    routing_overhead_ms = 0.0
    unroutable = []
    forbidden_physical_by_qpu = {
        qid: set(physical)
        for qid, physical in (forbidden_physical_by_qpu or {}).items()
    }
    physical_footprint = defaultdict(set)
    for logical, physical in physical_mapping.items():
        physical_footprint[logical_partition[logical]].add(physical)
    for (logical_left, logical_right), edge in interaction_graph["edges"].items():
        left_qpu = logical_partition[logical_left]
        right_qpu = logical_partition[logical_right]
        if left_qpu != right_qpu:
            continue  # Inter-QPU gates use teleportation/EPR, not local SWAPs.
        left_physical = physical_mapping[logical_left]
        right_physical = physical_mapping[logical_right]
        forbidden = forbidden_physical_by_qpu.get(left_qpu, set())
        topology = [
            topology_edge
            for topology_edge in system.qpus[left_qpu].get("topology", [])
            if topology_edge[0] not in forbidden
            and topology_edge[1] not in forbidden
        ]
        adjacency = defaultdict(list)
        direct_edges = set()
        for source, target in topology:
            adjacency[source].append(target)
            adjacency[target].append(source)
            direct_edges.add(frozenset((source, target)))
        gate_count = edge["two_qubit_gates"]
        physical_pair = (left_physical, right_physical)
        if frozenset(physical_pair) in direct_edges:
            continue

        path = _physical_shortest_path(
            topology, left_physical, right_physical
        )
        if path is None:
            unroutable.append({
                "logical_pair": (logical_left, logical_right),
                "physical_pair": physical_pair,
                "qpu": left_qpu,
                "two_qubit_gates": gate_count,
            })
            continue
        distance = len(path) - 1
        physical_footprint[left_qpu].update(path)
        swap_count = max(distance - 1, 0) * gate_count
        swap_gate_count = 3 * swap_count
        total_swap_gates += swap_gate_count
        calibration_policy = getattr(system, "sim_params", {}).get(
            "qpu_calibration_policy",
            {
                "mode": "configured_qpu_average",
                "cx_direction_policy": "conservative_max",
                "missing_calibration_policy": "error",
            },
        )
        path_gate_ns = sum(
            resolve_cx_calibration(
                system.qpus[left_qpu],
                path[index],
                path[index + 1],
                calibration_policy,
            )["length_ns"]
            for index in range(max(len(path) - 2, 0))
        )
        routing_overhead_ms += 3 * gate_count * path_gate_ns / 1e6
        routed_edges.append({
            "logical_pair": (logical_left, logical_right),
            "physical_pair": physical_pair,
            "qpu": left_qpu,
            "distance": distance,
            "path": path,
            "swap_count": swap_count,
            "swap_gate_count": swap_gate_count,
            "two_qubit_gates": gate_count,
        })
    return {
        "routing_mode": "static_aggregate",
        "sequence_available": interaction_graph.get("two_qubit_operations") is not None,
        "initial_physical_mapping": dict(physical_mapping),
        "final_physical_mapping": dict(physical_mapping),
        "operation_count": sum(
            edge["two_qubit_gates"] for edge in interaction_graph["edges"].values()
        ),
        "routed_edges": routed_edges,
        "unroutable_edges": unroutable,
        "non_adjacent_pairs": len(routed_edges) + len(unroutable),
        "swap_count": total_swap_gates // 3,
        "swap_gate_count": total_swap_gates,
        "routing_overhead_ms": routing_overhead_ms,
        "physical_footprint_by_qpu": {
            qid: sorted(physical)
            for qid, physical in physical_footprint.items()
        },
        "routing_path_incomplete_qpus": sorted({
            item["qpu"] for item in unroutable
        }),
    }


def _analyze_sequence_physical_routing(interaction_graph, logical_partition,
                                       physical_mapping, system,
                                       forbidden_physical_by_qpu=None):
    operations = interaction_graph.get("two_qubit_operations")
    if operations is None:
        raise ValueError(
            "sequence_exact physical routing requires an ordered circuit operations list."
        )
    expected = sum(
        edge["two_qubit_gates"] for edge in interaction_graph["edges"].values()
    )
    if len(operations) != expected:
        raise ValueError(
            "sequence_exact physical routing requires the ordered two-qubit "
            f"operation count ({len(operations)}) to match the interaction graph ({expected})."
        )

    current_mapping = dict(physical_mapping)
    forbidden_physical_by_qpu = {
        qid: set(physical)
        for qid, physical in (forbidden_physical_by_qpu or {}).items()
    }
    physical_footprint = defaultdict(set)
    inverse_by_qpu = defaultdict(dict)
    for logical, physical in current_mapping.items():
        qid = logical_partition[logical]
        inverse_by_qpu[qid][physical] = logical
        physical_footprint[qid].add(physical)

    routed_operations = []
    unroutable = []
    swap_count = 0
    routing_overhead_ms = 0.0
    local_operation_count = 0
    for operation in operations:
        logical_left, logical_right = operation["logical_pair"]
        left_qpu = logical_partition[logical_left]
        right_qpu = logical_partition[logical_right]
        if left_qpu != right_qpu:
            continue
        local_operation_count += 1
        qpu = system.qpus[left_qpu]
        forbidden = forbidden_physical_by_qpu.get(left_qpu, set())
        topology = [
            topology_edge for topology_edge in qpu.get("topology", [])
            if topology_edge[0] not in forbidden
            and topology_edge[1] not in forbidden
        ]
        before = (current_mapping[logical_left], current_mapping[logical_right])
        path = _physical_shortest_path(topology, before[0], before[1])
        if path is None:
            unroutable.append({
                "operation_index": operation["index"],
                "logical_pair": operation["logical_pair"],
                "physical_pair": before,
                "qpu": left_qpu,
                "two_qubit_gates": 1,
            })
            continue
        physical_footprint[left_qpu].update(path)

        swaps = []
        inverse = inverse_by_qpu[left_qpu]
        for position in range(max(len(path) - 2, 0)):
            physical_left = path[position]
            physical_right = path[position + 1]
            logical_at_left = inverse.get(physical_left)
            logical_at_right = inverse.get(physical_right)
            if logical_at_left is not None:
                current_mapping[logical_at_left] = physical_right
            if logical_at_right is not None:
                current_mapping[logical_at_right] = physical_left
            if logical_at_right is None:
                inverse.pop(physical_left, None)
            else:
                inverse[physical_left] = logical_at_right
            if logical_at_left is None:
                inverse.pop(physical_right, None)
            else:
                inverse[physical_right] = logical_at_left
            swaps.append((physical_left, physical_right))

        after = (current_mapping[logical_left], current_mapping[logical_right])
        swap_count += len(swaps)
        calibration_policy = getattr(system, "sim_params", {}).get(
            "qpu_calibration_policy",
            {
                "mode": "configured_qpu_average",
                "cx_direction_policy": "conservative_max",
                "missing_calibration_policy": "error",
            },
        )
        routing_overhead_ms += 3 * sum(
            resolve_cx_calibration(
                qpu, physical_left, physical_right, calibration_policy
            )["length_ns"]
            for physical_left, physical_right in swaps
        ) / 1e6
        routed_operations.append({
            "operation_index": operation["index"],
            "gate": operation["gate"],
            "logical_pair": operation["logical_pair"],
            "qpu": left_qpu,
            "physical_pair_before": before,
            "path": path,
            "swaps": swaps,
            "physical_pair_after": after,
            "swap_count": len(swaps),
            "swap_gate_count": 3 * len(swaps),
        })

    for logical, physical in current_mapping.items():
        physical_footprint[logical_partition[logical]].add(physical)

    return {
        "routing_mode": "sequence_exact",
        "sequence_available": True,
        "initial_physical_mapping": dict(physical_mapping),
        "final_physical_mapping": current_mapping,
        "operation_count": len(operations),
        "local_operation_count": local_operation_count,
        "routed_operations": routed_operations,
        "routed_edges": [],
        "unroutable_edges": unroutable,
        "non_adjacent_pairs": sum(bool(item["swaps"]) for item in routed_operations) + len(unroutable),
        "swap_count": swap_count,
        "swap_gate_count": 3 * swap_count,
        "routing_overhead_ms": routing_overhead_ms,
        "physical_footprint_by_qpu": {
            qid: sorted(physical)
            for qid, physical in physical_footprint.items()
        },
        "routing_path_incomplete_qpus": sorted({
            item["qpu"] for item in unroutable
        }),
    }


def analyze_physical_routing(interaction_graph, logical_partition,
                             physical_mapping, system, routing_mode="auto",
                             forbidden_physical_by_qpu=None):
    """Analyze C10 routing using aggregate or ordered-gate semantics."""
    if routing_mode not in {"auto", "static_aggregate", "sequence_exact"}:
        raise ValueError(
            "physical_routing_mode must be auto, static_aggregate, or sequence_exact."
        )
    sequence_available = interaction_graph.get("two_qubit_operations") is not None
    effective_mode = (
        "sequence_exact" if routing_mode == "auto" and sequence_available
        else "static_aggregate" if routing_mode == "auto"
        else routing_mode
    )
    if effective_mode == "sequence_exact":
        return _analyze_sequence_physical_routing(
            interaction_graph,
            logical_partition,
            physical_mapping,
            system,
            forbidden_physical_by_qpu=forbidden_physical_by_qpu,
        )
    return _analyze_static_physical_routing(
        interaction_graph,
        logical_partition,
        physical_mapping,
        system,
        forbidden_physical_by_qpu=forbidden_physical_by_qpu,
    )
