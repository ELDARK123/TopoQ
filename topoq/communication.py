"""Shared physical communication models for TopoQ and all baselines."""

import heapq
import math
from collections import defaultdict

import networkx as nx

from .config import (
    PAPER_CLASSICAL_EFFECTIVE_FACTOR,
    PAPER_ENTANGLEMENT_EFFECTIVE_FACTOR,
    validate_classical_bandwidth_sharing_mode,
)


def annotate_communication_overlap(schedule, mode):
    """Attach paper Eq. (40) proxy metrics without changing physical time."""
    enabled = mode == "paper_fixed_report"
    entanglement_credit_fraction = (
        1.0 - PAPER_ENTANGLEMENT_EFFECTIVE_FACTOR if enabled else 0.0
    )
    classical_credit_fraction = (
        1.0 - PAPER_CLASSICAL_EFFECTIVE_FACTOR if enabled else 0.0
    )
    for item in schedule.values():
        entanglement_wait = max(
            float(item.get("entanglement_wait", 0.0)), 0.0
        )
        classical_comm = max(float(item.get("comm_time", 0.0)), 0.0)
        entanglement_credit = (
            entanglement_wait * entanglement_credit_fraction
        )
        classical_credit = classical_comm * classical_credit_fraction
        item["paper_entanglement_overlap_credit_ms"] = (
            entanglement_credit
        )
        item["paper_classical_overlap_credit_ms"] = classical_credit
        item["paper_entanglement_effective_wait_ms"] = (
            entanglement_wait - entanglement_credit
        )
        item["paper_classical_effective_comm_ms"] = (
            classical_comm - classical_credit
        )
    return schedule


def edge_data_mb(workload, predecessor, successor):
    """Return dependency payload in decimal megabytes."""
    edge = workload.edge_data.get((predecessor, successor), {})
    return max(edge.get("data_volume_bits", 0), 0) / 8e6


def interface_transfer_ms(system, qpu_id, data_mb, interface=None):
    """Paper Eq. (4): QPU-interface latency plus serialization time."""
    iface = (
        interface if interface is not None
        else system.interface_links.get(qpu_id)
    )
    if not iface:
        raise ValueError(f"QPU {qpu_id} has no classical interface.")
    latency_us = iface.get("effective_latency_us")
    if latency_us is None:
        latency_us = iface["readout_latency_us"]
    return (
        latency_us / 1000.0
        + 8.0 * max(data_mb, 0.0)
        / max(iface["control_bandwidth_gbps"], 1e-12)
    )


def select_widest_path(system, source_node, target_node):
    """Select widest network path, breaking ties by latency and hop count."""
    source = system.get_network_node_id(source_node)
    target = system.get_network_node_id(target_node)
    if source == target:
        return {"path": [source], "bottleneck_gbps": math.inf,
                "latency_us": 0.0}
    queue = [(-math.inf, 0.0, 0, str(source), source, [source])]
    best = {}
    while queue:
        neg_bw, latency, hops, _, node, path = heapq.heappop(queue)
        bandwidth = -neg_bw
        state = (bandwidth, -latency, -hops)
        if node in best and best[node] >= state:
            continue
        best[node] = state
        if node == target:
            return {"path": path, "bottleneck_gbps": bandwidth,
                    "latency_us": latency}
        for neighbor, edge_bw, edge_latency in system.net_adj[node]:
            next_bw = edge_bw if math.isinf(bandwidth) else min(
                bandwidth, edge_bw
            )
            heapq.heappush(
                queue,
                (-next_bw, latency + edge_latency, hops + 1,
                 str(neighbor), neighbor, path + [neighbor]),
            )
    raise ValueError(
        f"No classical network path from {source_node} to {target_node}."
    )


def reserve_classical_path(system, timeline, path_info, earliest_start,
                           data_mb, task_id, predecessor_id,
                           duration_scale=1.0, sharing_mode=None):
    """Reserve fixed bottleneck bandwidth on every selected path link."""
    sharing_mode = validate_classical_bandwidth_sharing_mode(sharing_mode)
    if timeline is None:
        timeline = defaultdict(list)
    path = path_info["path"]
    bandwidth = path_info["bottleneck_gbps"]
    data_mb = max(data_mb, 0.0)
    duration_scale = float(duration_scale)
    if duration_scale <= 0:
        raise ValueError("duration_scale must be positive.")
    if len(path) < 2 or data_mb <= 0:
        return {
            "predecessor": predecessor_id, "path": path,
            "start": earliest_start, "end": earliest_start,
            "duration": 0.0, "bandwidth_reserved_gbps": 0.0,
            "data_mb": data_mb,
            "bandwidth_sharing_mode": sharing_mode,
        }
    duration = (
        8.0 * data_mb / max(bandwidth, 1e-12)
        + path_info["latency_us"] / 1000.0
    ) * duration_scale
    link_keys = [
        frozenset((path[index], path[index + 1]))
        for index in range(len(path) - 1)
    ]
    start = earliest_start
    while True:
        end = start + duration
        conflicts = [
            entry for key in link_keys for entry in timeline[key]
            if start < entry["end"] and entry["start"] < end
        ]
        if not conflicts:
            break
        start = max(entry["end"] for entry in conflicts)
    end = start + duration
    entry = {
        "start": start, "end": end, "bandwidth_gbps": bandwidth,
        "task": task_id, "predecessor": predecessor_id,
        "bandwidth_sharing_mode": sharing_mode,
    }
    for key in link_keys:
        timeline[key].append(dict(entry))
    return {
        "predecessor": predecessor_id, "path": path,
        "start": start, "end": end, "duration": duration,
        "bandwidth_reserved_gbps": bandwidth, "data_mb": data_mb,
        "duration_scale": duration_scale,
        "bandwidth_sharing_mode": sharing_mode,
    }


def _networkx_classical_graph(system):
    """Build a deterministic weighted graph from the system network."""
    graph = nx.Graph()
    edges = []
    for link_key, info in system.net_edge_info.items():
        left, right = sorted(link_key, key=str)
        edges.append((str(left), str(right), left, right, info))
    for _, _, left, right, info in sorted(edges):
        graph.add_edge(
            left,
            right,
            bandwidth_gbps=float(info["bw"]),
            latency_us=float(info["lat"]),
        )
    return graph


def _widest_low_latency_path(graph, source, target):
    """Return a widest path, breaking bottleneck ties by path latency."""
    if not nx.has_path(graph, source, target):
        return None
    spanning = nx.maximum_spanning_tree(
        graph, weight="bandwidth_gbps", algorithm="kruskal"
    )
    tree_path = nx.shortest_path(spanning, source, target)
    bottleneck = min(
        graph[left][right]["bandwidth_gbps"]
        for left, right in zip(tree_path, tree_path[1:])
    )

    threshold = nx.Graph()
    threshold.add_nodes_from(sorted(graph.nodes, key=str))
    eligible = []
    for left, right, data in graph.edges(data=True):
        if data["bandwidth_gbps"] + 1e-12 < bottleneck:
            continue
        ordered = tuple(sorted((left, right), key=str))
        eligible.append((
            str(ordered[0]), str(ordered[1]),
            ordered[0], ordered[1], data,
        ))
    for _, _, left, right, data in sorted(eligible):
        threshold.add_edge(left, right, **data)

    path = nx.shortest_path(
        threshold,
        source,
        target,
        weight=lambda left, right, data: data["latency_us"] + 1e-12,
        method="dijkstra",
    )
    latency_us = sum(
        graph[left][right]["latency_us"]
        for left, right in zip(path, path[1:])
    )
    return {
        "path": path,
        "bottleneck_gbps": bottleneck,
        "latency_us": latency_us,
    }


def select_edge_disjoint_paths(system, source_node, target_node, path_limit):
    """Select up to path_limit widest edge-disjoint classical paths."""
    source = system.get_network_node_id(source_node)
    target = system.get_network_node_id(target_node)
    if source == target:
        return [{
            "path": [source],
            "bottleneck_gbps": math.inf,
            "latency_us": 0.0,
            "path_rank": 0,
        }]
    graph = _networkx_classical_graph(system)
    if source not in graph or target not in graph:
        raise ValueError(
            f"No classical network path from {source_node} to {target_node}."
        )

    paths = []
    for rank in range(int(path_limit)):
        selected = _widest_low_latency_path(graph, source, target)
        if selected is None:
            break
        selected["path_rank"] = rank
        paths.append(selected)
        graph.remove_edges_from(zip(
            selected["path"], selected["path"][1:]
        ))
    if not paths:
        raise ValueError(
            f"No classical network path from {source_node} to {target_node}."
        )
    return paths


def split_data_across_paths(path_infos, data_mb):
    """Water-fill data across paths to minimize the ideal completion time."""
    data_mb = max(float(data_mb), 0.0)
    if not path_infos:
        raise ValueError("At least one classical path is required.")
    if data_mb == 0.0:
        return [0.0 for _ in path_infos]

    bandwidths = [float(path["bottleneck_gbps"]) for path in path_infos]
    latencies_ms = [float(path["latency_us"]) / 1000.0 for path in path_infos]
    for index, bandwidth in enumerate(bandwidths):
        if math.isinf(bandwidth):
            allocations = [0.0 for _ in path_infos]
            allocations[index] = data_mb
            return allocations
        if bandwidth <= 0:
            raise ValueError("Classical path bandwidth must be positive.")

    lower = min(latencies_ms)
    upper = min(
        latency + 8.0 * data_mb / bandwidth
        for latency, bandwidth in zip(latencies_ms, bandwidths)
    )
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        capacity_mb = sum(
            bandwidth * max(0.0, midpoint - latency) / 8.0
            for latency, bandwidth in zip(latencies_ms, bandwidths)
        )
        if capacity_mb >= data_mb:
            upper = midpoint
        else:
            lower = midpoint

    allocations = [
        bandwidth * max(0.0, upper - latency) / 8.0
        for latency, bandwidth in zip(latencies_ms, bandwidths)
    ]
    allocated = sum(allocations)
    if allocated <= 0:
        raise ValueError("Classical multi-path allocation produced no flow.")
    return [data_mb * allocation / allocated for allocation in allocations]


def select_classical_route(system, source_node, target_node, mode,
                           multipath_k=None):
    """Build one TopoQ classical routing plan under the selected policy."""
    if mode == "edge_disjoint_split":
        paths = select_edge_disjoint_paths(
            system, source_node, target_node, multipath_k
        )
        return {"routing_mode": mode, "paths": paths}
    if mode in {"single_widest", "paper_fixed_15"}:
        return {
            "routing_mode": mode,
            "paths": [select_widest_path(system, source_node, target_node)],
        }
    raise ValueError(f"Unknown classical routing mode {mode!r}.")


def reserve_classical_route(system, timeline, route_info, earliest_start,
                            data_mb, task_id, predecessor_id,
                            sharing_mode=None):
    """Reserve a single-path or split-flow TopoQ classical transfer."""
    sharing_mode = validate_classical_bandwidth_sharing_mode(sharing_mode)
    mode = route_info["routing_mode"]
    paths = route_info["paths"]
    if mode in {"single_widest", "paper_fixed_15"}:
        reservation = reserve_classical_path(
            system,
            timeline,
            paths[0],
            earliest_start,
            data_mb,
            task_id,
            predecessor_id,
            duration_scale=0.85 if mode == "paper_fixed_15" else 1.0,
            sharing_mode=sharing_mode,
        )
        reservation.update({
            "routing_mode": mode,
            "path_count": 1 if len(reservation["path"]) > 1 else 0,
            "subflows": [dict(reservation)],
        })
        return reservation

    if mode != "edge_disjoint_split":
        raise ValueError(f"Unknown classical routing mode {mode!r}.")
    allocations = split_data_across_paths(paths, data_mb)
    subflows = []
    for index, (path_info, allocation) in enumerate(zip(paths, allocations)):
        if allocation <= 1e-15:
            continue
        subflow = reserve_classical_path(
            system,
            timeline,
            path_info,
            earliest_start,
            allocation,
            task_id,
            predecessor_id,
            sharing_mode=sharing_mode,
        )
        subflow["path_index"] = index
        subflows.append(subflow)
    if not subflows:
        return {
            "predecessor": predecessor_id,
            "path": [],
            "paths": [],
            "subflows": [],
            "start": earliest_start,
            "end": earliest_start,
            "duration": 0.0,
            "bandwidth_reserved_gbps": 0.0,
            "data_mb": max(float(data_mb), 0.0),
            "routing_mode": mode,
            "path_count": 0,
            "bandwidth_sharing_mode": sharing_mode,
        }
    start = min(subflow["start"] for subflow in subflows)
    end = max(subflow["end"] for subflow in subflows)
    return {
        "predecessor": predecessor_id,
        "path": None,
        "paths": [subflow["path"] for subflow in subflows],
        "subflows": subflows,
        "start": start,
        "end": end,
        "duration": end - start,
        "bandwidth_reserved_gbps": sum(
            subflow["bandwidth_reserved_gbps"] for subflow in subflows
        ),
        "data_mb": sum(subflow["data_mb"] for subflow in subflows),
        "routing_mode": mode,
        "path_count": len(subflows),
        "bandwidth_sharing_mode": sharing_mode,
    }
