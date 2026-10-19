"""
System & Workload Models
========================
Data classes representing the hybrid quantum-classical computing system
and the workload DAG.
"""

import math
from collections import defaultdict, deque

from .calibration import load_ibm_chip_calibration


class HybridSystem:
    """Represents a hybrid quantum-classical computing system H = (Q, C, N)."""

    def __init__(self, config, network, sim_params):
        self.config_name = config["name"]
        self.n_qpus = config["n_qpus"]
        self.n_classical_nodes = config["n_classical_nodes"]

        # QPUs
        self.qpus = {}
        for q in config["qpu_instances"]:
            chip_type = q.get("chip_type")
            calibration = (
                load_ibm_chip_calibration(chip_type) if chip_type else None
            )
            self.qpus[q["qpu_id"]] = {
                "id": q["qpu_id"],
                "chip_type": chip_type,
                "n_qubits": q["n_qubits"],
                "coherence_time_us": q["coherence_time_us"],
                "gate_times_ns": q["gate_times_ns"],
                "gate_error_rates": q["gate_error_rates"],
                "topology": q["topology"],
                "ibm_calibration": calibration,
            }

        # Classical nodes
        self.nodes = {}
        for p in config["classical_instances"]:
            self.nodes[p["node_id"]] = {
                "id": p["node_id"],
                "type": p["type"],
                "compute_speed_tflops": p["compute_speed_tflops"],
                "memory_gb": p["memory_gb"],
            }

        # Quantum links (entanglement)
        self.quantum_links = {}
        self.quantum_link_channels = {}
        for link_index, link in enumerate(config["quantum_links"]):
            key = frozenset([link["qpu_i"], link["qpu_j"]])
            channels = self.quantum_link_channels.setdefault(key, [])
            channel_id = str(link.get(
                "channel_id",
                f"{link['qpu_i']}--{link['qpu_j']}#{len(channels)}",
            ))
            if any(item["channel_id"] == channel_id for item in channels):
                raise ValueError(
                    f"Duplicate quantum channel_id {channel_id!r} for "
                    f"{tuple(sorted(key))}."
                )
            channel = {
                "channel_id": channel_id,
                "config_index": link_index,
                "rate_hz": link["entanglement_rate_hz"],
                "decoherence_rate_per_ms": link["decoherence_rate_per_ms"],
                "max_concurrent_epr": link["max_concurrent_epr"],
                "distance_km": link["distance_km"],
            }
            for field_name in ("bsm_latency_ms", "switch_latency_ms"):
                if field_name not in link:
                    continue
                latency = float(link[field_name])
                if not math.isfinite(latency) or latency < 0.0:
                    raise ValueError(
                        f"{field_name} must be a non-negative finite number "
                        f"for quantum channel {channel_id!r}."
                    )
                channel[field_name] = latency
            if "signal_speed_m_s" in link:
                signal_speed = float(link["signal_speed_m_s"])
                if not math.isfinite(signal_speed) or signal_speed <= 0.0:
                    raise ValueError(
                        "signal_speed_m_s must be a positive finite number "
                        f"for quantum channel {channel_id!r}."
                    )
                channel["signal_speed_m_s"] = signal_speed
            if "signal_traversals" in link:
                signal_traversals = float(link["signal_traversals"])
                if (not math.isfinite(signal_traversals)
                        or signal_traversals <= 0.0
                        or not signal_traversals.is_integer()):
                    raise ValueError(
                        "signal_traversals must be a positive integer for "
                        f"quantum channel {channel_id!r}."
                    )
                channel["signal_traversals"] = int(signal_traversals)
            if "memory_capacity_epr" in link:
                memory_capacity = float(link["memory_capacity_epr"])
                if (not math.isfinite(memory_capacity)
                        or memory_capacity <= 0.0
                        or not memory_capacity.is_integer()):
                    raise ValueError(
                        "memory_capacity_epr must be a positive integer for "
                        f"quantum channel {channel_id!r}."
                    )
                channel["memory_capacity_epr"] = int(memory_capacity)
            if "epr_memory_lifetime_ms" in link:
                memory_lifetime = float(link["epr_memory_lifetime_ms"])
                if (not math.isfinite(memory_lifetime)
                        or memory_lifetime <= 0.0):
                    raise ValueError(
                        "epr_memory_lifetime_ms must be a positive finite "
                        f"number for quantum channel {channel_id!r}."
                    )
                channel["epr_memory_lifetime_ms"] = memory_lifetime
            if "epr_fidelity" in link:
                channel["epr_fidelity"] = link["epr_fidelity"]
            channels.append(channel)
            # Existing mapping, fidelity, and baseline interfaces continue to
            # use the first configured channel as the primary direct link.
            self.quantum_links.setdefault(key, channel)

        # Interface links (QPU <-> classical node)
        self.interface_links = {}
        self.interface_candidates = defaultdict(list)
        for link in config["interface_links"]:
            candidate = {
                "classical_node_id": link["classical_node_id"],
                "readout_latency_us": link["readout_latency_us"],
                "control_bandwidth_gbps": link["control_bandwidth_gbps"],
            }
            if "distance_km" in link:
                candidate["distance_km"] = link["distance_km"]
            qpu_id = link["qpu_id"]
            self.interface_candidates[qpu_id].append(candidate)
            self.interface_links.setdefault(qpu_id, candidate)

        # Classical network graph
        self.net_nodes = network["nodes"]
        self.net_edges = network["edges"]
        self._build_network_graph()
        self._map_classical_nodes_to_network()

        # Store sim params
        self.sim_params = sim_params

        # Precompute topology metrics (Phase 1)
        self._topology_metrics = None

    def _build_network_graph(self):
        """Build adjacency list for classical network."""
        self.net_adj = defaultdict(list)
        self.net_edge_info = {}
        for e in self.net_edges:
            src, tgt = e["source"], e["target"]
            bw = e.get("bandwidth_gbps", 10)
            lat = e.get("latency_us", 1)
            self.net_adj[src].append((tgt, bw, lat))
            self.net_adj[tgt].append((src, bw, lat))
            self.net_edge_info[frozenset([src, tgt])] = {"bw": bw, "lat": lat}

    def _map_classical_nodes_to_network(self):
        """Attach P-node resources to actual leaf endpoints of the network."""
        unseen = set(self.net_adj)
        component_candidates = []
        while unseen:
            start = min(unseen, key=str)
            component = {start}
            stack = [start]
            unseen.remove(start)
            while stack:
                node = stack.pop()
                for neighbor, _, _ in self.net_adj[node]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            leaves = sorted(
                node for node in component
                if (str(node).startswith("host_") or str(node).startswith("n_"))
                and len(self.net_adj[node]) == 1
            )
            if len(leaves) >= len(self.nodes):
                component_candidates.append((len(leaves), len(component), leaves))
        if not component_candidates:
            raise ValueError(
                "No connected network component exposes enough compute leaves "
                f"for {len(self.nodes)} classical resources."
            )
        _, _, leaf_nodes = max(component_candidates, key=lambda item: (item[0], item[1]))
        for node_id, network_id in zip(sorted(self.nodes), leaf_nodes):
            self.nodes[node_id]["network_node_id"] = network_id

    def get_network_node_id(self, classical_node_id):
        """Resolve a scheduler P-node identifier to a network graph endpoint."""
        return self.nodes[classical_node_id]["network_node_id"]

    def get_quantum_link(self, qpu_i, qpu_j):
        """Get quantum link info between two QPUs."""
        key = frozenset([qpu_i, qpu_j])
        return self.quantum_links.get(key)

    def get_quantum_link_channels(self, qpu_i, qpu_j):
        """Return all configured independent direct channels for a QPU pair."""
        key = frozenset([qpu_i, qpu_j])
        return tuple(self.quantum_link_channels.get(key, ()))

    def get_qpu_ids(self):
        return sorted(self.qpus.keys())

    def get_node_ids(self):
        return sorted(self.nodes.keys())


class Workload:
    """Represents a hybrid workload W = (T, E_D, prec, mu)."""

    def __init__(self, dag_data):
        self.id = dag_data["id"]
        self.type = dag_data["type"]
        self.n_qubits = dag_data["n_qubits"]
        self.n_iterations = dag_data.get("n_iterations", 1)

        # Tasks
        self.tasks = {}
        self.quantum_tasks = []
        self.classical_tasks = []
        for t in dag_data["tasks"]:
            self.tasks[t["id"]] = t
            if t["type"] == "quantum":
                self.quantum_tasks.append(t["id"])
            else:
                self.classical_tasks.append(t["id"])

        # Edges (dependencies)
        self.edges = dag_data["edges"]
        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        self.edge_data = {}
        for e in self.edges:
            src, tgt = e["source"], e["target"]
            self.successors[src].append(tgt)
            self.predecessors[tgt].append(src)
            self.edge_data[(src, tgt)] = e

    def get_task(self, task_id):
        return self.tasks[task_id]

    def get_shared_logical_qubits(self, predecessor, successor):
        """Return shared logical-qubit ids and whether they were inferred.

        New datasets can declare ``shared_logical_qubits`` explicitly on a
        quantum-to-quantum dependency edge.  For legacy datasets, equal
        circuit identifiers imply the overlapping logical-id range.
        """
        edge = self.edge_data.get((predecessor, successor), {})
        left = self.tasks.get(predecessor, {})
        right = self.tasks.get(successor, {})
        if left.get("type") != "quantum" or right.get("type") != "quantum":
            return [], False
        if "shared_logical_qubits" in edge:
            shared = sorted(set(edge["shared_logical_qubits"]))
            limit = min(left.get("n_qubits", 0), right.get("n_qubits", 0))
            invalid = [q for q in shared if not isinstance(q, int) or q < 0 or q >= limit]
            if invalid:
                raise ValueError(
                    f"Invalid shared logical qubits on edge {predecessor}->{successor}: {invalid}"
                )
            return shared, False
        if (left.get("circuit_id")
                and left.get("circuit_id") == right.get("circuit_id")):
            limit = min(left.get("n_qubits", 0), right.get("n_qubits", 0))
            return list(range(limit)), True
        return [], False

    def get_all_task_ids(self):
        return list(self.tasks.keys())

    def get_entry_tasks(self):
        """Tasks with no predecessors."""
        return [t for t in self.tasks if not self.predecessors.get(t)]

    def get_exit_tasks(self):
        """Tasks with no successors."""
        return [t for t in self.tasks if not self.successors.get(t)]
