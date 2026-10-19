"""
Baseline Schedulers
===================
Four baseline scheduling algorithms for comparison with TopoQ:

  - RandomScheduler:  Random task placement without topology awareness
  - GreedyScheduler:  Greedy load balancing without topology awareness
  - QOnlyScheduler:   Optimize quantum scheduling only, ignore classical placement
  - DQTetrisScheduler: Circuit mapping with inter-module communication optimization
"""

import json
import math
import random
from collections import Counter, defaultdict, deque
from copy import deepcopy

from .config import (
    DEFAULT_PARAMS,
    load_circuit,
    validate_execution_time_policy,
    validate_qpu_calibration_policy,
)
from .mapping import (
    analyze_physical_routing,
    balanced_capacity_partition,
    build_interaction_graph,
    count_remote_gates,
    dqtetris_partition,
    dqtetris_segmented_remote_count,
    interaction_only_partition,
    map_logical_to_physical,
    project_partition_to_direct_links,
    project_quantum_task_assignments,
    random_capacity_partition,
)
from .communication import (
    edge_data_mb,
    interface_transfer_ms,
    reserve_classical_path,
    select_widest_path,
)
from .fidelity import build_epr_event
from .entanglement import (
    deterministic_epr_service_window_ms,
    resolve_quantum_link_latency_policy,
    resolve_service_latency,
)
from .timing import (
    classical_task_duration_ms,
    quantum_task_duration_ms,
    quantum_task_timing_details,
    reference_task_duration_ms,
)


class RandomScheduler:
    """Random task placement without topology awareness."""
    def __init__(self, system, workload, seed=42, params=None):
        self.system = system
        self.workload = workload
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
        self.rng = random.Random(seed)
        self.classical_link_timeline = defaultdict(list)
        self.baseline_physical_feasibility = {}
        self.dqtetris_segmentation = {}
        self._dqtetris_segmentation_cache = {}
        self.external_dqtetris = {}
        external_path = self.params.get("dqtetris_external_outputs")
        if external_path:
            with open(external_path, "r", encoding="utf-8-sig") as handle:
                external_payload = json.load(handle)
            self.external_dqtetris = external_payload.get(
                "circuits", external_payload
            )
        self.quantum_link_latency_policy = (
            resolve_quantum_link_latency_policy(
                self.system.sim_params.get(
                    "quantum_link_latency_policy", {}
                ),
                self.system.sim_params,
            )
        )

    def _external_dqtetris_partition(self, record, required_qubits):
        """Convert a DQTetris ``final_path`` partition to our logical map.

        ``record["partition"]`` is DQTetris's list-of-lists partition for the
        first subcircuit (in DQTetris QPU order); ``qpu_order`` maps those
        indices to our QPU ids.
        """
        qpu_order = record.get("qpu_order")
        partition = record.get("partition")
        if not qpu_order or not partition:
            return None
        logical_partition = {}
        for index, logical_group in enumerate(partition):
            if index >= len(qpu_order):
                break
            for logical in logical_group:
                logical_partition[int(logical)] = qpu_order[index]
        missing = [
            logical for logical in range(required_qubits)
            if logical not in logical_partition
        ]
        if missing:
            # DQTetris should cover every qubit; fall back to a capacity-
            # feasible fill if a segment omits isolated qubits.
            counts = Counter(logical_partition.values())
            for logical in missing:
                candidates = [
                    qid for qid in qpu_order
                    if counts[qid] < self.system.qpus[qid]["n_qubits"]
                ]
                if not candidates:
                    return None
                chosen = min(candidates, key=lambda qid: (counts[qid], qid))
                logical_partition[logical] = chosen
                counts[chosen] += 1
        return logical_partition

    def _dqtetris_segment_for(self, circuit_id, circuit, graph, qpu_set):
        """Return (cached) task-level segmentation/teleportation estimate."""
        key = (circuit_id, tuple(sorted(qpu_set)))
        if key not in self._dqtetris_segmentation_cache:
            self._dqtetris_segmentation_cache[key] = (
                dqtetris_segmented_remote_count(
                    circuit, graph, qpu_set, self.system, rng=self.rng
                )
            )
        return self._dqtetris_segmentation_cache[key]

    def _quantum_duration(self, task_or_tid, qpu_set,
                          routing_overhead_ms=0.0, **mapping_context):
        task = (
            self.workload.tasks[task_or_tid]
            if isinstance(task_or_tid, str) else task_or_tid
        )
        return quantum_task_duration_ms(
            task,
            self.system,
            qpu_set,
            self.params,
            routing_overhead_ms=routing_overhead_ms,
            **mapping_context,
        )

    def _quantum_timing_details(self, task_or_tid, qpu_set,
                                routing_overhead_ms=0.0,
                                **mapping_context):
        task = (
            self.workload.tasks[task_or_tid]
            if isinstance(task_or_tid, str) else task_or_tid
        )
        return quantum_task_timing_details(
            task,
            self.system,
            qpu_set,
            self.params,
            routing_overhead_ms=routing_overhead_ms,
            **mapping_context,
        )

    def _classical_duration(self, task_or_tid, node_id):
        task = (
            self.workload.tasks[task_or_tid]
            if isinstance(task_or_tid, str) else task_or_tid
        )
        return classical_task_duration_ms(
            task, self.system.nodes[node_id], self.params
        )

    def _reference_duration(self, task_or_tid):
        task = (
            self.workload.tasks[task_or_tid]
            if isinstance(task_or_tid, str) else task_or_tid
        )
        return reference_task_duration_ms(task, self.system, self.params)

    def schedule(self):
        assignment = {}
        qpu_ids = self.system.get_qpu_ids()
        node_ids = self.system.get_node_ids()

        for tid in self.workload.tasks:
            if self.workload.tasks[tid]["type"] == "quantum":
                assignment[tid] = self.rng.choice(qpu_ids)
            else:
                assignment[tid] = self.rng.choice(node_ids)

        return self._run_scheduling(assignment)

    def _run_scheduling(self, assignment):
        sched = {}
        self.classical_link_timeline = defaultdict(list)
        assignment, assignment_projection = project_quantum_task_assignments(
            self.workload, assignment, self.system
        )
        self.baseline_physical_feasibility = {
            "method": "minimum_hamming_direct_link_projection",
            "partition_method": (
                "dqtetris_inspired_capacity_cut_refinement"
                if isinstance(self, DQTetrisScheduler)
                else "greedy_capacity_interaction"
            ),
            "task_assignment": assignment_projection,
            "partition_tasks_projected": 0,
            "partition_logical_qubits_moved": 0,
            "partition_before_violating_edges": 0,
            "partition_before_violating_gates": 0,
            "partition_after_violating_edges": 0,
            "partition_after_violating_gates": 0,
        }
        qpu_busy = {q: 0.0 for q in self.system.qpus}
        node_busy = {n: 0.0 for n in self.system.nodes}

        priorities = self._compute_priorities()
        all_tasks = sorted(self.workload.get_all_task_ids(), key=lambda t: priorities.get(t, 0), reverse=True)
        scheduled = set()

        # Quantum tasks that share logical state must use one coherent mapping.
        # Union-find over quantum-to-quantum edges with shared logical qubits.
        parent = {tid: tid for tid in self.workload.quantum_tasks}

        def find(tid):
            while parent[tid] != tid:
                parent[tid] = parent[parent[tid]]
                tid = parent[tid]
            return tid

        def union(left, right):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for (pred, succ) in self.workload.edge_data:
            if pred in parent and succ in parent:
                shared, _ = self.workload.get_shared_logical_qubits(pred, succ)
                if shared:
                    union(pred, succ)
        quantum_group = {tid: find(tid) for tid in parent}
        group_mappings = {}

        # Build explicit, capacity-feasible mappings from the same circuit data.
        intra_remote = {}
        mappings = {}
        circuit_cache = {}
        task_graphs = {}
        projected_mapping_cache = {}
        for tid in self.workload.quantum_tasks:
            task = self.workload.tasks[tid]
            required = task.get("n_qubits", 1)
            if required > sum(q["n_qubits"] for q in self.system.qpus.values()):
                raise ValueError(
                    f"Task {tid} requires {required} logical qubits, but the "
                    f"system provides only {sum(q['n_qubits'] for q in self.system.qpus.values())}."
                )
            circuit_id = task.get("circuit_id")
            if circuit_id not in circuit_cache:
                circuit_cache[circuit_id] = load_circuit(circuit_id)
            graph = build_interaction_graph(task, circuit_cache[circuit_id], DEFAULT_PARAMS["alpha"])
            task_graphs[tid] = graph

            group_key = quantum_group.get(tid, tid)
            inherited = group_mappings.get(group_key)
            required_qubits = task.get("n_qubits", 1)
            if inherited is not None and all(
                    logical in inherited["logical_partition"]
                    for logical in range(required_qubits)):
                mappings[tid] = deepcopy(inherited)
                assignment[tid] = inherited["primary_qpu"]
                remote, by_link = count_remote_gates(
                    graph, inherited["logical_partition"]
                )
                mappings[tid]["remote_gates_by_link"] = by_link
                intra_remote[tid] = remote
                if isinstance(self, DQTetrisScheduler):
                    external = self.external_dqtetris.get(circuit_id)
                    if external:
                        external_total = int(
                            external.get("total_communications", remote)
                        )
                        intra_remote[tid] = external_total
                        external_by_link = external.get("remote_by_link")
                        if external_by_link:
                            mappings[tid]["remote_gates_by_link"] = {
                                tuple(sorted(
                                    key.split("|") if isinstance(key, str)
                                    else key
                                )): value
                                for key, value in external_by_link.items()
                            }
                        self.dqtetris_segmentation[tid] = {
                            "model": "external_dqtetris_final_path",
                            "segments": len(external.get("segments", [])),
                            "total": external_total,
                            "teleports": external.get(
                                "teleport_communications"
                            ),
                            "cut_gates": external.get("cut_communications"),
                            "remote_by_link": mappings[tid][
                                "remote_gates_by_link"
                            ],
                        }
                    else:
                        segment = self._dqtetris_segment_for(
                            circuit_id, circuit_cache[circuit_id], graph,
                            inherited["qpu_set"],
                        )
                        self.dqtetris_segmentation[tid] = segment
                        intra_remote[tid] = segment["total"]
                        mappings[tid]["remote_gates_by_link"] = (
                            segment["remote_by_link"]
                        )
                continue

            external_record = None
            external_partition = None
            if isinstance(self, DQTetrisScheduler):
                external_record = self.external_dqtetris.get(circuit_id)
                if external_record:
                    external_partition = self._external_dqtetris_partition(
                        external_record, required_qubits
                    )
            if external_partition is not None:
                partition = external_partition
                primary = partition[0]
                assignment[tid] = primary
                qpu_set = sorted(set(partition.values()))
            else:
                qpu_ids = list(self.system.get_qpu_ids())
                primary = assignment[tid]
                others = [qid for qid in qpu_ids if qid != primary]
                if isinstance(self, RandomScheduler) and self.__class__ is RandomScheduler:
                    self.rng.shuffle(others)
                elif isinstance(self, QOnlyScheduler):
                    others.sort(key=lambda qid: self.system.qpus[qid]["gate_error_rates"]["2q"])
                elif isinstance(self, DQTetrisScheduler):
                    others.sort(key=lambda qid: -sum(
                        1 for other in qpu_ids
                        if other != qid and self.system.get_quantum_link(qid, other)
                    ))
                qpu_set = [primary]
                capacity = self.system.qpus[primary]["n_qubits"]
                for qid in others:
                    if capacity >= required:
                        break
                    qpu_set.append(qid)
                    capacity += self.system.qpus[qid]["n_qubits"]
                if self.__class__ is RandomScheduler:
                    partition = random_capacity_partition(graph, qpu_set, self.system, self.rng)
                elif isinstance(self, GreedyScheduler) and not isinstance(self, (QOnlyScheduler, DQTetrisScheduler)):
                    partition = balanced_capacity_partition(graph, qpu_set, self.system)
                elif isinstance(self, DQTetrisScheduler):
                    partition = dqtetris_partition(
                        graph, qpu_set, self.system, rng=self.rng
                    )
                    primary = partition[0]
                    assignment[tid] = primary
                else:
                    partition = interaction_only_partition(graph, qpu_set, self.system)
            graph_signature = tuple(
                (source, target, edge.get("two_qubit_gates", 1))
                for (source, target), edge in sorted(graph["edges"].items())
            )
            mapping_cache_key = (
                circuit_id,
                task.get("n_qubits"),
                task.get("n_gates"),
                graph_signature,
                tuple(sorted(partition.items())),
            )
            cached_mapping = projected_mapping_cache.get(mapping_cache_key)
            if cached_mapping is None:
                partition, partition_projection = (
                    project_partition_to_direct_links(
                        graph, partition, self.system
                    )
                )
                physical = map_logical_to_physical(
                    graph, partition, self.system
                )
                remote, by_link = count_remote_gates(graph, partition)
                routing = analyze_physical_routing(
                    graph,
                    partition,
                    physical,
                    self.system,
                    routing_mode=self.params.get(
                        "physical_routing_mode", "auto"
                    ),
                )
                if routing.get("unroutable_edges"):
                    raise ValueError(
                        f"Baseline task {tid} has unroutable physical "
                        "interactions."
                    )
                physical = routing["final_physical_mapping"]
                projected_mapping_cache[mapping_cache_key] = (
                    dict(partition),
                    dict(partition_projection),
                    dict(physical),
                    remote,
                    dict(by_link),
                    deepcopy(routing),
                )
            else:
                (partition, partition_projection, physical, remote, by_link,
                 routing) = cached_mapping
                partition = dict(partition)
                partition_projection = dict(partition_projection)
                physical = dict(physical)
                by_link = dict(by_link)
                routing = deepcopy(routing)
            self.baseline_physical_feasibility[
                "partition_tasks_projected"
            ] += int(partition_projection["projection_applied"])
            self.baseline_physical_feasibility[
                "partition_logical_qubits_moved"
            ] += partition_projection["moved_logical_qubits"]
            for stage in ("before", "after"):
                for quantity in ("violating_edges", "violating_gates"):
                    self.baseline_physical_feasibility[
                        f"partition_{stage}_{quantity}"
                    ] += partition_projection[stage][quantity]
            used_qpus = sorted(set(partition.values()))
            mappings[tid] = {
                "primary_qpu": primary, "qpu_set": used_qpus,
                "logical_partition": partition, "physical_mapping": physical,
                "remote_gates_by_link": by_link,
                "physical_routing": routing,
                "n_swap_gates": routing["swap_gate_count"],
                "routing_overhead_ms": routing["routing_overhead_ms"],
                "physical_feasibility_projection": partition_projection,
            }
            intra_remote[tid] = remote
            if isinstance(self, DQTetrisScheduler):
                if external_record and external_partition is not None:
                    external_total = int(
                        external_record.get("total_communications", remote)
                    )
                    intra_remote[tid] = external_total
                    external_by_link = external_record.get("remote_by_link")
                    if external_by_link:
                        mappings[tid]["remote_gates_by_link"] = {
                            tuple(sorted(key.split("|")))
                            if isinstance(key, str) else tuple(key): value
                            for key, value in external_by_link.items()
                        }
                    self.dqtetris_segmentation[tid] = {
                        "model": "external_dqtetris_final_path",
                        "segments": len(external_record.get("segments", [])),
                        "total": external_total,
                        "teleports": external_record.get("teleport_communications"),
                        "cut_gates": external_record.get("cut_communications"),
                        "remote_by_link": mappings[tid][
                            "remote_gates_by_link"
                        ],
                    }
                else:
                    segment = self._dqtetris_segment_for(
                        circuit_id, circuit_cache[circuit_id], graph, qpu_set
                    )
                    self.dqtetris_segmentation[tid] = segment
                    intra_remote[tid] = segment["total"]
                    mappings[tid]["remote_gates_by_link"] = (
                        segment["remote_by_link"]
                    )
            group_mappings.setdefault(group_key, mappings[tid])

        while len(scheduled) < len(all_tasks):
            for tid in all_tasks:
                if tid in scheduled:
                    continue
                preds = self.workload.predecessors.get(tid, [])
                if not all(p in scheduled for p in preds):
                    continue

                task = self.workload.tasks[tid]
                is_quantum = task["type"] == "quantum"

                if is_quantum:
                    qid = assignment[tid]
                    task_mapping = mappings[tid]
                    partition_sizes = Counter(task_mapping["logical_partition"].values())
                    timing_details = self._quantum_timing_details(
                        task,
                        partition_sizes,
                        routing_overhead_ms=task_mapping[
                            "routing_overhead_ms"
                        ],
                        logical_partition=task_mapping["logical_partition"],
                        physical_mapping=task_mapping["physical_mapping"],
                        interaction_graph=task_graphs[tid],
                        physical_routing=task_mapping["physical_routing"],
                    )
                    est_time = timing_details["duration_ms"]
                    est_start = max(qpu_busy[q] for q in partition_sizes)
                    ent_wait = 0.0
                    dependency_transfers = []
                    pending_epr_batches = []
                    for pred in dict.fromkeys(preds):
                        if pred in sched:
                            if self.workload.tasks[pred]["type"] == "classical":
                                transfer = self._schedule_classical_dependency(
                                    pred, tid, sched, assignment
                                )
                                dependency_transfers.append(transfer)
                                est_start = max(est_start, transfer["end"])
                                continue
                            est_start = max(est_start, sched[pred]["end"])
                            pred_q = assignment[pred]
                            transfer = {
                                "predecessor": pred,
                                "start": sched[pred]["end"],
                                "end": sched[pred]["end"],
                                "source_resource": pred_q,
                                "target_resource": qid,
                                "entanglement_reservations": [],
                            }
                            if pred_q != qid:
                                transfer["type"] = "quantum-quantum"
                            dependency_transfers.append(transfer)
                            if pred_q != qid:
                                link = self.system.get_quantum_link(qid, pred_q)
                                if not link:
                                    raise ValueError(
                                        "Missing direct quantum link "
                                        f"{tuple(sorted((qid, pred_q)))} for "
                                        f"baseline dependency {pred}->{tid}."
                                    )
                                timing = self._entanglement_timing(
                                    link, epr_count=1
                                )
                                duration = timing["scheduled_window_ms"]
                                ent_wait += duration
                                pending_epr_batches.append({
                                    "link_key": (qid, pred_q),
                                    "epr_count": 1,
                                    "duration": duration,
                                    "timing": timing,
                                    "predecessor": pred,
                                    "transfer": transfer,
                                })

                    # Intra-task remote gates
                    n_remote = intra_remote.get(tid, 0)
                    for link_key, link_remote in task_mapping["remote_gates_by_link"].items():
                        link = self.system.get_quantum_link(*link_key)
                        if not link:
                            raise ValueError(f"Missing quantum link {link_key} for baseline task {tid}.")
                        capacity = max(link["max_concurrent_epr"], 1)
                        remaining = max(int(link_remote), 0)
                        while remaining:
                            count = min(capacity, remaining)
                            timing = self._entanglement_timing(
                                link, epr_count=count
                            )
                            duration = timing["scheduled_window_ms"]
                            ent_wait += duration
                            pending_epr_batches.append({
                                "link_key": link_key,
                                "epr_count": count,
                                "duration": duration,
                                "timing": timing,
                                "predecessor": None,
                                "transfer": None,
                            })
                            remaining -= count

                    start = est_start + ent_wait
                    end = start + est_time
                    entanglement_reservations = []
                    generation_cursor = est_start
                    for batch_index, pending in enumerate(
                            pending_epr_batches):
                        generation_start = generation_cursor
                        generation_cursor += pending["duration"]
                        event = build_epr_event(
                            self.system,
                            pending["link_key"],
                            pending["epr_count"],
                            generation_cursor,
                            start,
                            task_id=tid,
                            predecessor=pending["predecessor"],
                            source="baseline_sequential_generation",
                        )
                        link = self.system.get_quantum_link(
                            *pending["link_key"]
                        )
                        event.update({
                            "batch_index": batch_index,
                            "start": generation_start,
                            "generation_start": generation_start,
                            "end": generation_cursor,
                            "deadline": start,
                            "effective_rate_hz": link["rate_hz"],
                            "duration_scale": 1.0,
                            "quantum_link_latency_mode": pending[
                                "timing"
                            ]["quantum_link_latency_mode"],
                            "stage_overhead_ms": pending["timing"][
                                "stage_overhead_ms"
                            ],
                            "selected_channel_stage_components": pending[
                                "timing"
                            ]["selected_channel_stage_components"],
                            "multi_channel_stage_aggregation": pending[
                                "timing"
                            ]["multi_channel_stage_aggregation"],
                            "capacity_epr": max(
                                int(link["max_concurrent_epr"]), 1
                            ),
                        })
                        if pending["transfer"] is None:
                            entanglement_reservations.append(event)
                        else:
                            pending["transfer"][
                                "entanglement_reservations"
                            ].append(event)
                            pending["transfer"]["end"] = generation_cursor
                            pending["transfer"]["quantum_link"] = tuple(
                                sorted(pending["link_key"])
                            )
                    sched[tid] = {"start": start, "end": end, "qpu": qid,
                                  "qpu_set": list(partition_sizes),
                                  "partition_sizes": dict(partition_sizes),
                                  "mapped_physical_qubits_by_qpu": {
                                      part_qid: sorted(
                                          physical
                                          for logical, physical in task_mapping[
                                              "physical_mapping"
                                          ].items()
                                          if task_mapping["logical_partition"][
                                              logical
                                          ] == part_qid
                                      )
                                      for part_qid in partition_sizes
                                  },
                                  "entanglement_wait": ent_wait,
                                  "n_remote_gates": n_remote,
                                  "n_swap_gates": task_mapping[
                                      "n_swap_gates"
                                  ],
                                  "routing_overhead_ms": task_mapping[
                                      "routing_overhead_ms"
                                  ],
                                  "quantum_timing": timing_details,
                                  "dependency_transfers": dependency_transfers,
                                  "entanglement_reservations": entanglement_reservations}
                    for part_qid in partition_sizes:
                        qpu_busy[part_qid] = end
                else:
                    nid = assignment[tid]
                    est_time = self._classical_duration(task, nid)
                    predecessor_ready = max(
                        (sched[pred]["end"] for pred in preds if pred in sched),
                        default=0.0,
                    )
                    base_ready = max(node_busy[nid], predecessor_ready)
                    dependency_ready = predecessor_ready
                    dependency_transfers = []
                    for pred in dict.fromkeys(preds):
                        if pred in sched:
                            transfer = self._schedule_classical_dependency(
                                pred, tid, sched, assignment
                            )
                            dependency_transfers.append(transfer)
                            dependency_ready = max(
                                dependency_ready, transfer["end"]
                            )
                    start = max(node_busy[nid], dependency_ready)
                    comm = max(0.0, start - base_ready)
                    end = start + est_time
                    sched[tid] = {
                        "start": start, "end": end, "node": nid,
                        "comm_time": comm,
                        "dependency_transfers": dependency_transfers,
                    }
                    node_busy[nid] = end

                scheduled.add(tid)

        return sched, assignment, mappings

    def _compute_ent_time(self, link, epr_count=1):
        """Compute entanglement time for baselines (no optimization)."""
        return self._entanglement_timing(
            link, epr_count=epr_count
        )["scheduled_window_ms"]

    def _entanglement_timing(self, link, epr_count=1):
        service_latency = resolve_service_latency(
            (link,),
            self.quantum_link_latency_policy,
            self.system.sim_params,
        )
        return {
            **service_latency,
            "scheduled_window_ms": deterministic_epr_service_window_ms(
                link["rate_hz"],
                DEFAULT_PARAMS["p_threshold"],
                1.0,
                service_latency["stage_overhead_ms"],
                epr_count=epr_count,
            ),
            "epr_count": int(epr_count),
            "generation_distribution": "erlang_poisson_success_count",
        }

    def _schedule_classical_dependency(self, predecessor, successor, sched,
                                       assignment):
        """Schedule Q-C, C-C, or C-Q data transfer on physical resources."""
        pred_type = self.workload.tasks[predecessor]["type"]
        succ_type = self.workload.tasks[successor]["type"]
        pred_end = sched[predecessor]["end"]
        data_mb = edge_data_mb(self.workload, predecessor, successor)
        source_resource = assignment[predecessor]
        target_resource = assignment[successor]
        transfer = {
            "predecessor": predecessor,
            "start": pred_end,
            "end": pred_end,
            "data_mb": data_mb,
            "source_resource": source_resource,
            "target_resource": target_resource,
        }
        if pred_type == "quantum" and succ_type == "quantum":
            if source_resource == target_resource:
                return transfer
            return dict(transfer, type="quantum-quantum")

        if pred_type == "quantum":
            qpu_id = assignment[predecessor]
            target_node = assignment[successor]
            controller = self.system.interface_links[qpu_id]["classical_node_id"]
            cursor = pred_end + interface_transfer_ms(
                self.system, qpu_id, data_mb
            )
            network = None
            if controller != target_node:
                path = select_widest_path(
                    self.system, controller, target_node
                )
                network = reserve_classical_path(
                    self.system, self.classical_link_timeline, path, cursor,
                    data_mb, successor, predecessor,
                )
                cursor = network["end"]
            return dict(
                transfer, type="quantum-classical", end=cursor,
                interface_ms=cursor - pred_end - (
                    network["duration"] if network else 0.0
                ), network=network,
            )

        source_node = assignment[predecessor]
        if succ_type == "classical":
            target_node = assignment[successor]
            if source_node == target_node:
                return transfer
            path = select_widest_path(self.system, source_node, target_node)
            network = reserve_classical_path(
                self.system, self.classical_link_timeline, path, pred_end,
                data_mb, successor, predecessor,
            )
            return dict(
                transfer, type="classical-classical", end=network["end"],
                network=network,
            )

        target_qpu = assignment[successor]
        controller = self.system.interface_links[target_qpu]["classical_node_id"]
        cursor = pred_end
        network = None
        if source_node != controller:
            path = select_widest_path(self.system, source_node, controller)
            network = reserve_classical_path(
                self.system, self.classical_link_timeline, path, cursor,
                data_mb, successor, predecessor,
            )
            cursor = network["end"]
        interface_ms = interface_transfer_ms(
            self.system, target_qpu, data_mb
        )
        cursor += interface_ms
        return dict(
            transfer, type="classical-quantum", end=cursor,
            interface_ms=interface_ms, network=network,
        )

    def _compute_priorities(self):
        """Compute a topological (HEFT-style) upward-rank priority.

        The previous implementation walked predecessors with a DFS and could
        read a successor's priority before that successor had been computed,
        which made the dispatch order topology-dependent instead of
        critical-path correct.  Kahn's algorithm guarantees that every
        successor is finalised before its predecessors are ranked.
        """
        predecessors = self.workload.predecessors
        successors = self.workload.successors
        in_degree = {
            tid: len(set(predecessors.get(tid, [])))
            for tid in self.workload.tasks
        }
        queue = deque(
            sorted(tid for tid, degree in in_degree.items() if degree == 0)
        )
        topo_order = []
        while queue:
            tid = queue.popleft()
            topo_order.append(tid)
            for succ in dict.fromkeys(successors.get(tid, [])):
                if succ not in in_degree:
                    continue
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        if len(topo_order) != len(in_degree):
            # A cyclic workload is invalid input; keep a deterministic
            # fallback rather than silently producing partial priorities.
            topo_order = sorted(in_degree)

        priorities = {}
        for tid in reversed(topo_order):
            task_time = self._reference_duration(tid)
            max_succ = max(
                (
                    priorities.get(succ, 0.0)
                    for succ in dict.fromkeys(successors.get(tid, []))
                ),
                default=0.0,
            )
            priorities[tid] = task_time + max_succ
        return priorities


class GreedyScheduler(RandomScheduler):
    """Greedy load balancing without topology awareness."""
    def schedule(self):
        assignment = {}
        qpu_loads = {q: 0.0 for q in self.system.qpus}
        node_loads = {n: 0.0 for n in self.system.nodes}

        # Sort tasks by the shared resource-aware reference duration.
        sorted_tasks = sorted(self.workload.tasks.values(),
                              key=self._reference_duration,
                              reverse=True)

        for task in sorted_tasks:
            tid = task["id"]
            if task["type"] == "quantum":
                # Assign to least loaded QPU
                best_q = min(self.system.qpus.keys(), key=lambda q: qpu_loads[q])
                assignment[tid] = best_q
                est_time = self._quantum_duration(task, [best_q])
                qpu_loads[best_q] += est_time
            else:
                # Assign to least loaded node
                best_n = min(self.system.nodes.keys(), key=lambda n: node_loads[n])
                assignment[tid] = best_n
                est_time = self._classical_duration(task, best_n)
                node_loads[best_n] += est_time

        return self._run_scheduling(assignment)


class QOnlyScheduler(RandomScheduler):
    """Optimize quantum scheduling only, ignore classical task placement."""
    def schedule(self):
        assignment = {}
        qpu_ids = self.system.get_qpu_ids()
        node_ids = self.system.get_node_ids()

        # Use topology-aware quantum assignment (like TopoQ Phase 2)
        for tid in self.workload.quantum_tasks:
            task = self.workload.tasks[tid]
            n_qubits = task.get("n_qubits", 1)
            # Find QPU with most qubits and good fidelity
            candidates = [q for q in qpu_ids if self.system.qpus[q]["n_qubits"] >= n_qubits]
            if candidates:
                # Pick QPU with best gate fidelity
                best_q = max(candidates, key=lambda q: 1 - self.system.qpus[q]["gate_error_rates"]["2q"])
                assignment[tid] = best_q
            else:
                assignment[tid] = self.rng.choice(qpu_ids)

        # Classical: random assignment (no optimization)
        for tid in self.workload.classical_tasks:
            assignment[tid] = self.rng.choice(node_ids)

        return self._run_scheduling(assignment)


class DQTetrisScheduler(RandomScheduler):
    """DQTetris-inspired partition-level baseline (Xu et al., SC 2025).

    This is a partition-level port: logical qubits are assigned to module
    capacity-feasible groups with a multi-start, cut-minimising partitioner
    (``dqtetris_partition``).  It does not reproduce DQTetris's gate-level
    circuit slicing / adaptive gate teleportation; cross-module interactions
    are charged as remote gates by the shared timing and fidelity model.  The
    manuscript must describe it as "DQTetris-inspired" rather than as an exact
    reproduction.
    """
    def schedule(self):
        assignment = {}
        qpu_ids = self.system.get_qpu_ids()
        node_ids = self.system.get_node_ids()

        # Seed assignment on the most connected QPU, then let the
        # cut-minimising partitioner spread logical qubits across a
        # capacity-feasible, directly connected QPU set.
        most_connected_q = max(qpu_ids, key=lambda q: sum(
            1 for other in qpu_ids if other != q and self.system.get_quantum_link(q, other)
        ))

        for tid in self.workload.quantum_tasks:
            assignment[tid] = most_connected_q

        # Classical: nearest to QPU
        for tid in self.workload.classical_tasks:
            related_q = None
            for pred in self.workload.predecessors.get(tid, []):
                if pred in assignment:
                    related_q = assignment[pred]
                    break
            if related_q and related_q in self.system.interface_links:
                node_id = self.system.interface_links[related_q]["classical_node_id"]
            else:
                node_id = self.rng.choice(node_ids)
            assignment[tid] = node_id

        return self._run_scheduling(assignment)

    def _run_scheduling_dqtetris(self, assignment):
        """Compatibility wrapper for the shared explicit-mapping scheduler."""
        return self._run_scheduling(assignment)
