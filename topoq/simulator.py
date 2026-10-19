"""
Discrete-Event Simulator & Metrics
==================================
Simulates the scheduled hybrid quantum-classical system and computes
all metrics: makespan, communication overhead, fidelity, utilization.
"""

import math
import random
from collections import defaultdict
from copy import deepcopy

from .communication import annotate_communication_overlap
from .config import (
    validate_communication_overlap_mode,
    validate_error_mitigation_policy,
)
from .fidelity import (
    aggregate_task_fidelities,
    estimate_schedule_task_fidelities,
    estimate_purification_scenario_fidelities,
)
from .entanglement import (
    resolve_quantum_link_latency_policy,
    resolve_service_latency,
)
from .mitigation import estimate_error_mitigation_metrics
from .resources import (
    find_quantum_capacity_start,
    physical_qubits_by_qpu,
    reserve_quantum_capacity,
)


class Simulator:
    """Discrete-event simulator for hybrid quantum-classical systems."""

    def __init__(self, system, sim_params, seed=42, algorithm_name="",
                 entanglement_mode="deterministic",
                 error_mitigation_policy=None,
                 communication_overlap_mode="none"):
        self.system = system
        self.sim_params = sim_params
        self.rng = random.Random(seed)
        if entanglement_mode not in {"deterministic", "poisson"}:
            raise ValueError(
                "entanglement simulation mode must be deterministic or poisson"
            )
        self.entanglement_mode = entanglement_mode
        self.quantum_link_latency_policy = (
            resolve_quantum_link_latency_policy(
                sim_params.get("quantum_link_latency_policy", {}),
                sim_params,
            )
        )
        self.communication_overlap_mode = validate_communication_overlap_mode(
            communication_overlap_mode
        )
        self.error_mitigation_policy = (
            dict(error_mitigation_policy)
            if error_mitigation_policy is not None
            else validate_error_mitigation_policy()
        )

    def simulate(self, sched, assignment, workload, mapping=None):
        """Run simulation and compute metrics."""
        sim_sched = deepcopy(sched)
        scheduled_makespan = max(s["end"] for s in sim_sched.values())
        stochastic = {
            "mode": self.entanglement_mode,
            "sampled_batches": 0,
            "sampled_generation_ms": 0.0,
            "sampled_stage_overhead_ms": 0.0,
            "sampled_service_ms": 0.0,
            "schedule_shift_ms": 0.0,
        }
        if self.entanglement_mode == "poisson":
            sim_sched, stochastic = self._replay_poisson_schedule(
                sim_sched, workload, assignment, mapping or {}
            )
        annotate_communication_overlap(
            sim_sched, self.communication_overlap_mode
        )

        # Compute metrics
        metrics = {}

        # Makespan
        metrics["makespan"] = max(s["end"] for s in sim_sched.values())
        stochastic["schedule_shift_ms"] = (
            metrics["makespan"] - scheduled_makespan
        )

        # Communication overhead breakdown
        ent_wait_total = sum(s.get("entanglement_wait", 0) for s in sim_sched.values())
        qc_transfer_total = sum(s.get("comm_time", 0) for s in sim_sched.values()
                               if self.workload_type(workload, s) == "classical")
        classical_net_total = sum(s.get("comm_time", 0) for s in sim_sched.values()
                                 if self.workload_type(workload, s) == "classical_net")

        # More accurate breakdown
        ent_wait = 0
        qc_transfer = 0
        classical_net = 0
        for tid, s in sim_sched.items():
            task = workload.tasks[tid]
            if task["type"] == "quantum":
                ent_wait += s.get("entanglement_wait", 0)
            else:
                # Check if comm is QC or classical
                preds = workload.predecessors.get(tid, [])
                has_quantum_pred = any(workload.tasks[p]["type"] == "quantum" for p in preds if p in workload.tasks)
                if has_quantum_pred:
                    qc_transfer += s.get("comm_time", 0)
                else:
                    classical_net += s.get("comm_time", 0)

        metrics["comm_entanglement"] = ent_wait
        metrics["comm_qc_transfer"] = qc_transfer
        metrics["comm_classical_net"] = classical_net
        metrics["comm_total"] = ent_wait + qc_transfer + classical_net

        entanglement_credit = sum(
            item.get("paper_entanglement_overlap_credit_ms", 0.0)
            for item in sim_sched.values()
        )
        qc_credit = 0.0
        classical_credit = 0.0
        for tid, item in sim_sched.items():
            if workload.tasks[tid]["type"] == "quantum":
                continue
            predecessors = workload.predecessors.get(tid, [])
            has_quantum_predecessor = any(
                workload.tasks[pred]["type"] == "quantum"
                for pred in predecessors if pred in workload.tasks
            )
            credit = item.get("paper_classical_overlap_credit_ms", 0.0)
            if has_quantum_predecessor:
                qc_credit += credit
            else:
                classical_credit += credit
        total_credit = entanglement_credit + qc_credit + classical_credit
        metrics["communication_overlap_mode"] = (
            self.communication_overlap_mode
        )
        metrics["paper_entanglement_overlap_credit_ms"] = (
            entanglement_credit
        )
        metrics["paper_qc_overlap_credit_ms"] = qc_credit
        metrics["paper_classical_overlap_credit_ms"] = classical_credit
        metrics["paper_overlap_credit_total_ms"] = total_credit
        metrics["paper_effective_comm_entanglement_ms"] = (
            ent_wait - entanglement_credit
        )
        metrics["paper_effective_comm_qc_transfer_ms"] = (
            qc_transfer - qc_credit
        )
        metrics["paper_effective_comm_classical_net_ms"] = (
            classical_net - classical_credit
        )
        metrics["paper_effective_comm_total_ms"] = (
            metrics["comm_total"] - total_credit
        )

        # Fidelity: stable task-level lower bounds plus aggregate diagnostics.
        metrics.update(self._compute_fidelity_metrics(
            sim_sched, assignment, workload, mapping or {}
        ))

        # Resource utilization
        metrics["utilization"] = self._compute_utilization(sim_sched, assignment, workload)
        metrics["entanglement_sampled_batches"] = stochastic["sampled_batches"]
        metrics["entanglement_sampled_generation_ms"] = stochastic["sampled_generation_ms"]
        metrics["entanglement_sampled_stage_overhead_ms"] = stochastic[
            "sampled_stage_overhead_ms"
        ]
        metrics["entanglement_sampled_service_ms"] = stochastic[
            "sampled_service_ms"
        ]
        metrics["entanglement_schedule_shift_ms"] = stochastic["schedule_shift_ms"]
        metrics["entanglement_simulation_mode"] = stochastic["mode"]
        metrics["quantum_link_latency_mode"] = (
            self.quantum_link_latency_policy["mode"]
        )
        quantum_count = sum(tid in sim_sched for tid in workload.quantum_tasks)
        shared_count = sum(
            bool(sim_sched[tid].get("space_shared"))
            for tid in workload.quantum_tasks if tid in sim_sched
        )
        metrics["space_shared_tasks"] = shared_count
        metrics["space_shared_task_fraction"] = (
            shared_count / quantum_count if quantum_count else 0.0
        )
        metrics["qpu_capacity_wait_ms"] = sum(
            sim_sched[tid].get("qpu_capacity_wait", 0.0)
            for tid in workload.quantum_tasks if tid in sim_sched
        )
        metrics["space_sharing_exclusive_routing_tasks"] = sum(
            bool(sim_sched[tid].get("space_sharing_exclusive_routing"))
            for tid in workload.quantum_tasks if tid in sim_sched
        )
        self.last_sim_sched = sim_sched

        return metrics

    def _sample_epr_generation_ms(self, link_key, effective_rate_hz=None,
                                  duration_scale=1.0, epr_count=1):
        link = self.system.quantum_links.get(frozenset(link_key))
        if link is None:
            raise ValueError(f"Unknown quantum link for stochastic replay: {link_key}")
        rate = float(
            link["rate_hz"]
            if effective_rate_hz is None else effective_rate_hz
        )
        if rate <= 0:
            raise ValueError(f"Quantum link {tuple(link_key)} has non-positive EPR rate")
        duration_scale = float(duration_scale)
        if duration_scale <= 0:
            raise ValueError("EPR duration scale must be positive")
        epr_count = int(epr_count)
        if epr_count <= 0:
            raise ValueError("epr_count must be a positive integer")
        unit_rate_wait = sum(
            -math.log1p(-self.rng.random()) for _ in range(epr_count)
        )
        return unit_rate_wait / rate * 1000.0 * duration_scale

    @staticmethod
    def _topological_order(workload, sched):
        tasks = set(sched)
        in_degree = {
            tid: sum(pred in tasks for pred in workload.predecessors.get(tid, []))
            for tid in tasks
        }
        ready = sorted(
            (tid for tid, degree in in_degree.items() if degree == 0),
            key=lambda tid: (sched[tid]["start"], tid),
        )
        order = []
        while ready:
            tid = ready.pop(0)
            order.append(tid)
            for successor in workload.successors.get(tid, []):
                if successor not in in_degree:
                    continue
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=lambda item: (sched[item]["start"], item))
        if len(order) != len(tasks):
            raise ValueError("Cannot replay a cyclic or incomplete workload schedule")
        return order

    def _sample_reservation_end(self, record, cache, statistics):
        statistics.setdefault("sampled_generation_ms", 0.0)
        statistics.setdefault("sampled_stage_overhead_ms", 0.0)
        statistics.setdefault("sampled_service_ms", 0.0)
        key = id(record)
        if key in cache:
            return cache[key]
        link_key = record.get("link") or record.get("quantum_link")
        if not link_key:
            cache[key] = float(record.get("end", record.get("ready", 0.0)))
            return cache[key]
        generation_ms = self._sample_epr_generation_ms(
            link_key,
            effective_rate_hz=record.get("effective_rate_hz"),
            duration_scale=record.get("duration_scale", 1.0),
            epr_count=record.get("epr_count", 1),
        )
        stage_overhead_ms = record.get("stage_overhead_ms")
        if stage_overhead_ms is None:
            stage_overhead_ms = self._link_stage_overhead_ms(link_key)
        stage_overhead_ms = float(stage_overhead_ms)
        duration = generation_ms + stage_overhead_ms
        start = float(record.get("generation_start", record.get("start", 0.0)))
        end = start + duration
        cache[key] = end
        statistics["sampled_batches"] += 1
        statistics["sampled_generation_ms"] += generation_ms
        statistics["sampled_stage_overhead_ms"] += stage_overhead_ms
        statistics["sampled_service_ms"] += duration
        record["scheduled_end"] = record.get("end")
        record["sampled_generation_ms"] = generation_ms
        record["sampled_stage_overhead_ms"] = stage_overhead_ms
        record["sampled_duration_ms"] = duration
        record["actual_end"] = end
        return end

    def _sample_aggregate_quantum_wait(self, tid, item, workload, assignment,
                                       mapping, statistics):
        """Sample baseline or legacy schedules that lack reservation records."""
        statistics.setdefault("sampled_generation_ms", 0.0)
        statistics.setdefault("sampled_stage_overhead_ms", 0.0)
        statistics.setdefault("sampled_service_ms", 0.0)
        total_wait = 0.0
        for predecessor in workload.predecessors.get(tid, []):
            if (predecessor not in workload.quantum_tasks
                    or assignment.get(predecessor) == assignment.get(tid)):
                continue
            link_key = (assignment[predecessor], assignment[tid])
            generation_ms = self._sample_epr_generation_ms(link_key)
            stage_overhead_ms = self._link_stage_overhead_ms(link_key)
            total_wait += generation_ms + stage_overhead_ms
            statistics["sampled_batches"] += 1
            statistics["sampled_generation_ms"] += generation_ms
            statistics["sampled_stage_overhead_ms"] += stage_overhead_ms
            statistics["sampled_service_ms"] += (
                generation_ms + stage_overhead_ms
            )
        remote_by_link = mapping.get(tid, {}).get("remote_gates_by_link", {})
        for link_key, remote_gates in remote_by_link.items():
            link = self.system.quantum_links.get(frozenset(link_key))
            if link is None:
                raise ValueError(f"Unknown quantum link for task {tid}: {link_key}")
            capacity = max(int(link["max_concurrent_epr"]), 1)
            remaining = max(int(remote_gates), 0)
            while remaining:
                epr_count = min(capacity, remaining)
                generation_ms = self._sample_epr_generation_ms(
                    link_key, epr_count=epr_count
                )
                stage_overhead_ms = self._link_stage_overhead_ms(link_key)
                total_wait += generation_ms + stage_overhead_ms
                statistics["sampled_batches"] += 1
                statistics["sampled_generation_ms"] += generation_ms
                statistics["sampled_stage_overhead_ms"] += (
                    stage_overhead_ms
                )
                statistics["sampled_service_ms"] += (
                    generation_ms + stage_overhead_ms
                )
                remaining -= epr_count
        return total_wait

    def _link_stage_overhead_ms(self, link_key):
        link = self.system.quantum_links.get(frozenset(link_key))
        if link is None:
            raise ValueError(f"Unknown quantum link: {link_key}")
        return resolve_service_latency(
            (link,),
            self.quantum_link_latency_policy,
            getattr(self.system, "sim_params", {}),
        )["stage_overhead_ms"]

    def _replay_poisson_schedule(self, sched, workload, assignment, mapping):
        """Replay sampled EPR completion times through resources and the DAG."""
        original = deepcopy(sched)
        statistics = {
            "mode": "poisson",
            "sampled_batches": 0,
            "sampled_generation_ms": 0.0,
            "sampled_stage_overhead_ms": 0.0,
            "sampled_service_ms": 0.0,
            "schedule_shift_ms": 0.0,
        }
        sampled = {}
        qpu_ready = {qid: 0.0 for qid in self.system.qpus}
        qpu_capacity = {
            qid: qpu["n_qubits"] for qid, qpu in self.system.qpus.items()
        }
        quantum_task_timeline = defaultdict(list)
        capacity_sharing_enabled = any(
            item.get("space_shared", False) for item in original.values()
        )
        node_ready = {nid: 0.0 for nid in self.system.nodes}

        for tid in self._topological_order(workload, original):
            item = sched[tid]
            original_item = original[tid]
            task = workload.tasks[tid]
            predecessors = workload.predecessors.get(tid, [])
            predecessor_ready = max(
                (sched[pred]["end"] for pred in predecessors if pred in sched),
                default=0.0,
            )
            transfer_ready = predecessor_ready
            transfers_by_pred = {
                transfer.get("predecessor"): transfer
                for transfer in item.get("dependency_transfers", [])
            }
            for predecessor in predecessors:
                if predecessor not in sched:
                    continue
                pred_end = sched[predecessor]["end"]
                transfer = transfers_by_pred.get(predecessor)
                if not transfer:
                    transfer_ready = max(transfer_ready, pred_end)
                    continue
                original_duration = max(
                    float(transfer.get("end", pred_end))
                    - float(transfer.get("start", pred_end)),
                    0.0,
                )
                if transfer.get("type") == "quantum-quantum":
                    reservations = transfer.get("entanglement_reservations", [])
                    actual_end = max(
                        (self._sample_reservation_end(
                            record, sampled, statistics
                        ) for record in reservations),
                        default=pred_end,
                    )
                    actual_end = max(actual_end, pred_end)
                else:
                    actual_end = pred_end + original_duration
                transfer["actual_start"] = pred_end
                transfer["actual_end"] = actual_end
                transfer_ready = max(transfer_ready, actual_end)

            if task["type"] == "quantum":
                qpu_set = item.get("qpu_set") or [item.get("qpu")]
                partition_sizes = item.get("partition_sizes") or {
                    item.get("qpu"): task.get("n_qubits", 0)
                }
                physical_requirements = item.get(
                    "physical_qubits_by_qpu"
                )
                if physical_requirements is None:
                    task_mapping = mapping.get(tid, {})
                    physical_requirements = physical_qubits_by_qpu(
                        task_mapping.get("logical_partition", {}),
                        task_mapping.get("physical_mapping", {}),
                    )
                exclusive_qpus = item.get("space_sharing_exclusive_qpus")
                if exclusive_qpus is None:
                    exclusive_qpus = (
                        list(partition_sizes)
                        if item.get("space_sharing_exclusive_routing") else []
                    )
                exclusive_qpus = set(exclusive_qpus)
                occupancy_requirements = {
                    qid: (
                        qpu_capacity[qid]
                        if qid in exclusive_qpus else qubits
                    )
                    for qid, qubits in partition_sizes.items()
                }
                compute_duration = max(
                    original_item["end"] - original_item["start"],
                    0.0,
                )
                if capacity_sharing_enabled:
                    resource_ready, _ = find_quantum_capacity_start(
                        quantum_task_timeline,
                        qpu_capacity,
                        occupancy_requirements,
                        transfer_ready,
                        compute_duration,
                        physical_requirements=physical_requirements,
                    )
                    base_ready = max(resource_ready, transfer_ready)
                else:
                    resource_ready = max(
                        (qpu_ready.get(qid, 0.0) for qid in qpu_set if qid),
                        default=0.0,
                    )
                    base_ready = max(resource_ready, transfer_ready)
                epr_ready = base_ready
                reservations = item.get("entanglement_reservations", [])
                for record in reservations:
                    epr_ready = max(
                        epr_ready,
                        self._sample_reservation_end(
                            record, sampled, statistics
                        ),
                    )
                if not reservations and item.get("entanglement_wait", 0.0) > 0:
                    epr_ready = max(
                        epr_ready,
                        base_ready + self._sample_aggregate_quantum_wait(
                            tid, item, workload, assignment, mapping, statistics
                        ),
                    )
                ready_without_capacity = max(base_ready, epr_ready)
                if capacity_sharing_enabled:
                    start, shared_with = find_quantum_capacity_start(
                        quantum_task_timeline,
                        qpu_capacity,
                        occupancy_requirements,
                        ready_without_capacity,
                        compute_duration,
                        physical_requirements=physical_requirements,
                    )
                else:
                    start = ready_without_capacity
                    shared_with = []
                end = start + compute_duration
                item["entanglement_wait"] = max(0.0, epr_ready - base_ready)
                for record in item.get("entanglement_reservations", []):
                    record["actual_consume"] = start
                for transfer in item.get("dependency_transfers", []):
                    if transfer.get("type") != "quantum-quantum":
                        continue
                    for record in transfer.get(
                            "entanglement_reservations", []):
                        record["actual_consume"] = start
                if capacity_sharing_enabled:
                    item["space_shared"] = bool(shared_with)
                    item["space_shared_with"] = shared_with
                    item["qpu_capacity_wait"] = max(
                        0.0, start - ready_without_capacity
                    )
                    reserve_quantum_capacity(
                        quantum_task_timeline,
                        occupancy_requirements,
                        start,
                        end,
                        tid,
                        physical_qubits=physical_requirements,
                    )
                else:
                    for qid in qpu_set:
                        if qid:
                            qpu_ready[qid] = end
            else:
                nid = item.get("node")
                resource_ready = node_ready.get(nid, 0.0)
                base_ready = max(resource_ready, predecessor_ready)
                start = max(resource_ready, transfer_ready)
                compute_duration = max(
                    original_item["end"] - original_item["start"],
                    0.0,
                )
                end = start + compute_duration
                item["comm_time"] = max(0.0, start - base_ready)
                if nid:
                    node_ready[nid] = end

            shift = start - original_item["start"]
            item["scheduled_start"] = original_item["start"]
            item["scheduled_end"] = original_item["end"]
            item["start"] = start
            item["end"] = end
            item["stochastic_shift_ms"] = shift

        return sched, statistics

    def workload_type(self, workload, sched_item):
        return "quantum" if "qpu" in sched_item else "classical"

    def _compute_fidelity(self, sched, assignment, workload, mapping=None):
        """Compatibility scalar: arithmetic mean of task lower bounds."""
        return self._compute_fidelity_metrics(
            sched, assignment, workload, mapping
        )["fidelity"]

    def _compute_fidelity_metrics(self, sched, assignment, workload,
                                  mapping=None):
        """Return physical fidelity and separate mitigation diagnostics."""
        task_estimates = estimate_schedule_task_fidelities(
            self.system, workload, sched, assignment, mapping=mapping
        )
        self.last_task_estimates = task_estimates
        default_fidelity = self.system.sim_params[
            "fidelity_parameters"
        ]["F_prep"]
        physical = aggregate_task_fidelities(
            task_estimates, default_fidelity
        )
        purification_estimates = estimate_purification_scenario_fidelities(
            self.system, task_estimates
        )
        purification_scenario = aggregate_task_fidelities(
            purification_estimates,
            default_fidelity,
        )
        mitigation = estimate_error_mitigation_metrics(
            task_estimates,
            sched,
            self.error_mitigation_policy,
            default_fidelity=default_fidelity,
        )
        plan = self.system.sim_params.get("purification_policy", {})
        remote_pairs = sum(
            max(int(item.get("n_remote_epr_pairs", 0)), 0)
            for item in task_estimates.values()
        )
        expected_raw_pairs = sum(
            float(item.get("expected_raw_epr_pairs", 0.0))
            for item in purification_estimates.values()
        )
        expected_attempts = sum(
            float(item.get("expected_purification_attempts", 0.0))
            for item in purification_estimates.values()
        )
        actual_rounds = max(
            (
                int(item.get("purification_rounds", 0))
                for item in purification_estimates.values()
            ),
            default=0,
        )
        eq5_task_count = sum(
            item.get("fidelity_model") == "eq5_per_epr_events"
            for item in task_estimates.values()
        )
        f_min = float(
            self.system.sim_params["fidelity_parameters"].get("F_min", 0.0)
        )
        task_fidelities = [
            item["fidelity"] for item in task_estimates.values()
        ]
        task_pass_rate = (
            sum(value >= f_min for value in task_fidelities)
            / len(task_fidelities)
            if task_fidelities else 0.0
        )
        return {
            **physical,
            "fidelity_log10_geometric_mean": (
                physical["fidelity_log10_product"]
                / max(int(physical.get("fidelity_task_count", 1)), 1)
            ),
            "fidelity_threshold_used": f_min,
            "fidelity_task_pass_rate": task_pass_rate,
            "purification_adjusted_fidelity": purification_scenario[
                "fidelity"
            ],
            "purification_adjusted_fidelity_min_task": (
                purification_scenario["fidelity_min_task"]
            ),
            "purification_adjusted_fidelity_geometric_mean": (
                purification_scenario["fidelity_geometric_mean"]
            ),
            "purification_adjusted_fidelity_log10_product": (
                purification_scenario["fidelity_log10_product"]
            ),
            "epr_purification_mode": plan.get("mode", "none"),
            "epr_purification_rounds": actual_rounds,
            "epr_purification_raw_fidelity": float(
                plan.get("raw_fidelity", min(
                    self.system.sim_params["fidelity_parameters"][
                        "F_EPR_range"
                    ]
                ))
            ),
            "epr_purification_output_fidelity": float(
                plan.get("output_fidelity", plan.get(
                    "raw_fidelity", min(
                        self.system.sim_params["fidelity_parameters"][
                            "F_EPR_range"
                        ]
                    )
                ))
            ),
            "epr_remote_output_pairs": remote_pairs,
            "epr_purification_expected_raw_epr_pairs": (
                expected_raw_pairs
            ),
            "epr_purification_expected_attempts": (
                expected_attempts
            ),
            "fidelity_eq5_task_count": eq5_task_count,
            "fidelity_eq7_fallback_task_count": (
                len(task_estimates) - eq5_task_count
            ),
            **mitigation,
        }

    def _compute_utilization(self, sched, assignment, workload):
        """Compute average resource utilization."""
        makespan = max(s["end"] for s in sched.values())
        if makespan <= 0:
            return 0.0

        # QPU utilization uses occupied qubit-time, including every partition.
        qpu_qubit_time = 0.0
        has_quantum_tasks = False
        for tid in workload.quantum_tasks:
            if tid in sched:
                item = sched[tid]
                duration = max(item["end"] - item["start"], 0.0)
                partition_sizes = item.get("partition_sizes")
                if not partition_sizes:
                    qid = assignment.get(tid)
                    partition_sizes = {
                        qid: workload.tasks[tid].get("n_qubits", 0)
                    }
                qpu_qubit_time += duration * sum(partition_sizes.values())
                has_quantum_tasks = True

        total_qpu_capacity = sum(
            qpu["n_qubits"] for qpu in self.system.qpus.values()
        )
        qpu_utils = []
        if has_quantum_tasks and total_qpu_capacity > 0:
            qpu_utils.append(
                qpu_qubit_time / (total_qpu_capacity * makespan)
            )

        # Node utilization
        node_busy_time = defaultdict(float)
        for tid in workload.classical_tasks:
            if tid in sched:
                nid = assignment.get(tid)
                if nid:
                    s = sched[tid]
                    node_busy_time[nid] += s["end"] - s["start"]

        node_utils = []
        for nid in self.system.nodes:
            if node_busy_time[nid] > 0:
                node_utils.append(node_busy_time[nid] / makespan)

        all_utils = qpu_utils + node_utils
        return sum(all_utils) / len(all_utils) if all_utils else 0.0
