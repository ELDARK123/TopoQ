"""Run TopoQ schedulers on public nonlinear hybrid DAGs.

Public inputs remain outside the legacy dataset tree. The TopoQ package sees
them through a temporary data-source context, so original paper experiments
continue to load their bundled inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path


TOPOQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOPOQ_ROOT.parent
PUBLIC_ROOT = PROJECT_ROOT / "public_benchmark_experiments"
if str(TOPOQ_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPOQ_ROOT))

from topoq.config import (  # noqa: E402
    DEFAULT_ALGORITHMS,
    DEFAULT_PARAMS,
    ENTANGLEMENT_RESERVATION_MODES,
    EPR_ADAPTIVE_HIGH_GATE_MODES,
    IBM_CX_DIRECTION_POLICIES,
    IBM_MISSING_CALIBRATION_POLICIES,
    QPU_CALIBRATION_MODES,
    RANDOM_SEED_BASE,
    dataset_source_context,
    load_dag,
)
from topoq.reporting import make_json_safe  # noqa: E402
from topoq.runner import run_single_experiment  # noqa: E402
from topoq.provenance import build_result_provenance  # noqa: E402


DAG_DIR = PUBLIC_ROOT / "normalized" / "hybrid_workload_dags"
CIRCUIT_DIR = PUBLIC_ROOT / "normalized" / "quantum_circuits"
SYSTEM_DATA_DIR = PUBLIC_ROOT / "raw" / "current_topoq_support_data"
RESULT_DIR = TOPOQ_ROOT / "experiment_results" / "public_scheduler_runs"
REPORT_DIR = PUBLIC_ROOT / "reports"

PUBLIC_EXPERIMENTS = {
    "public_vqe_50q_10iter": {
        "label": "VQE-50",
        "system_config": "small",
        "network_topology": "fattree_k4",
    },
    "public_vqe_100q_10iter": {
        "label": "VQE-100",
        "system_config": "medium",
        "network_topology": "dragonfly_4x4x2",
    },
    "public_vqe_200q_10iter": {
        "label": "VQE-200",
        "system_config": "medium",
        "network_topology": "dragonfly_4x4x2",
    },
    "public_qaoa_200q_p1_10iter": {
        "label": "QAOA-200",
        "system_config": "large",
        "network_topology": "dragonflyplus_8x8x4",
    },
    "public_qml_150q_10iter": {
        "label": "QML-150",
        "system_config": "large",
        "network_topology": "dragonflyplus_8x8x4",
    },
}

RUN_PROFILES = {
    "smoke": {
        "repetitions": 1,
        "description": "End-to-end interface validation with bounded TopoQ search",
        "params": {
            "entanglement_simulation_mode": "deterministic",
            "SA_max_non_improve": 20,
            "max_iterations": 1,
        },
    },
    "paper_smoke": {
        "repetitions": 1,
        "description": (
            "Bounded validation of the revised-paper execution-time "
            "equations before a formal repeated run"
        ),
        "params": {
            "entanglement_simulation_mode": "deterministic",
            "execution_time_mode": "paper_equation",
            "quantum_shots": 1000,
            "classical_compute_efficiency": 1.0,
            "qpu_calibration_mode": "ibm_physical_qubit",
            "ibm_cx_direction_policy": "conservative_max",
            "ibm_missing_calibration_policy": "error",
            "SA_max_non_improve": 20,
            "max_iterations": 1,
        },
    },
    "paper": {
        "repetitions": 3,
        "description": (
            "Paper-style repeated experiment with revised-paper "
            "resource-aware execution-time equations"
        ),
        "params": {
            "entanglement_simulation_mode": "poisson",
            "execution_time_mode": "paper_equation",
            "quantum_shots": 1000,
            "classical_compute_efficiency": 1.0,
            "qpu_calibration_mode": "ibm_physical_qubit",
            "ibm_cx_direction_policy": "conservative_max",
            "ibm_missing_calibration_policy": "error",
            "SA_max_non_improve": DEFAULT_PARAMS["SA_max_non_improve"],
            "max_iterations": DEFAULT_PARAMS["max_iterations"],
        },
    },
    "large": {
        "repetitions": 30,
        "description": "Large-sample statistics with full TopoQ search",
        "params": {
            "entanglement_simulation_mode": "poisson",
            "SA_max_non_improve": DEFAULT_PARAMS["SA_max_non_improve"],
            "max_iterations": DEFAULT_PARAMS["max_iterations"],
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_manifest(paths: list[Path]) -> dict:
    files = {}
    for path in sorted({Path(item).resolve() for item in paths}):
        files[relative(path)] = sha256_file(path)
    payload = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "file_count": len(files),
        "aggregate_sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def metric(metrics: dict, *names: str):
    for name in names:
        if name in metrics and isinstance(metrics[name], (int, float)):
            value = float(metrics[name])
            if math.isfinite(value):
                return value
    return None


def summarize_algorithm(metrics: dict) -> dict:
    return {
        "makespan_ms": metric(metrics, "makespan"),
        "fidelity": metric(metrics, "fidelity"),
        "quantum_communication_ms": metric(
            metrics,
            "comm_entanglement",
            "quantum_communication_time",
            "quantum_comm_time",
            "epr_generation_time",
        ),
        "classical_communication_ms": metric(
            metrics,
            "comm_classical_net",
            "classical_communication_time",
            "classical_comm_time",
            "communication_time",
        ),
        "quantum_classical_transfer_ms": metric(
            metrics,
            "comm_qc_transfer",
        ),
        "total_communication_ms": metric(
            metrics,
            "comm_total",
        ),
        "runtime_seconds": metric(metrics, "runtime_seconds"),
        "entanglement_simulation_mode": metrics.get(
            "entanglement_simulation_mode"
        ),
        "communication_overlap_mode": metrics.get(
            "communication_overlap_mode"
        ),
        "execution_time_mode": metrics.get("execution_time_mode"),
        "quantum_shots": metric(metrics, "quantum_shots"),
        "classical_compute_efficiency": metric(
            metrics, "classical_compute_efficiency"
        ),
        "entanglement_reservation_mode": metrics.get(
            "entanglement_reservation_mode"
        ),
        "epr_proactive_reservation_batches": metric(
            metrics, "epr_proactive_reservation_batches"
        ),
        "epr_on_demand_reservation_batches": metric(
            metrics, "epr_on_demand_reservation_batches"
        ),
        "epr_just_in_time_reservation_batches": metric(
            metrics, "epr_just_in_time_reservation_batches"
        ),
        "epr_mean_hold_ms": metric(metrics, "epr_mean_hold_ms"),
        "epr_max_hold_ms": metric(metrics, "epr_max_hold_ms"),
        "swap_gates_total": metric(metrics, "swap_gates_total"),
        "effective_two_qubit_gates_max": metric(
            metrics, "effective_two_qubit_gates_max"
        ),
        "baseline_task_assignment_projected": metrics.get(
            "baseline_task_assignment_projected"
        ),
        "baseline_quantum_tasks_moved": metrics.get(
            "baseline_quantum_tasks_moved"
        ),
    }


def audit_public_dag(dag: dict) -> dict:
    task_types = {task["id"]: task["type"] for task in dag["tasks"]}
    cq_edges = [
        edge for edge in dag["edges"]
        if task_types[edge["source"]] == "classical"
        and task_types[edge["target"]] == "quantum"
    ]
    qc_edges = [
        edge for edge in dag["edges"]
        if task_types[edge["source"]] == "quantum"
        and task_types[edge["target"]] == "classical"
    ]
    return {
        "id": dag["id"],
        "n_tasks": len(dag["tasks"]),
        "n_edges": len(dag["edges"]),
        "n_quantum_tasks": dag.get("n_quantum_tasks"),
        "n_classical_tasks": dag.get("n_classical_tasks"),
        "classical_to_quantum_edges": len(cq_edges),
        "quantum_to_classical_edges": len(qc_edges),
        "iteration_feedback_edges": dag.get("n_iteration_feedback_edges"),
        "input_is_hybrid": bool(cq_edges and qc_edges),
    }


def aggregate_samples(samples: list[dict]) -> dict:
    if len(samples) == 1:
        return summarize_algorithm(samples[0])
    summaries = [summarize_algorithm(sample) for sample in samples]
    aggregate = {}
    keys = set().union(*(summary.keys() for summary in summaries))
    for key in sorted(keys):
        values = [summary.get(key) for summary in summaries]
        numeric = [
            value for value in values
            if isinstance(value, (int, float)) and math.isfinite(value)
        ]
        if numeric:
            mean = sum(numeric) / len(numeric)
            aggregate[key] = mean
            aggregate[f"{key}_std"] = (
                sum((value - mean) ** 2 for value in numeric)
                / len(numeric)
            ) ** 0.5
        elif values:
            aggregate[key] = values[0]
    return aggregate


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_summary(results: dict) -> dict:
    return {
        "run_id": results["run_id"],
        "started_at": results["started_at"],
        "profile": results["profile"],
        "repetitions": results["repetitions"],
        "random_seed_base": results["random_seed_base"],
        "algorithms": results["algorithms"],
        "parameters": results["parameters"],
        "workloads": {
            dag_id: {
                "label": record["label"],
                "dag_sha256": record["dag_sha256"],
                "dag_audit": record["dag_audit"],
                "algorithm_summaries": {
                    algorithm: {
                        "status": data["status"],
                        **(
                            data["summary"]
                            if data["status"] == "completed"
                            else {"error": data["error"]}
                        ),
                    }
                    for algorithm, data in record["results"].items()
                },
            }
            for dag_id, record in results["workloads"].items()
        },
        "input_manifest": results["input_manifest"],
        "provenance": results["provenance"],
    }


def run_public_suite(
    selected_workloads,
    selected_algorithms,
    repetitions,
    seed_base,
    profile,
    param_overrides=None,
    diagnostics=False,
):
    profile_config = RUN_PROFILES[profile]
    if repetitions is None:
        repetitions = profile_config["repetitions"]
    params = dict(DEFAULT_PARAMS)
    params.update(profile_config["params"])
    params.update({
        "fidelity_constraint_mode": "report_only",
        "fidelity_threshold": None,
    })
    params.update(param_overrides or {})

    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("%Y%m%d-%H%M%S-%f")
    output_dir = RESULT_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "checkpoint_results.json"

    input_paths = [
        PUBLIC_ROOT / "ARCHIVE_MANIFEST.json",
        PUBLIC_ROOT / "DATASET_MANIFEST.json",
        REPORT_DIR / "public_workload_audit.json",
    ]
    input_paths.extend(sorted(DAG_DIR.glob("public_*.json")))
    input_paths.extend(sorted(CIRCUIT_DIR.rglob("*.json")))
    input_paths.extend(sorted(SYSTEM_DATA_DIR.rglob("*.json")))
    code_paths = sorted((TOPOQ_ROOT / "topoq").rglob("*.py"))
    code_paths.append(Path(__file__).resolve())
    provenance = build_result_provenance(
        protocol={
            "kind": "public_scheduler_suite",
            "profile": profile,
            "workloads": list(selected_workloads),
            "algorithms": list(selected_algorithms),
            "repetitions": repetitions,
            "random_seed_base": seed_base,
            "random_seeds": [
                seed_base + index for index in range(repetitions)
            ],
            "effective_parameters": params,
        },
        input_paths=input_paths,
        project_root=PROJECT_ROOT,
        code_paths=code_paths,
        dataset_paths=input_paths,
        input_scope="public_benchmark_snapshot",
    )

    results = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "runner": relative(Path(__file__)),
        "topoq_code_root": relative(TOPOQ_ROOT),
        "profile": profile,
        "profile_description": profile_config["description"],
        "repetitions": repetitions,
        "random_seed_base": seed_base,
        "algorithms": list(selected_algorithms),
        "data_roots": {
            "dag_dir": relative(DAG_DIR),
            "circuit_dir": relative(CIRCUIT_DIR),
            "system_support_dir": relative(SYSTEM_DATA_DIR),
        },
        "parameters": params,
        "input_manifest": file_manifest(input_paths),
        "provenance": provenance,
        "workloads": {},
    }
    write_json(checkpoint_path, results)

    with dataset_source_context(
        dag_dir=DAG_DIR,
        circuit_dirs=[CIRCUIT_DIR],
        system_data_root=SYSTEM_DATA_DIR,
    ):
        for dag_id in selected_workloads:
            spec = PUBLIC_EXPERIMENTS[dag_id]
            dag = load_dag(dag_id)
            workload_record = {
                "label": spec["label"],
                "dag_id": dag_id,
                "system_config": spec["system_config"],
                "network_topology": spec["network_topology"],
                "dag_sha256": sha256_file(DAG_DIR / f"{dag_id}.json"),
                "dag_audit": audit_public_dag(dag),
                "results": {},
            }
            results["workloads"][dag_id] = workload_record
            write_json(checkpoint_path, results)
            for algorithm in selected_algorithms:
                print(
                    f"[START] {dag_id} / {algorithm} / "
                    f"{repetitions} repetition(s)",
                    flush=True,
                )
                samples = []
                workload_record["results"][algorithm] = {
                    "status": "running",
                    "samples": samples,
                }
                write_json(checkpoint_path, results)
                try:
                    for index in range(repetitions):
                        seed = seed_base + index
                        start = time.perf_counter()
                        metrics = dict(run_single_experiment(
                            dag_id,
                            spec["system_config"],
                            spec["network_topology"],
                            algorithm,
                            params=params,
                            ablation=None,
                            seed=seed,
                            n_qpu_limit=None,
                            record_diagnostics=diagnostics,
                        ))
                        if diagnostics:
                            diagnostics_dir = output_dir / "diagnostics"
                            diagnostics_dir.mkdir(exist_ok=True)
                            diagnostics_payload = {
                                key: metrics.pop(key)
                                for key in (
                                    "task_diagnostics",
                                    "critical_path",
                                    "diagnostics_summary",
                                )
                                if key in metrics
                            }
                            diagnostics_path = diagnostics_dir / (
                                f"{dag_id}__{algorithm}__seed{seed}.json"
                            )
                            write_json(diagnostics_path, diagnostics_payload)
                            metrics["diagnostics_path"] = relative(
                                diagnostics_path
                            )
                        metrics["runtime_seconds"] = (
                            time.perf_counter() - start
                        )
                        metrics["seed"] = seed
                        samples.append(metrics)
                        workload_record["results"][algorithm] = {
                            "status": "running",
                            "completed_repetitions": len(samples),
                            "samples": samples,
                        }
                        write_json(checkpoint_path, results)
                    workload_record["results"][algorithm] = {
                        "status": "completed",
                        "summary": aggregate_samples(samples),
                        "samples": samples,
                    }
                    elapsed = sum(
                        sample["runtime_seconds"] for sample in samples
                    )
                    print(
                        f"[DONE] {dag_id} / {algorithm} / "
                        f"{elapsed:.3f} s",
                        flush=True,
                    )
                except Exception as exc:
                    workload_record["results"][algorithm] = {
                        "status": "failed",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "samples": samples,
                    }
                    print(
                        f"[FAILED] {dag_id} / {algorithm}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                write_json(checkpoint_path, results)

    summary = build_summary(results)

    detailed_path = output_dir / "public_scheduler_results.json"
    summary_path = output_dir / "public_scheduler_summary.json"
    write_json(detailed_path, results)
    write_json(summary_path, summary)
    latest_path = RESULT_DIR / "latest_public_scheduler_summary.json"
    write_json(latest_path, summary)
    return summary_path, detailed_path, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILES),
        default="smoke",
    )
    parser.add_argument(
        "--workload",
        action="append",
        choices=sorted(PUBLIC_EXPERIMENTS),
        help="run only selected workload id; repeat to select several",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=DEFAULT_ALGORITHMS,
        help="run only selected algorithm; repeat to select several",
    )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed-base", type=int, default=RANDOM_SEED_BASE)
    parser.add_argument(
        "--reservation-mode",
        choices=ENTANGLEMENT_RESERVATION_MODES,
        help=(
            "EPR reservation timing: proactive, on_demand, "
            "just_in_time, or gate-load adaptive"
        ),
    )
    parser.add_argument("--proactive-lead-fraction", type=float)
    parser.add_argument("--jit-memory-fraction", type=float)
    parser.add_argument("--adaptive-2q-threshold", type=int)
    parser.add_argument(
        "--adaptive-high-gate-mode",
        choices=EPR_ADAPTIVE_HIGH_GATE_MODES,
    )
    parser.add_argument(
        "--qpu-calibration-mode", choices=QPU_CALIBRATION_MODES
    )
    parser.add_argument(
        "--ibm-cx-direction-policy", choices=IBM_CX_DIRECTION_POLICIES
    )
    parser.add_argument(
        "--ibm-missing-calibration-policy",
        choices=IBM_MISSING_CALIBRATION_POLICIES,
    )
    parser.add_argument(
        "--dqtetris-external",
        help=(
            "JSON produced by analysis/convert_dqtetris_outputs.py; "
            "DQTetris uses its own final_path mapping and communication count"
        ),
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Persist per-task start/end/wait/EPR/comm/fidelity records and "
            "critical-path decomposition for each run"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.repetitions is not None and args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    workloads = args.workload or list(PUBLIC_EXPERIMENTS)
    algorithms = args.algorithm or list(DEFAULT_ALGORITHMS)
    param_overrides = {
        key: value for key, value in {
            "entanglement_reservation_mode": args.reservation_mode,
            "epr_proactive_lead_fraction": args.proactive_lead_fraction,
            "epr_jit_memory_fraction": args.jit_memory_fraction,
            "epr_adaptive_2q_gate_threshold": args.adaptive_2q_threshold,
            "epr_adaptive_high_gate_mode": args.adaptive_high_gate_mode,
            "qpu_calibration_mode": args.qpu_calibration_mode,
            "ibm_cx_direction_policy": args.ibm_cx_direction_policy,
            "ibm_missing_calibration_policy": (
                args.ibm_missing_calibration_policy
            ),
            "dqtetris_external_outputs": args.dqtetris_external,
        }.items() if value is not None
    }
    summary_path, detailed_path, summary = run_public_suite(
        workloads,
        algorithms,
        args.repetitions,
        args.seed_base,
        args.profile,
        param_overrides=param_overrides,
        diagnostics=args.diagnostics,
    )
    print(json.dumps({
        "summary": relative(summary_path),
        "details": relative(detailed_path),
        "workloads": len(summary["workloads"]),
        "algorithms": algorithms,
        "profile": args.profile,
        "repetitions": summary["repetitions"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
