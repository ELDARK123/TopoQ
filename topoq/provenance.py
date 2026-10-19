"""Content-addressed provenance for reproducible experiment results."""

import hashlib
import json
import platform
import sys
from pathlib import Path


PROVENANCE_SCHEMA_VERSION = "topoq-result-provenance-v1"
EXPERIMENT_PROTOCOL_VERSION = "topoq-paper-experiment-v1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path):
    """Return the SHA-256 digest of a file's raw bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    """Hash a JSON-compatible value using a deterministic encoding."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_file_manifest(paths, project_root=PROJECT_ROOT):
    """Build an ordered relative-path manifest and aggregate digest."""
    root = Path(project_root).resolve()
    files = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Provenance input is not a file: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Provenance file must be inside project root {root}: {path}"
            ) from exc
        files[relative] = sha256_file(path)

    ordered_files = dict(sorted(files.items()))
    aggregate_payload = [
        {"path": path, "sha256": digest}
        for path, digest in ordered_files.items()
    ]
    return {
        "algorithm": "sha256",
        "file_count": len(ordered_files),
        "aggregate_sha256": canonical_sha256(aggregate_payload),
        "files": ordered_files,
    }


def collect_code_paths(project_root=PROJECT_ROOT):
    """Return Python sources that define the executable experiment code."""
    root = Path(project_root).resolve()
    paths = list((root / "topoq").rglob("*.py"))
    entrypoint = root / "topoq_experiment.py"
    if entrypoint.is_file():
        paths.append(entrypoint)
    return sorted(set(paths), key=lambda path: path.as_posix())


def collect_experiment_input_paths(
        dag_id, system_config, network_topology,
        project_root=PROJECT_ROOT):
    """Return the exact JSON inputs loaded by one registered experiment."""
    dataset = Path(project_root).resolve() / "dataset"
    return [
        dataset / "workload_dags" / f"{dag_id}.json",
        dataset / "system_configs" / f"{system_config.lower()}.json",
        dataset / "network_topologies" / f"{network_topology}.json",
        dataset / "sim_params" / "sim_params.json",
    ]


def collect_dataset_json_paths(project_root=PROJECT_ROOT):
    """Return the complete JSON input snapshot used by the full suite."""
    dataset = Path(project_root).resolve() / "dataset"
    return sorted(dataset.rglob("*.json"), key=lambda path: path.as_posix())


def build_result_provenance(
        protocol, input_paths, project_root=PROJECT_ROOT, code_paths=None,
        input_scope="exact_experiment_inputs", dataset_paths=None):
    """Build protocol, code, input, and environment provenance metadata."""
    protocol_record = {
        "version": EXPERIMENT_PROTOCOL_VERSION,
        **protocol,
    }
    protocol_record["sha256"] = canonical_sha256(protocol_record)
    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
    }
    environment["sha256"] = canonical_sha256(environment)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "protocol": protocol_record,
        "code": build_file_manifest(
            code_paths if code_paths is not None
            else collect_code_paths(project_root),
            project_root=project_root,
        ),
        "inputs": {
            "scope": input_scope,
            **build_file_manifest(input_paths, project_root=project_root),
        },
        "dataset_snapshot": build_file_manifest(
            dataset_paths if dataset_paths is not None
            else collect_dataset_json_paths(project_root),
            project_root=project_root,
        ),
        "environment": environment,
    }


def report_compatibility_key(result):
    """Return the provenance fields that must match within one report."""
    try:
        provenance = result["metadata"]["provenance"]
        return {
            "schema_version": provenance["schema_version"],
            "protocol_version": provenance["protocol"]["version"],
            "code_sha256": provenance["code"]["aggregate_sha256"],
            "dataset_sha256": provenance[
                "dataset_snapshot"
            ]["aggregate_sha256"],
            "environment_sha256": provenance["environment"]["sha256"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Result lacks complete metadata.provenance compatibility fields."
        ) from exc


def validate_report_provenance(results):
    """Reject a report assembled from incompatible result snapshots."""
    if not results:
        raise ValueError("At least one result is required for a report.")
    items = list(results.items()) if isinstance(results, dict) else list(
        enumerate(results)
    )
    reference_name, reference_result = items[0]
    reference = report_compatibility_key(reference_result)
    for name, result in items[1:]:
        current = report_compatibility_key(result)
        if current != reference:
            differences = sorted(
                key for key in reference if reference[key] != current[key]
            )
            raise ValueError(
                "Incompatible result provenance for report entries "
                f"{reference_name!r} and {name!r}: "
                + ", ".join(differences)
            )
    return reference
