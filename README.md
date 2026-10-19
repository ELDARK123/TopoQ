# TopoQ

TopoQ is a topology-aware scheduling framework for hybrid
quantum-classical computing systems. It models a hybrid workflow as a DAG of
quantum and classical tasks, partitions and maps quantum circuits onto
distributed QPUs, provisions EPR pairs for remote gates, places classical
tasks on the classical network, and repairs cross-domain timing conflicts
with a joint objective.

This repository contains the current TopoQ implementation and simulator used
for the revision experiments. It is a research prototype; APIs may change.

## Features

- Hybrid quantum-classical DAG model with explicit resources, payloads,
  communication modes, and feedback edges.
- TopoQ five-phase scheduler with topology-aware preprocessing,
  entanglement-aware quantum scheduling, locality-aware classical scheduling,
  and feedback-driven joint refinement.
- Baselines: Random, Greedy, Q-Only, and a DQTetris-style partition baseline.
- Discrete-event timing and fidelity simulator with an event-level fidelity
  model (preparation, local gates, EPR generation/holding, teleportation, and
  data-qubit idle time).
- Per-task diagnostics and critical-path decomposition.
- Reproducible experiment runner with provenance manifests.

## Requirements

- Python 3.10 or newer
- `networkx`

The core package has no other mandatory third-party dependencies. Optional
public-benchmark and DQTetris comparison scripts require additional data and
tools that are not bundled here.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

For development, install the package in editable mode:

```bash
pip install -e .
```

## Quick start

Run a short synthetic TopoQ experiment from the repository root:

```bash
python -m topoq \
  --profile quick \
  --experiment VQE-50_Small \
  --algorithm TopoQ \
  --repetitions 1 \
  --output results/quick_vqe50.json \
  --overwrite
```

List the available experiments and algorithms:

```bash
python -m topoq --list-experiments
```

Run the unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Repository layout

```text
topoq/                         Core scheduler, simulator, baselines, metrics
scripts/
  run_public_scheduler_experiments.py
                                Public-benchmark runner (external data required)
tests/                         Regression and unit tests
dataset/                       Bundled synthetic DAG/circuit/system data
generate_dataset.py            Synthetic dataset generator
topoq_experiment.py            CLI wrapper for the experiment suite
```

## Reproducibility

The simulator records the effective parameters, input hashes, code hashes,
and random seeds in its result provenance. Public-benchmark runs accept
`--diagnostics` to persist per-task start/end/wait/EPR/communication/fidelity
records and a critical-path decomposition alongside the aggregate metrics.

The bundled `dataset/` is synthetic and self-contained. The public MQT Bench
inputs, DQTetris source, and DQTetris external-output adapter are kept outside
this repository because they have separate provenance and third-party
licenses.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

Third-party data and tools used in the paper (including MQT Bench and
DQTetris) retain their own licenses and are not redistributed here.
