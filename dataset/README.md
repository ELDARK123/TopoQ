# TopoQ Experimental Dataset

## License note

The repository's MIT license covers the TopoQ source code. The bundled JSON
data is provided for reproducibility; IBM/Qiskit-derived calibration
snapshots and MQT Bench circuit specifications retain their original terms
and references listed at the end of this file.

## Overview

This dataset supports the experimental evaluation of **TopoQ: Topology-Aware Scheduling for Hybrid Quantum-Classical Computing Systems**, submitted to IEEE INFOCOM.

The dataset is constructed based on the system model and problem formulation defined in **Section 3** of the paper, covering:

1. **IBM quantum chip topologies** — Real coupling maps from IBM Quantum devices
2. **Classical network topologies** — Fat-Tree, Dragonfly, Dragonfly+
3. **Hybrid system configurations** — Small / Medium / Large (matching Table I)
4. **Quantum circuit benchmarks** — VQE, QAOA, QML, QFT, GHZ (MQT Bench compatible)
5. **Hybrid workload DAGs** — 48 DAGs with quantum-classical feedback edges
6. **Simulation parameters** — Section 3 parameters (λ, η, generation capacity g, optional memory capacity M, B, L, ρ, R, F, ε)

## Directory Structure

```
dataset/
├── index.json                      # Master index
├── ibm_chips/                      # IBM quantum chip topologies
│   ├── index.json
│   ├── ibm_guadalupe.json          # 16 qubits, heavy-hex
│   ├── ibm_sydney.json             # 27 qubits, heavy-hex
│   ├── ibm_tokyo.json              # 20 qubits, grid+diagonal
│   ├── ibm_rochester.json          # 53 qubits, heavy-hex
│   ├── ibm_montreal.json           # 27 qubits, heavy-hex
│   ├── ibm_washington.json         # 127 qubits, heavy-hex
│   └── ibm_brooklyn.json           # 65 qubits, heavy-hex
├── network_topologies/             # Classical network topologies
│   ├── index.json
│   ├── fattree_k4.json             # Fat-Tree k=4 (8 hosts)
│   ├── fattree_k8.json             # Fat-Tree k=8 (32 hosts)
│   ├── dragonfly_4x4x2.json        # Dragonfly (32 hosts)
│   └── dragonflyplus_8x8x4.json    # Dragonfly+ (256 hosts)
├── system_configs/                 # Hybrid system configurations
│   ├── index.json
│   ├── small.json                  # 4 QPUs × 16q, 8 nodes, Fat-Tree
│   ├── medium.json                 # 8 QPUs × 27q, 16 nodes, Dragonfly
│   └── large.json                  # 16 QPUs × 127q, 32 nodes, Dragonfly+
├── quantum_circuits/               # Quantum circuit benchmarks
│   ├── index.json
│   ├── vqe_50q.json ... vqe_200q.json    # 6 VQE circuits
│   ├── qaoa_100q.json ... qaoa_500q.json # 5 QAOA circuits
│   ├── qml_80q.json ... qml_300q.json    # 5 QML circuits
│   ├── qft_50q.json ... qft_200q.json    # 3 QFT circuits
│   └── ghz_50q.json ... ghz_200q.json    # 3 GHZ circuits
├── workload_dags/                  # Hybrid workload DAGs
│   ├── index.json
│   ├── vqe_dag_*.json              # 18 VQE DAGs
│   ├── qaoa_dag_*.json             # 15 QAOA DAGs
│   └── qml_dag_*.json              # 15 QML DAGs
└── sim_params/
    └── sim_params.json             # All simulation parameters
```

## Data Sources

### IBM Quantum Chip Topologies

Extracted from **Qiskit's fake provider** module (`qiskit.providers.fake_provider`), which contains system snapshots of real IBM Quantum devices including coupling maps, qubit properties (T1, T2, frequencies), and gate properties (error rates, durations).

| Chip | Qubits | Topology | Used For |
|------|--------|----------|----------|
| IBM Guadalupe | 16 | Heavy-hex | Small config QPU template |
| IBM Sydney | 27 | Heavy-hex | Medium config QPU template |
| IBM Tokyo | 20 | Grid+diagonal | Small config alternative |
| IBM Rochester | 53 | Heavy-hex | Large config QPU template |
| IBM Montreal | 27 | Heavy-hex | Medium config alternative |
| IBM Washington | 127 | Heavy-hex | Large config (paper's 127-qubit QPUs) |
| IBM Brooklyn | 65 | Heavy-hex | Large config alternative |

The executable dataset uses the physical IBM chip capacities represented by
the included coupling maps: Small uses 16-qubit Guadalupe QPUs, Medium uses
27-qubit Sydney QPUs, and Large uses 127-qubit Washington QPUs.  This differs
from the paper's abstract 50/100/127 qubits-per-QPU configuration so that
logical-to-physical mapping can be checked against complete coupling graphs.

Synthetic circuit JSON files contain an explicit `interactions` list.  VQE
and QML use deterministic ring entanglers, GHZ uses a chain, QFT uses the
complete controlled-rotation graph, and QAOA uses a circuit-ID-seeded MaxCut
graph.  Interaction weights always sum to the declared two-qubit gate count.

### Classical Network Topologies

Generated using standard HPC network topology constructions:

| Topology | Nodes | Hosts | Diameter | Used For |
|----------|-------|-------|----------|----------|
| Fat-Tree k=4 | 40 | 8 | 6 | Small config |
| Fat-Tree k=8 | 216 | 32 | 6 | Medium config (alternative) |
| Dragonfly 4×4×2 | 48 | 32 | 3 | Medium config |
| Dragonfly+ 8×8×4 | 320 | 256 | 4 | Large config |

### Quantum Circuit Benchmarks

Circuit parameters are based on **MQT Bench** (Munich Quantum Toolkit Benchmark Library, https://www.cda.cit.tum.de/mqtbench/) circuit specifications. The MQT Bench library provides VQE, QAOA, and QNN circuits at various qubit counts.

To use real MQT Bench QASM files:
```bash
# Install MQT Bench
pip install mqt.bench

# Or download from https://www.mqt-bench.app/
# Then regenerate with:
python generate_dataset.py --output-dir ./dataset --mqt-bench /path/to/mqt-bench/qasm
```

### Workload DAG Generation

DAGs are generated using methods inspired by classical workflow scheduling benchmarks:

- **STG (Standard Task Graph Set)**: Random DAG generation with configurable parameters
- **DAGGEN**: Layer-based DAG generation for HPC workflows
- **WfCommons**: Workflow-based DAG generation for scientific applications

Each DAG follows the formal model **W = (T, E_D, ≺, μ)** from Section 3.3:
- T = T_Q ∪ T_C (quantum + classical tasks)
- E_D includes special E_D^QC feedback edges (quantum↔classical)
- μ assigns data volume to each dependency edge

## Mapping to Paper

| Paper Component | Dataset Component |
|-----------------|-------------------|
| §3.1 System H=(Q,C,N) | `system_configs/*.json` |
| §3.2 Topology G=(G_Q,G_C,G_I) | `ibm_chips/*.json` + `network_topologies/*.json` |
| §3.3 Workload W=(T,E_D,≺,μ) | `workload_dags/*.json` |
| §3.4 Communication Model | `sim_params/sim_params.json` (quantum_link_parameters, classical_link_parameters, interface_link_parameters) |
| §3.5 Fidelity Model | `sim_params/sim_params.json` (fidelity_parameters) |
| §3.6 Problem Formulation | `sim_params/sim_params.json` (algorithm_parameters) |
| Table I (System Configs) | `system_configs/{small,medium,large}.json` |
| §6.1 Workloads (VQE/QAOA/QML) | `quantum_circuits/*.json` + `workload_dags/*.json` |
| §6.1 Baselines | `sim_params/sim_params.json` (baselines) |

## Dataset Statistics

- **IBM chips**: 7 (16–127 qubits)
- **Network topologies**: 4 (8–256 hosts)
- **System configs**: 3 (Small/Medium/Large)
- **Quantum circuits**: 22 (50–500 qubits, 5 types)
- **Workload DAGs**: 48 (18 VQE + 15 QAOA + 15 QML)
- **Total tasks across all DAGs**: ~15,000+
- **Total DAG edges**: ~30,000+

## Reproducibility

```bash
# Generate dataset (requires qiskit, networkx, numpy)
python generate_dataset.py --output-dir ./dataset --seed 42

# Regenerate with the project-local IBM historical calibration snapshots
python generate_dataset.py --output-dir ./dataset \
  --ibm-snapshot-dir ./dataset/ibm_calibration_snapshots --seed 42

# With real MQT Bench circuits
python generate_dataset.py --output-dir ./dataset --mqt-bench /path/to/mqt-bench
```

## File Formats

All files are JSON with consistent structure:

**Chip files** (`ibm_chips/*.json`):
```json
{
  "name": "ibm_guadalupe",
  "n_qubits": 16,
  "coupling_map": [[0,1],[1,0],...],        // directed
  "coupling_map_undirected": [[0,1],[1,2],...], // deduplicated
  "basis_gates": ["id","rz","sx","x","cx","reset"],
  "qubit_properties": [{"qubit":0,"t1_us":...,"t2_us":...,"freq_ghz":...,"readout_error":...,"readout_length_ns":...}],
  "gate_properties": [{"name":"cx0_1","gate":"cx","qubits":[0,1],"parameters":{"gate_error":0.0068,"gate_length":334.0},"parameter_units":{"gate_error":"dimensionless","gate_length":"ns"}}],
  "calibration_metadata": {"last_update_date":"...","standardized_units":{"t1_us":"us","t2_us":"us","freq_ghz":"GHz","readout_length_ns":"ns","gate_length":"ns"}},
  "stats": {"avg_cx_error":0.0108, "avg_cx_duration_ns":393.8, "avg_t1_us":..., "avg_t2_us":..., "avg_degree":2.0}
}
```

The raw IBM snapshot files preserve their original explicit units. The generator
normalizes relaxation and dephasing times to microseconds, frequencies to GHz,
and gate/readout durations to nanoseconds before writing `ibm_chips/*.json`.

**DAG files** (`workload_dags/*.json`):
```json
{
  "id": "vqe_dag_100q_10iter",
  "type": "VQE",
  "n_tasks": 230,
  "n_quantum_tasks": 210,
  "n_classical_tasks": 20,
  "n_edges": 419,
  "n_qc_feedback_edges": 19,
  "tasks": [{"id":"t0","type":"quantum","name":"vqe_ansatz_iter0","n_qubits":100,...}],
  "edges": [{"source":"t0","target":"t1","type":"quantum-quantum","data_volume_bits":100}]
}
```

## References

- MQT Bench: Quetschlich et al., "MQT Bench: Benchmarking Software and Design Automation Tools for Quantum Computing," Quantum Software Lab, TUM, 2023. https://www.cda.cit.tum.de/mqtbench/
- IBM Quantum: https://quantum.ibm.com/
- STG: Tobita & Kasahara, "A Standard Task Graph Set for Fair Evaluations of Multiprocessor Scheduling Algorithms," Journal of Scheduling, 2002.
- DAGGEN: https://github.com/frs69wq/daggen
## Quantum-link EPR capacity fields

`max_concurrent_epr` is the required positive generation-service
parallelism. It limits EPR batches over the half-open interval
`[generation_start, ready)`.

`memory_capacity_epr` is an optional positive integer calibration for
successfully generated EPR pairs stored over `[ready, consume_or_discard)`.
It is not inferred from `max_concurrent_epr`. The current Small, Medium, and
Large configurations omit this field; experiments must report the missing
calibration or request strict rejection through
`--epr-memory-capacity-mode strict`.

## Quantum-link timing fields

`entanglement_rate_hz` is interpreted by the runtime
`quantum_link_latency_mode`:

- `delivered_pair_rate`: the rate already describes end-to-end EPR pairs ready
  for consumption. No fixed BSM or propagation stage is added.
- `physical_stage`: the rate describes stochastic generation before explicit
  fixed stages.

Physical-stage timing requires either per-channel fields
`bsm_latency_ms`, `switch_latency_ms`, `signal_speed_m_s`, and
`signal_traversals`, or their `quantum_link_*` runtime parameters. Latencies
must be non-negative and finite; speed and the integer traversal count must be
positive. Current Small, Medium, and Large configurations omit these stage
calibrations and therefore use `delivered_pair_rate` by default.
