"""
TopoQ Experiment Framework
==========================
Implements the full experimental pipeline described in the paper:
  - TopoQ algorithm (5 phases with iterative co-optimization loop)
  - 4 baselines: Random, Greedy, Q-Only, DQTetris
  - Discrete-event simulator for hybrid quantum-classical systems
  - Metrics: makespan, communication overhead, quantum fidelity, resource utilization
  - Ablation study, scalability analysis, parameter sensitivity

All experimental data is computed from the generated dataset -- no hardcoded numbers.

Modules
-------
config          Algorithm parameters, dataset paths, data loaders, experiment configs
models          HybridSystem and Workload data models
topoq_scheduler TopoQ 5-phase scheduling algorithm
baselines       Random / Greedy / Q-Only / DQTetris baseline schedulers
simulator       Discrete-event simulator and metric computation
runner          Single-experiment and repeated-experiment runners
experiments     Paper table/figure generation functions
reporting       Summary table printing and JSON export
main            Top-level entry point

Usage
-----
    python topoq_experiment.py          # run all experiments
    python -m topoq                     # same, via package
    from topoq.runner import run_single_experiment
"""

from .config import (
    DEFAULT_PARAMS,
    ENTANGLEMENT_SIMULATION_MODES,
    ENTANGLEMENT_MEMORY_CAPACITY_MODES,
    COHERENCE_EVENT_MODES,
    COHERENCE_TASK_WINDOW_MODES,
    FIDELITY_CONSTRAINT_MODES,
    PHYSICAL_ROUTING_MODES,
    QUANTUM_LINK_LATENCY_MODES,
    N_REPETITIONS,
    RANDOM_SEED_BASE,
    EXPERIMENT_CONFIGS,
    SCALABILITY_CONFIGS,
    ABLATION_CONFIGS,
    validate_fidelity_policy,
)
from .models import HybridSystem, Workload
from .topoq_scheduler import TopoQScheduler
from .baselines import (
    RandomScheduler,
    GreedyScheduler,
    QOnlyScheduler,
    DQTetrisScheduler,
)
from .simulator import Simulator
from .runner import run_single_experiment, run_with_repetitions
from .main import main

__all__ = [
    "DEFAULT_PARAMS",
    "ENTANGLEMENT_SIMULATION_MODES",
    "ENTANGLEMENT_MEMORY_CAPACITY_MODES",
    "COHERENCE_EVENT_MODES",
    "COHERENCE_TASK_WINDOW_MODES",
    "FIDELITY_CONSTRAINT_MODES",
    "PHYSICAL_ROUTING_MODES",
    "QUANTUM_LINK_LATENCY_MODES",
    "N_REPETITIONS",
    "RANDOM_SEED_BASE",
    "EXPERIMENT_CONFIGS",
    "SCALABILITY_CONFIGS",
    "ABLATION_CONFIGS",
    "validate_fidelity_policy",
    "HybridSystem",
    "Workload",
    "TopoQScheduler",
    "RandomScheduler",
    "GreedyScheduler",
    "QOnlyScheduler",
    "DQTetrisScheduler",
    "Simulator",
    "run_single_experiment",
    "run_with_repetitions",
    "main",
]
