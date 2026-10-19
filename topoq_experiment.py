#!/usr/bin/env python3
"""
TopoQ Experiment Framework — Entry Point
========================================
This file is a thin wrapper. The implementation has been split into the
``topoq`` package for readability and maintainability.

Package structure:
    topoq/
    ├── config.py            Algorithm parameters, data loading, experiment configs
    ├── models.py            HybridSystem, Workload
    ├── topoq_scheduler.py   TopoQ 5-phase algorithm
    ├── baselines.py         Random / Greedy / Q-Only / DQTetris
    ├── simulator.py         Discrete-event simulator & metrics
    ├── runner.py            Single & repeated experiment runners
    ├── experiments.py       Paper table/figure generation
    ├── reporting.py         Summary tables & JSON export
    └── main.py              Top-level orchestration

Usage:
    python topoq_experiment.py      # run all experiments
    python -m topoq                 # same, via package
"""

from topoq.main import main

if __name__ == "__main__":
    main()
