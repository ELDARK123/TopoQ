"""
Reporting
=========
Print summary tables matching the paper format and save results to JSON.
"""

from .config import EXPERIMENT_CONFIGS, ABLATION_CONFIGS


def make_json_safe(obj):
    """Recursively convert dict keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def print_summary(results):
    """Print summary tables matching paper format."""
    print("\n" + "=" * 60)
    print("SUMMARY: Experiment Results")
    print("=" * 60)

    # Table III: Makespan
    print("\n--- Table III: Makespan Comparison (ms) ---")
    print(f"{'Workload':<20} {'System':<10} {'Random':>8} {'Greedy':>8} {'Q-Only':>8} {'DQTetris':>8} {'TopoQ':>8}")
    for label in EXPERIMENT_CONFIGS:
        parts = label.rsplit("_", 1)
        wl, sys_name = parts[0], parts[1]
        row = results["makespan"][label]
        print(f"{wl:<20} {sys_name:<10} "
              f"{row['Random']['makespan']:>8.0f} {row['Greedy']['makespan']:>8.0f} "
              f"{row['Q-Only']['makespan']:>8.0f} {row['DQTetris']['makespan']:>8.0f} "
              f"{row['TopoQ']['makespan']:>8.0f}")

    # Table IV: Communication overhead
    print("\n--- Table IV: Communication Overhead Breakdown for VQE-200 (ms) ---")
    print(f"{'Component':<20} {'Random':>8} {'Greedy':>8} {'Q-Only':>8} {'DQTetris':>8} {'TopoQ':>8}")
    comm = results["communication"]
    components = [
        ("Entanglement Wait", "comm_entanglement"),
        ("Quantum-Classical", "comm_qc_transfer"),
        ("Classical Network", "comm_classical_net"),
        ("Total", "comm_total"),
    ]
    for name, key in components:
        vals = [comm[algo][key] for algo in ["Random", "Greedy", "Q-Only", "DQTetris", "TopoQ"]]
        print(f"{name:<20} {vals[0]:>8.0f} {vals[1]:>8.0f} {vals[2]:>8.0f} {vals[3]:>8.0f} {vals[4]:>8.0f}")

    # Table V: Fidelity
    print("\n--- Table V: Fidelity Comparison (%) ---")
    print(f"{'Workload':<15} {'Random':>8} {'Greedy':>8} {'Q-Only':>8} {'DQTetris':>8} {'TopoQ':>8}")
    for label in results["fidelity"]:
        row = results["fidelity"][label]
        print(f"{label:<15} {row['Random']:>8.1f} {row['Greedy']:>8.1f} {row['Q-Only']:>8.1f} {row['DQTetris']:>8.1f} {row['TopoQ']:>8.1f}")

    # Table VI: Ablation
    print("\n--- Table VI: Ablation Study for VQE-200 ---")
    print(f"{'Config':<25} {'Makespan':>10} {'Comm':>10} {'Fidelity':>10}")
    for ablation_name in ABLATION_CONFIGS:
        row = results["ablation"][ablation_name]
        print(f"{ablation_name:<25} {row['makespan']:>10.0f} {row['comm_total']:>10.0f} {row['fidelity']*100:>10.1f}")

    # Scalability
    print("\n--- Scalability Efficiency (%) ---")
    print(f"{'QPU count':<12} {'Greedy':>10} {'DQTetris':>10} {'TopoQ':>10}")
    eff = results["scalability_efficiency"]
    for n in sorted(eff["TopoQ"].keys(), key=int):
        print(f"{n:<12} {eff['Greedy'][n]:>10.1f} {eff['DQTetris'][n]:>10.1f} {eff['TopoQ'][n]:>10.1f}")
