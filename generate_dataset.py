#!/usr/bin/env python3
"""
TopoQ Dataset Generator
========================
Generates a complete experimental dataset for the TopoQ paper based on
Section 3 (Problem Formulation).

Components:
  1. IBM quantum chip topologies (from Qiskit fake providers)
  2. Classical network topologies (Fat-Tree, Dragonfly, Dragonfly+)
  3. Hybrid system configurations (Small / Medium / Large)
  4. Quantum circuits (from MQT Bench / QASMBench)
  5. Hybrid workload DAGs (VQE / QAOA / QML)
  6. Simulation parameters (entanglement rates, decoherence, fidelity)

Usage:
  python generate_dataset.py --output-dir ./dataset
  python generate_dataset.py --output-dir ./dataset --mqt-bench /path/to/mqt-bench

Requires: qiskit, networkx, numpy
"""

import os
import sys
import json
import random
import argparse
import hashlib
import re
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# ============================================================================
# 1. IBM Quantum Chip Topologies
# ============================================================================

IBM_CHIP_SNAPSHOTS = {
    'ibm_guadalupe': 'guadalupe',
    'ibm_sydney': 'sydney',
    'ibm_tokyo': 'tokyo',
    'ibm_rochester': 'rochester',
    'ibm_montreal': 'montreal',
    'ibm_washington': 'washington',
    'ibm_brooklyn': 'brooklyn',
}

DEFAULT_IBM_SNAPSHOT_DIR = (
    Path(__file__).resolve().parent / 'dataset' / 'ibm_calibration_snapshots'
)

_TIME_UNIT_TO_SECONDS = {
    's': 1.0,
    'ms': 1e-3,
    'us': 1e-6,
    'ns': 1e-9,
    'ps': 1e-12,
}
_FREQUENCY_UNIT_TO_HZ = {
    'hz': 1.0,
    'khz': 1e3,
    'mhz': 1e6,
    'ghz': 1e9,
}


def convert_calibration_value(value: float, source_unit: str,
                              target_unit: str) -> float:
    """Convert an IBM calibration value using its explicit source unit."""
    normalized_source = (source_unit or '').strip().lower().replace('μ', 'u')
    normalized_target = (target_unit or '').strip().lower().replace('μ', 'u')

    for micro_codepoint in (0x00B5, 0x03BC):
        normalized_source = normalized_source.replace(chr(micro_codepoint), 'u')
        normalized_target = normalized_target.replace(chr(micro_codepoint), 'u')

    if normalized_source == normalized_target:
        return float(value)

    if (normalized_source in _TIME_UNIT_TO_SECONDS and
            normalized_target in _TIME_UNIT_TO_SECONDS):
        seconds = float(value) * _TIME_UNIT_TO_SECONDS[normalized_source]
        return seconds / _TIME_UNIT_TO_SECONDS[normalized_target]

    if (normalized_source in _FREQUENCY_UNIT_TO_HZ and
            normalized_target in _FREQUENCY_UNIT_TO_HZ):
        hz = float(value) * _FREQUENCY_UNIT_TO_HZ[normalized_source]
        return hz / _FREQUENCY_UNIT_TO_HZ[normalized_target]

    if not normalized_source and not normalized_target:
        return float(value)

    raise ValueError(
        f"Unsupported calibration conversion: {source_unit!r} -> {target_unit!r}"
    )


def _calibration_parameter(parameters: List[dict], name: str) -> Optional[dict]:
    target = name.lower()
    for parameter in parameters:
        if str(parameter.get('name', '')).lower() == target:
            return parameter
    return None


def _converted_parameter(parameters: List[dict], name: str,
                         target_unit: str) -> Optional[float]:
    parameter = _calibration_parameter(parameters, name)
    if parameter is None or parameter.get('value') is None:
        return None
    return convert_calibration_value(
        parameter['value'], parameter.get('unit', ''), target_unit
    )


def _dimensionless_parameter(parameters: List[dict], name: str) -> Optional[float]:
    parameter = _calibration_parameter(parameters, name)
    if parameter is None or parameter.get('value') is None:
        return None
    unit = str(parameter.get('unit', '')).strip()
    if unit:
        raise ValueError(f"Expected a dimensionless {name}, found unit {unit!r}")
    return float(parameter['value'])


def _ibm_chip_stats(chip_data: dict) -> dict:
    cx_errors = [
        g['parameters']['gate_error'] for g in chip_data['gate_properties']
        if g['gate'] == 'cx' and g['parameters'].get('gate_error') is not None
    ]
    cx_durations = [
        g['parameters']['gate_length'] for g in chip_data['gate_properties']
        if g['gate'] == 'cx' and g['parameters'].get('gate_length') is not None
    ]
    t1_values = [
        q['t1_us'] for q in chip_data['qubit_properties']
        if q.get('t1_us') is not None
    ]
    t2_values = [
        q['t2_us'] for q in chip_data['qubit_properties']
        if q.get('t2_us') is not None
    ]
    readout_errors = [
        q['readout_error'] for q in chip_data['qubit_properties']
        if q.get('readout_error') is not None
    ]
    readout_lengths = [
        q['readout_length_ns'] for q in chip_data['qubit_properties']
        if q.get('readout_length_ns') is not None
    ]

    return {
        'n_edges_undirected': len(chip_data['coupling_map_undirected']),
        'avg_cx_error': round(float(np.mean(cx_errors)), 6) if cx_errors else None,
        'max_cx_error': round(max(cx_errors), 6) if cx_errors else None,
        'min_cx_error': round(min(cx_errors), 6) if cx_errors else None,
        'avg_cx_duration_ns': round(float(np.mean(cx_durations)), 1) if cx_durations else None,
        'max_cx_duration_ns': round(max(cx_durations), 1) if cx_durations else None,
        'avg_t1_us': round(float(np.mean(t1_values)), 2) if t1_values else None,
        'avg_t2_us': round(float(np.mean(t2_values)), 2) if t2_values else None,
        'avg_readout_error': round(float(np.mean(readout_errors)), 6) if readout_errors else None,
        'avg_readout_length_ns': round(float(np.mean(readout_lengths)), 1) if readout_lengths else None,
        'avg_degree': round(
            2 * len(chip_data['coupling_map_undirected']) / chip_data['n_qubits'], 2
        ),
    }


def _validate_ibm_chip_data(chip_data: dict) -> None:
    if len(chip_data['qubit_properties']) != chip_data['n_qubits']:
        raise ValueError(
            f"{chip_data['name']}: qubit property count does not match n_qubits"
        )

    for qprop in chip_data['qubit_properties']:
        qubit = qprop['qubit']
        for field in ('t1_us', 't2_us'):
            value = qprop.get(field)
            if value is not None and value <= 0:
                raise ValueError(f"{chip_data['name']} q{qubit}: invalid {field}={value}")
        frequency = qprop.get('freq_ghz')
        if frequency is not None and not 1.0 <= frequency <= 20.0:
            raise ValueError(
                f"{chip_data['name']} q{qubit}: implausible freq_ghz={frequency}"
            )
        readout_error = qprop.get('readout_error')
        if readout_error is not None and not 0.0 <= readout_error <= 1.0:
            raise ValueError(
                f"{chip_data['name']} q{qubit}: invalid readout_error={readout_error}"
            )
        readout_length = qprop.get('readout_length_ns')
        if readout_length is not None and readout_length <= 0:
            raise ValueError(
                f"{chip_data['name']} q{qubit}: invalid readout_length_ns={readout_length}"
            )

    for gate in chip_data['gate_properties']:
        error = gate['parameters'].get('gate_error')
        if error is not None and not 0.0 <= error <= 1.0:
            raise ValueError(f"{chip_data['name']} {gate['name']}: invalid gate_error={error}")
        duration = gate['parameters'].get('gate_length')
        if duration is not None and duration < 0:
            raise ValueError(
                f"{chip_data['name']} {gate['name']}: invalid gate_length={duration}"
            )
        if gate['gate'] == 'cx' and duration is not None and duration <= 0:
            raise ValueError(
                f"{chip_data['name']} {gate['name']}: CX duration must be positive"
            )


def _write_ibm_chip_dataset(result: Dict[str, dict], output_dir: str) -> None:
    chip_dir = Path(output_dir) / 'ibm_chips'
    chip_dir.mkdir(parents=True, exist_ok=True)
    for name, chip_data in result.items():
        with (chip_dir / f'{name}.json').open('w', encoding='utf-8') as f:
            json.dump(chip_data, f, indent=2)

    index = {
        name: {
            'n_qubits': chip['n_qubits'],
            'topology_type': chip['topology_type'],
            'calibration_last_update': chip.get('calibration_metadata', {}).get(
                'last_update_date'
            ),
            'stats': chip['stats'],
        }
        for name, chip in result.items()
    }
    with (chip_dir / 'index.json').open('w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)


def _extract_ibm_chips_from_snapshots(output_dir: str,
                                      snapshot_dir: Path) -> Dict[str, dict]:
    result = {}
    for name, snapshot_name in IBM_CHIP_SNAPSHOTS.items():
        config_path = snapshot_dir / f'conf_{snapshot_name}.json'
        props_path = snapshot_dir / f'props_{snapshot_name}.json'
        with config_path.open(encoding='utf-8') as f:
            config = json.load(f)
        with props_path.open(encoding='utf-8') as f:
            properties = json.load(f)

        n_qubits = int(config['n_qubits'])
        coupling_map = [list(edge) for edge in config.get('coupling_map', [])]
        chip_data = {
            'name': name,
            'n_qubits': n_qubits,
            'basis_gates': list(config.get('basis_gates', [])),
            'coupling_map': coupling_map,
            'coupling_map_undirected': _dedupe_edges(coupling_map),
            'qubit_properties': [],
            'gate_properties': [],
            'source': 'IBM Quantum historical calibration snapshot (Qiskit fake provider)',
            'topology_type': _classify_topology(name, n_qubits),
            'calibration_metadata': {
                'backend_name': properties.get('backend_name', config.get('backend_name')),
                'backend_version': properties.get(
                    'backend_version', config.get('backend_version')
                ),
                'last_update_date': properties.get('last_update_date'),
                'snapshot_properties_file': props_path.name,
                'snapshot_configuration_file': config_path.name,
                'standardized_units': {
                    't1_us': 'us',
                    't2_us': 'us',
                    'freq_ghz': 'GHz',
                    'readout_error': 'dimensionless',
                    'readout_length_ns': 'ns',
                    'gate_error': 'dimensionless',
                    'gate_length': 'ns',
                },
            },
        }

        for qubit, parameters in enumerate(properties.get('qubits', [])):
            chip_data['qubit_properties'].append({
                'qubit': qubit,
                't1_us': _converted_parameter(parameters, 'T1', 'us'),
                't2_us': _converted_parameter(parameters, 'T2', 'us'),
                'freq_ghz': _converted_parameter(parameters, 'frequency', 'GHz'),
                'readout_error': _dimensionless_parameter(parameters, 'readout_error'),
                'readout_length_ns': _converted_parameter(
                    parameters, 'readout_length', 'ns'
                ),
            })

        for gate in properties.get('gates', []):
            parameters = {}
            parameter_units = {}
            for parameter in gate.get('parameters', []):
                parameter_name = parameter.get('name')
                if parameter_name == 'gate_length':
                    parameters[parameter_name] = convert_calibration_value(
                        parameter['value'], parameter.get('unit', ''), 'ns'
                    )
                    parameter_units[parameter_name] = 'ns'
                elif parameter_name == 'gate_error':
                    parameters[parameter_name] = _dimensionless_parameter(
                        gate.get('parameters', []), 'gate_error'
                    )
                    parameter_units[parameter_name] = 'dimensionless'
                else:
                    parameters[parameter_name] = parameter.get('value')
                    parameter_units[parameter_name] = parameter.get('unit', '')
            chip_data['gate_properties'].append({
                'name': gate.get('name'),
                'gate': gate.get('gate'),
                'qubits': list(gate.get('qubits', [])),
                'parameters': parameters,
                'parameter_units': parameter_units,
            })

        _validate_ibm_chip_data(chip_data)
        chip_data['stats'] = _ibm_chip_stats(chip_data)
        result[name] = chip_data
        print(
            f"  [OK] {name}: {n_qubits} qubits, "
            f"{len(chip_data['coupling_map_undirected'])} edges, "
            f"avg CX error={chip_data['stats'].get('avg_cx_error', 'N/A')}"
        )

    _write_ibm_chip_dataset(result, output_dir)
    return result


def extract_ibm_chips(output_dir: str,
                      snapshot_dir: Optional[str] = None) -> Dict[str, dict]:
    """
    Extract real IBM quantum chip topologies from Qiskit fake providers.
    
    Chips extracted:
      - ibm_guadalupe  (16 qubits, heavy-hex)     → Small config QPU
      - ibm_sydney     (27 qubits, heavy-hex)     → Medium config QPU
      - ibm_tokyo      (20 qubits, grid+diag)     → Small config alt
      - ibm_rochester  (53 qubits, heavy-hex)     → Large config QPU
      - ibm_montreal   (27 qubits, heavy-hex)     → Medium config alt
      - ibm_washington (127 qubits, heavy-hex)    → Large config QPU
      - ibm_brooklyn   (65 qubits, heavy-hex)     → Large config alt
    
    Returns dict of chip_name -> {n_qubits, coupling_map, basis_gates,
                                  qubit_properties, gate_properties}
    """
    resolved_snapshot_dir = Path(snapshot_dir) if snapshot_dir else DEFAULT_IBM_SNAPSHOT_DIR
    snapshot_complete = all(
        (resolved_snapshot_dir / f'conf_{snapshot}.json').is_file() and
        (resolved_snapshot_dir / f'props_{snapshot}.json').is_file()
        for snapshot in IBM_CHIP_SNAPSHOTS.values()
    )
    if snapshot_complete:
        print(f"  [INFO] Reading local IBM calibration snapshots: {resolved_snapshot_dir}")
        return _extract_ibm_chips_from_snapshots(output_dir, resolved_snapshot_dir)

    try:
        from qiskit.providers.fake_provider import (
            FakeGuadalupe, FakeSydney, FakeTokyo, FakeRochester,
            FakeMontreal, FakeWashington, FakeBrooklyn
        )
    except ImportError:
        print("[WARN] qiskit not available. Using pre-built topologies.")
        return _fallback_ibm_chips(output_dir)

    chips = {
        'ibm_guadalupe':  FakeGuadalupe,
        'ibm_sydney':     FakeSydney,
        'ibm_tokyo':      FakeTokyo,
        'ibm_rochester':  FakeRochester,
        'ibm_montreal':   FakeMontreal,
        'ibm_washington': FakeWashington,
        'ibm_brooklyn':   FakeBrooklyn,
    }

    chip_dir = os.path.join(output_dir, 'ibm_chips')
    os.makedirs(chip_dir, exist_ok=True)
    result = {}

    for name, backend_cls in chips.items():
        backend = backend_cls()
        config = backend.configuration()
        props = backend.properties()

        chip_data = {
            'name': name,
            'n_qubits': config.n_qubits,
            'basis_gates': list(config.basis_gates),
            'coupling_map': config.coupling_map,
            'coupling_map_undirected': _dedupe_edges(config.coupling_map),
            'qubit_properties': [],
            'gate_properties': [],
            'source': 'IBM Quantum (via Qiskit fake provider)',
            'topology_type': _classify_topology(name, config.n_qubits),
        }

        # BackendProperties exposes time in seconds and frequency in Hz.
        for q in range(config.n_qubits):
            qprop = {'qubit': q}
            try:
                qprop['t1_us'] = props.t1(q) * 1e6
                qprop['t2_us'] = props.t2(q) * 1e6
                qprop['freq_ghz'] = props.frequency(q) / 1e9
                qprop['readout_error'] = props.readout_error(q)
                qprop['readout_length_ns'] = props.readout_length(q) * 1e9
            except Exception:
                qprop['t1_us'] = None
                qprop['t2_us'] = None
                qprop['freq_ghz'] = None
                qprop['readout_error'] = None
                qprop['readout_length_ns'] = None
            chip_data['qubit_properties'].append(qprop)

        # Extract gate properties (error rates, durations)
        try:
            for inst in props.gates:
                gprop = {
                    'name': inst.name,
                    'gate': inst.gate,
                    'qubits': list(inst.qubits) if hasattr(inst, 'qubits') else [],
                    'parameters': {}
                }
                for p in inst.parameters:
                    if p.name == 'gate_length':
                        gprop['parameters'][p.name] = convert_calibration_value(
                            p.value, getattr(p, 'unit', ''), 'ns'
                        )
                    else:
                        gprop['parameters'][p.name] = p.value
                chip_data['gate_properties'].append(gprop)
        except Exception:
            pass

        chip_data['calibration_metadata'] = {
            'backend_name': getattr(config, 'backend_name', name),
            'backend_version': getattr(config, 'backend_version', None),
            'last_update_date': str(getattr(props, 'last_update_date', '')) or None,
            'standardized_units': {
                't1_us': 'us',
                't2_us': 'us',
                'freq_ghz': 'GHz',
                'readout_error': 'dimensionless',
                'readout_length_ns': 'ns',
                'gate_error': 'dimensionless',
                'gate_length': 'ns',
            },
        }
        _validate_ibm_chip_data(chip_data)
        chip_data['stats'] = _ibm_chip_stats(chip_data)

        result[name] = chip_data
        fpath = os.path.join(chip_dir, f'{name}.json')
        with open(fpath, 'w') as f:
            json.dump(chip_data, f, indent=2)
        print(f"  [OK] {name}: {config.n_qubits} qubits, "
              f"{len(chip_data['coupling_map_undirected'])} edges, "
              f"avg CX error={chip_data['stats'].get('avg_cx_error', 'N/A')}")

    _write_ibm_chip_dataset(result, output_dir)

    return result


def _dedupe_edges(coupling_map: List[List[int]]) -> List[List[int]]:
    """Convert directed coupling map to undirected edge list."""
    edges = set()
    for a, b in coupling_map:
        edge = tuple(sorted([a, b]))
        edges.add(edge)
    return [list(e) for e in sorted(edges)]


def _classify_topology(name: str, n_qubits: int) -> str:
    """Classify the chip topology type based on known IBM architectures."""
    if name in ('ibm_guadalupe', 'ibm_sydney', 'ibm_montreal', 'ibm_rochester',
                'ibm_washington', 'ibm_brooklyn'):
        return 'heavy-hex'
    elif name == 'ibm_tokyo':
        return 'grid-with-diagonal'
    else:
        return 'unknown'


def _fallback_ibm_chips(output_dir: str) -> Dict[str, dict]:
    """Fallback: use hardcoded IBM chip topologies if Qiskit is unavailable."""
    chips = {
        'ibm_guadalupe': {
            'n_qubits': 16,
            'coupling_map_undirected': [
                [0,1],[1,2],[1,4],[2,3],[3,5],[4,7],[5,8],[6,7],[7,10],
                [8,9],[8,11],[10,12],[11,14],[12,13],[12,15],[13,14]
            ],
            'topology_type': 'heavy-hex',
        },
        'ibm_sydney': {
            'n_qubits': 27,
            'coupling_map_undirected': [
                [0,1],[1,2],[1,4],[2,3],[3,5],[4,7],[5,8],[6,7],[7,10],
                [8,9],[8,11],[10,12],[11,14],[12,13],[12,15],[13,14],
                [14,16],[15,18],[16,19],[17,18],[18,21],[19,20],[19,22],
                [21,23],[22,25],[23,24],[24,25],[25,26]
            ],
            'topology_type': 'heavy-hex',
        },
        'ibm_rochester': {
            'n_qubits': 53,
            'coupling_map_undirected': [
                [0,1],[0,5],[1,2],[2,3],[3,4],[4,6],[5,9],[6,13],[7,8],
                [7,16],[8,9],[9,10],[10,11],[11,12],[11,17],[12,13],
                [13,14],[14,15],[15,18],[16,19],[17,23],[18,27],[19,20],
                [20,21],[21,22],[21,28],[22,23],[23,24],[24,25],[25,26],
                [25,29],[26,27],[28,32],[29,36],[30,31],[30,39],[31,32],
                [32,33],[33,34],[34,35],[34,40],[35,36],[36,37],[37,38],
                [38,41],[39,42],[40,46],[41,50],[42,43],[43,44],[44,45],
                [44,51],[45,46],[46,47],[47,48],[48,49],[48,52],[49,50]
            ],
            'topology_type': 'heavy-hex',
        },
        'ibm_washington': {
            'n_qubits': 127,
            'coupling_map_undirected': [
                [0,1],[0,14],[1,2],[2,3],[3,4],[4,5],[4,15],[5,6],[6,7],
                [7,8],[8,16],[9,10],[10,11],[11,12],[12,13],[12,17],
                [14,18],[15,22],[16,26],[17,30],[18,19],[19,20],[20,21],
                [20,33],[21,22],[22,23],[23,24],[24,25],[24,34],[25,26],
                [26,27],[27,28],[28,29],[28,35],[29,30],[30,31],[31,32],
                [32,36],[33,39],[34,43],[35,47],[36,51],[37,38],[37,52],
                [38,39],[39,40],[40,41],[41,42],[41,53],[42,43],[43,44],
                [44,45],[45,46],[45,54],[46,47],[47,48],[48,49],[49,50],
                [49,55],[50,51],[52,56],[53,60],[54,64],[55,68],[56,57],
                [57,58],[58,59],[58,71],[59,60],[60,61],[61,62],[62,63],
                [62,72],[63,64],[64,65],[65,66],[66,67],[66,73],[67,68],
                [68,69],[69,70],[70,74],[71,77],[72,81],[73,85],[74,89],
                [75,76],[75,90],[76,77],[77,78],[78,79],[79,80],[79,91],
                [80,81],[81,82],[82,83],[83,84],[83,92],[84,85],[85,86],
                [86,87],[87,88],[87,93],[88,89],[90,94],[91,98],[92,102],
                [93,106],[94,95],[95,96],[96,97],[96,109],[97,98],[98,99],
                [99,100],[100,101],[100,110],[101,102],[102,103],[103,104],
                [104,105],[104,111],[105,106],[106,107],[107,108],[108,112],
                [109,96],[110,118],[111,122],[112,126],[113,114],[114,115],
                [115,116],[116,117],[117,118],[118,119],[119,120],[120,121],
                [121,122],[122,123],[123,124],[124,125],[125,126]
            ],
            'topology_type': 'heavy-hex',
        },
    }

    chip_dir = os.path.join(output_dir, 'ibm_chips')
    os.makedirs(chip_dir, exist_ok=True)
    result = {}
    for name, data in chips.items():
        data['name'] = name
        data['basis_gates'] = ['id', 'rz', 'sx', 'x', 'cx', 'reset']
        data['source'] = 'IBM Quantum (hardcoded fallback)'
        result[name] = data
        with open(os.path.join(chip_dir, f'{name}.json'), 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  [OK] {name}: {data['n_qubits']} qubits (fallback)")

    return result


# ============================================================================
# 2. Classical Network Topologies
# ============================================================================

def generate_fattree(switches_per_level: int = 4) -> nx.Graph:
    """
    Generate a Fat-Tree topology.
    k-port Fat-Tree has:
      - k^2/4 core switches
      - k aggregation switches
      - k edge switches
      - k^2/2 hosts
    
    For k=4: 4 core, 4 agg, 4 edge, 8 hosts = 20 nodes
    For k=8: 16 core, 8 agg, 8 edge, 32 hosts = 64 nodes
    """
    k = switches_per_level
    G = nx.Graph()
    
    core_id = 0
    agg_start = k * k // 4
    edge_start = agg_start + k
    host_start = edge_start + k
    
    # Core switches
    for i in range(k * k // 4):
        G.add_node(f'core_{i}', type='core', id=core_id + i)
    
    # Aggregation switches
    for i in range(k):
        G.add_node(f'agg_{i}', type='aggregation', id=agg_start + i)
    
    # Edge switches
    for i in range(k):
        G.add_node(f'edge_{i}', type='edge', id=edge_start + i)
    
    # Hosts
    for i in range(k * k // 2):
        G.add_node(f'host_{i}', type='host', id=host_start + i)
    
    # Core-Aggregation links (each core connects to k/2 pods)
    for i in range(k * k // 4):
        for j in range(k):
            pod = j
            agg_in_pod = agg_start + pod
            if (i % (k // 2)) == (j % (k // 2)):
                G.add_edge(f'core_{i}', f'agg_{j}',
                           bandwidth_gbps=100, latency_us=2)
    
    # Aggregation-Edge links (within each pod)
    for pod in range(k):
        for i in range(k // 2):
            agg = agg_start + pod
            edge = edge_start + pod * (k // 2) + i
            G.add_edge(f'agg_{pod}', f'edge_{pod * (k // 2) + i}',
                       bandwidth_gbps=40, latency_us=1)
    
    # Edge-Host links
    for pod in range(k):
        for i in range(k // 2):
            edge = edge_start + pod * (k // 2) + i
            for j in range(k // 2):
                host = host_start + pod * (k // 2) * (k // 2) + i * (k // 2) + j
                G.add_edge(f'edge_{pod * (k // 2) + i}', f'host_{host}',
                           bandwidth_gbps=10, latency_us=0.5)
    
    return G


def generate_dragonfly(groups: int = 4, routers_per_group: int = 4,
                       nodes_per_router: int = 2) -> nx.Graph:
    """
    Generate a Dragonfly topology.
    
    For groups=4, routers_per_group=4, nodes_per_router=2:
      - 4 groups × 4 routers × 2 nodes = 32 nodes
    Each router has:
      - a = routers_per_group-1 intra-group links
      - h = groups-1 inter-group links (simplified)
      - p = nodes_per_router local links
    """
    G = nx.Graph()
    node_id = 0
    router_id = 0
    
    for g in range(groups):
        for r in range(routers_per_group):
            G.add_node(f'r_{g}_{r}', type='router', group=g, id=router_id)
            router_id += 1
            for n in range(nodes_per_router):
                G.add_node(f'n_{g}_{r}_{n}', type='host', group=g, id=node_id)
                G.add_edge(f'r_{g}_{r}', f'n_{g}_{r}_{n}',
                           bandwidth_gbps=10, latency_us=0.5)
                node_id += 1
    
    # Intra-group links (full mesh within each group)
    for g in range(groups):
        for r1 in range(routers_per_group):
            for r2 in range(r1 + 1, routers_per_group):
                G.add_edge(f'r_{g}_{r1}', f'r_{g}_{r2}',
                           bandwidth_gbps=40, latency_us=1)
    
    # Inter-group links (each group connects to every other group)
    for g1 in range(groups):
        for g2 in range(g1 + 1, groups):
            # Connect first router of g1 to first router of g2
            r1 = 0
            r2 = (g2 - g1) % routers_per_group
            G.add_edge(f'r_{g1}_{r1}', f'r_{g2}_{r2}',
                       bandwidth_gbps=100, latency_us=2)
    
    return G


def generate_dragonfly_plus(groups: int = 8, routers_per_group: int = 8,
                             nodes_per_router: int = 4) -> nx.Graph:
    """
    Generate a Dragonfly+ topology (3-level hierarchy).
    
    For groups=8, routers_per_group=8, nodes_per_router=4:
      - Level 1: 8 groups (each with 8 routers)
      - Level 2: inter-group links
      - Level 3: global interconnect
    Total: 8 × 8 × 4 = 256 nodes
    """
    G = nx.Graph()
    node_id = 0
    router_id = 0
    
    for g in range(groups):
        for r in range(routers_per_group):
            G.add_node(f'r_{g}_{r}', type='router', group=g, level=1, id=router_id)
            router_id += 1
            for n in range(nodes_per_router):
                G.add_node(f'n_{g}_{r}_{n}', type='host', group=g, id=node_id)
                G.add_edge(f'r_{g}_{r}', f'n_{g}_{r}_{n}',
                           bandwidth_gbps=25, latency_us=0.5)
                node_id += 1
    
    # Intra-group: full mesh
    for g in range(groups):
        for r1 in range(routers_per_group):
            for r2 in range(r1 + 1, routers_per_group):
                G.add_edge(f'r_{g}_{r1}', f'r_{g}_{r2}',
                           bandwidth_gbps=100, latency_us=0.5)
    
    # Inter-group: connect via dedicated inter-group routers
    for g1 in range(groups):
        for g2 in range(g1 + 1, groups):
            r1 = g2 % routers_per_group
            r2 = g1 % routers_per_group
            G.add_edge(f'r_{g1}_{r1}', f'r_{g2}_{r2}',
                       bandwidth_gbps=200, latency_us=2)
    
    return G


def networkx_to_json(G: nx.Graph, name: str) -> dict:
    """Convert NetworkX graph to JSON-serializable dict with link parameters."""
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({'id': node_id, **data})
    
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({'source': u, 'target': v, **data})
    
    # Compute graph statistics
    host_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'host']
    router_nodes = [n for n, d in G.nodes(data=True) if d.get('type') in ('router', 'core', 'aggregation', 'edge')]
    
    return {
        'name': name,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_hosts': len(host_nodes),
        'n_routers': len(router_nodes),
        'nodes': nodes,
        'edges': edges,
        'diameter': nx.diameter(G) if nx.is_connected(G) else -1,
        'avg_shortest_path': round(np.mean(list(
            nx.average_shortest_path_length(G) for _ in [0]
        )), 2) if nx.is_connected(G) else -1,
    }


def generate_network_topologies(output_dir: str) -> Dict[str, dict]:
    """Generate all classical network topologies."""
    net_dir = os.path.join(output_dir, 'network_topologies')
    os.makedirs(net_dir, exist_ok=True)
    result = {}

    # Fat-Tree k=4 (Small: 8 hosts)
    G_ft4 = generate_fattree(4)
    ft4_data = networkx_to_json(G_ft4, 'fattree_k4')
    result['fattree_k4'] = ft4_data
    with open(os.path.join(net_dir, 'fattree_k4.json'), 'w') as f:
        json.dump(ft4_data, f, indent=2)
    print(f"  [OK] fattree_k4: {ft4_data['n_nodes']} nodes, {ft4_data['n_edges']} edges, "
          f"{ft4_data['n_hosts']} hosts")

    # Fat-Tree k=8 (Medium: 32 hosts)
    G_ft8 = generate_fattree(8)
    ft8_data = networkx_to_json(G_ft8, 'fattree_k8')
    result['fattree_k8'] = ft8_data
    with open(os.path.join(net_dir, 'fattree_k8.json'), 'w') as f:
        json.dump(ft8_data, f, indent=2)
    print(f"  [OK] fattree_k8: {ft8_data['n_nodes']} nodes, {ft8_data['n_edges']} edges, "
          f"{ft8_data['n_hosts']} hosts")

    # Dragonfly (Medium: 32 hosts)
    G_df = generate_dragonfly(groups=4, routers_per_group=4, nodes_per_router=2)
    df_data = networkx_to_json(G_df, 'dragonfly_4x4x2')
    result['dragonfly_4x4x2'] = df_data
    with open(os.path.join(net_dir, 'dragonfly_4x4x2.json'), 'w') as f:
        json.dump(df_data, f, indent=2)
    print(f"  [OK] dragonfly_4x4x2: {df_data['n_nodes']} nodes, {df_data['n_edges']} edges, "
          f"{df_data['n_hosts']} hosts")

    # Dragonfly+ (Large: 256 hosts)
    G_dfp = generate_dragonfly_plus(groups=8, routers_per_group=8, nodes_per_router=4)
    dfp_data = networkx_to_json(G_dfp, 'dragonflyplus_8x8x4')
    result['dragonflyplus_8x8x4'] = dfp_data
    with open(os.path.join(net_dir, 'dragonflyplus_8x8x4.json'), 'w') as f:
        json.dump(dfp_data, f, indent=2)
    print(f"  [OK] dragonflyplus_8x8x4: {dfp_data['n_nodes']} nodes, {dfp_data['n_edges']} edges, "
          f"{dfp_data['n_hosts']} hosts")

    # Write index
    index = {name: {'n_nodes': d['n_nodes'], 'n_hosts': d['n_hosts'],
                     'diameter': d['diameter']}
             for name, d in result.items()}
    with open(os.path.join(net_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)

    return result


# ============================================================================
# 3. Hybrid System Configurations
# ============================================================================

def generate_system_configs(chips: Dict, networks: Dict, output_dir: str) -> Dict:
    """
    Generate three system configurations matching Table I in the paper:
      - Small:  4 QPUs, 16 qubits/QPU, 8 classical nodes, Fat-Tree
      - Medium: 8 QPUs, 27 qubits/QPU, 16 classical nodes, Dragonfly
      - Large:  16 QPUs, 127 qubits/QPU, 32 classical nodes, Dragonfly+
    """
    cfg_dir = os.path.join(output_dir, 'system_configs')
    os.makedirs(cfg_dir, exist_ok=True)

    configs = {
        'small': {
            'name': 'Small',
            'n_qpus': 4,
            'qpu_type': 'ibm_guadalupe',  # 16 qubits each
            'n_classical_nodes': 8,
            'classical_node_types': {
                'CPU': 4, 'GPU': 3, 'TPU': 1
            },
            'network_topology': 'fattree_k4',
            'network_type': 'Fat-Tree',
        },
        'medium': {
            'name': 'Medium',
            'n_qpus': 8,
            'qpu_type': 'ibm_sydney',  # 27 qubits each
            'n_classical_nodes': 16,
            'classical_node_types': {
                'CPU': 8, 'GPU': 6, 'TPU': 2
            },
            'network_topology': 'dragonfly_4x4x2',
            'network_type': 'Dragonfly',
        },
        'large': {
            'name': 'Large',
            'n_qpus': 16,
            'qpu_type': 'ibm_washington',  # 127 qubits each
            'n_classical_nodes': 32,
            'classical_node_types': {
                'CPU': 16, 'GPU': 12, 'TPU': 4
            },
            'network_topology': 'dragonflyplus_8x8x4',
            'network_type': 'Dragonfly+',
        },
    }

    # Enrich with actual chip and network data
    for cfg_name, cfg in configs.items():
        chip = chips.get(cfg['qpu_type'], {})
        net = networks.get(cfg['network_topology'], {})

        # Generate QPU instances using the physical capacity of the selected
        # IBM chip.  Never claim more qubits than its coupling map describes.
        qpu_instances = []
        target_qubits = chip['n_qubits']
        topology = chip.get('coupling_map_undirected', [])
        topology_vertices = {q for edge in topology for q in edge}
        if topology_vertices and max(topology_vertices) >= target_qubits:
            raise ValueError(
                f"{cfg['qpu_type']} coupling map exceeds declared capacity "
                f"{target_qubits}"
            )
        for i in range(cfg['n_qpus']):
            qpu = {
                'qpu_id': f'Q{i}',
                'chip_type': cfg['qpu_type'],
                'n_qubits': target_qubits,
                'coherence_time_us': _sample_coherence_time(),
                'gate_times_ns': {
                    '1q': 35.6,   # typical IBM single-qubit gate
                    '2q': 334.0,  # typical IBM CX gate
                    'measure': 1200.0,
                },
                'gate_error_rates': {
                    '1q': round(random.uniform(1e-4, 5e-4), 6),
                    '2q': round(random.uniform(1e-3, 1e-2), 6),
                    'measure': round(random.uniform(1e-3, 5e-3), 6),
                },
                'topology': topology,
                'topology_type': chip.get('topology_type', 'heavy-hex'),
                'topology_source': 'IBM Quantum coupling map',
            }
            qpu_instances.append(qpu)

        # Generate classical node instances
        classical_instances = []
        node_idx = 0
        for node_type, count in cfg['classical_node_types'].items():
            for i in range(count):
                node = {
                    'node_id': f'P{node_idx}',
                    'type': node_type,
                    'compute_speed_tflops': _sample_compute_speed(node_type),
                    'memory_gb': _sample_memory(node_type),
                }
                classical_instances.append(node)
                node_idx += 1

        # Generate quantum interconnect (entanglement links between QPU pairs)
        quantum_links = []
        for i in range(cfg['n_qpus']):
            for j in range(i + 1, cfg['n_qpus']):
                if random.random() > 0.3:  # ~70% of pairs have direct quantum links
                    link = {
                        'qpu_i': f'Q{i}',
                        'qpu_j': f'Q{j}',
                        'entanglement_rate_hz': random.choice(
                            [100, 500, 1000, 5000, 10000]),
                        'decoherence_rate_per_ms': round(random.uniform(0.01, 0.1), 4),
                        'epr_fidelity': round(random.uniform(0.85, 0.95), 6),
                        'max_concurrent_epr': random.choice([4, 8, 16, 32]),
                        'distance_km': round(random.uniform(1, 50), 1),
                    }
                    quantum_links.append(link)

        # Generate interface links (QPU to classical controller)
        interface_links = []
        for i in range(cfg['n_qpus']):
            # Each QPU connects to its nearest classical node
            j = i % cfg['n_classical_nodes']
            interface_links.append({
                'qpu_id': f'Q{i}',
                'classical_node_id': f'P{j}',
                'readout_latency_us': round(random.uniform(1, 10), 2),
                'control_bandwidth_gbps': random.choice([1, 5, 10, 25]),
            })

        cfg_full = {
            **cfg,
            'qpu_instances': qpu_instances,
            'classical_instances': classical_instances,
            'quantum_links': quantum_links,
            'interface_links': interface_links,
            'network_stats': {
                'n_nodes': net.get('n_nodes', 0),
                'n_edges': net.get('n_edges', 0),
                'diameter': net.get('diameter', 0),
            },
        }

        with open(os.path.join(cfg_dir, f'{cfg_name}.json'), 'w') as f:
            json.dump(cfg_full, f, indent=2)
        print(f"  [OK] {cfg_name}: {cfg['n_qpus']} QPUs × {target_qubits} qubits, "
              f"{cfg['n_classical_nodes']} classical nodes, {len(quantum_links)} quantum links")

    # Write index
    with open(os.path.join(cfg_dir, 'index.json'), 'w') as f:
        json.dump({k: {'n_qpus': v['n_qpus'], 'n_classical_nodes': v['n_classical_nodes'],
                        'network_type': v['network_type']}
                   for k, v in configs.items()}, f, indent=2)

    return configs


def _sample_coherence_time() -> float:
    """Sample coherence time in microseconds (IBM range: 50-300 us)."""
    return round(random.uniform(50, 300), 1)


def _sample_compute_speed(node_type: str) -> float:
    """Sample compute speed in TFLOPS based on node type."""
    if node_type == 'CPU':
        return round(random.uniform(0.5, 3.0), 2)
    elif node_type == 'GPU':
        return round(random.uniform(10, 80), 2)
    elif node_type == 'TPU':
        return round(random.uniform(50, 200), 2)
    return 1.0


def _sample_memory(node_type: str) -> int:
    """Sample memory in GB based on node type."""
    if node_type == 'CPU':
        return random.choice([64, 128, 256])
    elif node_type == 'GPU':
        return random.choice([32, 40, 80])
    elif node_type == 'TPU':
        return random.choice([16, 32])
    return 64


# ============================================================================
# 4. Quantum Circuit Benchmarks (MQT Bench integration)
# ============================================================================

def generate_quantum_circuits(output_dir: str, mqt_bench_dir: str = None) -> Dict:
    """
    Generate/download quantum circuit benchmarks.
    
    Primary: MQT Bench (https://www.cda.cit.tum.de/mqtbench/)
    Fallback: Generate synthetic circuits matching VQE/QAOA/QML patterns.
    
    Circuit types (matching paper workloads):
      - VQE:   50-200 qubit Hamiltonians, iterative measurement-feedback
      - QAOA:  100-500 variables, alternating operator layers
      - QML:   80-300 qubit QNN, gradient computation
      - QFT:   For ablation/diversity
      - GHZ:   For ablation/diversity
    """
    circ_dir = os.path.join(output_dir, 'quantum_circuits')
    os.makedirs(circ_dir, exist_ok=True)
    result = {}

    # Try to load from MQT Bench directory if provided
    if mqt_bench_dir and os.path.exists(mqt_bench_dir):
        result = _load_mqt_bench(mqt_bench_dir, circ_dir)
    
    # Generate synthetic circuit specs if MQT Bench not available
    if not result:
        result = _generate_synthetic_circuits(circ_dir)

    # Write index
    with open(os.path.join(circ_dir, 'index.json'), 'w') as f:
        json.dump({k: {'type': v['type'], 'n_qubits': v['n_qubits'],
                        'depth': v.get('depth', 0),
                        'n_gates': v.get('n_gates', 0),
                        'source': v.get('source', 'synthetic')}
                   for k, v in result.items()}, f, indent=2)

    return result


def _load_mqt_bench(mqt_dir: str, output_dir: str) -> Dict:
    """Load circuits from MQT Bench directory."""
    result = {}
    for root, dirs, files in os.walk(mqt_dir):
        for fname in files:
            if fname.endswith('.qasm'):
                fpath = os.path.join(root, fname)
                # Parse QASM to extract circuit info
                try:
                    n_qubits, depth, n_gates, gate_counts, interactions = _parse_qasm(fpath)
                    circuit_id = fname.replace('.qasm', '')
                    result[circuit_id] = {
                        'id': circuit_id,
                        'file': fname,
                        'path': fpath,
                        'type': _classify_circuit(circuit_id),
                        'n_qubits': n_qubits,
                        'depth': depth,
                        'n_gates': n_gates,
                        'gate_counts': gate_counts,
                        'interactions': interactions,
                        'interaction_source': 'parsed from QASM two-qubit gates',
                        'source': 'MQT Bench',
                    }
                except Exception as e:
                    print(f"  [WARN] Failed to parse {fname}: {e}")
    return result


def _parse_qasm(fpath: str) -> Tuple[int, int, int, Dict, List[Dict]]:
    """Parse a QASM file to extract circuit statistics."""
    n_qubits = 0
    n_gates = 0
    gate_counts = {}
    interaction_counts = {}
    
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('qreg'):
                # qreg q[20];
                n_qubits = int(line.split('[')[1].split(']')[0])
            elif line.startswith('OPENQASM') or line.startswith('include') or line.startswith('//'):
                continue
            elif line and not line.startswith('creg') and not line.startswith('measure'):
                gate_name = line.split()[0] if line.split() else 'unknown'
                if gate_name in ('cx', 'cz', 'rzz', 'swap'):
                    gate_counts['2q'] = gate_counts.get('2q', 0) + 1
                    qubits = [int(q) for q in re.findall(r'\[(\d+)\]', line)]
                    if len(qubits) >= 2:
                        edge = tuple(sorted(qubits[:2]))
                        interaction_counts[edge] = interaction_counts.get(edge, 0) + 1
                elif gate_name in ('h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'sx'):
                    gate_counts['1q'] = gate_counts.get('1q', 0) + 1
                n_gates += 1
    
    # Depth estimation (simplified)
    depth = max(1, n_gates // max(1, n_qubits))
    
    interactions = _interaction_records(interaction_counts)
    return n_qubits, depth, n_gates, gate_counts, interactions


def _classify_circuit(name: str) -> str:
    """Classify a circuit by its name."""
    name_lower = name.lower()
    if 'vqe' in name_lower:
        return 'VQE'
    elif 'qaoa' in name_lower:
        return 'QAOA'
    elif 'qnn' in name_lower or 'qml' in name_lower:
        return 'QML'
    elif 'qft' in name_lower:
        return 'QFT'
    elif 'ghz' in name_lower:
        return 'GHZ'
    elif 'grover' in name_lower:
        return 'Grover'
    elif 'ae' in name_lower:
        return 'AE'
    else:
        return 'other'


def _interaction_records(interaction_counts: Dict[Tuple[int, int], int]) -> List[Dict]:
    """Convert aggregated two-qubit counts to stable JSON records."""
    total = sum(interaction_counts.values())
    if total <= 0:
        return []
    return [
        {
            'source': source,
            'target': target,
            'two_qubit_gates': count,
            'frequency': count / total,
        }
        for (source, target), count in sorted(interaction_counts.items())
    ]


def _distribute_gate_counts(edges: List[Tuple[int, int]], n_2q: int) -> Dict[Tuple[int, int], int]:
    """Distribute exactly ``n_2q`` gates over a deterministic edge list."""
    if n_2q <= 0 or not edges:
        return {}
    used_edges = edges[:min(len(edges), n_2q)]
    base, remainder = divmod(n_2q, len(used_edges))
    return {
        edge: base + (1 if index < remainder else 0)
        for index, edge in enumerate(used_edges)
    }


def _build_circuit_interactions(circuit: Dict) -> List[Dict]:
    """Build reproducible interactions for a synthetic circuit family."""
    n_qubits = circuit['n_qubits']
    n_2q = circuit.get('gate_counts', {}).get('2q', 0)
    circuit_type = circuit.get('type', '').upper()

    if n_qubits < 2 or n_2q <= 0:
        return []

    if circuit_type in ('VQE', 'QML'):
        edges = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
        edges = [tuple(sorted(edge)) for edge in edges]
        edges = sorted(set(edges))
    elif circuit_type == 'GHZ':
        edges = [(q, q + 1) for q in range(n_qubits - 1)]
    elif circuit_type == 'QFT':
        edges = [(source, target)
                 for source in range(n_qubits)
                 for target in range(source + 1, n_qubits)]
    elif circuit_type == 'QAOA':
        all_edges = [(source, target)
                     for source in range(n_qubits)
                     for target in range(source + 1, n_qubits)]
        density = circuit.get('graph_density', 0.1)
        n_problem_edges = min(
            n_2q,
            max(1, round(density * len(all_edges))),
        )
        digest = hashlib.sha256(circuit['id'].encode('utf-8')).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        edges = sorted(rng.sample(all_edges, n_problem_edges))
    else:
        raise ValueError(f"No interaction model for circuit type {circuit_type!r}")

    return _interaction_records(_distribute_gate_counts(edges, n_2q))


def _validate_interactions(circuit: Dict) -> None:
    """Ensure interaction endpoints and weights match the circuit metadata."""
    n_qubits = circuit['n_qubits']
    expected = circuit.get('gate_counts', {}).get('2q', 0)
    interactions = circuit.get('interactions', [])
    actual = sum(edge['two_qubit_gates'] for edge in interactions)
    if actual != expected:
        raise ValueError(
            f"{circuit['id']}: interaction gate total {actual} != {expected}"
        )
    for edge in interactions:
        source, target = edge['source'], edge['target']
        if not (0 <= source < target < n_qubits):
            raise ValueError(
                f"{circuit['id']}: invalid interaction edge {(source, target)}"
            )


def _generate_synthetic_circuits(output_dir: str) -> Dict:
    """
    Generate synthetic circuit specifications matching the paper's workload
    descriptions when MQT Bench is not available.
    
    These are realistic parameter ranges based on published VQE/QAOA/QML circuits.
    """
    circuits = {}

    # VQE circuits (molecular simulation, 50-200 qubits)
    vqe_sizes = [50, 80, 100, 120, 150, 200]
    for nq in vqe_sizes:
        cid = f'vqe_{nq}q'
        n_layers = max(2, nq // 20)
        n_1q = nq * n_layers * 2  # 2 single-qubit gates per qubit per layer
        n_2q = nq * n_layers  # entangling layer
        circuits[cid] = {
            'id': cid,
            'type': 'VQE',
            'n_qubits': nq,
            'depth': n_layers * 3,
            'n_gates': n_1q + n_2q,
            'gate_counts': {'1q': n_1q, '2q': n_2q},
            'source': 'synthetic (parameterized from literature)',
            'description': f'VQE molecular simulation, {nq} qubits, {n_layers} ansatz layers',
            'n_measurement_terms': nq * (nq - 1) // 2,  # Pauli string count
            'n_iterations': random.randint(50, 500),  # VQE optimization iterations
        }

    # QAOA circuits (combinatorial optimization, 100-500 variables)
    qaoa_sizes = [100, 150, 200, 300, 500]
    for nq in qaoa_sizes:
        cid = f'qaoa_{nq}q'
        p_layers = random.choice([1, 2, 3, 5])  # QAOA depth parameter p
        n_1q = nq * p_layers * 2
        n_2q = nq * p_layers  # one entangler per qubit per layer
        circuits[cid] = {
            'id': cid,
            'type': 'QAOA',
            'n_qubits': nq,
            'depth': p_layers * 2,
            'n_gates': n_1q + n_2q,
            'gate_counts': {'1q': n_1q, '2q': n_2q},
            'source': 'synthetic (parameterized from literature)',
            'description': f'QAOA MaxCut, {nq} variables, p={p_layers}',
            'problem': 'MaxCut',
            'graph_density': round(random.uniform(0.1, 0.5), 2),
        }

    # QML circuits (quantum neural network, 80-300 qubits)
    qml_sizes = [80, 120, 150, 200, 300]
    for nq in qml_sizes:
        cid = f'qml_{nq}q'
        n_layers = max(3, nq // 30)
        n_1q = nq * n_layers * 3  # more single-qubit rotations for QML
        n_2q = nq * n_layers
        circuits[cid] = {
            'id': cid,
            'type': 'QML',
            'n_qubits': nq,
            'depth': n_layers * 4,
            'n_gates': n_1q + n_2q,
            'gate_counts': {'1q': n_1q, '2q': n_2q},
            'source': 'synthetic (parameterized from literature)',
            'description': f'Quantum neural network, {nq} qubits, {n_layers} layers',
            'n_training_params': nq * n_layers,
            'n_epochs': random.randint(10, 100),
        }

    # Additional circuits for diversity
    # QFT
    for nq in [50, 100, 200]:
        cid = f'qft_{nq}q'
        n_1q = nq * (nq + 1) // 2  # Hadamards + rotations
        n_2q = nq * (nq - 1) // 2  # controlled rotations
        circuits[cid] = {
            'id': cid,
            'type': 'QFT',
            'n_qubits': nq,
            'depth': nq,
            'n_gates': n_1q + n_2q,
            'gate_counts': {'1q': n_1q, '2q': n_2q},
            'source': 'synthetic',
            'description': f'Quantum Fourier Transform, {nq} qubits',
        }

    # GHZ state preparation
    for nq in [50, 100, 200]:
        cid = f'ghz_{nq}q'
        circuits[cid] = {
            'id': cid,
            'type': 'GHZ',
            'n_qubits': nq,
            'depth': nq - 1,
            'n_gates': nq,
            'gate_counts': {'1q': 1, '2q': nq - 1},
            'source': 'synthetic',
            'description': f'GHZ state preparation, {nq} qubits',
        }

    # Save individual circuit specs
    for cid, cdata in circuits.items():
        cdata['interactions'] = _build_circuit_interactions(cdata)
        cdata['interaction_source'] = 'deterministic synthetic circuit-family model'
        _validate_interactions(cdata)
        with open(os.path.join(output_dir, f'{cid}.json'), 'w') as f:
            json.dump(cdata, f, indent=2)
        print(f"  [OK] {cid}: {cdata['n_qubits']} qubits, {cdata['n_gates']} gates, "
              f"depth={cdata['depth']}")

    return circuits


# ============================================================================
# 5. Hybrid Workload DAG Generation
# ============================================================================

def generate_workload_dags(circuits: Dict, output_dir: str, seed: int = 42) -> Dict:
    """
    Generate hybrid quantum-classical workload DAGs.
    
    References classical DAG/workflow scheduling methods:
      - STG (Standard Task Graph Set): random DAG generation with parameters
      - DAGGEN: layer-based DAG generation
      - WfCommons: workflow-based DAG generation
    
    For TopoQ, we generate three types of hybrid DAGs:
      - VQE DAG:  iterative measurement → classical optimization → parameter update → quantum execution
      - QAOA DAG: alternating operator layers with classical parameter optimization between layers
      - QML DAG:  forward pass (quantum) → gradient computation (classical) → parameter update → next epoch
    
    Each DAG follows the formal model: W = (T, E_D, ≺, μ)
    with T = T_Q ∪ T_C and special E_D^QC feedback edges.
    """
    random.seed(seed)
    np.random.seed(seed)

    dag_dir = os.path.join(output_dir, 'workload_dags')
    os.makedirs(dag_dir, exist_ok=True)
    result = {}

    # Select circuits for each workload type
    vqe_circuits = {k: v for k, v in circuits.items() if v['type'] == 'VQE'}
    qaoa_circuits = {k: v for k, v in circuits.items() if v['type'] == 'QAOA'}
    qml_circuits = {k: v for k, v in circuits.items() if v['type'] == 'QML'}

    # Generate VQE DAGs
    for cid, circuit in vqe_circuits.items():
        for n_iter in [5, 10, 20]:  # VQE iterations represented as DAG layers
            dag_id = f'vqe_dag_{circuit["n_qubits"]}q_{n_iter}iter'
            dag = _generate_vqe_dag(circuit, n_iter)
            result[dag_id] = dag
            with open(os.path.join(dag_dir, f'{dag_id}.json'), 'w') as f:
                json.dump(dag, f, indent=2)
            print(f"  [OK] {dag_id}: {dag['n_tasks']} tasks "
                  f"({dag['n_quantum_tasks']}Q + {dag['n_classical_tasks']}C), "
                  f"{dag['n_edges']} edges")

    # Generate QAOA DAGs
    for cid, circuit in qaoa_circuits.items():
        p = circuit.get('depth', 2) // 2  # QAOA p parameter
        for n_iter in [5, 10, 20]:
            dag_id = f'qaoa_dag_{circuit["n_qubits"]}q_p{p}_{n_iter}iter'
            dag = _generate_qaoa_dag(circuit, n_iter)
            result[dag_id] = dag
            with open(os.path.join(dag_dir, f'{dag_id}.json'), 'w') as f:
                json.dump(dag, f, indent=2)
            print(f"  [OK] {dag_id}: {dag['n_tasks']} tasks "
                  f"({dag['n_quantum_tasks']}Q + {dag['n_classical_tasks']}C), "
                  f"{dag['n_edges']} edges")

    # Generate QML DAGs
    for cid, circuit in qml_circuits.items():
        n_epochs = circuit.get('n_epochs', 20)
        for n_iter in [5, 10, 20]:
            dag_id = f'qml_dag_{circuit["n_qubits"]}q_{n_iter}iter'
            dag = _generate_qml_dag(circuit, n_iter)
            result[dag_id] = dag
            with open(os.path.join(dag_dir, f'{dag_id}.json'), 'w') as f:
                json.dump(dag, f, indent=2)
            print(f"  [OK] {dag_id}: {dag['n_tasks']} tasks "
                  f"({dag['n_quantum_tasks']}Q + {dag['n_classical_tasks']}C), "
                  f"{dag['n_edges']} edges")

    # Write index
    with open(os.path.join(dag_dir, 'index.json'), 'w') as f:
        json.dump({k: {'type': v['type'], 'n_tasks': v['n_tasks'],
                        'n_quantum_tasks': v['n_quantum_tasks'],
                        'n_classical_tasks': v['n_classical_tasks'],
                        'n_edges': v['n_edges']}
                   for k, v in result.items()}, f, indent=2)

    return result


def _generate_vqe_dag(circuit: dict, n_iterations: int) -> dict:
    """
    Generate a VQE workload DAG.
    
    Structure (per iteration):
      1. [Quantum] Prepare ansatz state (quantum circuit execution)
      2. [Quantum] Measure Hamiltonian terms (multiple measurement tasks, can be parallelized)
      3. [Classical] Aggregate measurement results (classical reduction)
      4. [Classical] Classical optimizer step (gradient computation + parameter update)
      5. [Q→C feedback] Updated parameters sent to next iteration's quantum task
    
    Total: n_iterations × (1 + n_measurements + 1 + 1) tasks
    """
    tasks = []
    edges = []
    task_id = 0

    n_qubits = circuit['n_qubits']
    # Number of measurement groups (Pauli terms that can be measured simultaneously)
    n_measure_groups = max(1, min(20, n_qubits // 5))
    n_measurement_terms = circuit.get('n_measurement_terms', n_qubits * (n_qubits - 1) // 2)

    for it in range(n_iterations):
        # 1. Ansatz state preparation (quantum)
        ansatz_task = {
            'id': f't{task_id}',
            'type': 'quantum',
            'name': f'vqe_ansatz_iter{it}',
            'circuit_id': circuit['id'],
            'n_qubits': n_qubits,
            'n_gates': circuit.get('n_gates', n_qubits * 4),
            'depth': circuit.get('depth', 6),
            'estimated_time_ms': round(n_qubits * 0.5 + circuit.get('depth', 6) * 0.1, 2),
        }
        tasks.append(ansatz_task)
        ansatz_id = task_id
        task_id += 1

        # 2. Measurement tasks (quantum, parallelized by measurement group)
        measure_task_ids = []
        for mg in range(n_measure_groups):
            measure_task = {
                'id': f't{task_id}',
                'type': 'quantum',
                'name': f'vqe_measure_iter{it}_group{mg}',
                'circuit_id': circuit['id'],
                'n_qubits': n_qubits,
                'n_gates': n_qubits,  # measurement circuits
                'depth': 1,
                'estimated_time_ms': round(n_qubits * 0.05, 2),
                'measurement_group': mg,
                'n_terms': n_measurement_terms // n_measure_groups,
            }
            tasks.append(measure_task)
            measure_task_ids.append(task_id)
            # Edge: ansatz → measure (data dependency: quantum state)
            edges.append({
                'source': f't{ansatz_id}',
                'target': f't{task_id}',
                'type': 'quantum-quantum',
                'data_volume_bits': n_qubits,  # quantum state handoff
            })
            task_id += 1

        # 3. Aggregate measurement results (classical)
        agg_task = {
            'id': f't{task_id}',
            'type': 'classical',
            'name': f'vqe_aggregate_iter{it}',
            'workload_flops': n_measure_groups * 1e6,  # 1 MFLOP per group
            'memory_mb': 100,
            'estimated_time_ms': round(n_measure_groups * 0.5, 2),
        }
        tasks.append(agg_task)
        agg_id = task_id
        for mtid in measure_task_ids:
            # Edge: measure → aggregate (Q→C feedback edge)
            edges.append({
                'source': f't{mtid}',
                'target': f't{agg_id}',
                'type': 'quantum-classical',  # E_D^QC edge
                'data_volume_bits': n_qubits * 1000,  # measurement results
            })
        task_id += 1

        # 4. Classical optimizer step (gradient + parameter update)
        opt_task = {
            'id': f't{task_id}',
            'type': 'classical',
            'name': f'vqe_optimize_iter{it}',
            'workload_flops': n_qubits * 1e7,  # 10 MFLOP per qubit
            'memory_mb': 500,
            'estimated_time_ms': round(n_qubits * 0.3, 2),
        }
        tasks.append(opt_task)
        opt_id = task_id
        # Edge: aggregate → optimize
        edges.append({
            'source': f't{agg_id}',
            'target': f't{opt_id}',
            'type': 'classical-classical',
            'data_volume_bits': n_measure_groups * 64,
        })
        task_id += 1

        # 5. If not last iteration, edge from optimizer to next iteration's ansatz
        if it < n_iterations - 1:
            # This edge will be created when next iteration's ansatz is added
            # We store it as a pending edge
            pass

    # Add inter-iteration feedback edges (C→Q)
    for it in range(n_iterations - 1):
        # Find optimizer task of iteration it
        opt_task_id = 3 + it * (2 + n_measure_groups)  # pattern: ansatz(1) + measures(n_mg) + agg(1) + opt(1)
        # Find ansatz task of iteration it+1
        next_ansatz_id = (it + 1) * (2 + n_measure_groups)
        edges.append({
            'source': f't{opt_task_id}',
            'target': f't{next_ansatz_id}',
            'type': 'classical-quantum',  # E_D^QC feedback edge
            'data_volume_bits': n_qubits * 64,  # parameter update
        })

    n_q = sum(1 for t in tasks if t['type'] == 'quantum')
    n_c = sum(1 for t in tasks if t['type'] == 'classical')
    n_qc_edges = sum(1 for e in edges if e['type'] in ('quantum-classical', 'classical-quantum'))

    return {
        'id': f'vqe_dag_{n_qubits}q_{n_iterations}iter',
        'type': 'VQE',
        'circuit_id': circuit['id'],
        'n_qubits': n_qubits,
        'n_iterations': n_iterations,
        'n_tasks': len(tasks),
        'n_quantum_tasks': n_q,
        'n_classical_tasks': n_c,
        'n_edges': len(edges),
        'n_qc_feedback_edges': n_qc_edges,
        'tasks': tasks,
        'edges': edges,
        'description': f'VQE DAG: {n_iterations} iterations, {n_qubits} qubits, '
                       f'{n_measure_groups} measurement groups per iteration',
    }


def _generate_qaoa_dag(circuit: dict, n_iterations: int) -> dict:
    """
    Generate a QAOA workload DAG.
    
    Structure (per iteration):
      1. [Quantum] Execute cost Hamiltonian layer (problem-specific gates)
      2. [Quantum] Execute mixer Hamiltonian layer
      3. [Quantum] Measure expectation value
      4. [Classical] Compute objective function
      5. [Classical] Update parameters (β, γ)
    
    The quantum tasks have alternating structure (cost → mixer → measure).
    """
    tasks = []
    edges = []
    task_id = 0

    n_qubits = circuit['n_qubits']
    p = circuit.get('depth', 2) // 2  # QAOA p parameter

    for it in range(n_iterations):
        # Execute p layers of (cost, mixer) pairs
        prev_task_id = None
        for layer in range(p):
            # Cost Hamiltonian layer
            cost_task = {
                'id': f't{task_id}',
                'type': 'quantum',
                'name': f'qaoa_cost_iter{it}_layer{layer}',
                'circuit_id': circuit['id'],
                'n_qubits': n_qubits,
                'n_gates': n_qubits,
                'depth': 1,
                'estimated_time_ms': round(n_qubits * 0.1, 2),
                'qaoa_layer': 'cost',
                'layer_idx': layer,
            }
            tasks.append(cost_task)
            if prev_task_id is not None:
                edges.append({
                    'source': f't{prev_task_id}',
                    'target': f't{task_id}',
                    'type': 'quantum-quantum',
                    'data_volume_bits': n_qubits,
                })
            prev_task_id = task_id
            task_id += 1

            # Mixer Hamiltonian layer
            mixer_task = {
                'id': f't{task_id}',
                'type': 'quantum',
                'name': f'qaoa_mixer_iter{it}_layer{layer}',
                'circuit_id': circuit['id'],
                'n_qubits': n_qubits,
                'n_gates': n_qubits,
                'depth': 1,
                'estimated_time_ms': round(n_qubits * 0.05, 2),
                'qaoa_layer': 'mixer',
                'layer_idx': layer,
            }
            tasks.append(mixer_task)
            edges.append({
                'source': f't{prev_task_id}',
                'target': f't{task_id}',
                'type': 'quantum-quantum',
                'data_volume_bits': n_qubits,
            })
            prev_task_id = task_id
            task_id += 1

        # Measurement task
        measure_task = {
            'id': f't{task_id}',
            'type': 'quantum',
            'name': f'qaoa_measure_iter{it}',
            'circuit_id': circuit['id'],
            'n_qubits': n_qubits,
            'n_gates': n_qubits,
            'depth': 1,
            'estimated_time_ms': round(n_qubits * 0.05, 2),
        }
        tasks.append(measure_task)
        edges.append({
            'source': f't{prev_task_id}',
            'target': f't{task_id}',
            'type': 'quantum-quantum',
            'data_volume_bits': n_qubits,
        })
        measure_id = task_id
        task_id += 1

        # Classical: compute objective
        obj_task = {
            'id': f't{task_id}',
            'type': 'classical',
            'name': f'qaoa_objective_iter{it}',
            'workload_flops': n_qubits * 1e6,
            'memory_mb': 200,
            'estimated_time_ms': round(n_qubits * 0.1, 2),
        }
        tasks.append(obj_task)
        edges.append({
            'source': f't{measure_id}',
            'target': f't{task_id}',
            'type': 'quantum-classical',
            'data_volume_bits': n_qubits * 1000,
        })
        obj_id = task_id
        task_id += 1

        # Classical: update parameters
        update_task = {
            'id': f't{task_id}',
            'type': 'classical',
            'name': f'qaoa_update_iter{it}',
            'workload_flops': p * 1e7,
            'memory_mb': 100,
            'estimated_time_ms': round(p * 0.5, 2),
        }
        tasks.append(update_task)
        edges.append({
            'source': f't{obj_id}',
            'target': f't{task_id}',
            'type': 'classical-classical',
            'data_volume_bits': 64,
        })
        update_id = task_id
        task_id += 1

    # Add inter-iteration feedback edges
    tasks_per_iter = 2 * p + 3  # cost/mixer pairs + measure + objective + update
    for it in range(n_iterations - 1):
        update_id_this = (it + 1) * tasks_per_iter - 1
        cost_id_next = (it + 1) * tasks_per_iter
        edges.append({
            'source': f't{update_id_this}',
            'target': f't{cost_id_next}',
            'type': 'classical-quantum',
            'data_volume_bits': p * 64 * 2,  # β and γ parameters
        })

    n_q = sum(1 for t in tasks if t['type'] == 'quantum')
    n_c = sum(1 for t in tasks if t['type'] == 'classical')
    n_qc_edges = sum(1 for e in edges if e['type'] in ('quantum-classical', 'classical-quantum'))

    return {
        'id': f'qaoa_dag_{n_qubits}q_p{p}_{n_iterations}iter',
        'type': 'QAOA',
        'circuit_id': circuit['id'],
        'n_qubits': n_qubits,
        'p': p,
        'n_iterations': n_iterations,
        'n_tasks': len(tasks),
        'n_quantum_tasks': n_q,
        'n_classical_tasks': n_c,
        'n_edges': len(edges),
        'n_qc_feedback_edges': n_qc_edges,
        'tasks': tasks,
        'edges': edges,
        'description': f'QAOA DAG: {n_iterations} iterations, p={p}, {n_qubits} qubits',
    }


def _generate_qml_dag(circuit: dict, n_iterations: int) -> dict:
    """
    Generate a QML (Quantum Neural Network) workload DAG.
    
    Structure (per epoch):
      1. [Quantum] Forward pass: execute quantum circuit (ansatz + encoding)
      2. [Quantum] Measurement: read out predictions
      3. [Classical] Loss computation: compute loss from predictions
      4. [Classical] Backward pass: classical gradient computation
         (parameter-shift rule requires additional quantum executions)
      5. [Quantum] Parameter-shift evaluations (parallelized)
      6. [Classical] Gradient aggregation and parameter update
    """
    tasks = []
    edges = []
    task_id = 0

    n_qubits = circuit['n_qubits']
    n_params = circuit.get('n_training_params', n_qubits * 3)
    # Parameter-shift rule: 2 evaluations per parameter
    n_shift_evals = min(n_params * 2, 50)  # cap for tractability

    for it in range(n_iterations):
        # 1. Forward pass
        forward_task = {
            'id': f't{task_id}',
            'type': 'quantum',
            'name': f'qml_forward_iter{it}',
            'circuit_id': circuit['id'],
            'n_qubits': n_qubits,
            'n_gates': circuit.get('n_gates', n_qubits * 4),
            'depth': circuit.get('depth', 8),
            'estimated_time_ms': round(n_qubits * 0.3, 2),
        }
        tasks.append(forward_task)
        forward_id = task_id
        task_id += 1

        # 2. Measurement
        measure_task = {
            'id': f't{task_id}',
            'type': 'quantum',
            'name': f'qml_measure_iter{it}',
            'circuit_id': circuit['id'],
            'n_qubits': n_qubits,
            'n_gates': n_qubits,
            'depth': 1,
            'estimated_time_ms': round(n_qubits * 0.05, 2),
        }
        tasks.append(measure_task)
        edges.append({
            'source': f't{forward_id}',
            'target': f't{task_id}',
            'type': 'quantum-quantum',
            'data_volume_bits': n_qubits,
        })
        measure_id = task_id
        task_id += 1

        # 3. Loss computation (classical)
        loss_task = {
            'id': f't{task_id}',
            'type': 'classical',
            'name': f'qml_loss_iter{it}',
            'workload_flops': n_qubits * 1e6,
            'memory_mb': 100,
            'estimated_time_ms': round(n_qubits * 0.1, 2),
        }
        tasks.append(loss_task)
        edges.append({
            'source': f't{measure_id}',
            'target': f't{task_id}',
            'type': 'quantum-classical',
            'data_volume_bits': n_qubits * 1000,
        })
        loss_id = task_id
        task_id += 1

        # 4. Parameter-shift evaluations (quantum, parallelized)
        shift_task_ids = []
        for s in range(n_shift_evals):
            shift_task = {
                'id': f't{task_id}',
                'type': 'quantum',
                'name': f'qml_shift_iter{it}_eval{s}',
                'circuit_id': circuit['id'],
                'n_qubits': n_qubits,
                'n_gates': circuit.get('n_gates', n_qubits * 4),
                'depth': circuit.get('depth', 8),
                'estimated_time_ms': round(n_qubits * 0.3, 2),
                'shift_idx': s,
            }
            tasks.append(shift_task)
            shift_task_ids.append(task_id)
            # Dependency: loss → shift (need to know which params to shift)
            edges.append({
                'source': f't{loss_id}',
                'target': f't{task_id}',
                'type': 'classical-quantum',
                'data_volume_bits': n_params * 64,
            })
            task_id += 1

        # 5. Gradient aggregation and parameter update (classical)
        grad_task = {
            'id': f't{task_id}',
            'type': 'classical',
            'name': f'qml_gradient_update_iter{it}',
            'workload_flops': n_params * 1e7,
            'memory_mb': 500,
            'estimated_time_ms': round(n_params * 0.2, 2),
        }
        tasks.append(grad_task)
        for stid in shift_task_ids:
            edges.append({
                'source': f't{stid}',
                'target': f't{task_id}',
                'type': 'quantum-classical',
                'data_volume_bits': n_qubits * 1000,
            })
        edges.append({
            'source': f't{loss_id}',
            'target': f't{task_id}',
            'type': 'classical-classical',
            'data_volume_bits': 64,
        })
        grad_id = task_id
        task_id += 1

    # Inter-iteration feedback
    tasks_per_iter = 4 + n_shift_evals  # forward + measure + loss + shifts + gradient
    for it in range(n_iterations - 1):
        grad_id_this = (it + 1) * tasks_per_iter - 1
        forward_id_next = (it + 1) * tasks_per_iter
        edges.append({
            'source': f't{grad_id_this}',
            'target': f't{forward_id_next}',
            'type': 'classical-quantum',
            'data_volume_bits': n_params * 64,
        })

    n_q = sum(1 for t in tasks if t['type'] == 'quantum')
    n_c = sum(1 for t in tasks if t['type'] == 'classical')
    n_qc_edges = sum(1 for e in edges if e['type'] in ('quantum-classical', 'classical-quantum'))

    return {
        'id': f'qml_dag_{n_qubits}q_{n_iterations}iter',
        'type': 'QML',
        'circuit_id': circuit['id'],
        'n_qubits': n_qubits,
        'n_iterations': n_iterations,
        'n_training_params': n_params,
        'n_shift_evals': n_shift_evals,
        'n_tasks': len(tasks),
        'n_quantum_tasks': n_q,
        'n_classical_tasks': n_c,
        'n_edges': len(edges),
        'n_qc_feedback_edges': n_qc_edges,
        'tasks': tasks,
        'edges': edges,
        'description': f'QML DAG: {n_iterations} iterations, {n_qubits} qubits, '
                       f'{n_shift_evals} parameter-shift evaluations per iteration',
    }


# ============================================================================
# 6. Simulation Parameters
# ============================================================================

def generate_sim_params(output_dir: str) -> Dict:
    """
    Generate simulation parameters matching Section 3 of the paper.
    
    Parameters cover:
      - Quantum link parameters (λ, η, b) for each quantum topology edge
      - Classical link parameters (B, L) for each classical topology edge
      - Interface link parameters (ρ, R) for each QPU-classical connection
      - Fidelity parameters (F_prep, gate error rates, EPR fidelity, decoherence)
      - Algorithm parameters (weight values, beta, etc.)
    """
    param_dir = os.path.join(output_dir, 'sim_params')
    os.makedirs(param_dir, exist_ok=True)

    params = {
        'fidelity_parameters': {
            'F_prep': 0.999,          # State preparation fidelity (superconducting)
            'epsilon_1q': 1e-4,        # Single-qubit gate error rate
            'epsilon_2q_range': [1e-3, 1e-2],  # Two-qubit gate error rate range
            'epsilon_measure': 5e-3,   # Measurement error rate
            'F_EPR_range': [0.85, 0.95],  # Initial EPR fidelity range (photonic links)
            'F_min': 0.99,             # Minimum fidelity threshold
            'F_min_qaoa': 0.985,       # Relaxed threshold for QAOA (deeper circuits)
        },
        'quantum_link_parameters': {
            # Entanglement generation rate λ (EPR pairs/second)
            # Based on photonic link experiments: 100 Hz to 10 kHz
            'lambda_range_hz': [100, 10000],
            'lambda_typical_hz': 1000,
            
            # Memory decoherence rate η (per millisecond)
            # Based on NV center / superconducting memory experiments
            'eta_range_per_ms': [0.01, 0.1],
            'eta_typical_per_ms': 0.05,
            
            # Maximum concurrent EPR pairs (buffer capacity)
            'b_range': [4, 32],
            'b_typical': 16,
            
            # Classical signal propagation speed (fiber)
            'v_signal_ms': 2e8,  # m/s
        },
        'classical_link_parameters': {
            # Bandwidth (Gbps)
            'B_range_gbps': [10, 200],
            'B_host_gbps': 10,
            'B_edge_gbps': 40,
            'B_agg_gbps': 100,
            'B_core_gbps': 200,
            
            # Latency (microseconds)
            'L_range_us': [0.5, 5],
            'L_host_us': 0.5,
            'L_edge_us': 1,
            'L_agg_us': 2,
            'L_core_us': 2,
        },
        'interface_link_parameters': {
            # Readout latency (microseconds)
            'rho_range_us': [1, 10],
            'rho_typical_us': 5,
            
            # Control bandwidth (Gbps)
            'R_range_gbps': [1, 25],
            'R_typical_gbps': 10,
        },
        'qpu_parameters': {
            # Coherence time (microseconds) - IBM range
            'T_coh_range_us': [50, 300],
            'T_coh_typical_us': 100,
            
            # Gate execution times (nanoseconds)
            'tau_1q_ns': 35.6,          # Single-qubit gate
            'tau_2q_ns': 334.0,         # Two-qubit gate (CX)
            'tau_measure_ns': 1200.0,   # Measurement
            'tau_Bell_ns': 100.0,       # Bell measurement for teleportation
            'tau_swap_ns': 200.0,       # Entanglement swapping
            
            # Qubit counts per QPU (matching paper Table I)
            'n_qubits_small': 50,
            'n_qubits_medium': 100,
            'n_qubits_large': 127,
        },
        'classical_node_parameters': {
            # Compute speed (TFLOPS)
            'p_cpu_tflops': [0.5, 3.0],
            'p_gpu_tflops': [10, 80],
            'p_tpu_tflops': [50, 200],
            
            # Memory (GB)
            'm_cpu_gb': [64, 256],
            'm_gpu_gb': [32, 80],
            'm_tpu_gb': [16, 32],
        },
        'algorithm_parameters': {
            # Topology score weights (Section 4.2)
            'w1': 0.5,   # Quantum centrality weight
            'w2': 0.3,   # Classical access latency weight
            'w3': 0.2,   # Qubit count weight
            
            # Partitioning sensitivity (Section 4.3)
            'beta': 0.5,
            
            # Gate scheduling priority weights (Section 4.4)
            'mu1': 0.6,  # Critical path length weight
            'mu2': 0.4,  # Entanglement urgency weight
            
            # Coherence safety margin (Section 4.4)
            'T_safety_fraction': 0.1,  # 10% of T_coh
        },
        'evaluation_parameters': {
            'n_repetitions': 30,        # Number of runs per experiment
            'random_seed_base': 42,
            'confidence_level': 0.95,
        },
        'baselines': {
            'Random': {'description': 'Random task placement without topology awareness'},
            'Greedy': {'description': 'Greedy load balancing without topology awareness'},
            'Q-Only': {'description': 'Optimize quantum scheduling only, ignoring classical placement'},
            'DQTetris': {'description': 'State-of-the-art circuit mapping (Xu et al. SC 2025)',
                         'reference': '[7] in paper'},
        },
    }

    with open(os.path.join(param_dir, 'sim_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  [OK] sim_params.json: {len(params)} parameter groups")

    return params


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TopoQ Dataset Generator')
    parser.add_argument('--output-dir', default='./dataset',
                        help='Output directory for dataset files')
    parser.add_argument(
        '--ibm-snapshot-dir', default=None,
        help='Directory containing conf_*.json and props_*.json IBM snapshots'
    )
    parser.add_argument('--mqt-bench', default=None,
                        help='Path to MQT Bench circuits directory (optional)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    print("=" * 70)
    print("TopoQ Dataset Generator")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Step 1: Extract IBM quantum chip topologies
    print("[1/6] Extracting IBM quantum chip topologies...")
    chips = extract_ibm_chips(args.output_dir, args.ibm_snapshot_dir)
    print()

    # Step 2: Generate classical network topologies
    print("[2/6] Generating classical network topologies...")
    networks = generate_network_topologies(args.output_dir)
    print()

    # Step 3: Generate hybrid system configurations
    print("[3/6] Generating hybrid system configurations...")
    configs = generate_system_configs(chips, networks, args.output_dir)
    print()

    # Step 4: Generate quantum circuit benchmarks
    print("[4/6] Generating quantum circuit benchmarks...")
    circuits = generate_quantum_circuits(args.output_dir, args.mqt_bench)
    print()

    # Step 5: Generate hybrid workload DAGs
    print("[5/6] Generating hybrid workload DAGs...")
    dags = generate_workload_dags(circuits, args.output_dir, args.seed)
    print()

    # Step 6: Generate simulation parameters
    print("[6/6] Generating simulation parameters...")
    sim_params = generate_sim_params(args.output_dir)
    print()

    # Write master index
    print("\nWriting master index...")
    master_index = {
        'description': 'TopoQ Experimental Dataset',
        'paper': 'TopoQ: Topology-Aware Scheduling for Hybrid Quantum-Classical Computing Systems',
        'venue': 'IEEE INFOCOM',
        'generation_date': str(np.datetime64('today')),
        'random_seed': args.seed,
        'components': {
            'ibm_chips': {
                'description': 'IBM quantum chip topologies from Qiskit fake providers',
                'count': len(chips),
                'chips': list(chips.keys()),
                'directory': 'ibm_chips/',
            },
            'network_topologies': {
                'description': 'Classical network topologies (Fat-Tree, Dragonfly, Dragonfly+)',
                'count': len(networks),
                'topologies': list(networks.keys()),
                'directory': 'network_topologies/',
            },
            'system_configs': {
                'description': 'Hybrid system configurations (Small, Medium, Large)',
                'count': len(configs),
                'configs': list(configs.keys()),
                'directory': 'system_configs/',
            },
            'quantum_circuits': {
                'description': 'Quantum circuit benchmarks (VQE, QAOA, QML, QFT, GHZ)',
                'count': len(circuits),
                'circuit_types': list(set(c['type'] for c in circuits.values())),
                'directory': 'quantum_circuits/',
            },
            'workload_dags': {
                'description': 'Hybrid quantum-classical workload DAGs',
                'count': len(dags),
                'dag_types': list(set(d['type'] for d in dags.values())),
                'directory': 'workload_dags/',
            },
            'sim_params': {
                'description': 'Simulation parameters matching Section 3 of the paper',
                'directory': 'sim_params/',
            },
        },
        'usage': {
            'compile': 'pdflatex TopoQ_INFOCOM && bibtex TopoQ_INFOCOM && pdflatex TopoQ_INFOCOM',
            'dataset_generation': 'python generate_dataset.py --output-dir ./dataset',
            'with_mqt_bench': 'python generate_dataset.py --output-dir ./dataset --mqt-bench /path/to/mqt-bench',
        },
    }

    with open(os.path.join(args.output_dir, 'index.json'), 'w') as f:
        json.dump(master_index, f, indent=2)

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print(f"  Total chips:       {len(chips)}")
    print(f"  Total networks:    {len(networks)}")
    print(f"  Total configs:     {len(configs)}")
    print(f"  Total circuits:    {len(circuits)}")
    print(f"  Total DAGs:        {len(dags)}")
    print(f"  Output directory:  {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
