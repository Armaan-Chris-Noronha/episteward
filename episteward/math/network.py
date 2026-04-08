"""
Contact-graph network epidemiology for inter-ward/hospital transmission.

The hospital network is modelled as a weighted directed graph loaded once from
``episteward/data/hospital_network.json`` at first use (module-level singleton).

Transmission probability per edge per 24-h step:
    P(i→j) = w(i,j) * β_pathogen * isolation_factor
    isolation_factor = 0.1 if isolation_active else 1.0

Public API
----------
compute_transmission_probability(source_ward, target_ward,
    pathogen_name, isolation_active) -> float
get_at_risk_wards(infected_wards, graph)           -> List[str]
simulate_spread_step(infected_set, pathogen_name,
    isolation_map, rng)                            -> set[str]
get_transmission_chain(transfer_logs,
    culture_results)                               -> List[str]
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA: Dict[str, float] = {
    "ESBL": 0.15,
    "CRK":  0.08,
    "MRSA": 0.12,
    "VRE":  0.10,
    "MDR":  0.09,
    "default": 0.10,
}

_ISOLATION_FACTOR = 0.1   # 90% reduction when source ward is isolated
_DATA_PATH = Path(__file__).parent.parent / "data" / "hospital_network.json"


# ---------------------------------------------------------------------------
# Graph singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph:
    """Load and cache the hospital network graph from disk."""
    data = json.loads(_DATA_PATH.read_text())
    return build_graph(data)


# ---------------------------------------------------------------------------
# Graph constructor
# ---------------------------------------------------------------------------

def build_graph(hospital_network: Dict[str, Any]) -> nx.DiGraph:
    """
    Construct a networkx DiGraph from a hospital_network dict.

    Nodes may be plain strings or dicts with an ``id`` key plus optional
    attributes (``ward_capacity``, ``isolation_beds``, ``average_los_days``).
    Beta values in ``hospital_network["beta_values"]`` override module BETA
    and are stored on ``G.graph["beta_values"]``.
    """
    G = nx.DiGraph()
    G.graph["beta_values"] = {**BETA, **hospital_network.get("beta_values", {})}

    for node in hospital_network["nodes"]:
        if isinstance(node, dict):
            node_id = node["id"]
            attrs = {k: v for k, v in node.items() if k not in ("id", "_note")}
            G.add_node(node_id, **attrs)
        else:
            G.add_node(node)

    for edge in hospital_network["edges"]:
        G.add_edge(edge["from"], edge["to"], weight=edge["weight"])

    return G


# ---------------------------------------------------------------------------
# Public API — new functions
# ---------------------------------------------------------------------------

def compute_transmission_probability(
    source_ward: str,
    target_ward: str,
    pathogen_name: str,
    isolation_active: bool = False,
    graph: nx.DiGraph | None = None,
) -> float:
    """
    Return the per-step transmission probability from source to target ward.

    Formula:
        P = w(source→target) × β_pathogen × isolation_factor

        isolation_factor = 0.1 if isolation_active else 1.0
        (isolation on the *source* ward reduces outbound spread by 90%)

    Parameters
    ----------
    source_ward      : originating ward ID
    target_ward      : receiving ward ID
    pathogen_name    : key for beta lookup (e.g. "ESBL", "CRK", "MRSA")
    isolation_active : True when source ward has active contact precautions
    graph            : DiGraph to query; defaults to module-level singleton

    Returns
    -------
    float in [0, 1], or 0.0 if no edge exists
    """
    G = graph if graph is not None else _load_graph()

    if not G.has_edge(source_ward, target_ward):
        return 0.0

    w = G[source_ward][target_ward]["weight"]
    beta_map = G.graph.get("beta_values", BETA)
    beta = beta_map.get(pathogen_name, beta_map.get("default", BETA["default"]))
    isolation_factor = _ISOLATION_FACTOR if isolation_active else 1.0

    return float(np.clip(w * beta * isolation_factor, 0.0, 1.0))


def get_at_risk_wards(
    infected_wards: Set[str] | List[str],
    graph: nx.DiGraph | None = None,
) -> List[str]:
    """
    Return all wards reachable (one hop) from any ward in *infected_wards*.

    Excludes wards that are already infected.

    Parameters
    ----------
    infected_wards : set or list of ward IDs currently carrying infection
    graph          : DiGraph; defaults to module-level singleton

    Returns
    -------
    Sorted list of unique at-risk ward IDs
    """
    G = graph if graph is not None else _load_graph()
    infected = set(infected_wards)
    at_risk: Set[str] = set()

    for ward in infected:
        if ward not in G:
            continue
        for _, neighbour in G.out_edges(ward):
            if neighbour not in infected:
                at_risk.add(neighbour)

    return sorted(at_risk)


def simulate_spread_step(
    infected_set: Set[str],
    pathogen_name: str,
    isolation_map: Dict[str, bool],
    rng: np.random.Generator,
    graph: nx.DiGraph | None = None,
) -> Set[str]:
    """
    Simulate one 24-h transmission step; return the set of *newly* infected wards.

    Each infected source ward independently attempts to transmit to each
    out-neighbour. The event is a Bernoulli draw with probability
    ``compute_transmission_probability``.

    Parameters
    ----------
    infected_set   : ward IDs currently carrying infection
    pathogen_name  : determines beta value
    isolation_map  : ward_id → bool (True = isolation active on that ward)
    rng            : seeded numpy Generator (no global state)
    graph          : DiGraph; defaults to module-level singleton

    Returns
    -------
    set[str] of ward IDs newly infected this step
    (does not include wards already in *infected_set*)
    """
    G = graph if graph is not None else _load_graph()
    newly_infected: Set[str] = set()

    for source in infected_set:
        if source not in G:
            continue
        source_isolated = isolation_map.get(source, False)
        for _, target in G.out_edges(source):
            if target in infected_set or target in newly_infected:
                continue
            p = compute_transmission_probability(
                source, target, pathogen_name,
                isolation_active=source_isolated,
                graph=G,
            )
            if p > 0 and rng.random() < p:
                newly_infected.add(target)

    return newly_infected


def get_transmission_chain(
    transfer_logs: List[Dict[str, Any]],
    culture_results: Dict[str, Any],
) -> List[str]:
    """
    Reconstruct the transmission chain (source → … → current) from clinical data.

    Algorithm
    ---------
    1. Find the index patient: earliest positive culture time.
    2. Trace their ward transfers forward in chronological order.
    3. Return the ordered list of wards visited from that patient's first
       positive culture onwards.

    Parameters
    ----------
    transfer_logs : list of dicts, each with keys:
        ``patient_id``, ``from_ward``, ``to_ward``, ``timestamp`` (ISO str)
    culture_results : dict mapping patient_id → dict with keys:
        ``result`` ("positive" | "negative"), ``timestamp`` (ISO str)

    Returns
    -------
    List[str] of ward IDs in chronological transmission order.
    Empty list if no positive cultures are present.
    """
    # Find index patient — earliest positive culture
    positive_patients = {
        pid: info
        for pid, info in culture_results.items()
        if info.get("result") == "positive"
    }
    if not positive_patients:
        return []

    index_pid = min(
        positive_patients,
        key=lambda pid: positive_patients[pid].get("timestamp", ""),
    )
    index_time = positive_patients[index_pid].get("timestamp", "")

    # Trace ward path of index patient from culture time onward
    patient_transfers = [
        log for log in transfer_logs
        if log.get("patient_id") == index_pid
        and log.get("timestamp", "") >= index_time
    ]
    patient_transfers.sort(key=lambda x: x.get("timestamp", ""))

    chain: List[str] = []
    for log in patient_transfers:
        if not chain:
            from_ward = log.get("from_ward", "")
            if from_ward:
                chain.append(from_ward)
        to_ward = log.get("to_ward", "")
        if to_ward and (not chain or chain[-1] != to_ward):
            chain.append(to_ward)

    return chain


# ---------------------------------------------------------------------------
# Backward-compatible helpers
# ---------------------------------------------------------------------------

def transmission_probability(
    graph: nx.DiGraph,
    source_ward: str,
    target_ward: str,
    infected_count: int,
    pathogen_type: str = "ESBL",
    immune: bool = False,
) -> float:
    """
    Legacy: P = β × w × infected_count, with immune flag.

    Kept for backward compatibility. Prefer ``compute_transmission_probability``.
    """
    if not graph.has_edge(source_ward, target_ward):
        return 0.0
    if immune:
        return 0.0

    w = graph[source_ward][target_ward]["weight"]
    beta_map = graph.graph.get("beta_values", BETA)
    beta = beta_map.get(pathogen_type, beta_map.get("default", BETA["default"]))
    return float(np.clip(beta * w * infected_count, 0.0, 1.0))


def simulate_spread(
    graph: nx.DiGraph,
    infected_wards: Dict[str, int],
    pathogen_type: str,
    isolated_wards: Set[str],
    rng: np.random.Generator,
) -> Dict[str, int]:
    """
    Legacy spread simulation returning delta dict.

    Kept for backward compatibility. Prefer ``simulate_spread_step``.
    """
    new_infections: Dict[str, int] = {}

    for source, count in infected_wards.items():
        if source in isolated_wards or count == 0:
            continue
        for _, target, data in graph.out_edges(source, data=True):
            immune = target in isolated_wards
            p = transmission_probability(graph, source, target, count, pathogen_type, immune)
            if p > 0:
                events = int(rng.binomial(count, p))
                if events > 0:
                    new_infections[target] = new_infections.get(target, 0) + events

    return new_infections


def shortest_transmission_path(
    graph: nx.DiGraph,
    source: str,
    target: str,
) -> List[str]:
    """Return the highest-weight (most likely) path between two wards."""
    try:
        inv_graph = nx.DiGraph()
        for u, v, data in graph.edges(data=True):
            w = data.get("weight", 0.0)
            inv_graph.add_edge(u, v, weight=1.0 / (w + 1e-9))
        return nx.shortest_path(inv_graph, source, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
