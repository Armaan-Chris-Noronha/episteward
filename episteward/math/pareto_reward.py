"""
Multi-objective Pareto reward module.

Models antibiotic stewardship as a four-objective optimisation problem:

    objectives = [clinical, ecology, economics, stewardship]

all in [0, 1].  Adaptive weights shift as ward-level antimicrobial resistance
(captured by WRPI) worsens: clinical weight decreases and ecology weight
increases to penalise further resistance selection.

Ward Resistance Pressure Index (WRPI)
--------------------------------------
    WRPI = Σ_k prev_R(k) · severity_weight(k)  /  Σ_k severity_weight(k)

Adaptive weights (before softmax normalisation)
-----------------------------------------------
    λ_1 ← λ_1_base · (1 − 0.3 · WRPI)   — clinical drops in AMR crisis
    λ_2 ← λ_2_base · (1 + 0.6 · WRPI)   — ecology rises with resistance
    λ_3, λ_4 unchanged.

A softmax is then applied so the vector sums to 1.

Pareto front
------------
A point is *non-dominated* if no other known point weakly dominates it in all
objectives and strictly dominates it in at least one.  The Pareto front is the
set of all non-dominated points.

Hypervolume indicator
---------------------
Computed via a recursive dimension-slicing algorithm (HSO-style):

    HV_d(S, ref) = ∫_{ref[d-1]}^{∞} HV_{d-1}(S|_z, ref[d-1:]) dz

where S|_z = {p[:d-1] : p ∈ S, p[d-1] ≥ z}.  Runs in O(n^d) time;
acceptable for fronts < 50 points, d = 4.

Public API
----------
compute_reward_vector   — package four scalars into an ndarray shape (4,)
compute_wrpi            — Ward Resistance Pressure Index ∈ [0, 1]
compute_adaptive_weights — softmax-normalised weights shifted by WRPI
scalarize               — weighted sum of a reward vector
is_dominated            — check if a point is dominated by any point in a front
update_pareto_front     — add a point to the front, pruning dominated solutions
compute_hypervolume     — hypervolume indicator of the Pareto front
"""

from __future__ import annotations

from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BASE_WEIGHTS: np.ndarray = np.array([0.40, 0.30, 0.15, 0.15])
_WRPI_CLINICAL_DECAY: float  = 0.30   # λ_1 multiplied by (1 - 0.3·WRPI)
_WRPI_ECOLOGY_BOOST:  float  = 0.60   # λ_2 multiplied by (1 + 0.6·WRPI)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_reward_vector(
    clinical:    float,
    ecology:     float,
    economics:   float,
    stewardship: float,
) -> np.ndarray:
    """
    Package four scalar objectives into a reward vector, clamping to [0, 1].

    Parameters
    ----------
    clinical    : treatment success / clinical outcome score ∈ [0, 1]
    ecology     : ecological footprint / resistance pressure penalty ∈ [0, 1]
    economics   : cost-effectiveness / resource use score ∈ [0, 1]
    stewardship : guideline adherence / de-escalation score ∈ [0, 1]

    Returns
    -------
    np.ndarray shape (4,), all values in [0.0, 1.0]
    """
    vec = np.array([clinical, ecology, economics, stewardship], dtype=float)
    return np.clip(vec, 0.0, 1.0)


def compute_wrpi(
    ward_resistance_prevalences: dict,
    severity_weights: dict,
) -> float:
    """
    Compute the Ward Resistance Pressure Index (WRPI) ∈ [0, 1].

        WRPI = Σ_k prev_R(k) · severity_weight(k)  /  Σ_k severity_weight(k)

    The denominator normalises so that WRPI = 1 only when every pathogen has
    100 % resistance prevalence at maximum severity weight.

    Parameters
    ----------
    ward_resistance_prevalences : {pathogen: resistance_prevalence ∈ [0, 1]}
    severity_weights            : {pathogen: clinical_severity_weight ≥ 0}

    Returns
    -------
    float in [0.0, 1.0]; 0.0 if severity_weights is empty.
    """
    weight_total = sum(severity_weights.values())
    if weight_total < 1e-300:
        return 0.0

    numerator = sum(
        ward_resistance_prevalences.get(k, 0.0) * w
        for k, w in severity_weights.items()
    )
    return float(np.clip(numerator / weight_total, 0.0, 1.0))


def compute_adaptive_weights(
    wrpi: float,
    base_weights: np.ndarray = _DEFAULT_BASE_WEIGHTS,
) -> np.ndarray:
    """
    Compute WRPI-adaptive objective weights via softmax normalisation.

    Adjustment rules:
        λ_clinical ← λ_clinical_base · (1 − 0.3 · WRPI)
        λ_ecology  ← λ_ecology_base  · (1 + 0.6 · WRPI)
        λ_economics, λ_stewardship unchanged.

    The adjusted values are then passed through softmax so the output sums to 1.

    Parameters
    ----------
    wrpi        : Ward Resistance Pressure Index ∈ [0, 1]
    base_weights: shape (4,) base weights [clinical, ecology, economics, stewardship];
                  defaults to [0.40, 0.30, 0.15, 0.15]

    Returns
    -------
    np.ndarray shape (4,), summing to 1.0
    """
    wrpi = float(np.clip(wrpi, 0.0, 1.0))
    w = np.array(base_weights, dtype=float).copy()
    w[0] *= (1.0 - _WRPI_CLINICAL_DECAY * wrpi)   # clinical drops
    w[1] *= (1.0 + _WRPI_ECOLOGY_BOOST  * wrpi)   # ecology rises
    # Softmax
    w_exp = np.exp(w - w.max())                    # numerically stable
    return w_exp / w_exp.sum()


def scalarize(reward_vector: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted-sum scalarisation of a reward vector.

        scalar = clip(reward_vector · weights, 0, 1)

    Parameters
    ----------
    reward_vector : shape (4,), objectives in [0, 1]
    weights       : shape (4,), non-negative; need not sum to 1

    Returns
    -------
    float in [0.0, 1.0]
    """
    return float(np.clip(float(np.dot(reward_vector, weights)), 0.0, 1.0))


def is_dominated(point: np.ndarray, front: List[np.ndarray]) -> bool:
    """
    Return True if *point* is (weakly) dominated by at least one point in *front*.

    Point q dominates p iff:
        q[i] ≥ p[i] for all i  AND  q[j] > p[j] for some j.

    Parameters
    ----------
    point : shape (4,)
    front : list of shape-(4,) arrays (the current Pareto front)

    Returns
    -------
    bool
    """
    p = np.asarray(point)
    for q in front:
        q = np.asarray(q)
        if np.all(q >= p) and np.any(q > p):
            return True
    return False


def update_pareto_front(
    front: List[np.ndarray],
    new_point: np.ndarray,
) -> List[np.ndarray]:
    """
    Add *new_point* to *front* if it is non-dominated, removing any existing
    points that the new point dominates.

    Parameters
    ----------
    front     : current Pareto front (list of arrays)
    new_point : candidate point, shape (4,)

    Returns
    -------
    Updated Pareto front (new list; original is not modified).
    """
    p = np.asarray(new_point)

    # Reject new point if it is dominated by any existing front member
    if is_dominated(p, front):
        return list(front)

    # Keep only existing points NOT dominated by the new point
    surviving = [q for q in front if not is_dominated(np.asarray(q), [p])]
    return surviving + [p]


# ---------------------------------------------------------------------------
# Hypervolume — recursive dimension-slicing
# ---------------------------------------------------------------------------


def _non_dominated_nd(pts: list) -> list:
    """Return the non-dominated subset of *pts* (all dimensions, maximisation)."""
    result: list = []
    for i, p in enumerate(pts):
        if not any(
            i != j
            and np.all(pts[j] >= p)
            and np.any(pts[j] > p)
            for j in range(len(pts))
        ):
            result.append(p)
    return result


def _hv_1d(pts: list, ref: np.ndarray) -> float:
    vals = [float(p[0]) for p in pts if float(p[0]) > float(ref[0])]
    return max(vals) - float(ref[0]) if vals else 0.0


def _hv_recursive(pts: list, ref: np.ndarray) -> float:
    """
    Recursive hypervolume via dimension slicing (HSO-style).

    Slice on the LAST dimension:
        HV_d(S, ref) = Σ_i HV_{d-1}(nd(active_i), ref[:d-1]) · (z_i − z_{i+1})

    where active_i = {p[:d-1] : p ∈ S_sorted[:i+1]} and z_{last+1} = ref[d-1].
    Points not strictly dominating ref in every dimension are filtered first.
    """
    # Filter: keep only points strictly dominating ref in all dims
    pts = [np.asarray(p) for p in pts if np.all(np.asarray(p) > ref)]
    if not pts:
        return 0.0

    d = len(ref)
    if d == 1:
        return _hv_1d(pts, ref)

    # Sort by last coordinate DESCENDING
    pts = sorted(pts, key=lambda p: float(p[-1]), reverse=True)

    vol = 0.0
    active: list = []

    for i, p in enumerate(pts):
        active.append(p)

        # Next z boundary (or reference boundary)
        z_next = float(pts[i + 1][-1]) if i + 1 < len(pts) else float(ref[-1])
        delta_z = float(p[-1]) - z_next

        if delta_z <= 0.0:
            continue  # Degenerate slab (tied last coordinate)

        # (d-1)-D non-dominated projection of current active set
        proj = [q[:-1] for q in active]
        nd_proj = _non_dominated_nd(proj)
        vol += _hv_recursive(nd_proj, ref[:-1]) * delta_z

    return float(vol)


def compute_hypervolume(
    front: List[np.ndarray],
    reference: np.ndarray = None,
) -> float:
    """
    Compute the hypervolume indicator of a Pareto front relative to a reference.

    Uses a recursive dimension-slicing algorithm (O(n^d) in the worst case).
    Practical for fronts with < 50 points in 4 dimensions.

    Parameters
    ----------
    front     : list of shape-(4,) arrays representing non-dominated solutions
    reference : reference point, shape (4,); defaults to zeros(4)

    Returns
    -------
    float ≥ 0 — hypervolume dominated by *front* above *reference*
    """
    if not front:
        return 0.0
    if reference is None:
        reference = np.zeros(4)

    ref = np.asarray(reference, dtype=float)
    pts = [np.asarray(p, dtype=float) for p in front]
    return _hv_recursive(pts, ref)
