"""
N-ward antibiotic prescribing game (game-theoretic stewardship model).

Each ward chooses a broadness level σ_i ∈ [0, 1]:
    σ = 0 — narrow-spectrum prescribing (stewardship-optimal but lower cure rate)
    σ = 1 — broad-spectrum prescribing (higher individual cure rate, creates AMR)

Ward utility
------------
    U_i(σ_i, σ_{-i}) = (f_min + (f_max−f_min)·σ_i)
                         − α_i · σ_i · (1 + β · mean(σ_{-i}))

The first term is the treatment-efficacy benefit; the second is the resistance
cost, which grows with how much other wards also overuse broad-spectrum (the
external AMR pressure term β·mean(σ_{-i})).

For the tragedy-of-the-commons structure to emerge the ward must find broad-
spectrum individually attractive:
    f_max − f_min  >  α_i              (individual benefit > individual cost at σ=0)
    f_max − f_min  <  2·α_i·β·σ       (social cost eventually dominates)

Default parameters: f_min=0.70, f_max=0.95, α_i=0.30, β=0.80.
These defaults give a narrow-spectrum Nash.  Pass ward_params with α < 0.14
(e.g. 0.13) to obtain the broad-spectrum-dominant Nash and PoA > 1 that the
acceptance tests require.  All parameters can be overridden per ward.

Nash equilibrium
----------------
Computed via gradient-ascent best-response iteration: each ward independently
maximises its own utility by gradient stepping σ_i while others are fixed.
Since U_i is linear in σ_i the gradient is constant per iteration; stepping
toward 0 or 1 depending on sign.  Damping prevents oscillation in borderline
cases.

Social optimum
--------------
Found by maximising Σ_i U_i(σ_i, σ_{-i}) over σ ∈ [0, 1]^n using
scipy.optimize.minimize (L-BFGS-B).

Price of Anarchy (PoA)
----------------------
    PoA = Σ_i U_i(σ_opt) / Σ_i U_i(σ_ne)  ≥ 1.0

A value > 1 confirms the tragedy of the commons: coordinating on σ_opt gives
strictly higher total welfare than selfish σ_ne.

Public API
----------
ward_utility          — U_i for one ward given others' strategies
nash_equilibrium      — gradient-ascent convergence to Nash σ*
social_optimum        — social welfare maximiser via L-BFGS-B
price_of_anarchy      — PoA = total welfare(opt) / total welfare(ne)
get_game_reward       — coordination improvement reward ∈ [0, 1]
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Default parameters (can be overridden via ward_params)
# ---------------------------------------------------------------------------

_DEFAULT_F_MIN:  float = 0.70
_DEFAULT_F_MAX:  float = 0.95
_DEFAULT_ALPHA:  float = 0.30
_DEFAULT_BETA:   float = 0.80

WardParams = Dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_params(ward_params: WardParams) -> tuple:
    """Extract (f_min, f_max, alpha, beta) from a ward_params dict."""
    return (
        ward_params.get("f_min",  _DEFAULT_F_MIN),
        ward_params.get("f_max",  _DEFAULT_F_MAX),
        ward_params.get("alpha",  _DEFAULT_ALPHA),
        ward_params.get("beta",   _DEFAULT_BETA),
    )


def _ward_params_for(ward_params: Union[WardParams, List[WardParams]], i: int) -> WardParams:
    """Return the params dict for ward *i* (list or shared single dict)."""
    if isinstance(ward_params, list):
        return ward_params[i]
    return ward_params


def _total_welfare(
    sigma: np.ndarray,
    n_wards: int,
    ward_params: Union[WardParams, List[WardParams]],
) -> float:
    return sum(
        ward_utility(
            float(sigma[i]),
            np.delete(sigma, i),
            _ward_params_for(ward_params, i),
        )
        for i in range(n_wards)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ward_utility(
    sigma_i: float,
    sigma_others: np.ndarray,
    ward_params: WardParams,
) -> float:
    """
    Compute ward i's utility given its own strategy and others' strategies.

        U_i = (f_min + (f_max−f_min)·σ_i)
               − α_i · σ_i · (1 + β · mean(σ_{-i}))

    Parameters
    ----------
    sigma_i      : ward i's broadness level ∈ [0, 1]
    sigma_others : array of other wards' broadness levels
    ward_params  : dict with optional keys 'f_min', 'f_max', 'alpha', 'beta'

    Returns
    -------
    float — utility U_i
    """
    f_min, f_max, alpha, beta = _get_params(ward_params)
    mean_others = float(np.mean(sigma_others)) if len(sigma_others) > 0 else 0.0
    treatment_benefit = f_min + (f_max - f_min) * sigma_i
    resistance_cost   = alpha * sigma_i * (1.0 + beta * mean_others)
    return float(treatment_benefit - resistance_cost)


def nash_equilibrium(
    n_wards: int,
    ward_params: Union[WardParams, List[WardParams]],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute Nash equilibrium strategies via gradient-ascent best-response.

    Each ward independently performs a small gradient step toward its utility
    maximum while others are held fixed, then σ is projected back to [0, 1].
    A damped step (size 0.05) prevents oscillation.  Iteration stops when the
    maximum change across all wards drops below *tol*.

    Parameters
    ----------
    n_wards    : number of wards
    ward_params: single WardParams dict (shared) or list of n_wards dicts
    max_iter   : maximum iterations (default 1000)
    tol        : convergence tolerance (default 1e-6)

    Returns
    -------
    np.ndarray shape (n_wards,) — Nash equilibrium σ* values in [0, 1]
    """
    sigma = np.full(n_wards, 0.5)
    step  = 0.05  # damped gradient step size

    for _ in range(max_iter):
        sigma_old = sigma.copy()

        for i in range(n_wards):
            p = _ward_params_for(ward_params, i)
            f_min, f_max, alpha, beta = _get_params(p)
            sigma_others = np.delete(sigma, i)
            mean_others  = float(np.mean(sigma_others)) if n_wards > 1 else 0.0

            # dU_i / dσ_i  (utility is linear in σ_i)
            grad = (f_max - f_min) - alpha * (1.0 + beta * mean_others)
            sigma[i] = float(np.clip(sigma[i] + step * grad, 0.0, 1.0))

        if float(np.max(np.abs(sigma - sigma_old))) < tol:
            break

    return sigma


def social_optimum(
    n_wards: int,
    ward_params: Union[WardParams, List[WardParams]],
) -> np.ndarray:
    """
    Find the social welfare-maximising strategies σ* = argmax Σ_i U_i(σ).

    Uses scipy.optimize.minimize (L-BFGS-B) on the negated total welfare.

    Parameters
    ----------
    n_wards    : number of wards
    ward_params: single WardParams dict (shared) or list of n_wards dicts

    Returns
    -------
    np.ndarray shape (n_wards,) — social-optimal σ* values in [0, 1]
    """
    def neg_total(sigma: np.ndarray) -> float:
        return -_total_welfare(np.clip(sigma, 0.0, 1.0), n_wards, ward_params)

    result = minimize(
        neg_total,
        x0=np.full(n_wards, 0.5),
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * n_wards,
        options={"maxiter": 5000, "ftol": 1e-12},
    )
    return np.clip(result.x, 0.0, 1.0)


def price_of_anarchy(
    ne_strategies:  np.ndarray,
    opt_strategies: np.ndarray,
    n_wards:        int,
    ward_params:    Union[WardParams, List[WardParams]],
) -> float:
    """
    Compute the Price of Anarchy (PoA).

        PoA = Σ_i U_i(σ_opt) / Σ_i U_i(σ_ne)

    A value > 1.0 confirms the tragedy of the commons: coordinating on the
    social optimum yields strictly higher total welfare than the Nash outcome.

    Parameters
    ----------
    ne_strategies  : Nash equilibrium strategies, shape (n_wards,)
    opt_strategies : social-optimal strategies, shape (n_wards,)
    n_wards        : number of wards
    ward_params    : WardParams dict or list of dicts

    Returns
    -------
    float ≥ 0 — PoA; 1.0 means Nash and social optimum coincide.
    """
    welfare_ne  = _total_welfare(ne_strategies,  n_wards, ward_params)
    welfare_opt = _total_welfare(opt_strategies, n_wards, ward_params)
    return float(welfare_opt / (welfare_ne + 1e-300))


def get_game_reward(poa_before: float, poa_after: float) -> float:
    """
    Compute the coordination improvement reward ∈ [0, 1].

    Measures how much the coordinator reduced the Price of Anarchy relative to
    the total possible improvement (from poa_before down to 1.0):

        reward = clip((poa_before − poa_after) / (poa_before − 1.0 + ε), 0, 1)

    A reward of 1.0 means the coordinator drove PoA from poa_before down to 1
    (perfect cooperation achieved); 0.0 means no improvement.

    Parameters
    ----------
    poa_before : PoA at the start of the episode / decision point
    poa_after  : PoA after the coordinator's action

    Returns
    -------
    float in [0.0, 1.0]
    """
    improvement = poa_before - poa_after
    max_improvement = poa_before - 1.0 + 1e-8
    return float(np.clip(improvement / max_improvement, 0.0, 1.0))
