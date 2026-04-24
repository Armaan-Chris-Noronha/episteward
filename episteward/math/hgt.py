"""
Plasmid-mediated Horizontal Gene Transfer (HGT) model.

Models the spread of antibiotic resistance plasmids among three bacterial
sub-populations:
    S — susceptible  (no resistance plasmid)
    R — transconjugant (received plasmid from donor D)
    D — donor         (carries and transmits resistance plasmid)

Three-population ODE:
    dS/dt = r_S·S·(1-N/K) - k_kill(C)·S     - γ(C)·D·S + δ·R
    dR/dt = r_R·R·(1-N/K) - k_kill(C)·R/f   + γ(C)·D·S - δ·R
    dD/dt = r_D·D·(1-N/K) - k_kill(C)·D/f   + μ·S

where N = S + R + D.

Key biological features
-----------------------
SOS response (γ):   antibiotic stress INCREASES plasmid transfer rate
    γ(C) = γ_baseline · (1 + α_SOS · C/MIC)

Pharmacodynamic killing (k_kill):
    k_kill(C) = E_max · C^n / (EC50^n + C^n)   with EC50 = MIC

Resistance protection: R and D bacteria are killed at rate k_kill/f (factor f
less susceptible than S).

Fixed model parameters
----------------------
r_S = 0.5   (susceptible growth rate, /h)
r_R = 0.45  (resistant growth rate, /h — fitness cost)
r_D = 0.45  (donor growth rate, /h)
K   = 1e9   (carrying capacity, cells)
f   = 2.0   (resistance protection factor; R/D killed at k_kill/f)
δ   = 0.01  (conjugant-to-susceptible reversion rate, /h)
μ   = 1e-8  (spontaneous mutation rate S→D, per cell per hour)
E_max = 0.8, EC50 = MIC, n = 1.5  (Hill killing parameters)

Public API
----------
hgt_rate                — γ(C) HGT rate with SOS response
solve_hgt_ode           — integrate three-population ODE over a C(t) trajectory
cross_species_transfer  — single-step inter-species plasmid transfer event
get_hgt_pressure_score  — scalar [0,1] HGT pressure score
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Fixed model parameters
# ---------------------------------------------------------------------------

_r_S: float = 0.50    # susceptible growth rate (/h)
_r_R: float = 0.45    # transconjugant growth rate (/h)
_r_D: float = 0.45    # donor growth rate (/h)
_K:   float = 1e9     # carrying capacity (cells)
_f:   float = 2.0     # resistance protection factor
_delta: float = 0.01  # reversion rate R→S (/h)
_mu:    float = 1e-8  # spontaneous S→D mutation rate (per cell per hour)

# Killing Hill equation parameters
_E_MAX: float = 0.8
_N_HILL: float = 1.5

# Normalisation reference for hgt_pressure_score
_SCORE_REF: float = 1e7   # R increase of this magnitude → score ≈ 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _k_kill(C: float, mic: float) -> float:
    """Pharmacodynamic killing rate at concentration C."""
    cn = C ** _N_HILL
    ec50n = mic ** _N_HILL
    return float(_E_MAX * cn / (ec50n + cn)) if C > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hgt_rate(
    C: float,
    mic: float,
    gamma_baseline: float = 0.001,
    alpha_sos: float = 2.0,
) -> float:
    """
    Compute the plasmid transfer (conjugation) rate γ(C) with SOS induction.

        γ(C) = γ_baseline · (1 + α_SOS · C / MIC)

    Antibiotic stress up-regulates SOS response, which in turn increases
    expression of transfer genes — so γ increases with drug concentration.

    Parameters
    ----------
    C             : drug concentration (mg/L)
    mic           : MIC for this drug/pathogen combination (mg/L)
    gamma_baseline: conjugation rate at zero drug pressure (default 0.001 /h)
    alpha_sos     : SOS fold-increase coefficient (default 2.0)

    Returns
    -------
    float — γ(C) ≥ gamma_baseline
    """
    return float(gamma_baseline * (1.0 + alpha_sos * C / (mic + 1e-12)))


def solve_hgt_ode(
    S0: float,
    R0: float,
    D0: float,
    C_series: np.ndarray,
    drug_name: str,
    pathogen: str,
    dt: float = 1.0,
    mic: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the three-population HGT ODE over a drug concentration trajectory.

    The concentration is taken as piecewise-constant: C = C_series[i] during
    the interval [i·dt, (i+1)·dt].

    Uses scipy.integrate.solve_ivp (method='RK45').

    Parameters
    ----------
    S0        : initial susceptible population (cells)
    R0        : initial transconjugant population (cells)
    D0        : initial donor population (cells)
    C_series  : drug concentration at each time step (mg/L), shape (T,)
    drug_name : antibiotic name (reserved for future drug-specific params)
    pathogen  : pathogen name   (reserved for future pathogen-specific params)
    dt        : duration of each time step (hours, default 1.0)
    mic       : MIC for the drug/pathogen pair (mg/L, default 2.0)

    Returns
    -------
    S, R, D : np.ndarray of shape (T+1,) — populations at times
              [0, dt, 2·dt, …, T·dt], non-negative
    """
    T = len(C_series)
    if T == 0:
        return (
            np.array([float(S0)]),
            np.array([float(R0)]),
            np.array([float(D0)]),
        )

    t_total = T * dt
    t_eval = np.linspace(0.0, t_total, T + 1)

    def ode(t: float, y: np.ndarray) -> list:
        S, R, D = np.maximum(y, 0.0)
        N = S + R + D

        idx = min(int(t / (dt + 1e-12)), T - 1)
        C = float(C_series[idx])

        gamma = hgt_rate(C, mic)
        k = _k_kill(C, mic)
        lv = 1.0 - N / _K  # logistic factor

        dS = _r_S * S * lv - k * S - gamma * D * S + _delta * R
        dR = _r_R * R * lv - k * R / _f + gamma * D * S - _delta * R
        dD = _r_D * D * lv - k * D / _f + _mu * S

        return [dS, dR, dD]

    sol = solve_ivp(
        ode,
        (0.0, t_total),
        [float(S0), float(R0), float(D0)],
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )

    return (
        np.maximum(sol.y[0], 0.0),
        np.maximum(sol.y[1], 0.0),
        np.maximum(sol.y[2], 0.0),
    )


def cross_species_transfer(
    D_donor: float,
    S_recipient: float,
    contact_rate: float,
    gamma_cross: float = 0.001,
) -> float:
    """
    Compute the number of inter-species plasmid transfer events in one step.

        transfers = gamma_cross · contact_rate · D_donor · S_recipient

    Models transmission from a donor species to a phylogenetically distinct
    recipient population (e.g. E. coli donors to K. pneumoniae recipients).

    Parameters
    ----------
    D_donor       : donor population size (cells)
    S_recipient   : recipient susceptible population (cells)
    contact_rate  : species-contact scaling factor (dimensionless)
    gamma_cross   : inter-species conjugation rate (default 0.001 /h)

    Returns
    -------
    float — expected number of transfer events (≥ 0)
    """
    return float(gamma_cross * contact_rate * D_donor * S_recipient)


def get_hgt_pressure_score(
    R_final: float,
    R_initial: float,
    cross_transfers: float,
) -> float:
    """
    Compute an HGT pressure score in [0.0, 1.0].

    Combines the absolute increase in transconjugant population with
    inter-species transfer events, each normalised to a reference scale.

        delta_R        = max(0, R_final - R_initial)
        r_component    = clip(delta_R / _SCORE_REF, 0, 1)
        cross_component = clip(cross_transfers / _SCORE_REF, 0, 1)
        score          = clip(0.5·r_component + 0.5·cross_component, 0, 1)

    A score of 1.0 indicates both a large increase in transconjugant load
    and high cross-species transfer — maximum resistance spread risk.

    Parameters
    ----------
    R_final        : final transconjugant count (cells)
    R_initial      : initial transconjugant count (cells)
    cross_transfers: total inter-species transfer events

    Returns
    -------
    float in [0.0, 1.0]
    """
    delta_r = max(0.0, R_final - R_initial)
    r_component     = float(np.clip(delta_r       / (_SCORE_REF + 1e-12), 0.0, 1.0))
    cross_component = float(np.clip(cross_transfers / (_SCORE_REF + 1e-12), 0.0, 1.0))
    return float(np.clip(0.5 * r_component + 0.5 * cross_component, 0.0, 1.0))
