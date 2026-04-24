"""
Resistance dynamics via Itô stochastic differential equation (SDE).

Replaces the deterministic Wright-Fisher model (evolution.py) with a
continuous-time Itô SDE that couples drug concentration to resistance allele
frequency p_R ∈ [0, 1]:

    dp_R = s(C(t)) · p_R · (1-p_R) · dt  +  σ · √(p_R(1-p_R)) · dW_t

where
    s(C) — MSW-aware selection coefficient (see selection_coefficient)
    σ    — genetic drift diffusion coefficient = √(1 / (2 · N_eff))
    W_t  — standard Brownian motion

Euler–Maruyama discretisation:
    p_R(t+Δt) = p_R(t) + s · p_R · (1-p_R) · Δt
                        + σ · √(p_R(1-p_R)) · √Δt · ξ,    ξ ~ N(0,1)

    Result is clamped to [0, 1].

Selection coefficient — MSW-aware Hill function:
    s(C) = s_max · C^n / (EC50^n + C^n)   MIC ≤ C ≤ MPC  (mutant amplification)
    s(C) = 0                               C > MPC         (sterilising — no selection)
    s(C) = −fitness_cost                   C < MIC         (fitness cost to resistant allele)

    Defaults: s_max=0.3, n=2.0, EC50=MIC, fitness_cost=0.05

MPC is looked up from episteward.math.msw.compute_mpc (mutation_freq=1e-8,
hill_coeff=1.5).

Public API
----------
selection_coefficient         — MSW-aware s(C) for a single concentration
euler_maruyama_step           — one Δt step of the Itô SDE
simulate_resistance_trajectory — full trajectory from a concentration–time series
resistance_emerged             — boolean: has p_R exceeded threshold for ≥N steps?
"""

from __future__ import annotations

from typing import List

import numpy as np

from episteward.math.msw import compute_mpc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_EFF: float = 1e8                     # effective bacterial population size
_SIGMA: float = float(np.sqrt(1.0 / (2.0 * _N_EFF)))  # diffusion coefficient

_DEFAULT_S_MAX: float = 0.30
_DEFAULT_N: float = 2.0
_DEFAULT_FITNESS_COST: float = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def selection_coefficient(
    C: float,
    mic: float,
    mpc: float,
    s_max: float = _DEFAULT_S_MAX,
    n: float = _DEFAULT_N,
    fitness_cost: float = _DEFAULT_FITNESS_COST,
) -> float:
    """
    Compute the MSW-aware selection coefficient s(C) for a drug concentration C.

    Zone-based rules:
        C < MIC        →  s = −fitness_cost   (resistant allele pays growth cost)
        MIC ≤ C ≤ MPC  →  s = s_max · C^n / (EC50^n + C^n)  with EC50 = MIC
        C > MPC        →  s = 0               (sterilising; mutants also killed)

    Parameters
    ----------
    C           : drug concentration (mg/L)
    mic         : minimum inhibitory concentration (mg/L)
    mpc         : mutant prevention concentration (mg/L)
    s_max       : maximum selection coefficient in the MSW (default 0.3)
    n           : Hill exponent (default 2.0)
    fitness_cost: growth-rate penalty of the resistance allele below MIC
                  (default 0.05)

    Returns
    -------
    float — selection coefficient in [-fitness_cost, s_max]
    """
    if C < mic:
        return -fitness_cost
    if C > mpc:
        return 0.0
    # MSW zone: Hill function with EC50 = MIC
    ec50 = mic
    cn = C ** n
    ec50n = ec50 ** n
    return float(s_max * cn / (ec50n + cn))


def euler_maruyama_step(
    p_R: float,
    s: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    """
    Advance resistance allele frequency by one Euler–Maruyama step.

        p_R(t+Δt) = p_R(t) + s · p_R · (1-p_R) · Δt
                            + σ · √(p_R(1-p_R)) · √Δt · ξ

    ξ ~ N(0,1).  Result is clamped to [0, 1].

    Parameters
    ----------
    p_R   : current resistance allele frequency ∈ [0, 1]
    s     : selection coefficient (from selection_coefficient)
    sigma : diffusion coefficient (typically sqrt(1/(2*N_eff)))
    dt    : time step (hours)
    rng   : numpy Generator for reproducibility

    Returns
    -------
    float in [0.0, 1.0]
    """
    variance_term = p_R * (1.0 - p_R)
    drift = s * variance_term * dt
    diffusion = sigma * float(np.sqrt(variance_term)) * float(np.sqrt(dt)) * rng.normal()
    return float(np.clip(p_R + drift + diffusion, 0.0, 1.0))


def simulate_resistance_trajectory(
    p_init: float,
    C_series: np.ndarray,
    drug_name: str,
    pathogen: str,
    dt: float,
    rng: np.random.Generator,
    mic: float = 2.0,
) -> List[float]:
    """
    Simulate a resistance allele frequency trajectory under the Itô SDE.

    At each time step i the drug concentration C_series[i] determines the
    selection coefficient via selection_coefficient(), then one Euler–Maruyama
    step advances p_R.

    Parameters
    ----------
    p_init    : initial resistance frequency ∈ [0, 1]
    C_series  : drug concentration–time array (mg/L), shape (T,)
    drug_name : antibiotic name (forwarded to compute_mpc)
    pathogen  : pathogen name   (forwarded to compute_mpc)
    dt        : time step used by the ODE solver that produced C_series (hours)
    rng       : numpy Generator for reproducibility
    mic       : pathogen MIC (mg/L); default 2.0

    Returns
    -------
    List[float] of length len(C_series) + 1 — initial value followed by one
    entry per time step; all values in [0, 1].
    """
    mpc = compute_mpc(drug_name, pathogen, mic)
    trajectory: List[float] = [float(np.clip(p_init, 0.0, 1.0))]
    p_R = trajectory[0]

    for C in C_series:
        s = selection_coefficient(float(C), mic, mpc)
        p_R = euler_maruyama_step(p_R, s, _SIGMA, dt, rng)
        trajectory.append(p_R)

    return trajectory


def resistance_emerged(
    trajectory: List[float],
    threshold: float = 0.5,
    sustained_steps: int = 48,
) -> bool:
    """
    Return True if resistance allele frequency has exceeded *threshold* for at
    least *sustained_steps* consecutive time steps.

    The default threshold of 0.5 represents majority-resistant population.
    The default sustained_steps of 48 corresponds to 24 h at dt=0.5 h/step.

    Parameters
    ----------
    trajectory     : resistance frequency sequence (from simulate_resistance_trajectory)
    threshold      : frequency above which resistance is considered "emerged" (default 0.5)
    sustained_steps: minimum consecutive steps required above threshold (default 48)

    Returns
    -------
    bool — True if sustained resistance above threshold was detected
    """
    count = 0
    for p in trajectory:
        if p > threshold:
            count += 1
            if count >= sustained_steps:
                return True
        else:
            count = 0
    return False
