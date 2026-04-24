"""
Mutant Selection Window (MSW) model.

The MSW is the drug concentration range [MIC, MPC] within which resistant
mutants are selectively amplified from a susceptible bacterial population.

    MPC = MIC × (1 / mutation_freq)^(1 / hill_coeff)

Three concentration zones:
    sub_mic   — C < MIC           no drug effect; no selection pressure
    msw       — MIC ≤ C ≤ MPC    selective amplification of resistant mutants
    mpc_plus  — C > MPC           sterilising; mutant growth prevented

MSW risk = T_MSW / T_total — fraction of elapsed time spent in [MIC, MPC].
Dosing strategies that keep concentrations above MPC throughout the interval
minimise resistance selection; sub-MIC concentrations incur no MSW risk but
also have no bactericidal effect.

Hardcoded constants (consistent with pharmacodynamic literature defaults):
    mutation_freq = 1e-8   (probability per cell per division)
    hill_coeff    = 1.5    (Hill exponent for concentration–response curve)

These give MPC ≈ 215 443 × MIC, meaning the MSW spans a very wide range and
clinical concentrations rarely exceed MPC — modelling the real-world difficulty
of achieving sterilising concentrations in tissue.

Public API
----------
compute_mpc              — mutant prevention concentration from MIC
classify_zone            — zone label for a single concentration
compute_msw_risk         — T_MSW / T_total from a concentration–time series
get_msw_reward_component — reward in [0, 1] penalising MSW exposure
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MUTATION_FREQ: float = 1e-8  # mutation probability per cell per division
_HILL_COEFF: float = 1.5      # Hill coefficient for MSW boundary

Zone = Literal["sub_mic", "msw", "mpc_plus"]

# Fraction of dosing interval above MPC required for the sterilising bonus
_MPC_BONUS_THRESHOLD: float = 0.80
_MPC_BONUS_VALUE: float = 0.20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_mpc(drug_name: str, pathogen: str, mic: float) -> float:
    """
    Compute the Mutant Prevention Concentration (MPC).

        MPC = MIC × (1 / mutation_freq)^(1 / hill_coeff)

    With the hardcoded defaults (mutation_freq=1e-8, hill_coeff=1.5):
        MPC ≈ MIC × 215 443

    The drug_name and pathogen parameters are accepted for API consistency and
    future per-drug / per-pathogen mutation-frequency lookup; currently all
    drugs and pathogens use the same hardcoded constants.

    Parameters
    ----------
    drug_name : antibiotic name (reserved for future per-drug mutation_freq)
    pathogen  : pathogen name   (reserved for future per-pathogen mutation_freq)
    mic       : minimum inhibitory concentration (mg/L)

    Returns
    -------
    float — MPC in mg/L (always > mic)
    """
    return float(mic * (1.0 / _MUTATION_FREQ) ** (1.0 / _HILL_COEFF))


def classify_zone(C: float, mic: float, mpc: float) -> Zone:
    """
    Classify a single drug concentration into one of three MSW zones.

    Boundaries are inclusive:
        sub_mic  : C < MIC
        msw      : MIC ≤ C ≤ MPC
        mpc_plus : C > MPC

    Parameters
    ----------
    C   : drug concentration (mg/L)
    mic : minimum inhibitory concentration (mg/L)
    mpc : mutant prevention concentration (mg/L)

    Returns
    -------
    "sub_mic" | "msw" | "mpc_plus"
    """
    if C < mic:
        return "sub_mic"
    if C <= mpc:
        return "msw"
    return "mpc_plus"


def compute_msw_risk(
    C_series: np.ndarray,
    drug_name: str,
    pathogen: str,
    mic: float,
    dt: float = 0.5,
) -> float:
    """
    Compute MSW risk: the fraction of time spent in the mutant selection window.

        MSW_risk = T_MSW / T_total
                 = (number of points where MIC ≤ C ≤ MPC) / (total points)

    Since both the MSW count and the total are multiplied by dt, the time step
    dt cancels from the ratio; all equally-spaced series give the same result
    regardless of dt value.

    Parameters
    ----------
    C_series  : concentration–time array (mg/L), shape (n,)
    drug_name : antibiotic name (passed to compute_mpc)
    pathogen  : pathogen name   (passed to compute_mpc)
    mic       : pathogen MIC (mg/L)
    dt        : time step (hours) — accepted for API compatibility; does not
                affect the result for equally-spaced series

    Returns
    -------
    float in [0.0, 1.0]
        0.0 — never in MSW (all sub-MIC or all above MPC)
        1.0 — always in MSW
    """
    if len(C_series) == 0:
        return 0.0

    mpc = compute_mpc(drug_name, pathogen, mic)
    in_msw = (C_series >= mic) & (C_series <= mpc)
    return float(np.mean(in_msw))


def get_msw_reward_component(
    C_series: np.ndarray,
    drug_name: str,
    pathogen: str,
    mic: float,
) -> float:
    """
    Compute the MSW reward component for a concentration–time series.

    Scoring rule:
        base  = 1.0 − msw_risk
        bonus = +0.2 if fraction of points with C > MPC exceeds 0.80
        score = clip(base + bonus, 0.0, 1.0)

    A score of 1.0 means the drug achieved sterilising concentrations (above
    MPC) for >80 % of the interval and spent no time in the MSW zone.
    A score of 0.0 means the entire exposure was within the MSW.

    Parameters
    ----------
    C_series  : concentration–time array (mg/L)
    drug_name : antibiotic name
    pathogen  : pathogen name
    mic       : pathogen MIC (mg/L)

    Returns
    -------
    float in [0.0, 1.0]
    """
    if len(C_series) == 0:
        return 0.0

    mpc = compute_mpc(drug_name, pathogen, mic)
    msw_risk = compute_msw_risk(C_series, drug_name, pathogen, mic)

    base = 1.0 - msw_risk
    pct_above_mpc = float(np.mean(C_series > mpc))
    bonus = _MPC_BONUS_VALUE if pct_above_mpc > _MPC_BONUS_THRESHOLD else 0.0

    return float(np.clip(base + bonus, 0.0, 1.0))
