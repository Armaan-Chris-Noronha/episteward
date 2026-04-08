"""
Wright-Fisher model for resistance allele evolution under antibiotic pressure.

Model
-----
Each timestep advances the allele frequency by one Wright-Fisher generation:

    p_new ~ Binomial(2N, p_fitness) / 2N

    p_fitness = p * w_R / (p * w_R + (1-p) * w_S)
    w_R = 1.0          resistant fitness (unaffected by drug)
    w_S = 1.0 - s      sensitive fitness (reduced under drug pressure)

Selective coefficient *s* is derived from PK/PD pressure via %T>MIC:
    s = 0.9 * %T>MIC          (scales 0 → 0.9; never 1.0)

Rules
-----
- All RNG calls use an explicit ``rng`` argument — no global state.
- Population size N defaults to 1e8 (frequency-space binomial sampling).

Public API
----------
evolve_resistance(p_current, selective_coeff, rng, N=1e8) -> float
compute_selective_coefficient(drug_name, dose_mg, mic)    -> float
resistance_emerged(allele_freq, threshold=0.5)            -> bool
get_resistance_trajectory(drug_name, dose_mg,
    duration_steps, initial_freq, rng)                    -> List[float]
"""

from __future__ import annotations

from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_S = 0.9          # cap selective coefficient — never full extinction per step
_DEFAULT_N = int(1e8) # effective population size for Binomial sampling


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evolve_resistance(
    p_current: float,
    selective_coeff: float,
    rng: np.random.Generator,
    N: int = _DEFAULT_N,
) -> float:
    """
    Advance allele frequency by one Wright-Fisher generation.

    Parameters
    ----------
    p_current       : current resistant-allele frequency in [0, 1]
    selective_coeff : s ∈ [0, 1] — selection against the sensitive strain
                      0 = neutral drift, _MAX_S = maximum pressure
    rng             : seeded numpy Generator (no global state)
    N               : effective haploid population size (default 1e8)

    Returns
    -------
    float : new allele frequency in [0, 1]
    """
    p = float(np.clip(p_current, 0.0, 1.0))
    s = float(np.clip(selective_coeff, 0.0, 1.0))

    w_R = 1.0
    w_S = 1.0 - s
    mean_fitness = p * w_R + (1.0 - p) * w_S

    if mean_fitness <= 0.0:
        return 1.0  # degenerate: entire population resistant

    p_selected = p * w_R / mean_fitness

    # Binomial sampling in frequency space
    draws = rng.binomial(2 * N, p_selected)
    return float(draws / (2 * N))


def compute_selective_coefficient(
    drug_name: str,
    dose_mg: float,
    mic: float,
) -> float:
    """
    Compute the selective coefficient *s* from antibiotic pressure.

    Uses %T>MIC (from pkpd.get_pkpd_score) as the pressure proxy:
        s = _MAX_S * %T>MIC

    When dose_mg = 0 (no drug), %T>MIC = 0 → s = 0 (pure neutral drift).

    Parameters
    ----------
    drug_name : key in antibiotics.json (e.g. "meropenem")
    dose_mg   : administered dose in mg (0 = no antibiotic)
    mic       : pathogen MIC in mg/L

    Returns
    -------
    float : s in [0.0, _MAX_S]
    """
    if dose_mg <= 0.0:
        return 0.0

    # Import here to avoid circular imports at module level
    from episteward.math.pkpd import _get_drug, get_pkpd_score

    drug = _get_drug(drug_name)
    frequency_h = float(drug["frequencies"][0])
    pct_t_above = get_pkpd_score(drug_name, dose_mg, frequency_h, mic)
    return float(np.clip(_MAX_S * pct_t_above, 0.0, _MAX_S))


def resistance_emerged(
    allele_freq: float,
    threshold: float = 0.5,
) -> bool:
    """
    Return True when allele frequency has crossed the emergence threshold.

    Parameters
    ----------
    allele_freq : current resistant-allele frequency in [0, 1]
    threshold   : default 0.5 — majority-allele criterion

    Returns
    -------
    bool
    """
    return float(allele_freq) > float(threshold)


def get_resistance_trajectory(
    drug_name: str,
    dose_mg: float,
    duration_steps: int,
    initial_freq: float,
    rng: np.random.Generator,
    mic: float = 2.0,
    N: int = _DEFAULT_N,
) -> List[float]:
    """
    Simulate resistance evolution over *duration_steps* timesteps.

    Each step = one 24-h dosing cycle.

    Parameters
    ----------
    drug_name      : key in antibiotics.json
    dose_mg        : dose per cycle in mg (0 = no drug)
    duration_steps : number of 24-h steps to simulate
    initial_freq   : starting allele frequency in [0, 1]
    rng            : seeded numpy Generator (no global state)
    mic            : pathogen MIC in mg/L (default 2.0)
    N              : effective population size for Wright-Fisher

    Returns
    -------
    List[float] of length ``duration_steps + 1``
        trajectory[0] = initial_freq, trajectory[i] = freq after step i
    """
    s = compute_selective_coefficient(drug_name, dose_mg, mic)
    freq = float(np.clip(initial_freq, 0.0, 1.0))
    trajectory: List[float] = [freq]

    for _ in range(duration_steps):
        freq = evolve_resistance(freq, s, rng, N=N)
        trajectory.append(freq)

    return trajectory


# ---------------------------------------------------------------------------
# Backward-compatible helpers (kept for existing tests)
# ---------------------------------------------------------------------------

def selection_coefficient(dose_mg: float, standard_dose_mg: float) -> float:
    """
    Legacy: derive *s* from dose ratio (no MIC).

    Kept for backward compatibility. Prefer ``compute_selective_coefficient``.
    """
    ratio = dose_mg / max(standard_dose_mg, 1.0)
    return float(np.clip(_MAX_S * (1 - np.exp(-ratio)), 0.0, _MAX_S))


def wright_fisher_step(
    allele_freq: float,
    s: float,
    rng: np.random.Generator,
) -> float:
    """
    Legacy wrapper around ``evolve_resistance`` using the old signature.

    Kept for backward compatibility. Prefer ``evolve_resistance``.
    """
    return evolve_resistance(allele_freq, s, rng)


def evolve_resistance_legacy(
    allele_freq: float,
    treatment_hours: float,
    dose_mg: float,
    standard_dose_mg: float,
    rng: np.random.Generator,
    timestep_hours: float = 24.0,
) -> tuple[float, bool]:
    """
    Legacy: one-step evolution with hours-based emergence check.

    Kept for backward compatibility. Prefer ``get_resistance_trajectory``.
    """
    s = selection_coefficient(dose_mg, standard_dose_mg)
    new_freq = evolve_resistance(allele_freq, s, rng)
    emerged = resistance_emerged(new_freq) and (treatment_hours >= 48.0)
    return new_freq, emerged
