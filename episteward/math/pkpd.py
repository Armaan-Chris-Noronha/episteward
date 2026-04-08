"""
One-compartment pharmacokinetic / pharmacodynamic (PK/PD) model.

PK — concentration over time (solved with scipy.integrate.solve_ivp):
    C(t) = (F * D / Vd) * exp(-ke * t)
    ke   = CL / Vd

PD — drug effect via Hill equation:
    effect(C) = Emax * C^n / (EC50^n + C^n)

PK/PD scoring index — %T>MIC (time-dependent antibiotics):
    score = clip(%T>MIC / T_TARGET, 0, 1)
    T_TARGET = 0.40  (40% of dosing interval above MIC, standard beta-lactam target)

Public API (all params auto-loaded from antibiotics.json):
    get_concentration_curve(drug_name, dose_mg, duration_hours) -> np.ndarray
    is_in_therapeutic_window(drug_name, dose_mg, mic)          -> tuple[bool, float]
    get_pkpd_score(drug_name, dose_mg, frequency_h, pathogen_mic) -> float
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).parent.parent / "data" / "antibiotics.json"

# Standard PK/PD targets per drug class (fraction of dosing interval above MIC)
_T_ABOVE_MIC_TARGET: Dict[str, float] = {
    "carbapenem":                          0.40,
    "beta_lactam_beta_lactamase_inhibitor": 0.50,
    "third_gen_cephalosporin":             0.50,
    "first_gen_cephalosporin":             0.50,
    "penicillin":                          0.50,
    "fluoroquinolone":                     0.40,
    "glycopeptide":                        0.40,
    "oxazolidinone":                       0.40,
    "macrolide":                           0.40,
    "nitrofurantoin":                      0.40,
    "sulfonamide":                         0.40,
    "polymyxin":                           0.40,
}

_DEFAULT_T_TARGET = 0.40
_PATIENT_WEIGHT_KG = 70.0
_N_EVAL_POINTS = 241


@lru_cache(maxsize=1)
def _load_antibiotics() -> Dict[str, Any]:
    """Load antibiotics.json once and cache."""
    return json.loads(_DATA_PATH.read_text())


def _get_drug(drug_name: str) -> Dict[str, Any]:
    """Return the antibiotics.json entry for *drug_name*, or raise ValueError."""
    db = _load_antibiotics()
    entry = db.get(drug_name.lower())
    if entry is None:
        raise ValueError(
            f"Unknown drug '{drug_name}'. Known drugs: {sorted(db.keys())}"
        )
    return entry


def _ke_and_c0(pk_params: Dict[str, float], dose_mg: float) -> Tuple[float, float]:
    """Compute elimination rate constant ke and initial concentration C0."""
    Vd = pk_params["Vd_L_kg"] * _PATIENT_WEIGHT_KG
    CL = pk_params["CL_L_h_kg"] * _PATIENT_WEIGHT_KG
    ke = CL / Vd
    C0 = pk_params["F"] * dose_mg / Vd
    return ke, C0


# ---------------------------------------------------------------------------
# Core ODE solver (used by both internal and public API)
# ---------------------------------------------------------------------------

def concentration_profile(
    dose_mg: float,
    pk_params: Dict[str, float],
    t_span: Tuple[float, float] = (0.0, 24.0),
    n_points: int = _N_EVAL_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the one-compartment PK ODE for a single dose via scipy.integrate.solve_ivp.

    Parameters
    ----------
    dose_mg   : administered dose in milligrams
    pk_params : dict with keys F, Vd_L_kg, CL_L_h_kg (patient weight = 70 kg)
    t_span    : (t_start, t_end) in hours
    n_points  : evaluation points

    Returns
    -------
    t : np.ndarray  shape (n_points,)  time in hours
    C : np.ndarray  shape (n_points,)  concentration in mg/L
    """
    ke, C0 = _ke_and_c0(pk_params, dose_mg)

    def ode(_t: float, y: list[float]) -> list[float]:
        return [-ke * y[0]]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(ode, t_span, [C0], t_eval=t_eval, dense_output=False)
    return sol.t, sol.y[0]


# ---------------------------------------------------------------------------
# Hill equation
# ---------------------------------------------------------------------------

def hill_effect(
    concentration: float,
    emax: float,
    ec50: float,
    hill_n: float = 1.0,
) -> float:
    """
    Hill equation: effect = Emax * C^n / (EC50^n + C^n).

    Returns a value in [0, Emax].
    """
    if concentration <= 0:
        return 0.0
    cn = concentration**hill_n
    return emax * cn / (ec50**hill_n + cn)


# ---------------------------------------------------------------------------
# Internal scoring helper (peak/trough — kept for backward compatibility)
# ---------------------------------------------------------------------------

def therapeutic_score(
    dose_mg: float,
    pk_params: Dict[str, float],
    mic: float,
    frequency_hours: float,
) -> float:
    """
    Continuous [0.0, 1.0] score based on peak/trough vs MIC window.

    Kept for backward compatibility. Prefer ``get_pkpd_score`` for grading.
    """
    _, C = concentration_profile(dose_mg, pk_params, t_span=(0.0, frequency_hours))
    c_peak = C.max()
    c_trough = C.min()

    low_target = mic * 4
    high_target = mic * 64

    if c_trough < low_target:
        return float(np.clip(c_trough / low_target, 0.0, 1.0))
    if c_peak > high_target:
        overshoot = (c_peak - high_target) / high_target
        return float(np.clip(1.0 - overshoot, 0.0, 1.0))
    return 1.0


# ---------------------------------------------------------------------------
# Public API — drug-name-based, loads params from antibiotics.json
# ---------------------------------------------------------------------------

def get_concentration_curve(
    drug_name: str,
    dose_mg: float,
    duration_hours: float,
    n_points: int = _N_EVAL_POINTS,
) -> np.ndarray:
    """
    Return the concentration–time curve C(t) for *drug_name* over *duration_hours*.

    Parameters
    ----------
    drug_name      : key in antibiotics.json (e.g. "meropenem")
    dose_mg        : single dose in mg
    duration_hours : evaluation window in hours

    Returns
    -------
    C : np.ndarray  shape (n_points,)  concentration in mg/L
        (time axis runs 0 → duration_hours uniformly)
    """
    drug = _get_drug(drug_name)
    pk = drug["pk_params"]
    ke, C0 = _ke_and_c0(pk, dose_mg)
    t = np.linspace(0.0, duration_hours, n_points)
    return C0 * np.exp(-ke * t)


def is_in_therapeutic_window(
    drug_name: str,
    dose_mg: float,
    mic: float,
) -> Tuple[bool, float]:
    """
    Check whether *dose_mg* keeps concentration above *mic* for ≥40% of the
    standard dosing interval.

    Parameters
    ----------
    drug_name : key in antibiotics.json
    dose_mg   : dose in mg
    mic       : pathogen MIC in mg/L

    Returns
    -------
    in_window : bool   True if score ≥ 0.4
    score     : float  %T>MIC normalised to [0, 1]
    """
    drug = _get_drug(drug_name)
    freq_h = float(drug["frequencies"][0])  # use primary dosing interval
    score = get_pkpd_score(drug_name, dose_mg, freq_h, mic)
    return score >= 0.4, score


def get_pkpd_score(
    drug_name: str,
    dose_mg: float,
    frequency_h: float,
    pathogen_mic: float,
) -> float:
    """
    Compute a PK/PD efficacy score in [0.0, 1.0] using the %T>MIC index.

    Score = clip( %T>MIC / T_TARGET, 0, 1 )

    where T_TARGET is the class-specific target fraction (default 0.40).

    Examples
    --------
    >>> get_pkpd_score("meropenem", 1000.0, 8.0, 2.0)   # ≥ 0.8
    >>> get_pkpd_score("ampicillin", 2000.0, 6.0, 256.0) # ≤ 0.1

    Parameters
    ----------
    drug_name    : key in antibiotics.json
    dose_mg      : single dose in mg
    frequency_h  : dosing interval in hours (one of 4, 6, 8, 12, 24)
    pathogen_mic : MIC in mg/L

    Returns
    -------
    float in [0.0, 1.0]
    """
    drug = _get_drug(drug_name)
    pk = drug["pk_params"]
    drug_class = drug.get("class", "")
    t_target = _T_ABOVE_MIC_TARGET.get(drug_class, _DEFAULT_T_TARGET)

    ke, C0 = _ke_and_c0(pk, dose_mg)

    # If initial concentration never reaches MIC → score 0
    if C0 <= pathogen_mic:
        return 0.0

    # Time at which C(t) drops to MIC: C0 * exp(-ke * t_cross) = MIC
    t_cross = np.log(C0 / pathogen_mic) / ke  # hours

    # %T>MIC = min(t_cross, frequency_h) / frequency_h
    pct_t_above_mic = min(t_cross, frequency_h) / frequency_h

    return float(np.clip(pct_t_above_mic / t_target, 0.0, 1.0))
