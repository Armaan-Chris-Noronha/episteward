"""
Two-compartment population pharmacokinetic (PopPK) model with inter-individual
variability (IIV), Bayesian MAP estimation, and PK/PD efficacy scoring.

Two-compartment ODE (oral / IM absorption):
    dC1/dt = (F·D·ka·exp(-ka·t))/V1 - (k12+k10)·C1 + k21·(V2/V1)·C2
    dC2/dt =  k12·(V1/V2)·C1 - k21·C2
    k10 = CL/V1,  k12 = Q/V1,  k21 = Q/V2

For IV bolus (ka = 0): initial condition C1(0) = F·D/V1, C2(0) = 0.

IIV (log-normal):
    theta_i = theta_pop · exp(eta_i),  eta_i ~ N(0, omega_i^2)

MAP objective:
    OFV = Σ (C_obs - C_model)² / σ² + Σ (ln θ_i - ln θ_pop)² / ω_i²

Population parameters are hardcoded for meropenem, vancomycin, and ceftriaxone
based on published pharmacokinetic literature.

Public API
----------
PKParams              — PK parameter set (V1, V2, CL, Q, F, ka)
PatientCovariates     — patient demographics for covariate-adjusted PK
sample_individual_pk  — sample individual params with IIV from population priors
solve_two_compartment — RK45 ODE solver returning C1(t) as np.ndarray
bayesian_tdm_update   — MAP estimation of PKParams from TDM observations
get_pkpd_score        — PK/PD efficacy score in [0.0, 1.0]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PKParams:
    """Individual two-compartment PK parameter set."""

    V1: float  # central volume of distribution (L)
    V2: float  # peripheral volume of distribution (L)
    CL: float  # total clearance (L/h)
    Q: float   # inter-compartmental clearance (L/h)
    F: float   # bioavailability fraction [0, 1]
    ka: float  # first-order absorption rate constant (1/h); 0.0 for IV bolus


@dataclass
class PatientCovariates:
    """Patient covariates used to adjust population PK parameters."""

    age: float             # years
    weight_kg: float
    creatinine_mg_dl: float
    is_icu: bool


# ---------------------------------------------------------------------------
# Population PK parameters (hardcoded from published literature)
# ---------------------------------------------------------------------------
# theta_pop : population means for V1, V2, CL, Q, F, ka
# omega     : inter-individual variability (SD on log scale) per parameter
# sigma     : residual error SD (proportional, on linear scale)
# pd_type   : "time_dependent" (beta-lactams) | "auc_dependent" (vancomycin)
# t_target  : target %T>MIC (fraction of dosing interval) for time-dep drugs
# auc_target: target AUC24h/MIC for AUC-dependent drugs
# cl_renal_dependent: whether CL scales with creatinine clearance

_DRUG_PARAMS: Dict[str, Dict] = {
    "meropenem": {
        "theta_pop": {"V1": 12.5, "V2": 9.5, "CL": 9.0, "Q": 4.0, "F": 1.0, "ka": 0.0},
        "omega":     {"V1": 0.30,  "V2": 0.30, "CL": 0.25, "Q": 0.30},
        "sigma": 0.10,
        "pd_type": "time_dependent",
        "t_target": 0.40,
        "cl_renal_dependent": True,
    },
    "vancomycin": {
        "theta_pop": {"V1": 35.0, "V2": 30.0, "CL": 4.5, "Q": 3.5, "F": 1.0, "ka": 0.0},
        "omega":     {"V1": 0.30,  "V2": 0.30, "CL": 0.30, "Q": 0.30},
        "sigma": 0.15,
        "pd_type": "auc_dependent",
        "auc_target": 400.0,  # AUC24h / MIC target (mg·h/L per mg/L)
        "cl_renal_dependent": True,
    },
    "ceftriaxone": {
        "theta_pop": {"V1": 7.5, "V2": 5.5, "CL": 1.5, "Q": 2.0, "F": 1.0, "ka": 0.0},
        "omega":     {"V1": 0.25, "V2": 0.25, "CL": 0.20, "Q": 0.25},
        "sigma": 0.10,
        "pd_type": "time_dependent",
        "t_target": 0.50,
        "cl_renal_dependent": False,
    },
}

# Reference CrCl for clearance normalisation (mL/min, ~70 kg adult, normal renal)
_REFERENCE_CRCL_ML_MIN: float = 100.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cockcroft_gault(age: float, weight_kg: float, creatinine_mg_dl: float) -> float:
    """Estimate creatinine clearance (mL/min) via Cockcroft-Gault (male approx)."""
    scr = max(creatinine_mg_dl, 0.1)
    return (140.0 - age) * weight_kg / (72.0 * scr)


def _renal_cl_scale(patient: PatientCovariates) -> float:
    """
    Scalar factor for renal-dependent clearance relative to reference CrCl.

    Clamped to [0.20, 2.00] to avoid extreme extrapolation.
    """
    crcl = _cockcroft_gault(patient.age, patient.weight_kg, patient.creatinine_mg_dl)
    return float(np.clip(crcl / _REFERENCE_CRCL_ML_MIN, 0.20, 2.00))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_individual_pk(
    drug_name: str,
    patient: PatientCovariates,
    rng: np.random.Generator,
) -> PKParams:
    """
    Sample individual PK parameters with inter-individual variability (IIV).

    theta_i = theta_pop · exp(eta_i),  eta_i ~ N(0, omega_i²)

    For renally-eliminated drugs (meropenem, vancomycin), the population
    clearance is first scaled by the patient's estimated CrCl relative to
    the reference population (CrCl = 100 mL/min).

    Parameters
    ----------
    drug_name : one of "meropenem", "vancomycin", "ceftriaxone"
    patient   : patient covariates
    rng       : numpy Generator for reproducible sampling

    Returns
    -------
    PKParams with patient-specific sampled values
    """
    dn = drug_name.lower()
    if dn not in _DRUG_PARAMS:
        raise ValueError(
            f"Unknown drug '{drug_name}'. Known drugs: {sorted(_DRUG_PARAMS.keys())}"
        )

    dp = _DRUG_PARAMS[dn]
    theta = dp["theta_pop"]
    omega = dp["omega"]

    # Covariate-adjusted population clearance before IIV sampling
    cl_pop = theta["CL"]
    if dp["cl_renal_dependent"]:
        cl_pop = cl_pop * _renal_cl_scale(patient)

    return PKParams(
        V1=theta["V1"] * float(np.exp(rng.normal(0.0, omega["V1"]))),
        V2=theta["V2"] * float(np.exp(rng.normal(0.0, omega["V2"]))),
        CL=cl_pop      * float(np.exp(rng.normal(0.0, omega["CL"]))),
        Q =theta["Q"]  * float(np.exp(rng.normal(0.0, omega["Q"]))),
        F =theta["F"],
        ka=theta["ka"],
    )


def solve_two_compartment(
    pk: PKParams,
    dose_mg: float,
    duration_h: float,
    dt: float = 0.5,
) -> np.ndarray:
    """
    Solve the two-compartment PK ODE for central compartment concentration C1(t).

    Uses scipy.integrate.solve_ivp (method='RK45').

    ODE system
    ----------
    For oral / IM (ka > 0):
        dC1/dt = (F·D·ka·exp(-ka·t))/V1 - (k12+k10)·C1 + k21·(V2/V1)·C2
        dC2/dt =  k12·(V1/V2)·C1 - k21·C2

    For IV bolus (ka == 0.0):
        y0 = [F·D/V1, 0.0];  absorption term is zero

    Parameters
    ----------
    pk         : individual PK parameters
    dose_mg    : administered dose (mg)
    duration_h : simulation window (hours)
    dt         : time step between evaluation points (hours)

    Returns
    -------
    np.ndarray  shape (n,) — C1 concentrations at equally-spaced time points
                [0, dt, 2·dt, …, duration_h], non-negative
    """
    k10 = pk.CL / pk.V1
    k12 = pk.Q  / pk.V1
    k21 = pk.Q  / pk.V2
    r_v  = pk.V2 / pk.V1  # peripheral / central volume ratio

    n_points = max(2, int(round(duration_h / dt)) + 1)
    t_eval = np.linspace(0.0, duration_h, n_points)

    if pk.ka == 0.0:
        # IV bolus: all drug starts in central compartment
        y0 = [pk.F * dose_mg / pk.V1, 0.0]

        def ode(_t: float, y: List[float]) -> List[float]:
            C1, C2 = y
            return [
                -(k12 + k10) * C1 + k21 * r_v * C2,
                (k12 / r_v) * C1 - k21 * C2,
            ]

    else:
        # First-order oral / IM absorption
        y0 = [0.0, 0.0]

        def ode(t: float, y: List[float]) -> List[float]:
            C1, C2 = y
            absorption = pk.F * dose_mg * pk.ka * np.exp(-pk.ka * t) / pk.V1
            return [
                absorption - (k12 + k10) * C1 + k21 * r_v * C2,
                (k12 / r_v) * C1 - k21 * C2,
            ]

    sol = solve_ivp(
        ode,
        (0.0, duration_h),
        y0,
        method="RK45",
        t_eval=t_eval,
        dense_output=False,
        rtol=1e-6,
        atol=1e-8,
    )
    return np.maximum(sol.y[0], 0.0)


def bayesian_tdm_update(
    pk: PKParams,
    observed_levels: List[Tuple[float, float]],
    drug_name: str,
    dose_mg: float = 1000.0,
) -> PKParams:
    """
    Bayesian MAP update of individual PK parameters from TDM observations.

    Minimises the objective function:
        OFV = Σ_j (C_obs_j - C_model_j)² / σ²
            + Σ_i (ln θ_i - ln θ_pop_i)² / ω_i²

    using scipy.optimize.minimize (method='L-BFGS-B') in η-space.

    F and ka are held fixed from the input PKParams; only V1, V2, CL, Q are
    updated.

    Parameters
    ----------
    pk             : starting PK estimate (F and ka are used as-is)
    observed_levels: list of (time_h, concentration_mg_L) pairs
    drug_name      : must match a key in _DRUG_PARAMS
    dose_mg        : dose administered prior to TDM sampling (mg)

    Returns
    -------
    PKParams with MAP-estimated V1, V2, CL, Q
    """
    if not observed_levels:
        return pk

    dn = drug_name.lower()
    if dn not in _DRUG_PARAMS:
        raise ValueError(
            f"Unknown drug '{drug_name}'. Known drugs: {sorted(_DRUG_PARAMS.keys())}"
        )

    dp = _DRUG_PARAMS[dn]
    theta = dp["theta_pop"]
    omega = dp["omega"]
    sigma2 = dp["sigma"] ** 2 + 1e-12

    t_obs = np.array([t for t, _ in observed_levels])
    c_obs = np.array([c for _, c in observed_levels])
    t_max = float(t_obs.max()) + 1.0

    omega_sq = {k: v ** 2 + 1e-12 for k, v in omega.items()}

    def ofv(eta: np.ndarray) -> float:
        pk_trial = PKParams(
            V1=theta["V1"] * np.exp(eta[0]),
            V2=theta["V2"] * np.exp(eta[1]),
            CL=theta["CL"] * np.exp(eta[2]),
            Q =theta["Q"]  * np.exp(eta[3]),
            F =pk.F,
            ka=pk.ka,
        )
        C_series = solve_two_compartment(pk_trial, dose_mg, t_max, dt=0.5)
        t_series = np.linspace(0.0, t_max, len(C_series))
        C_model  = np.interp(t_obs, t_series, C_series)

        likelihood = float(np.sum((c_obs - C_model) ** 2) / sigma2)
        penalty = (
            eta[0] ** 2 / omega_sq["V1"]
            + eta[1] ** 2 / omega_sq["V2"]
            + eta[2] ** 2 / omega_sq["CL"]
            + eta[3] ** 2 / omega_sq["Q"]
        )
        return likelihood + penalty

    result = minimize(
        ofv,
        x0=np.zeros(4),
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    eta_map = result.x

    return PKParams(
        V1=theta["V1"] * float(np.exp(eta_map[0])),
        V2=theta["V2"] * float(np.exp(eta_map[1])),
        CL=theta["CL"] * float(np.exp(eta_map[2])),
        Q =theta["Q"]  * float(np.exp(eta_map[3])),
        F =pk.F,
        ka=pk.ka,
    )


def get_pkpd_score(
    pk: PKParams,
    dose_mg: float,
    frequency_h: float,
    duration_days: int,
    mic: float,
    drug_name: str,
) -> float:
    """
    Compute a PK/PD efficacy score in [0.0, 1.0] for one dosing interval.

    Scoring indices
    ---------------
    Time-dependent antibiotics (beta-lactams — meropenem, ceftriaxone, unknown):
        score = clip( (%T > MIC) / T_target,  0, 1 )

    AUC-dependent antibiotics (vancomycin):
        score = clip( AUC_24h / (MIC · AUC_target),  0, 1 )

    A fine time grid (dt = 0.1 h) is used internally for accurate integration.
    The score reflects first-dose kinetics; for accumulating drugs (low CL)
    steady-state scores may be higher.

    Parameters
    ----------
    pk           : individual PK parameters
    dose_mg      : single dose (mg)
    frequency_h  : dosing interval (h)
    duration_days: treatment duration — reserved; not used for per-interval score
    mic          : pathogen MIC (mg/L)
    drug_name    : used to look up PD type and targets; unknown drugs use
                   time-dependent defaults (T_target = 0.50)

    Returns
    -------
    float in [0.0, 1.0]
    """
    dn = drug_name.lower()

    if dn in _DRUG_PARAMS:
        pd_type    = _DRUG_PARAMS[dn]["pd_type"]
        t_target   = _DRUG_PARAMS[dn].get("t_target", 0.40)
        auc_target = _DRUG_PARAMS[dn].get("auc_target", 400.0)
    else:
        pd_type    = "time_dependent"
        t_target   = 0.50
        auc_target = 400.0

    # Fine grid for numerical accuracy
    C_series = solve_two_compartment(pk, dose_mg, frequency_h, dt=0.1)

    if pd_type == "time_dependent":
        pct_above = float(np.mean(C_series > mic))
        score = float(np.clip(pct_above / t_target, 0.0, 1.0))

    else:  # auc_dependent
        dt_actual = frequency_h / max(len(C_series) - 1, 1)
        auc_per_dose = float(np.trapz(C_series, dx=dt_actual))
        doses_per_day = 24.0 / frequency_h
        auc_24h = auc_per_dose * doses_per_day
        score = float(np.clip(auc_24h / (mic * auc_target + 1e-12), 0.0, 1.0))

    return score
