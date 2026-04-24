"""
Shared R2 helper utilities for all graders.

Provides deterministic (fixed-seed) wrappers around the math modules so
graders stay side-effect free and produce the same reward for the same
(action, state) pair regardless of external RNG state.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from episteward.math.population_pk import (
    PKParams,
    PatientCovariates,
    get_pkpd_score,
    sample_individual_pk,
    solve_two_compartment,
)
from episteward.math.msw import get_msw_reward_component
from episteward.math.bayesian_diagnostics import (
    PATHOGENS,
    get_diagnostic_reward,
    init_prior,
    shannon_entropy,
    update_posterior,
)
from episteward.math.pareto_reward import (
    compute_adaptive_weights,
    compute_reward_vector,
    compute_wrpi,
    scalarize,
)

# Fixed seed used for all grader PK sampling — ensures determinism
_GRADER_RNG_SEED: int = 42

# Reference maximum antibiotic duration for ecological footprint normalisation
_MAX_GUIDELINE_DAYS: float = 14.0

# WRPI severity weights keyed by internal pathogen names used in resistance_map
_SEVERITY_WEIGHTS: dict = {
    "E_coli_ESBL":       1.0,
    "K_pneumoniae_CRK":  1.5,
    "S_aureus_MRSA":     1.0,
    "E_faecium_VRE":     0.8,
    "P_aeruginosa_MDR":  1.2,
}

# Diagnostic test costs in reward units
_TEST_COSTS: dict = {
    "rapid_pcr":        0.05,
    "standard_culture": 0.10,
    "sensitivity_panel": 0.05,
}


# ---------------------------------------------------------------------------
# Patient covariates
# ---------------------------------------------------------------------------


def make_patient_covariates(state) -> PatientCovariates:
    """
    Build PatientCovariates from HospitalState, falling back to population
    defaults when specific vitals fields are absent.

    The standard vitals dict (temp_c, hr_bpm, wbc_k_ul, crp_mg_l,
    procalcitonin_ng_ml) does not include age / weight / creatinine, so those
    three fields default to a 50-year-old 70 kg patient with normal renal
    function unless the patient dict explicitly carries them.
    """
    vitals: dict = {}
    ward_id: str = ""
    if state.patients:
        p = state.patients[0]
        vitals = p.get("vitals", {})
        ward_id = p.get("ward_id", "")

    return PatientCovariates(
        age=float(vitals.get("age", 50.0)),
        weight_kg=float(vitals.get("weight_kg", 70.0)),
        creatinine_mg_dl=float(vitals.get("creatinine_mg_dl", 1.0)),
        is_icu="icu" in ward_id.lower(),
    )


# ---------------------------------------------------------------------------
# PK / PK-PD helpers
# ---------------------------------------------------------------------------


def get_pk_and_cseries(
    drug_name: str,
    dose_mg: float,
    frequency_h: float,
    state,
) -> Tuple[PKParams | None, np.ndarray]:
    """
    Sample individual PKParams and solve the two-compartment ODE for one
    dosing interval.

    Uses a fixed RNG seed (_GRADER_RNG_SEED) so graders are deterministic.
    Returns (None, empty array) for drugs not in _DRUG_PARAMS (e.g. ampicillin).
    """
    try:
        cov = make_patient_covariates(state)
        rng = np.random.default_rng(_GRADER_RNG_SEED)
        pk = sample_individual_pk(drug_name.lower(), cov, rng)
        C_series = solve_two_compartment(pk, dose_mg, frequency_h, dt=0.5)
        return pk, C_series
    except Exception:
        return None, np.array([])


def r2_pkpd_score(
    drug_name: str,
    dose_mg: float,
    frequency_h: float,
    duration_days: int,
    state,
    mic: float = 2.0,
) -> float:
    """
    PK/PD efficacy score in [0, 1] via population_pk.get_pkpd_score().
    Returns 0.0 when the drug is unknown to the PK library.
    """
    pk, _ = get_pk_and_cseries(drug_name, dose_mg, frequency_h, state)
    if pk is None:
        return 0.0
    return get_pkpd_score(pk, dose_mg, frequency_h, duration_days, mic, drug_name)


# ---------------------------------------------------------------------------
# MSW component
# ---------------------------------------------------------------------------


def r2_msw_component(
    drug_name: str,
    dose_mg: float,
    frequency_h: float,
    state,
    mic: float = 2.0,
    pathogen: str = "E_coli",
) -> float:
    """
    MSW reward component in [0, 1] from get_msw_reward_component().
    Returns 0.5 (neutral) when the drug has no PK data.
    """
    _, C_series = get_pk_and_cseries(drug_name, dose_mg, frequency_h, state)
    if len(C_series) == 0:
        return 0.0  # pessimistic: unknown PK → assume worst MSW exposure
    return get_msw_reward_component(C_series, drug_name, pathogen, mic)


# ---------------------------------------------------------------------------
# Bayesian entropy reward
# ---------------------------------------------------------------------------


def r2_bayesian_entropy_reward(action, state) -> float:
    """
    Compute get_diagnostic_reward() using episode-start prior entropy versus
    current posterior given available culture_results.

    Prior entropy is log₂(5) ≈ 2.322 bits (uniform over 5 pathogens).
    Posterior is derived by updating a site-specific prior with any confirmed
    culture result in the patient record.  If action.diagnostic_test is set,
    the matching test cost is applied as a resource penalty.
    """
    infection_site = "bloodstream"
    culture_pathogen: str | None = None

    if state.patients:
        p = state.patients[0]
        infection_site = p.get("infection_site", "bloodstream")
        # Only update if a definitive culture result is available
        raw_result = p.get("culture_result")
        if raw_result in PATHOGENS:
            culture_pathogen = raw_result
        # Also check if the stored pathogen maps to a PATHOGENS entry
        if culture_pathogen is None:
            stored = p.get("pathogen", "")
            # pathogen names in PATHOGENS: E_coli, K_pneumoniae, S_aureus, …
            # stored names: E_coli_ESBL, K_pneumoniae_CRK, … strip suffix
            for pg in PATHOGENS:
                if stored.startswith(pg):
                    culture_pathogen = pg
                    break

    prior_entropy = math.log2(len(PATHOGENS))  # ≈ 2.322 bits

    # Site-based prior as the reference point
    prior = init_prior(infection_site, [], {})
    if culture_pathogen is not None:
        posterior = update_posterior(prior, "culture", culture_pathogen)
    else:
        posterior = prior  # no new information

    posterior_entropy = shannon_entropy(posterior)

    test_cost = _TEST_COSTS.get(action.diagnostic_test or "", 0.0)
    test_ordered = action.diagnostic_test is not None

    return get_diagnostic_reward(prior_entropy, posterior_entropy, test_ordered, test_cost)


# ---------------------------------------------------------------------------
# Ecological footprint
# ---------------------------------------------------------------------------


def r2_ecological_footprint(duration_days: int) -> float:
    """
    Ecological footprint in [0, 1]: duration_days / 14.

    14 days is used as the normalisation reference (maximum guideline duration
    for most serious infections).  Short courses score near 0 (good); courses
    approaching 14 days score near 1 (high ecological burden).
    """
    return float(min(1.0, duration_days / _MAX_GUIDELINE_DAYS))


# ---------------------------------------------------------------------------
# WRPI from HospitalState
# ---------------------------------------------------------------------------


def r2_wrpi_from_state(state) -> float:
    """
    Compute WRPI from HospitalState.resistance_map.

    Uses the maximum allele frequency per pathogen across all wards, combined
    with the hardcoded clinical severity weights in _SEVERITY_WEIGHTS.
    """
    if not state.resistance_map:
        return 0.0
    prevalences: dict = {}
    for ward_data in state.resistance_map.values():
        for pathogen, freq in ward_data.items():
            prevalences[pathogen] = max(prevalences.get(pathogen, 0.0), float(freq))
    if not prevalences:
        return 0.0
    # Use only the severity weights for pathogens that are present
    severity = {p: _SEVERITY_WEIGHTS.get(p, 1.0) for p in prevalences}
    return compute_wrpi(prevalences, severity)


# ---------------------------------------------------------------------------
# App routing score
# ---------------------------------------------------------------------------


def r2_app_routing_score(action) -> float:
    """
    Return +0.05 if action.target_app matches the correct enterprise app:
      culture_requested → lab
      isolation_order   → ehr
      antibiotic set    → pharmacy

    Priority: lab > ehr > pharmacy (to handle actions that set multiple flags).
    Returns 0.0 when target_app is None or mismatched.
    """
    if action.target_app is None:
        return 0.0
    if action.culture_requested and action.target_app == "lab":
        return 0.05
    if action.isolation_order and action.target_app == "ehr":
        return 0.05
    if action.antibiotic and action.target_app == "pharmacy":
        return 0.05
    return 0.0


# ---------------------------------------------------------------------------
# Pareto scalarization
# ---------------------------------------------------------------------------


def r2_pareto_scalarize(
    clinical: float,
    ecology: float,
    economics: float,
    stewardship: float,
    wrpi: float,
) -> float:
    """
    Build a 4-component Pareto reward vector and scalarize with WRPI-adaptive
    weights.

    All four inputs must be in [0, 1].  Returns a float in [0, 1].
    """
    weights = compute_adaptive_weights(wrpi)
    reward_vec = compute_reward_vector(
        float(np.clip(clinical,    0.0, 1.0)),
        float(np.clip(ecology,     0.0, 1.0)),
        float(np.clip(economics,   0.0, 1.0)),
        float(np.clip(stewardship, 0.0, 1.0)),
    )
    return scalarize(reward_vec, weights)
