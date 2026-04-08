"""
Bayesian resistance probability estimator.

Uses ward-level antibiogram data as the prior, then updates on each
culture result via Bayes' rule:

    P(resistant | result) ∝ P(result | resistant) * P(resistant)

Likelihoods (symmetric by default):
    P("resistant" | truly resistant)  = sensitivity_accuracy  (default 0.95)
    P("resistant" | truly sensitive)  = 1 - sensitivity_accuracy
    P("sensitive" | truly resistant)  = 1 - sensitivity_accuracy
    P("sensitive" | truly sensitive)  = sensitivity_accuracy

Public API
----------
update_posterior(prior_prob, culture_result,
    sensitivity_accuracy=0.95)                         -> float
get_resistance_probability(pathogen_name, antibiotic_name,
    culture_results, ward_id="icu")                    -> tuple[float, float, float]
get_empiric_recommendation(pathogen_name,
    available_antibiotics, resistance_probs)           -> str
estimate_resistance(pathogen, antibiotic, ward_id,
    resistance_profiles, culture_result=None)          -> Dict[str, float]
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).parent.parent / "data" / "resistance_profiles.json"
_N_PSEUDO = 10.0   # pseudo-observation count for Beta prior
_DEFAULT_SENSITIVITY = 0.95


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_resistance_profiles() -> Dict[str, Any]:
    """Load resistance_profiles.json once and cache."""
    return json.loads(_DATA_PATH.read_text())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _beta_from_prior(prior_prob: float) -> Tuple[float, float]:
    """
    Convert a scalar prior probability to Beta(alpha, beta) parameters.

    Uses a weakly informative prior equivalent to _N_PSEUDO pseudo-observations.
    """
    alpha = float(prior_prob) * _N_PSEUDO
    b = (1.0 - float(prior_prob)) * _N_PSEUDO
    return float(alpha), float(b)


def _update_alpha_beta(
    alpha: float,
    b: float,
    culture_result: Optional[str],
    sensitivity: float,
) -> Tuple[float, float]:
    """
    Apply one Bayesian likelihood update to Beta parameters.

    Specificity is assumed equal to sensitivity (symmetric test characteristics).
    """
    specificity = sensitivity

    if culture_result == "resistant":
        lr = sensitivity / (1.0 - specificity)
        alpha *= lr
    elif culture_result == "sensitive":
        lr = (1.0 - sensitivity) / specificity
        alpha *= lr
        b /= lr
    # None → no update

    # Clamp to valid Beta support
    total = alpha + b
    alpha = float(np.clip(alpha, 0.01, total - 0.01))
    b = total - alpha
    return alpha, b


def _beta_ci(alpha: float, b: float) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) for Beta(alpha, b)."""
    dist = beta_dist(alpha, b)
    return float(dist.mean()), float(dist.ppf(0.025)), float(dist.ppf(0.975))


# ---------------------------------------------------------------------------
# Public API — new functions
# ---------------------------------------------------------------------------

def update_posterior(
    prior_prob: float,
    culture_result: Optional[str],
    sensitivity_accuracy: float = _DEFAULT_SENSITIVITY,
) -> float:
    """
    Return the posterior probability P(resistant) after one culture result.

    Parameters
    ----------
    prior_prob         : P(resistant) before the culture result
    culture_result     : "resistant", "sensitive", or None (no update)
    sensitivity_accuracy: test sensitivity = specificity (symmetric, default 0.95)

    Returns
    -------
    float : updated P(resistant) in [0, 1]
    """
    alpha, b = _beta_from_prior(float(np.clip(prior_prob, 0.0, 1.0)))
    alpha, b = _update_alpha_beta(alpha, b, culture_result, sensitivity_accuracy)
    return float(alpha / (alpha + b))


def get_resistance_probability(
    pathogen_name: str,
    antibiotic_name: str,
    culture_results: Optional[List[Optional[str]]] = None,
    ward_id: str = "icu",
) -> Tuple[float, float, float]:
    """
    Compute posterior resistance probability with 95% credible interval.

    Loads the prior from resistance_profiles.json for the given
    pathogen/antibiotic/ward, then applies sequential Bayesian updates
    from each culture result.

    Parameters
    ----------
    pathogen_name  : organism key (e.g. "E_coli_ESBL")
    antibiotic_name: drug key (e.g. "meropenem")
    culture_results: list of "resistant", "sensitive", or None values.
                     None or empty list → prior only.
    ward_id        : ward for antibiogram lookup (default "icu")

    Returns
    -------
    (posterior_mean, ci_lower_95, ci_upper_95) all in [0, 1]
    """
    profiles = _load_resistance_profiles()
    prior = prior_resistance_prob(pathogen_name, antibiotic_name, ward_id, profiles)

    alpha, b = _beta_from_prior(prior)
    for result in (culture_results or []):
        alpha, b = _update_alpha_beta(alpha, b, result, _DEFAULT_SENSITIVITY)

    return _beta_ci(alpha, b)


def get_empiric_recommendation(
    pathogen_name: str,
    available_antibiotics: List[str],
    resistance_probs: Dict[str, float],
) -> str:
    """
    Return the antibiotic with the highest coverage probability (lowest resistance).

    Parameters
    ----------
    pathogen_name       : organism key (for logging / future use)
    available_antibiotics: ordered list of candidate antibiotic names
    resistance_probs    : mapping antibiotic_name → P(resistant)

    Returns
    -------
    str : name of the recommended antibiotic (minimum resistance probability).
        Falls back to the first available antibiotic if none has a known probability.

    Raises
    ------
    ValueError : if available_antibiotics is empty
    """
    if not available_antibiotics:
        raise ValueError("available_antibiotics must not be empty")

    best = min(
        available_antibiotics,
        key=lambda ab: resistance_probs.get(ab, 1.0),
    )
    return best


# ---------------------------------------------------------------------------
# Backward-compatible helpers
# ---------------------------------------------------------------------------

def prior_resistance_prob(
    pathogen: str,
    antibiotic: str,
    ward_id: str,
    resistance_profiles: Dict[str, Any],
) -> float:
    """
    Return the antibiogram prior P(resistant) for a drug-bug-ward triplet.

    Falls back to organism-level, then global default of 0.2.
    """
    ward_data = resistance_profiles.get(ward_id, {})
    bug_data = ward_data.get(pathogen, resistance_profiles.get(pathogen, {}))
    return float(bug_data.get(antibiotic, bug_data.get("default", 0.2)))


def estimate_resistance(
    pathogen: str,
    antibiotic: str,
    ward_id: str,
    resistance_profiles: Dict[str, Any],
    culture_result: Optional[str] = None,
) -> Dict[str, float]:
    """
    Full pipeline: prior → Bayesian update → return summary dict.

    Kept for backward compatibility. Prefer ``get_resistance_probability``.

    Returns
    -------
    dict with keys: prior, posterior, ci_lower, ci_upper
    """
    prior = prior_resistance_prob(pathogen, antibiotic, ward_id, resistance_profiles)
    alpha, b = _beta_from_prior(prior)
    alpha, b = _update_alpha_beta(alpha, b, culture_result, _DEFAULT_SENSITIVITY)
    mean, ci_lower, ci_upper = _beta_ci(alpha, b)
    return {
        "prior": prior,
        "posterior": mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
