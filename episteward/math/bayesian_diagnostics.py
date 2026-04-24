"""
Bayesian pathogen identification model.

Maintains a probability distribution (posterior) over five clinically relevant
pathogens and updates it as diagnostic results become available.

    π_{t+1}(k) ∝ L(observation | pathogen=k) · π_t(k)

Pathogens modelled
------------------
    E_coli, K_pneumoniae, S_aureus, P_aeruginosa, E_faecalis

Observation types
-----------------
    gram_stain       — result: "gram_negative" | "gram_positive"
    culture          — result: one of the five pathogen names
    sensitivity_panel — result: "susceptible" | "resistant"

Likelihood tables
-----------------
Gram stain (hardcoded, biologically grounded):
    gram_negative  E_coli=0.95, K_pneumoniae=0.95, S_aureus=0.02,
                   P_aeruginosa=0.95, E_faecalis=0.02
    gram_positive  (reversed)

Culture (high-sensitivity confirmed identification):
    result=k  L(k | true=k) = 0.90, L(k | true=j≠k) = 0.025

Sensitivity panel (broad-spectrum susceptibility signal):
    susceptible  E_coli=0.70, K_pneumoniae=0.30, S_aureus=0.80,
                 P_aeruginosa=0.15, E_faecalis=0.75
    resistant    (complementary values)

Information measures
--------------------
Shannon entropy in bits:
    H(π) = −Σ_k π(k) · log₂(π(k))    (0·log₂(0) := 0)

Value of Information:
    VoI(test) = H(π) − E[H(π_posterior | test_result)] − test_cost
    where the expectation is over all possible test results weighted
    by their marginal probability P(result) = Σ_k L(result|k)·π(k).

Public API
----------
PATHOGENS                 — ordered list of the five pathogen names
init_prior                — site- and risk-factor-aware prior
update_posterior          — Bayesian update on one diagnostic result
shannon_entropy           — H(π) in bits
value_of_information      — expected information gain minus test cost
get_diagnostic_reward     — scalar [0, 1] reward for a diagnostic step
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATHOGENS: List[str] = [
    "E_coli",
    "K_pneumoniae",
    "S_aureus",
    "P_aeruginosa",
    "E_faecalis",
]

# Maximum possible entropy for 5 pathogens (uniform distribution), in bits
_MAX_ENTROPY: float = math.log2(len(PATHOGENS))  # ≈ 2.322 bits

# ---------------------------------------------------------------------------
# Hardcoded likelihood tables
# ---------------------------------------------------------------------------

# Gram stain: P(result | true_pathogen)
_GRAM_LIKELIHOODS: Dict[str, Dict[str, float]] = {
    "gram_negative": {
        "E_coli":        0.95,
        "K_pneumoniae":  0.95,
        "S_aureus":      0.02,
        "P_aeruginosa":  0.95,
        "E_faecalis":    0.02,
    },
    "gram_positive": {
        "E_coli":        0.02,
        "K_pneumoniae":  0.02,
        "S_aureus":      0.95,
        "P_aeruginosa":  0.02,
        "E_faecalis":    0.95,
    },
}

# Culture: P(reported_pathogen | true_pathogen)
#   Sensitivity 0.95 for the correct organism; 0.0125 split across the other 4.
#   (Modern automated species ID: sensitivity >95 %, specificity >99 %)
def _culture_likelihood(reported: str, true: str) -> float:
    return 0.95 if reported == true else 0.0125


# Sensitivity panel: P(result | true_pathogen)
# "susceptible" ≈ organism responds to empirical narrow-spectrum therapy
_SENSITIVITY_LIKELIHOODS: Dict[str, Dict[str, float]] = {
    "susceptible": {
        "E_coli":        0.70,
        "K_pneumoniae":  0.30,
        "S_aureus":      0.80,
        "P_aeruginosa":  0.15,
        "E_faecalis":    0.75,
    },
    "resistant": {
        "E_coli":        0.30,
        "K_pneumoniae":  0.70,
        "S_aureus":      0.20,
        "P_aeruginosa":  0.85,
        "E_faecalis":    0.25,
    },
}

# Possible results per test type (used by VoI enumeration)
_TEST_RESULTS: Dict[str, List[str]] = {
    "gram_stain":        ["gram_negative", "gram_positive"],
    "culture":           list(PATHOGENS),
    "sensitivity_panel": ["susceptible", "resistant"],
}

# ---------------------------------------------------------------------------
# Site-based empirical prior weights (unnormalised)
# Rows: infection site keywords; Columns: PATHOGENS order above.
# Sources: published empirical antibiogram/epidemiology tables.
# ---------------------------------------------------------------------------

_SITE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "urinary":     {"E_coli": 55, "K_pneumoniae": 15, "S_aureus":  5, "P_aeruginosa": 10, "E_faecalis": 15},
    "uti":         {"E_coli": 55, "K_pneumoniae": 15, "S_aureus":  5, "P_aeruginosa": 10, "E_faecalis": 15},
    "bloodstream": {"E_coli": 25, "K_pneumoniae": 15, "S_aureus": 35, "P_aeruginosa": 10, "E_faecalis": 15},
    "bacteremia":  {"E_coli": 25, "K_pneumoniae": 15, "S_aureus": 35, "P_aeruginosa": 10, "E_faecalis": 15},
    "respiratory": {"E_coli": 10, "K_pneumoniae": 25, "S_aureus": 20, "P_aeruginosa": 35, "E_faecalis": 10},
    "pneumonia":   {"E_coli": 10, "K_pneumoniae": 25, "S_aureus": 20, "P_aeruginosa": 35, "E_faecalis": 10},
    "wound":       {"E_coli": 10, "K_pneumoniae":  5, "S_aureus": 55, "P_aeruginosa": 20, "E_faecalis": 10},
    "skin":        {"E_coli": 10, "K_pneumoniae":  5, "S_aureus": 55, "P_aeruginosa": 20, "E_faecalis": 10},
}

# Risk factor multipliers applied to individual pathogen weights
_RISK_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "icu":               {"K_pneumoniae": 2.0, "P_aeruginosa": 2.0},
    "immunocompromised": {"P_aeruginosa": 1.5},
    "recent_antibiotics":{"K_pneumoniae": 1.5, "P_aeruginosa": 1.5},
    "catheter":          {"E_faecalis": 2.0, "K_pneumoniae": 1.5},
    "diabetes":          {"E_coli": 1.5, "S_aureus": 1.5},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of *weights* normalised to sum to 1.0."""
    total = sum(weights.values()) + 1e-300
    return {k: v / total for k, v in weights.items()}


def _likelihood_for(observation_type: str, result: str, pathogen: str) -> float:
    """Look up L(result | pathogen) for the given observation type."""
    if observation_type == "gram_stain":
        return _GRAM_LIKELIHOODS[result][pathogen]
    if observation_type == "culture":
        return _culture_likelihood(result, pathogen)
    if observation_type == "sensitivity_panel":
        return _SENSITIVITY_LIKELIHOODS[result][pathogen]
    raise ValueError(
        f"Unknown observation_type '{observation_type}'; "
        "expected 'gram_stain', 'culture', or 'sensitivity_panel'."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_prior(
    infection_site: str,
    risk_factors: List[str],
    ward_antibiogram: Dict[str, float],
) -> Dict[str, float]:
    """
    Build an informed prior over PATHOGENS from infection site and patient factors.

    Algorithm
    ---------
    1. Start from a site-specific empirical weight table (uniform if site unknown).
    2. Apply risk-factor multipliers to individual pathogen weights.
    3. Blend with *ward_antibiogram* (if non-empty): the antibiogram weights are
       treated as an additional observation that pulls the prior toward locally
       prevalent organisms.  Blend weight for the antibiogram is 30 %.
    4. Normalise to sum to 1.0.

    Parameters
    ----------
    infection_site    : clinical site string (e.g. "urinary_tract", "respiratory")
    risk_factors      : list of risk factor strings (e.g. ["icu", "diabetes"])
    ward_antibiogram  : dict mapping pathogen name → relative ward prevalence
                        (any non-negative floats; empty dict = no adjustment)

    Returns
    -------
    Dict[str, float] — posterior prior, keys = PATHOGENS, values sum to 1.0.
    """
    # Step 1 — site-based baseline weights
    site_key = infection_site.lower().replace(" ", "_").replace("-", "_")
    # Check partial match (e.g. "urinary_tract" matches "urinary")
    matched: Dict[str, float] | None = None
    for key, weights in _SITE_WEIGHTS.items():
        if key in site_key:
            matched = dict(weights)
            break
    if matched is None:
        matched = {p: 20.0 for p in PATHOGENS}  # uniform fallback

    # Step 2 — apply risk factor multipliers
    for rf in risk_factors:
        rf_lower = rf.lower().replace(" ", "_").replace("-", "_")
        if rf_lower in _RISK_MULTIPLIERS:
            for pathogen, mult in _RISK_MULTIPLIERS[rf_lower].items():
                matched[pathogen] = matched.get(pathogen, 20.0) * mult

    # Step 3 — blend with ward antibiogram (30% weight)
    if ward_antibiogram:
        ab_weights = {p: ward_antibiogram.get(p, 0.0) for p in PATHOGENS}
        ab_total = sum(ab_weights.values())
        if ab_total > 0:
            ab_norm = {k: v / ab_total for k, v in ab_weights.items()}
            prior_total = sum(matched.values())
            prior_norm  = {k: v / prior_total for k, v in matched.items()}
            matched = {
                p: 0.70 * prior_norm[p] + 0.30 * ab_norm[p]
                for p in PATHOGENS
            }

    return _normalise(matched)


def update_posterior(
    prior: Dict[str, float],
    observation_type: str,
    result: str,
) -> Dict[str, float]:
    """
    Apply one Bayesian update: π(k) ← L(result | k) · π(k), then normalise.

    Parameters
    ----------
    prior            : current probability distribution over PATHOGENS
    observation_type : "gram_stain" | "culture" | "sensitivity_panel"
    result           : observed result (type-specific — see module docstring)

    Returns
    -------
    Dict[str, float] — updated posterior, keys = PATHOGENS, values sum to 1.0.

    Raises
    ------
    ValueError  if observation_type or result is unrecognised.
    """
    if observation_type == "gram_stain" and result not in _GRAM_LIKELIHOODS:
        raise ValueError(
            f"Unknown gram_stain result '{result}'; "
            "expected 'gram_negative' or 'gram_positive'."
        )
    if observation_type == "sensitivity_panel" and result not in _SENSITIVITY_LIKELIHOODS:
        raise ValueError(
            f"Unknown sensitivity_panel result '{result}'; "
            "expected 'susceptible' or 'resistant'."
        )
    if observation_type == "culture" and result not in PATHOGENS:
        raise ValueError(
            f"Unknown culture result '{result}'; expected one of {PATHOGENS}."
        )

    unnorm = {
        p: _likelihood_for(observation_type, result, p) * prior.get(p, 0.0)
        for p in PATHOGENS
    }
    return _normalise(unnorm)


def shannon_entropy(posterior: Dict[str, float]) -> float:
    """
    Compute Shannon entropy of a pathogen posterior in bits.

        H(π) = −Σ_k π(k) · log₂(π(k))     (0 · log₂(0) := 0)

    Parameters
    ----------
    posterior : probability distribution over PATHOGENS

    Returns
    -------
    float — entropy in bits, in [0.0, log₂(|PATHOGENS|)]
    """
    h = 0.0
    for p in posterior.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def value_of_information(
    posterior: Dict[str, float],
    test_type: str,
    test_cost: float,
) -> float:
    """
    Compute the net Value of Information for ordering *test_type*.

        VoI = H(π) − E[H(π_posterior | test_result)] − test_cost

    where the expectation is taken over the marginal distribution of test
    results:
        P(result) = Σ_k L(result | k) · π(k)

    A positive VoI means ordering the test is expected to be worthwhile.

    Parameters
    ----------
    posterior : current probability distribution over PATHOGENS (the "prior"
                relative to this test decision)
    test_type : "gram_stain" | "culture" | "sensitivity_panel"
    test_cost : cost of the test in reward units (e.g. 0.1 for standard culture)

    Returns
    -------
    float — net VoI; clamped to [−test_cost, H(π)]
    """
    if test_type not in _TEST_RESULTS:
        raise ValueError(
            f"Unknown test_type '{test_type}'; "
            f"expected one of {list(_TEST_RESULTS)}."
        )

    h_prior = shannon_entropy(posterior)
    possible_results = _TEST_RESULTS[test_type]

    expected_h_post = 0.0
    for result in possible_results:
        # Marginal probability of this result
        p_result = sum(
            _likelihood_for(test_type, result, p) * posterior.get(p, 0.0)
            for p in PATHOGENS
        )
        if p_result < 1e-300:
            continue
        # Posterior conditioned on this result
        post_r = update_posterior(posterior, test_type, result)
        expected_h_post += p_result * shannon_entropy(post_r)

    information_gain = h_prior - expected_h_post
    return float(information_gain - test_cost)


def get_diagnostic_reward(
    prior_entropy: float,
    posterior_entropy: float,
    test_ordered: bool,
    test_cost: float,
) -> float:
    """
    Compute a scalar diagnostic reward in [0.0, 1.0].

    Rewards entropy reduction achieved this step, normalised by the maximum
    possible entropy (log₂ of the number of pathogens = 2.322 bits).
    When a test was ordered the test_cost is subtracted as a resource penalty.

        norm_gain   = max(0, prior_entropy − posterior_entropy) / _MAX_ENTROPY
        cost_penalty = test_cost  if test_ordered  else  0.0
        reward       = clip(norm_gain − cost_penalty, 0.0, 1.0)

    Parameters
    ----------
    prior_entropy    : H(π) before this step (bits)
    posterior_entropy: H(π) after this step (bits)
    test_ordered     : True if the agent ordered a diagnostic test this step
    test_cost        : cost of the test in normalised reward units

    Returns
    -------
    float in [0.0, 1.0]
    """
    info_gain  = max(0.0, prior_entropy - posterior_entropy)
    norm_gain  = info_gain / _MAX_ENTROPY
    cost_penalty = test_cost if test_ordered else 0.0
    return float(np.clip(norm_gain - cost_penalty, 0.0, 1.0))
