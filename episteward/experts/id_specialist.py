"""
Simulated ID specialist for Snorkel AI experts-in-the-loop bonus.

The specialist's clinical stance EVOLVES with the epidemiological situation:

  aggressive  — outbreak active, high resistance pressure.  Broad empiric coverage
                is endorsed; narrow-spectrum is questioned as too risky.

  moderate    — default.  Culture-first, balanced coverage, early de-escalation valued.

  conservative — resistance contained, no treatment failures.  Maximum stewardship:
                narrow-spectrum preferred, any broad-spectrum use requires justification.

The same antibiotic action (e.g. meropenem) receives *positive* feedback under
aggressive stance and *negative* feedback under conservative stance.  The RL agent
must learn to read the stance signal and adapt its prescribing strategy — this is
the "changing requirements" aspect required for the Snorkel bonus.

Stance-to-preference-weight mapping
------------------------------------
                   aggressive   moderate  conservative
narrow_spectrum        0.1        0.5        0.9
early_deescalation     0.2        0.5        0.9
culture_first          0.4        0.6        0.8
duration_minimisation  0.3        0.5        0.8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from episteward.models import EpiAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BROAD_SPECTRUM: frozenset = frozenset({
    "meropenem", "colistin", "ertapenem",
    "piperacillin-tazobactam", "vancomycin", "linezolid", "ceftriaxone",
})
_NARROW_SPECTRUM: frozenset = frozenset({
    "cefazolin", "ampicillin", "nitrofurantoin",
    "trimethoprim-sulfamethoxazole", "azithromycin", "ciprofloxacin",
})
_LAST_RESORT: frozenset = frozenset({"meropenem", "colistin"})

# Preference weights per stance
_WEIGHTS_BY_STANCE: Dict[str, Dict[str, float]] = {
    "aggressive": {
        "narrow_spectrum":        0.1,
        "early_deescalation":     0.2,
        "culture_first":          0.4,
        "duration_minimisation":  0.3,
    },
    "moderate": {
        "narrow_spectrum":        0.5,
        "early_deescalation":     0.5,
        "culture_first":          0.6,
        "duration_minimisation":  0.5,
    },
    "conservative": {
        "narrow_spectrum":        0.9,
        "early_deescalation":     0.9,
        "culture_first":          0.8,
        "duration_minimisation":  0.8,
    },
}

# Evidence-based max duration by infection site (same as OversightAgent)
_DURATION_LIMITS: Dict[str, int] = {
    "urinary_tract":    7,
    "bloodstream":      14,
    "respiratory":      7,
    "skin_soft_tissue": 10,
    "bone_joint":       42,
}
_DEFAULT_DURATION_LIMIT: int = 14

# Stance thresholds
_AGGRESSIVE_WRPI = 0.6
_CONSERVATIVE_WRPI = 0.2


# ---------------------------------------------------------------------------
# SpecialistFeedback
# ---------------------------------------------------------------------------

@dataclass
class SpecialistFeedback:
    """Structured evaluation from the ID specialist."""

    approved: bool
    comment: str                         # human-readable explanation
    stance: str                          # specialist's current stance at time of eval
    suggested_alternative: Optional[str] # antibiotic suggestion or None
    reward_signal: float                 # [0, 1] alignment reward for the RL agent


# ---------------------------------------------------------------------------
# IDSpecialist
# ---------------------------------------------------------------------------

class IDSpecialist:
    """
    Simulated infectious disease consultant with evolving clinical stance.

    Instantiate once per environment; call ``update_stance()`` each step before
    calling ``evaluate_action()`` so the feedback reflects the current situation.
    """

    STANCES = ["conservative", "moderate", "aggressive"]

    def __init__(self) -> None:
        self.stance: str = "moderate"
        self.stance_history: List[str] = []
        self.preference_weights: Dict[str, float] = dict(
            _WEIGHTS_BY_STANCE["moderate"]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_stance(
        self,
        wrpi: float,
        recent_failures: int,
        outbreak_active: bool,
    ) -> None:
        """
        Evolve the clinical stance based on current ward situation.

        Parameters
        ----------
        wrpi            : Ward Resistance Pressure Index [0, 1]
        recent_failures : number of treatment failures flagged in recent steps
        outbreak_active : True if any ward has a live network alert
        """
        if outbreak_active and wrpi > _AGGRESSIVE_WRPI:
            new_stance = "aggressive"
        elif wrpi < _CONSERVATIVE_WRPI and recent_failures == 0:
            new_stance = "conservative"
        else:
            new_stance = "moderate"

        if new_stance != self.stance:
            self.stance_history.append(new_stance)
            self.stance = new_stance
            self.preference_weights = dict(_WEIGHTS_BY_STANCE[new_stance])
        elif not self.stance_history:
            # Record initial stance on first call
            self.stance_history.append(self.stance)

    def evaluate_action(
        self,
        action: EpiAction,
        context: Dict[str, Any],
    ) -> SpecialistFeedback:
        """
        Evaluate an action against current specialist preferences.

        Parameters
        ----------
        action  : the EpiAction to evaluate
        context : dict with optional keys:
                  'resistance_flags' List[str], 'culture_result' str|None,
                  'infection_site' str, 'wrpi' float
        """
        drug = action.antibiotic.lower()
        resistance_flags: List[str] = context.get("resistance_flags", [])
        culture_result: Optional[str] = context.get("culture_result")
        infection_site: str = context.get("infection_site", "bloodstream")
        duration_limit = _DURATION_LIMITS.get(infection_site, _DEFAULT_DURATION_LIMIT)

        is_broad = drug in _BROAD_SPECTRUM
        is_narrow = drug in _NARROW_SPECTRUM
        is_last_resort = drug in _LAST_RESORT
        has_justification = bool(resistance_flags)
        culture_ordered = action.culture_requested or action.diagnostic_test is not None
        duration_ok = action.duration_days <= duration_limit

        if self.stance == "aggressive":
            return self._evaluate_aggressive(
                action, drug, is_broad, is_narrow, is_last_resort,
                has_justification, culture_ordered, duration_ok,
            )
        elif self.stance == "conservative":
            return self._evaluate_conservative(
                action, drug, is_broad, is_narrow, is_last_resort,
                has_justification, culture_ordered, duration_ok,
                culture_result,
            )
        else:  # moderate
            return self._evaluate_moderate(
                action, drug, is_broad, is_narrow, is_last_resort,
                has_justification, culture_ordered, duration_ok,
            )

    def get_feedback_reward(self, feedback: SpecialistFeedback) -> float:
        """
        Return the [0, 1] reward signal from a SpecialistFeedback object.

        Simply returns ``feedback.reward_signal``; provided as a named method
        to match the CLAUDE.md interface contract and allow future override.
        """
        return float(max(0.0, min(1.0, feedback.reward_signal)))

    # ------------------------------------------------------------------
    # Stance-specific evaluators
    # ------------------------------------------------------------------

    def _evaluate_aggressive(
        self,
        action: EpiAction,
        drug: str,
        is_broad: bool,
        is_narrow: bool,
        is_last_resort: bool,
        has_justification: bool,
        culture_ordered: bool,
        duration_ok: bool,
    ) -> SpecialistFeedback:
        """Aggressive: broad coverage endorsed; narrow questioned as too risky."""

        if is_last_resort and has_justification:
            return SpecialistFeedback(
                approved=True,
                comment=(
                    f"Agreed. {drug.capitalize()} is appropriate given confirmed "
                    "resistance flags and active outbreak conditions."
                ),
                stance="aggressive",
                suggested_alternative=None,
                reward_signal=1.0,
            )

        if is_broad:
            reward = 0.85 if duration_ok else 0.6
            comment = (
                f"Broad-spectrum empiric coverage with {drug} is reasonable given "
                "the current outbreak. Ensure culture is collected."
            )
            if not duration_ok:
                comment += f" Consider shortening duration to ≤{action.duration_days - 2} days."
            return SpecialistFeedback(
                approved=True,
                comment=comment,
                stance="aggressive",
                suggested_alternative=None,
                reward_signal=reward,
            )

        if is_narrow and not has_justification:
            return SpecialistFeedback(
                approved=False,
                comment=(
                    f"{drug.capitalize()} may be insufficient during active outbreak. "
                    "Consider escalating to piperacillin-tazobactam or meropenem "
                    "until cultures confirm susceptibility."
                ),
                stance="aggressive",
                suggested_alternative="piperacillin-tazobactam",
                reward_signal=0.2,
            )

        # Neutral (unknown drug or narrow with justification)
        return SpecialistFeedback(
            approved=True,
            comment="Acceptable given current outbreak conditions.",
            stance="aggressive",
            suggested_alternative=None,
            reward_signal=0.5,
        )

    def _evaluate_conservative(
        self,
        action: EpiAction,
        drug: str,
        is_broad: bool,
        is_narrow: bool,
        is_last_resort: bool,
        has_justification: bool,
        culture_ordered: bool,
        duration_ok: bool,
        culture_result: Optional[str],
    ) -> SpecialistFeedback:
        """Conservative: maximum stewardship — narrow preferred, broad requires justification."""

        if is_narrow and culture_ordered and duration_ok:
            return SpecialistFeedback(
                approved=True,
                comment=(
                    f"Excellent stewardship. {drug.capitalize()} with diagnostic "
                    "workup and appropriate duration is the ideal approach."
                ),
                stance="conservative",
                suggested_alternative=None,
                reward_signal=1.0,
            )

        if is_narrow and duration_ok:
            return SpecialistFeedback(
                approved=True,
                comment=(
                    f"{drug.capitalize()} is appropriately narrow. "
                    "I'd recommend ordering a culture to guide any future adjustments."
                ),
                stance="conservative",
                suggested_alternative=None,
                reward_signal=0.8,
            )

        if is_last_resort and not has_justification:
            return SpecialistFeedback(
                approved=False,
                comment=(
                    f"{drug.capitalize()} is a last-resort agent. Without confirmed "
                    "resistant organisms, this prescribing cannot be justified. "
                    "De-escalate immediately."
                ),
                stance="conservative",
                suggested_alternative="ceftriaxone",
                reward_signal=0.0,
            )

        if is_broad and not has_justification:
            alt = "cefazolin" if not culture_result else "nitrofurantoin"
            return SpecialistFeedback(
                approved=False,
                comment=(
                    f"{drug.capitalize()} is broader than necessary. With no resistance "
                    "flags, a narrower agent should be used. De-escalate to protect "
                    "ecological stewardship."
                ),
                stance="conservative",
                suggested_alternative=alt,
                reward_signal=0.1,
            )

        if is_broad and has_justification:
            reward = 0.55 if duration_ok else 0.3
            return SpecialistFeedback(
                approved=True,
                comment=(
                    f"{drug.capitalize()} is justified by resistance flags, but I expect "
                    "de-escalation as soon as culture sensitivities are available."
                ),
                stance="conservative",
                suggested_alternative=None,
                reward_signal=reward,
            )

        if not duration_ok:
            return SpecialistFeedback(
                approved=False,
                comment=(
                    f"Duration of {action.duration_days} days exceeds guideline. "
                    "Shorter courses reduce ecological footprint."
                ),
                stance="conservative",
                suggested_alternative=None,
                reward_signal=0.3,
            )

        return SpecialistFeedback(
            approved=True,
            comment="Prescription is within acceptable stewardship parameters.",
            stance="conservative",
            suggested_alternative=None,
            reward_signal=0.6,
        )

    def _evaluate_moderate(
        self,
        action: EpiAction,
        drug: str,
        is_broad: bool,
        is_narrow: bool,
        is_last_resort: bool,
        has_justification: bool,
        culture_ordered: bool,
        duration_ok: bool,
    ) -> SpecialistFeedback:
        """Moderate: balanced assessment. Culture-first, de-escalation valued."""

        if is_narrow and culture_ordered and duration_ok:
            return SpecialistFeedback(
                approved=True,
                comment=(
                    f"Good approach. {drug.capitalize()} with diagnostic workup "
                    "and guideline-consistent duration is well-balanced."
                ),
                stance="moderate",
                suggested_alternative=None,
                reward_signal=0.9,
            )

        if is_last_resort and not has_justification:
            return SpecialistFeedback(
                approved=False,
                comment=(
                    f"{drug.capitalize()} without resistance justification is "
                    "difficult to defend. Consider a step-down approach."
                ),
                stance="moderate",
                suggested_alternative="piperacillin-tazobactam",
                reward_signal=0.15,
            )

        if is_broad and not culture_ordered:
            return SpecialistFeedback(
                approved=True,
                comment=(
                    f"{drug.capitalize()} is acceptable empirically, but please ensure "
                    "cultures are collected before the next dose to enable de-escalation."
                ),
                stance="moderate",
                suggested_alternative=None,
                reward_signal=0.55,
            )

        if is_broad and culture_ordered and duration_ok:
            return SpecialistFeedback(
                approved=True,
                comment=(
                    f"{drug.capitalize()} with culture workup is reasonable. "
                    "Plan de-escalation once sensitivities return."
                ),
                stance="moderate",
                suggested_alternative=None,
                reward_signal=0.7,
            )

        if not duration_ok:
            return SpecialistFeedback(
                approved=False,
                comment=(
                    f"Duration of {action.duration_days} days exceeds the evidence-based "
                    "guideline. Prolonged courses increase resistance risk."
                ),
                stance="moderate",
                suggested_alternative=None,
                reward_signal=0.3,
            )

        return SpecialistFeedback(
            approved=True,
            comment="Prescription is clinically acceptable.",
            stance="moderate",
            suggested_alternative=None,
            reward_signal=0.6,
        )
