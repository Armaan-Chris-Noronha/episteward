"""
Enterprise hospital apps for Scaler AI Labs multi-app RL bonus.

Each app represents a distinct hospital system with its own business rules,
acceptance criteria, and reward signals.  The RL agent must learn to route
actions to the correct app and satisfy each system's domain rules.

Apps
----
EHRApp          — isolation orders, specialist consults, patient history
LabApp          — culture requests, diagnostic test orders
PharmacyApp     — antibiotic prescriptions with pre-authorisation rules
MicrobiologyApp — antibiogram queries, resistance surveillance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from episteward.models import EpiAction

# ---------------------------------------------------------------------------
# AppResponse
# ---------------------------------------------------------------------------

@dataclass
class AppResponse:
    """Structured response returned by every app process() call."""

    routed: bool                  # True if the action was addressed to this app
    processed: bool = False       # True if the app accepted and handled the action
    reward_delta: float = 0.0     # bonus/penalty added to step reward, clamped to [-0.1, 0.1]
    message: str = ""             # human-readable status from the app
    latency_steps: int = 0        # steps until result is available (0 = immediate)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_THERAPEUTIC_WINDOWS: Dict[str, tuple] = {
    # (min_dose_mg, max_dose_mg)
    "meropenem":                  (500.0, 2000.0),
    "vancomycin":                 (750.0, 3000.0),
    "ceftriaxone":                (1000.0, 4000.0),
    "colistin":                   (75.0, 300.0),
    "ertapenem":                  (500.0, 1000.0),
    "piperacillin-tazobactam":    (2250.0, 18000.0),
    "ampicillin":                 (500.0, 4000.0),
    "cefazolin":                  (500.0, 6000.0),
    "nitrofurantoin":             (50.0, 200.0),
    "ciprofloxacin":              (200.0, 1500.0),
    "linezolid":                  (600.0, 1200.0),
    "azithromycin":               (250.0, 500.0),
    "trimethoprim-sulfamethoxazole": (160.0, 960.0),
}


def _dose_in_window(action: EpiAction) -> bool:
    """Return True if action.dose_mg falls within the therapeutic window."""
    window = _THERAPEUTIC_WINDOWS.get(action.antibiotic.lower())
    if window is None:
        return True   # unknown drug — no range to enforce
    lo, hi = window
    return lo <= action.dose_mg <= hi


def _has_sirs(patient: Dict[str, Any]) -> bool:
    """
    Simplified SIRS check: any two of temp >38 °C, HR >90, WBC >12k.
    Uses patient dict vitals sub-dict.
    """
    vitals = patient.get("vitals", {})
    sirs_flags = 0
    if vitals.get("temp_c", 37.0) > 38.0:
        sirs_flags += 1
    if vitals.get("hr_bpm", 80) > 90:
        sirs_flags += 1
    if vitals.get("wbc_k_ul", 8.0) > 12.0:
        sirs_flags += 1
    return sirs_flags >= 2


# ---------------------------------------------------------------------------
# EHRApp
# ---------------------------------------------------------------------------

class EHRApp:
    """
    Electronic Health Record system.

    Handles: isolation orders, specialist consult documentation, patient history.

    Business rules
    --------------
    • isolation_order requires a documented clinical rationale (action.reasoning).
    • Antibiotic prescriptions must NOT be routed here — they belong to PharmacyApp.
    """

    def process(self, action: EpiAction, ctx: Dict[str, Any]) -> AppResponse:
        if action.target_app != "ehr":
            return AppResponse(routed=False)

        # Wrong-app: antibiotic prescription routed to EHR
        if action.antibiotic is not None and not action.isolation_order and not action.specialist_consult:
            return AppResponse(
                routed=True,
                processed=False,
                reward_delta=-0.02,
                message=(
                    "EHR does not process antibiotic prescriptions. "
                    "Route prescriptions to PharmacyApp."
                ),
            )

        # Wrong-app: culture/diagnostic request routed to EHR
        if action.culture_requested or action.diagnostic_test is not None:
            return AppResponse(
                routed=True,
                processed=False,
                reward_delta=-0.02,
                message=(
                    "EHR does not handle diagnostic test orders. "
                    "Route culture requests to LabApp."
                ),
            )

        reward = 0.0
        parts: list = []

        # Isolation order
        if action.isolation_order:
            if action.reasoning:
                reward += 0.05
                parts.append("Isolation order documented with clinical rationale — accepted.")
            else:
                reward -= 0.01
                parts.append(
                    "Isolation order recorded, but no clinical rationale documented. "
                    "Future orders require reasoning for audit compliance."
                )

        # Specialist consult
        if action.specialist_consult:
            reward += 0.03
            parts.append("ID specialist consult request logged.")

        if not parts:
            parts.append("EHR query processed — no administrative actions required.")

        return AppResponse(
            routed=True,
            processed=True,
            reward_delta=float(max(-0.1, min(0.1, reward))),
            message=" ".join(parts),
        )


# ---------------------------------------------------------------------------
# LabApp
# ---------------------------------------------------------------------------

class LabApp:
    """
    Laboratory Information System.

    Handles: culture requests, rapid PCR, standard culture, sensitivity panels.

    Business rules
    --------------
    • rapid_pcr requires the patient to be in ICU or meet SIRS criteria.
    • standard_culture introduces a 2-step delay before results are available.
    • +0.05 bonus if a test is ordered when pathogen uncertainty is high.
    """

    def process(self, action: EpiAction, ctx: Dict[str, Any]) -> AppResponse:
        if action.target_app != "lab":
            return AppResponse(routed=False)

        patient: Dict[str, Any] = ctx.get("patient", {})
        ward_id: str = patient.get("ward_id", "")
        in_icu = "icu" in ward_id.lower()
        sirs_met = _has_sirs(patient)
        high_uncertainty = ctx.get("high_pathogen_uncertainty", False)

        ordered_something = action.culture_requested or action.diagnostic_test is not None

        if not ordered_something:
            return AppResponse(
                routed=True,
                processed=False,
                message="No test order found. Set culture_requested=True or specify diagnostic_test.",
            )

        reward = 0.0
        parts: list = []
        latency = 0

        # rapid_pcr availability check
        if action.diagnostic_test == "rapid_pcr":
            if not (in_icu or sirs_met):
                return AppResponse(
                    routed=True,
                    processed=False,
                    reward_delta=-0.02,
                    message=(
                        "Rapid PCR is only available for ICU patients or those meeting "
                        "SIRS criteria. Test order denied."
                    ),
                )
            reward += 0.05
            parts.append("Rapid PCR ordered — result expected within 2 hours.")

        elif action.diagnostic_test == "standard_culture" or action.culture_requested:
            latency = 2
            parts.append("Standard culture ordered — result available in 2 steps.")

        elif action.diagnostic_test == "sensitivity_panel":
            reward += 0.03
            parts.append("Sensitivity panel ordered — result immediate.")

        # Bonus for testing when uncertainty is high
        if high_uncertainty and ordered_something:
            reward += 0.05
            parts.append("Testing at high-uncertainty decision point — stewardship bonus.")

        return AppResponse(
            routed=True,
            processed=True,
            reward_delta=float(max(-0.1, min(0.1, reward))),
            message=" ".join(parts) if parts else "Test order accepted.",
            latency_steps=latency,
        )


# ---------------------------------------------------------------------------
# PharmacyApp
# ---------------------------------------------------------------------------

class PharmacyApp:
    """
    Pharmacy system.

    Handles: antibiotic prescriptions, dose and frequency validation.

    Business rules
    --------------
    • meropenem requires pharmacy pre-authorisation: action.specialist_consult must be True.
    • colistin requires a prior specialist consult in this episode (ctx['specialist_consulted']).
    • +0.05 bonus if the dose falls within the PK/PD therapeutic window.
    """

    def process(self, action: EpiAction, ctx: Dict[str, Any]) -> AppResponse:
        if action.target_app != "pharmacy":
            return AppResponse(routed=False)

        drug = action.antibiotic.lower()

        # Pre-authorisation: meropenem needs specialist_consult in THIS action
        if drug == "meropenem" and not action.specialist_consult:
            return AppResponse(
                routed=True,
                processed=False,
                reward_delta=-0.05,
                message=(
                    "Meropenem requires pharmacy pre-authorisation. "
                    "Include specialist_consult=True to obtain approval."
                ),
            )

        # Colistin requires prior specialist consult in the episode
        if drug == "colistin" and not ctx.get("specialist_consulted", False):
            return AppResponse(
                routed=True,
                processed=False,
                reward_delta=-0.05,
                message=(
                    "Colistin requires a documented ID specialist consult prior to "
                    "prescribing. No prior consult found this episode."
                ),
            )

        # Dose therapeutic window check
        in_window = _dose_in_window(action)
        reward = 0.05 if in_window else -0.03

        parts = [f"{drug.capitalize()} {action.dose_mg} mg {action.route} prescription accepted."]
        if in_window:
            parts.append("Dose is within PK/PD therapeutic window.")
        else:
            parts.append(
                f"Dose {action.dose_mg} mg is outside the standard therapeutic range. "
                "Review dosing."
            )

        return AppResponse(
            routed=True,
            processed=True,
            reward_delta=float(max(-0.1, min(0.1, reward))),
            message=" ".join(parts),
        )


# ---------------------------------------------------------------------------
# MicrobiologyApp
# ---------------------------------------------------------------------------

class MicrobiologyApp:
    """
    Microbiology dashboard.

    Handles: antibiogram queries, ward resistance trends, outbreak alerts.

    Business rules
    --------------
    • Antibiogram data is stale if >7 steps have passed since the last query.
    • +0.03 bonus for consulting microbiology before the first prescription.
    """

    def process(self, action: EpiAction, ctx: Dict[str, Any]) -> AppResponse:
        if action.target_app != "microbiology":
            return AppResponse(routed=False)

        step: int = ctx.get("step", 0)
        microbiology_last_step: Optional[int] = ctx.get("microbiology_last_step")
        first_prescription_made: bool = ctx.get("first_prescription_made", False)

        reward = 0.0
        parts: list = []

        # Staleness check
        if microbiology_last_step is not None:
            steps_since = step - microbiology_last_step
            if steps_since > 7:
                reward -= 0.02
                parts.append(
                    f"Warning: antibiogram data is {steps_since} steps old (>7 step threshold). "
                    "Local resistance patterns may have shifted."
                )
            else:
                parts.append(
                    f"Antibiogram current (last updated {steps_since} step(s) ago)."
                )
        else:
            parts.append("Antibiogram loaded — initial consultation.")

        # Bonus for consulting before first prescription
        if not first_prescription_made:
            reward += 0.03
            parts.append("Stewardship bonus: microbiology consulted before first prescription.")

        return AppResponse(
            routed=True,
            processed=True,
            reward_delta=float(max(-0.1, min(0.1, reward))),
            message=" ".join(parts),
        )
