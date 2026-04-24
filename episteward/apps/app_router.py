"""
AppRouter — routes EpiAction to the correct enterprise app.

Routing priority (when action.target_app is None):
  culture_requested=True  → LabApp
  isolation_order=True    → EHRApp
  antibiotic is not None  → PharmacyApp
  (no clear signal)       → MicrobiologyApp (observation step)

Stateful tracking (reset each episode)
--------------------------------------
_specialist_consulted     — True after any action with specialist_consult=True
_microbiology_last_step   — step number of last MicrobiologyApp query
_first_prescription_made  — True after first successful PharmacyApp dispatch
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from episteward.apps.enterprise_apps import (
    AppResponse,
    EHRApp,
    LabApp,
    MicrobiologyApp,
    PharmacyApp,
)
from episteward.models import EpiAction


class AppRouter:
    """
    Routes EpiActions to the appropriate enterprise hospital app.

    Maintains per-episode state so business rules that span multiple steps
    (e.g. prior specialist consult required for colistin) work correctly.
    Reset between episodes by calling reset().
    """

    def __init__(self) -> None:
        self._ehr = EHRApp()
        self._lab = LabApp()
        self._pharmacy = PharmacyApp()
        self._microbiology = MicrobiologyApp()

        # Per-episode state — reset between episodes
        self._specialist_consulted: bool = False
        self._microbiology_last_step: Optional[int] = None
        self._first_prescription_made: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset episode-scoped state. Call at the start of each episode."""
        self._specialist_consulted = False
        self._microbiology_last_step = None
        self._first_prescription_made = False

    def route(self, action: EpiAction, state: Any) -> AppResponse:
        """
        Route action to the designated app (or infer the app if not specified).

        Parameters
        ----------
        action : EpiAction to evaluate
        state  : HospitalState or None — used to extract patient/step context
        """
        ctx = self._build_context(action, state)

        # Track specialist consults for colistin pre-auth across steps
        if action.specialist_consult:
            self._specialist_consulted = True

        target = action.target_app
        if target is None:
            response = self._infer_app(action, ctx)
        else:
            app_map = {
                "ehr":          self._ehr,
                "lab":          self._lab,
                "pharmacy":     self._pharmacy,
                "microbiology": self._microbiology,
            }
            app = app_map.get(target)
            if app is None:
                return AppResponse(
                    routed=False,
                    message=f"Unknown target_app '{target}'.",
                )
            response = app.process(action, ctx)

        # Post-process: track microbiology queries and prescription history
        if response.routed and response.processed:
            if target == "microbiology" or (
                target is None and response.message.startswith("Antibiogram")
            ):
                step = ctx.get("step", 0)
                self._microbiology_last_step = step

            if target == "pharmacy" or (
                target is None and action.antibiotic is not None
            ):
                self._first_prescription_made = True

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_app(self, action: EpiAction, ctx: Dict[str, Any]) -> AppResponse:
        """
        Infer the target app from action content when target_app is None.

        Because each app's process() guards on action.target_app, we create a
        modified action with the inferred target set before dispatching.
        """
        if action.culture_requested or action.diagnostic_test is not None:
            inferred = action.model_copy(update={"target_app": "lab"})
            return self._lab.process(inferred, ctx)
        if action.isolation_order:
            inferred = action.model_copy(update={"target_app": "ehr"})
            return self._ehr.process(inferred, ctx)
        if action.antibiotic is not None:
            inferred = action.model_copy(update={"target_app": "pharmacy"})
            return self._pharmacy.process(inferred, ctx)
        # Default: microbiology observation step
        inferred = action.model_copy(update={"target_app": "microbiology"})
        return self._microbiology.process(inferred, ctx)

    def _build_context(self, action: EpiAction, state: Any) -> Dict[str, Any]:
        """
        Extract the context dict that apps need from HospitalState.

        Uses duck-typing so tests can pass a plain dict as state.
        """
        ctx: Dict[str, Any] = {
            "specialist_consulted": self._specialist_consulted,
            "microbiology_last_step": self._microbiology_last_step,
            "first_prescription_made": self._first_prescription_made,
            "step": 0,
            "patient": {},
            "high_pathogen_uncertainty": False,
        }

        if state is None:
            return ctx

        # HospitalState
        if hasattr(state, "step_number"):
            ctx["step"] = state.step_number
            patients = getattr(state, "patients", [])
            if patients:
                ctx["patient"] = patients[0] if isinstance(patients[0], dict) else {}

        # Plain dict (for tests or Docker mode)
        elif isinstance(state, dict):
            ctx["step"] = state.get("step_number", 0)
            patients = state.get("patients", [])
            if patients:
                ctx["patient"] = patients[0] if isinstance(patients[0], dict) else {}

        return ctx

    # ------------------------------------------------------------------
    # Inference helpers — exposed for tests
    # ------------------------------------------------------------------

    def _infer_target(self, action: EpiAction) -> str:
        """Return the inferred target app name (for testing/logging)."""
        if action.culture_requested or action.diagnostic_test is not None:
            return "lab"
        if action.isolation_order:
            return "ehr"
        if action.antibiotic is not None:
            return "pharmacy"
        return "microbiology"
