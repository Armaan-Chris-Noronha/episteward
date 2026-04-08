"""
Pydantic v2 models for EpiSteward.

All API boundary types live here:
  - EpiObservation  — what the agent sees each step
  - EpiAction       — what the agent submits each step
  - EpiReward       — reward breakdown returned by graders
  - StepResult      — envelope returned by /step and /reset
  - StateResult     — envelope returned by /state (read-only)
  - ResetRequest    — optional body for POST /reset
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_VALID_ROUTES = {"IV", "PO", "IM"}
_VALID_FREQUENCIES = {4.0, 6.0, 8.0, 12.0, 24.0}


class EpiObservation(BaseModel):
    """Full observation delivered to the agent at each step."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "P001",
                "ward_id": "ICU",
                "infection_site": "bloodstream",
                "symptoms": ["fever", "hypotension", "tachycardia"],
                "vitals": {
                    "temp_c": 39.2,
                    "hr_bpm": 118,
                    "wbc_k_ul": 18.7,
                    "crp_mg_l": 120.0,
                    "procalcitonin_ng_ml": 4.2,
                },
                "culture_results": {"status": "pending"},
                "resistance_flags": ["ESBL"],
                "transfer_history": ["EmergencyDept", "ICU"],
                "antibiotic_history": [],
                "network_alert": None,
                "step_number": 1,
            }
        }
    )

    patient_id: str
    ward_id: str
    infection_site: str
    symptoms: List[str]
    vitals: Dict[str, float]         # temp_c, hr_bpm, wbc_k_ul, crp_mg_l, procalcitonin_ng_ml
    culture_results: Dict[str, Any]  # may have missing fields for realism
    resistance_flags: List[str]      # e.g. ["ESBL", "MRSA", "CRE"]
    transfer_history: List[str]      # ward IDs visited, chronological
    antibiotic_history: List[Dict[str, Any]]
    network_alert: Optional[str] = None
    step_number: int


class EpiAction(BaseModel):
    """Action submitted by the agent each step."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "antibiotic": "meropenem",
                "dose_mg": 1000.0,
                "frequency_hours": 8.0,
                "duration_days": 7,
                "route": "IV",
                "isolation_order": True,
                "culture_requested": True,
                "specialist_consult": False,
                "reasoning": "Broad-spectrum empiric coverage for suspected ESBL BSI",
            }
        }
    )

    antibiotic: str                   # must match a key in antibiotics.json
    dose_mg: float = Field(gt=0, description="Dose in milligrams — must be positive")
    frequency_hours: float = Field(description="Dosing interval — must be one of 4, 6, 8, 12, 24")
    duration_days: int = Field(gt=0, description="Treatment duration in days")
    route: str = Field(description='Administration route — "IV", "PO", or "IM"')
    isolation_order: bool = False
    culture_requested: bool = False
    specialist_consult: bool = False
    reasoning: Optional[str] = None  # logged but not graded

    @field_validator("frequency_hours")
    @classmethod
    def frequency_must_be_standard(cls, v: float) -> float:
        """Reject non-standard dosing intervals."""
        if float(v) not in _VALID_FREQUENCIES:
            raise ValueError(
                f"frequency_hours must be one of {sorted(_VALID_FREQUENCIES)}, got {v}"
            )
        return float(v)

    @field_validator("route")
    @classmethod
    def route_must_be_valid(cls, v: str) -> str:
        """Reject routes outside IV / PO / IM."""
        if v not in _VALID_ROUTES:
            raise ValueError(
                f"route must be one of {_VALID_ROUTES}, got '{v}'"
            )
        return v


class EpiReward(BaseModel):
    """Reward breakdown returned by a grader."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": 0.72,
                "components": {
                    "pkpd": 0.28,
                    "stewardship": 0.25,
                    "coverage": 0.19,
                },
                "done": False,
                "info": {"step": 2, "drug_class_matched": "carbapenem"},
            }
        }
    )

    value: float = Field(description="Scalar reward in [0.0, 1.0]")
    components: Dict[str, float] = Field(
        description="Per-component breakdown, e.g. pkpd, stewardship, resistance, coverage"
    )
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def clamp_value(self) -> "EpiReward":
        """Hard-clamp reward to [0, 1] to satisfy OpenEnv contract."""
        self.value = float(min(1.0, max(0.0, self.value)))
        return self


class StepResult(BaseModel):
    """Envelope returned by POST /reset and POST /step."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "observation": {"patient_id": "P001", "step_number": 1},
                "reward": 0.45,
                "done": False,
                "info": {"drug_class": "carbapenem"},
            }
        }
    )

    observation: EpiObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """Read-only snapshot returned by GET /state. Must not advance episode."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "task1_triage",
                "step_number": 3,
                "episode_seed": 42,
                "hospital_state": {"patients": {}, "ward_infection_counts": {}},
                "is_done": False,
            }
        }
    )

    task_id: str
    step_number: int
    episode_seed: int
    hospital_state: Dict[str, Any]  # serialized HospitalState
    is_done: bool


class ResetRequest(BaseModel):
    """Optional body for POST /reset. All fields optional so {} is valid."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"task_id": "task2_containment", "seed": 42}
        }
    )

    task_id: Optional[str] = "task1_triage"
    seed: Optional[int] = None
