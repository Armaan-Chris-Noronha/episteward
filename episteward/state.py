"""
HospitalState — the mutable episode state shared across tasks and graders.

This dataclass holds all simulation state for one episode. It is seeded in
reset() so episodes are reproducible. Tasks mutate it; /state serializes it
read-only.

Public API
----------
to_observation(patient_id)              -> EpiObservation
apply_action(action, math_modules=None) -> None
is_terminal(max_steps)                  -> bool
clone()                                 -> HospitalState   # deep copy
to_dict()                               -> Dict[str, Any]  # serialization
seed(seed)                              -> None
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# Patient record helper (internal)
# --------------------------------------------------------------------------

@dataclass
class PatientRecord:
    """Per-patient mutable state within an episode."""

    patient_id: str
    ward_id: str
    pathogen: Optional[str] = None
    resistance_frequency: float = 0.0  # Wright-Fisher allele frequency
    is_isolated: bool = False
    is_treated: bool = False
    culture_pending: bool = False
    culture_result: Optional[str] = None  # "resistant" | "sensitive" | None
    infection_site: str = "bloodstream"
    symptoms: List[str] = field(default_factory=list)
    vitals: Dict[str, float] = field(default_factory=lambda: {
        "temp_c": 37.0, "hr_bpm": 80, "wbc_k_ul": 9.0,
        "crp_mg_l": 10.0, "procalcitonin_ng_ml": 0.1,
    })
    treatment_hours_elapsed: float = 0.0
    transfer_history: List[str] = field(default_factory=list)
    antibiotic_history: List[Dict[str, Any]] = field(default_factory=list)
    alive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return plain-dict representation for List[Dict] storage."""
        return {
            "patient_id": self.patient_id,
            "ward_id": self.ward_id,
            "pathogen": self.pathogen,
            "resistance_frequency": self.resistance_frequency,
            "is_isolated": self.is_isolated,
            "is_treated": self.is_treated,
            "culture_pending": self.culture_pending,
            "culture_result": self.culture_result,
            "infection_site": self.infection_site,
            "symptoms": list(self.symptoms),
            "vitals": dict(self.vitals),
            "treatment_hours_elapsed": self.treatment_hours_elapsed,
            "transfer_history": list(self.transfer_history),
            "antibiotic_history": [dict(h) for h in self.antibiotic_history],
            "alive": self.alive,
        }


# --------------------------------------------------------------------------
# HospitalState
# --------------------------------------------------------------------------

@dataclass
class HospitalState:
    """
    Full mutable state for one EpiSteward episode.

    ``patients`` is a ``List[Dict]`` snapshot of per-patient data.
    ``ward_assignments`` provides fast patient_id → ward_id lookup.
    ``resistance_map`` tracks ward-level resistance allele frequencies per pathogen.
    ``isolation_map`` tracks contact-precaution status per ward.
    """

    # ---- Fields per spec -----------------------------------------------
    patients: List[Dict[str, Any]] = field(default_factory=list)
    ward_assignments: Dict[str, str] = field(default_factory=dict)
    resistance_map: Dict[str, Dict[str, float]] = field(default_factory=dict)
    isolation_map: Dict[str, bool] = field(default_factory=dict)
    colistin_budget: int = 10
    colistin_used: int = 0
    step_number: int = 0
    episode_seed: int = 0
    active_task: str = "task1_triage"
    new_resistance_events: int = 0
    transmission_chain: List[str] = field(default_factory=list)

    # ---- Backward-compatible fields (used by existing task implementations) --
    ward_infection_counts: Dict[str, int] = field(default_factory=dict)

    # ---- Derived / internal -----------------------------------------------
    is_done: bool = False
    rng: Any = field(default_factory=lambda: np.random.default_rng())

    # Cached static data (not copied in clone — shared references are safe)
    _antibiotics: Dict[str, Any] = field(default_factory=dict, repr=False)
    _pathogens: Dict[str, Any] = field(default_factory=dict, repr=False)
    _resistance_profiles: Dict[str, Any] = field(default_factory=dict, repr=False)
    _hospital_network: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Load static data files and initialize RNG from episode_seed."""
        self._antibiotics = json.loads((DATA_DIR / "antibiotics.json").read_text())
        self._pathogens = json.loads((DATA_DIR / "pathogens.json").read_text())
        self._resistance_profiles = json.loads(
            (DATA_DIR / "resistance_profiles.json").read_text()
        )
        self._hospital_network = json.loads(
            (DATA_DIR / "hospital_network.json").read_text()
        )
        self.rng = np.random.default_rng(self.episode_seed)

    # ---- Seed / reset -------------------------------------------------------

    def seed(self, seed: int) -> None:
        """Re-initialize the episode RNG for reproducibility."""
        self.episode_seed = seed
        self.rng = np.random.default_rng(seed)

    # ---- Core API -----------------------------------------------------------

    def to_observation(self, patient_id: str) -> "EpiObservation":
        """
        Build an EpiObservation for the given patient from current state.

        Imports EpiObservation lazily to avoid circular imports at module level.
        """
        from episteward.models import EpiObservation  # lazy to avoid circular

        patient = self._get_patient_dict(patient_id)
        if patient is None:
            raise KeyError(f"Patient '{patient_id}' not found in current state")

        ward_id = patient.get("ward_id", "")
        resistance_flags = self._infer_resistance_flags(patient)
        network_alert = self._network_alert_for_ward(ward_id)

        return EpiObservation(
            patient_id=patient_id,
            ward_id=ward_id,
            infection_site=patient.get("infection_site", "unknown"),
            symptoms=list(patient.get("symptoms", [])),
            vitals=dict(patient.get("vitals", {})),
            culture_results=self._culture_dict(patient),
            resistance_flags=resistance_flags,
            transfer_history=list(patient.get("transfer_history", [])),
            antibiotic_history=list(patient.get("antibiotic_history", [])),
            network_alert=network_alert,
            step_number=self.step_number,
        )

    def apply_action(
        self,
        action: "EpiAction",
        math_modules: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Advance state by one step given the agent's action.

        Side effects:
          - Sets isolation_map for the target patient's ward if isolation_order=True
          - Evolves resistance frequency via Wright-Fisher (if pathogen present)
          - Logs antibiotic to patient's antibiotic_history
          - Increments colistin_used if colistin prescribed
          - Runs one network spread step
          - Increments step_number
          - Updates new_resistance_events count

        Parameters
        ----------
        action       : validated EpiAction from the agent
        math_modules : optional dict overriding math callables for testing.
                       Recognized keys: "evolve_resistance",
                       "simulate_spread_step", "compute_selective_coefficient".
                       If None, imports from episteward.math directly.
        """
        mods = math_modules or {}

        # Resolve math functions (allow injection for tests)
        evolve_fn = mods.get("evolve_resistance", _default_evolve)
        selective_fn = mods.get("compute_selective_coefficient", _default_selective)
        spread_fn = mods.get("simulate_spread_step", _default_spread)

        # 1. Isolation order
        if action.isolation_order:
            # Isolate the ward of the *first* infected patient (task-agnostic fallback)
            for p in self.patients:
                if p.get("pathogen"):
                    self.isolation_map[p["ward_id"]] = True
                    break

        # 2. Evolve resistance for all infected patients on this antibiotic
        for p in self.patients:
            pathogen = p.get("pathogen")
            if not pathogen or not p.get("alive", True):
                continue
            try:
                s = selective_fn(action.antibiotic, action.dose_mg,
                                  mic=self._get_mic(pathogen, action.antibiotic))
            except Exception:
                s = 0.0
            old_freq = p.get("resistance_frequency", 0.0)
            new_freq = evolve_fn(old_freq, s, self.rng)
            p["resistance_frequency"] = new_freq

            # Track resistance emergence events
            if new_freq > 0.5 and old_freq <= 0.5:
                self.new_resistance_events += 1

            # Mirror into resistance_map
            ward = p.get("ward_id", "")
            if ward:
                self.resistance_map.setdefault(ward, {})[pathogen] = new_freq

            # Advance treatment clock
            p["treatment_hours_elapsed"] = (
                p.get("treatment_hours_elapsed", 0.0) + 24.0
            )
            p["is_treated"] = True

        # 3. Log antibiotic action to each patient's history
        action_record = {
            "antibiotic": action.antibiotic,
            "dose_mg": action.dose_mg,
            "frequency_hours": action.frequency_hours,
            "duration_days": action.duration_days,
            "route": action.route,
            "step": self.step_number,
        }
        for p in self.patients:
            if p.get("alive", True):
                p.setdefault("antibiotic_history", []).append(action_record)

        # 4. Colistin budget tracking
        if action.antibiotic.lower() == "colistin":
            self.colistin_used += 1

        # 5. Culture requests
        if action.culture_requested:
            for p in self.patients:
                if p.get("alive", True) and not p.get("culture_result"):
                    p["culture_pending"] = True

        # 6. Network spread step
        from episteward.math.network import build_graph
        G = build_graph(self._hospital_network)
        infected_wards = {
            p["ward_id"] for p in self.patients
            if p.get("pathogen") and p.get("alive", True)
        }
        pathogen_names = {
            p["pathogen"] for p in self.patients
            if p.get("pathogen") and p.get("alive", True)
        }
        for pathogen_name in pathogen_names:
            newly = spread_fn(
                infected_wards, pathogen_name, self.isolation_map, self.rng, graph=G
            )
            self.transmission_chain.extend(sorted(newly))

        # 7. Update ward_assignments mirror
        for p in self.patients:
            self.ward_assignments[p["patient_id"]] = p.get("ward_id", "")

        # 8. Advance step
        self.step_number += 1

    def is_terminal(self, max_steps: int) -> bool:
        """
        Return True when the episode should end.

        Conditions:
          - step_number has reached max_steps
          - all patients are dead or cured (no living infected patients remain)
        """
        if self.step_number >= max_steps:
            return True
        living_infected = any(
            p.get("pathogen") and p.get("alive", True)
            for p in self.patients
        )
        return not living_infected

    def clone(self) -> "HospitalState":
        """
        Return a deep copy of this state.

        Mutations to the clone do not affect the original and vice versa.
        The clone's RNG is seeded identically so it will produce the same
        sequence as the original from this point forward.
        """
        cloned = HospitalState.__new__(HospitalState)

        # Deep-copy all mutable episode state
        cloned.patients = copy.deepcopy(self.patients)
        cloned.ward_assignments = dict(self.ward_assignments)
        cloned.resistance_map = {
            ward: dict(pathogens)
            for ward, pathogens in self.resistance_map.items()
        }
        cloned.isolation_map = dict(self.isolation_map)
        cloned.colistin_budget = self.colistin_budget
        cloned.colistin_used = self.colistin_used
        cloned.step_number = self.step_number
        cloned.episode_seed = self.episode_seed
        cloned.active_task = self.active_task
        cloned.new_resistance_events = self.new_resistance_events
        cloned.transmission_chain = list(self.transmission_chain)
        cloned.is_done = self.is_done

        # Copy RNG state so clone produces same sequence
        cloned.rng = copy.deepcopy(self.rng)

        # Share read-only static data (safe — never mutated)
        cloned._antibiotics = self._antibiotics
        cloned._pathogens = self._pathogens
        cloned._resistance_profiles = self._resistance_profiles
        cloned._hospital_network = self._hospital_network

        return cloned

    # ---- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict for /state response (read-only snapshot)."""
        return {
            "task_id": self.active_task,
            "episode_seed": self.episode_seed,
            "step_number": self.step_number,
            "is_done": self.is_done,
            "colistin_budget": self.colistin_budget,
            "colistin_used": self.colistin_used,
            "new_resistance_events": self.new_resistance_events,
            "transmission_chain": list(self.transmission_chain),
            "patients": copy.deepcopy(self.patients),
            "ward_assignments": dict(self.ward_assignments),
            "resistance_map": {
                w: dict(p) for w, p in self.resistance_map.items()
            },
            "isolation_map": dict(self.isolation_map),
        }

    # ---- Internal helpers ---------------------------------------------------

    def _get_patient_dict(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Return the patient dict for the given ID, or None."""
        for p in self.patients:
            if p.get("patient_id") == patient_id:
                return p
        return None

    def _infer_resistance_flags(self, patient: Dict[str, Any]) -> List[str]:
        """Derive resistance flags from patient pathogen + allele frequency."""
        pathogen = patient.get("pathogen", "")
        if not pathogen:
            return []
        flags = []
        # Map pathogen names to AMR flags
        _flag_map = {
            "E_coli_ESBL": "ESBL",
            "K_pneumoniae_CRK": "CRK",
            "S_aureus_MRSA": "MRSA",
            "E_faecium_VRE": "VRE",
            "P_aeruginosa_MDR": "MDR",
        }
        flag = _flag_map.get(pathogen)
        if flag:
            flags.append(flag)
        # Add CRE if allele frequency crossed threshold
        if patient.get("resistance_frequency", 0.0) > 0.5:
            if "CRE" not in flags:
                flags.append("CRE")
        return flags

    def _culture_dict(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Build culture_results dict from patient state."""
        if patient.get("culture_result"):
            return {
                "status": "final",
                "result": patient["culture_result"],
                "pathogen": patient.get("pathogen"),
            }
        if patient.get("culture_pending"):
            return {"status": "pending"}
        return {"status": "not_requested"}

    def _network_alert_for_ward(self, ward_id: str) -> Optional[str]:
        """Return a network alert string if transmission was detected nearby."""
        if not self.transmission_chain:
            return None
        if ward_id in self.transmission_chain:
            return f"Transmission detected in {ward_id}"
        return None

    def _get_mic(self, pathogen: str, antibiotic: str) -> float:
        """Look up MIC breakpoint for pathogen/antibiotic pair."""
        drug_entry = self._antibiotics.get(antibiotic.lower(), {})
        mic_bp = drug_entry.get("mic_breakpoints", {})
        # Use susceptible breakpoint as the MIC target
        return float(mic_bp.get("susceptible", 2.0))


# --------------------------------------------------------------------------
# Default math callables (avoid circular imports at module level)
# --------------------------------------------------------------------------

def _default_evolve(p: float, s: float, rng: Any) -> float:
    from episteward.math.evolution import evolve_resistance
    return evolve_resistance(p, s, rng)


def _default_selective(drug_name: str, dose_mg: float, mic: float = 2.0) -> float:
    from episteward.math.evolution import compute_selective_coefficient
    return compute_selective_coefficient(drug_name, dose_mg, mic)


def _default_spread(
    infected_set: set,
    pathogen_name: str,
    isolation_map: Dict[str, bool],
    rng: Any,
    graph: Any = None,
) -> set:
    from episteward.math.network import simulate_spread_step
    return simulate_spread_step(infected_set, pathogen_name, isolation_map, rng, graph=graph)
