"""
Task 4 — MultiWardStewardshipGame (Expert).

5 ward agents + 1 coordinator agent (the LLM being trained).

Each ward agent has a broadness tendency σ_i ∈ [0, 1]:
  σ = 0 — narrow-spectrum prescribing (stewardship-optimal)
  σ = 1 — broad-spectrum prescribing (creates AMR over time)

Ward sigma update each step (peer pressure + coordinator signal):
  σ_new = clip(0.7·σ_old + 0.2·peer_avg + 0.1·coordinator_signal
               − 0.1·isolation_order − 0.1·specialist_consult, 0, 1)

Coordinator signal decoded from EpiAction.antibiotic:
  narrow  (nitrofurantoin, cefazolin, ampicillin, …)  → signal = 0.1
  moderate (ceftriaxone, piperacillin-tazobactam, …)  → signal = 0.5
  broad   (meropenem, colistin)                       → signal = 0.9
  target_app="ehr" → observation-only step (no sigma update)

Resistance dynamics each step:
  R_new = clip(R_old + σ·0.03 − 0.01, 0, 0.95)
  cure_rate = clip(base − R·0.30, 0, 1)  where base = 0.85 if σ≥0.5 else 0.75
  ward failure: cure_rate < 0.70

Episode: 20 steps.  Ground truth passes pre/post-step sigmas to MultiAgentGrader.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from episteward.models import EpiAction, EpiObservation
from episteward.state import HospitalState
from episteward.tasks.base import BaseTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WARDS: int = 5
_MAX_STEPS: int = 20

# Ward params that produce PoA > 1 (tragedy of commons)
# alpha=0.13 → Nash at σ=1, social optimum at σ≈0.58
_WARD_PARAMS: Dict[str, float] = {"f_min": 0.70, "f_max": 0.95, "alpha": 0.13, "beta": 0.80}

# Failure threshold: cure rate below this counts as a ward treatment failure
_FAILURE_THRESHOLD: float = 0.70

# Coordinator signal by antibiotic spectrum
_BROAD_DRUGS = {"meropenem", "colistin"}
_MODERATE_DRUGS = {"ceftriaxone", "piperacillin-tazobactam", "ertapenem"}
# All others → narrow signal


def _decode_signal(action: EpiAction) -> float:
    """Map coordinator antibiotic choice to broadness signal [0.1, 0.5, 0.9]."""
    drug = action.antibiotic.lower()
    if drug in _BROAD_DRUGS:
        return 0.9
    if drug in _MODERATE_DRUGS:
        return 0.5
    return 0.1  # narrow default


# ---------------------------------------------------------------------------
# MultiWardStewardshipGame task
# ---------------------------------------------------------------------------


class MultiWardStewardshipGame(BaseTask):
    """
    Coordinator controls 5 ward agents over 20 steps via EpiAction signals.

    The coordinator's job: reduce inter-ward broadness (σ), minimise resistance
    growth and treatment failures, and drive the prescribing game toward the
    socially optimal equilibrium (low Price of Anarchy).
    """

    max_steps = _MAX_STEPS
    name = "task4_multiagent"

    def __init__(self) -> None:
        super().__init__()
        self._ward_sigmas: np.ndarray = np.zeros(N_WARDS)
        self._ward_resistance: np.ndarray = np.zeros(N_WARDS)
        self._ward_cure_rates: np.ndarray = np.zeros(N_WARDS)
        # Snapshots for ground_truth after each step
        self._sigmas_before: List[float] = [0.0] * N_WARDS
        self._last_failures: int = 0

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0) -> EpiObservation:
        """Initialise a new 5-ward episode."""
        self.state = HospitalState(active_task=self.name, episode_seed=seed)
        self.state.step_number = 1
        rng = np.random.default_rng(seed)

        # Initialise wards near Nash (high σ) so coordinator has room to improve.
        # Resistance starts low enough that no wards begin in failure state.
        self._ward_sigmas = rng.uniform(0.70, 0.95, N_WARDS)
        self._ward_resistance = rng.uniform(0.08, 0.22, N_WARDS)
        self._ward_cure_rates = self._compute_cure_rates()

        # Snapshots — start of episode, no action yet
        self._sigmas_before = list(self._ward_sigmas.copy())
        self._last_failures = 0

        self._sync_hospital_state()
        return self._make_observation()

    def step(self, action: EpiAction) -> Tuple[EpiObservation, bool]:
        """Apply coordinator action, update ward sigmas and dynamics."""
        self._assert_ready()
        assert self.state is not None

        # Snapshot sigmas BEFORE this action
        self._sigmas_before = list(self._ward_sigmas.copy())

        # ---- Decode coordinator signal ----
        obs_only = action.target_app == "ehr"
        signal = _decode_signal(action) if not obs_only else float(np.mean(self._ward_sigmas))

        # ---- Update ward sigmas ----
        if not obs_only:
            peer_avg = float(np.mean(self._ward_sigmas))
            new_sigmas = np.zeros(N_WARDS)
            for i in range(N_WARDS):
                s = (
                    0.7 * self._ward_sigmas[i]
                    + 0.2 * peer_avg
                    + 0.1 * signal
                    - (0.1 if action.isolation_order else 0.0)
                    - (0.1 if action.specialist_consult else 0.0)
                )
                new_sigmas[i] = float(np.clip(s, 0.0, 1.0))
            self._ward_sigmas = new_sigmas

        # ---- Update resistance and cure rates ----
        for i in range(N_WARDS):
            delta = self._ward_sigmas[i] * 0.03 - 0.01
            self._ward_resistance[i] = float(
                np.clip(self._ward_resistance[i] + delta, 0.0, 0.95)
            )
        self._ward_cure_rates = self._compute_cure_rates()

        # ---- Count ward treatment failures ----
        self._last_failures = int(
            np.sum(self._ward_cure_rates < _FAILURE_THRESHOLD)
        )

        self.state.step_number += 1
        done = self.state.step_number > self.max_steps
        if done:
            self.state.is_done = True

        self._sync_hospital_state()
        return self._make_observation(), done

    @property
    def ground_truth(self) -> Dict[str, Any]:
        """Ground truth for MultiAgentGrader — never exposed to agent."""
        return {
            "sigmas_before": list(self._sigmas_before),
            "sigmas_after": list(self._ward_sigmas),
            "ward_params": dict(_WARD_PARAMS),
            "ward_failures": self._last_failures,
            "n_wards": N_WARDS,
            "ward_resistance": list(self._ward_resistance),
            "ward_cure_rates": list(self._ward_cure_rates),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_cure_rates(self) -> np.ndarray:
        # Cure rate depends only on resistance, not sigma.
        # Broad prescribing increases resistance → eventual cure rate decline.
        # Narrow prescribing reduces resistance → sustained cure rates.
        cure = np.zeros(N_WARDS)
        for i in range(N_WARDS):
            cure[i] = float(np.clip(0.85 - self._ward_resistance[i] * 0.40, 0.0, 1.0))
        return cure

    def _sync_hospital_state(self) -> None:
        """Keep HospitalState in sync with ward simulation for serialisation."""
        assert self.state is not None
        # Resistance map: ward-level resistance prevalence by pathogen
        self.state.resistance_map = {
            f"Ward_{i}": {"E_coli_ESBL": float(self._ward_resistance[i])}
            for i in range(N_WARDS)
        }
        # Patients: one synthetic agent per ward
        self.state.patients = [
            {
                "patient_id": f"WARD_{i}_AGENT",
                "ward_id": f"Ward_{i}",
                "pathogen": "E_coli_ESBL",
                "resistance_frequency": float(self._ward_sigmas[i]),
                "is_isolated": False,
                "is_treated": False,
                "culture_pending": False,
                "culture_result": None,
                "infection_site": "multi_ward",
                "symptoms": [],
                "vitals": {
                    "sigma": float(self._ward_sigmas[i]),
                    "cure_rate": float(self._ward_cure_rates[i]),
                    "resistance": float(self._ward_resistance[i]),
                },
                "treatment_hours_elapsed": 0.0,
                "transfer_history": [f"Ward_{i}"],
                "antibiotic_history": [],
                "alive": True,
            }
            for i in range(N_WARDS)
        ]

    def _make_observation(self) -> EpiObservation:
        """Package multi-ward state into a single EpiObservation."""
        assert self.state is not None
        step = self.state.step_number

        mean_sigma = float(np.mean(self._ward_sigmas))
        mean_resistance = float(np.mean(self._ward_resistance))
        mean_cure = float(np.mean(self._ward_cure_rates))

        # Compute WRPI from resistance map (severity-weighted sum, normalised)
        total_resistance = float(np.sum(self._ward_resistance))
        wrpi = float(min(1.0, total_resistance / (N_WARDS * 1.5)))

        # Compute approximate PoA for the observation using current sigmas
        try:
            from episteward.math.game_theory import (
                price_of_anarchy,
                social_optimum,
            )
            opt = social_optimum(N_WARDS, _WARD_PARAMS)
            current_poa = price_of_anarchy(
                self._ward_sigmas, opt, N_WARDS, _WARD_PARAMS
            )
        except Exception:
            current_poa = 1.0

        # Symptoms encode actionable alerts
        symptoms: List[str] = []
        if mean_sigma > 0.7:
            symptoms.append("broad_prescribing_detected")
        if mean_resistance > 0.5:
            symptoms.append("high_ward_resistance")
        if self._last_failures > 0:
            symptoms.append(f"treatment_failures_{self._last_failures}_wards")

        # Culture results: per-ward state (Dict[str, Any] allows nested dicts)
        culture_results: Dict[str, Any] = {
            f"Ward_{i}": {
                "sigma": round(float(self._ward_sigmas[i]), 3),
                "resistance_prevalence": round(float(self._ward_resistance[i]), 3),
                "cure_rate": round(float(self._ward_cure_rates[i]), 3),
                "treatment_failure": bool(self._ward_cure_rates[i] < _FAILURE_THRESHOLD),
            }
            for i in range(N_WARDS)
        }
        culture_results["summary"] = {
            "mean_sigma": round(mean_sigma, 3),
            "mean_resistance": round(mean_resistance, 3),
            "poa": round(current_poa, 4),
        }

        # Resistance flags for wards above threshold
        resistance_flags: List[str] = [
            f"Ward_{i}_HIGH_AMR"
            for i in range(N_WARDS)
            if self._ward_resistance[i] > 0.50
        ]
        if not resistance_flags:
            resistance_flags = ["no_critical_resistance"]

        network_alert = (
            f"5-ward game: mean_sigma={mean_sigma:.2f}, "
            f"mean_R={mean_resistance:.2f}, "
            f"poa={current_poa:.3f}, "
            f"failures={self._last_failures}, "
            f"wrpi={wrpi:.2f}"
        )

        return EpiObservation(
            patient_id="COORDINATOR",
            ward_id="NETWORK",
            infection_site="multi_ward",
            symptoms=symptoms,
            vitals={
                "wrpi": round(wrpi, 3),
                "poa": round(current_poa, 4),
                "mean_sigma": round(mean_sigma, 3),
                "mean_resistance": round(mean_resistance, 3),
                "mean_cure_rate": round(mean_cure, 3),
            },
            culture_results=culture_results,
            resistance_flags=resistance_flags,
            transfer_history=[f"Ward_{i}" for i in range(N_WARDS)],
            antibiotic_history=[],
            network_alert=network_alert,
            step_number=step,
        )
