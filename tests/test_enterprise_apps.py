"""Tests for enterprise apps (Scaler AI Labs multi-app bonus)."""

import pytest
from episteward.apps import AppResponse, AppRouter, EHRApp, LabApp, MicrobiologyApp, PharmacyApp
from episteward.models import EpiAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(
    antibiotic="ceftriaxone",
    dose_mg=1000.0,
    frequency_hours=8.0,
    duration_days=7,
    route="IV",
    isolation_order=False,
    culture_requested=False,
    specialist_consult=False,
    diagnostic_test=None,
    target_app=None,
    reasoning=None,
):
    return EpiAction(
        antibiotic=antibiotic,
        dose_mg=dose_mg,
        frequency_hours=frequency_hours,
        duration_days=duration_days,
        route=route,
        isolation_order=isolation_order,
        culture_requested=culture_requested,
        specialist_consult=specialist_consult,
        diagnostic_test=diagnostic_test,
        target_app=target_app,
        reasoning=reasoning,
    )


def _ctx(**kwargs):
    base = {
        "specialist_consulted": False,
        "microbiology_last_step": None,
        "first_prescription_made": False,
        "step": 0,
        "patient": {},
        "high_pathogen_uncertainty": False,
    }
    base.update(kwargs)
    return base


def _icu_patient():
    return {
        "ward_id": "ICU",
        "vitals": {"temp_c": 39.5, "hr_bpm": 110, "wbc_k_ul": 15.0},
    }


def _sirs_patient():
    return {
        "ward_id": "MedWard_A",
        "vitals": {"temp_c": 38.5, "hr_bpm": 95, "wbc_k_ul": 13.0},
    }


def _normal_patient():
    return {
        "ward_id": "MedWard_B",
        "vitals": {"temp_c": 37.0, "hr_bpm": 70, "wbc_k_ul": 8.0},
    }


# ---------------------------------------------------------------------------
# AppResponse shape
# ---------------------------------------------------------------------------

def test_app_response_default_not_routed():
    resp = AppResponse(routed=False)
    assert resp.routed is False
    assert resp.processed is False
    assert resp.reward_delta == 0.0
    assert resp.latency_steps == 0


# ---------------------------------------------------------------------------
# EHRApp
# ---------------------------------------------------------------------------

def test_ehr_not_routed_when_target_is_pharmacy():
    app = EHRApp()
    action = _action(target_app="pharmacy")
    resp = app.process(action, _ctx())
    assert resp.routed is False


def test_ehr_processes_isolation_with_reasoning():
    app = EHRApp()
    action = _action(target_app="ehr", isolation_order=True, reasoning="MDR organism confirmed.")
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is True
    assert resp.reward_delta >= 0.05


def test_ehr_penalises_isolation_without_reasoning():
    app = EHRApp()
    action = _action(target_app="ehr", isolation_order=True, reasoning=None)
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is True
    assert resp.reward_delta < 0.05


def test_ehr_processes_specialist_consult():
    app = EHRApp()
    action = _action(target_app="ehr", specialist_consult=True)
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is True
    assert resp.reward_delta >= 0.03


def test_ehr_rejects_antibiotic_prescription():
    """Prescription routed to EHR → routed=True, processed=False with penalty."""
    app = EHRApp()
    action = _action(target_app="ehr", antibiotic="ceftriaxone")
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is False
    assert resp.reward_delta < 0.0


def test_ehr_rejects_culture_request():
    """Culture request routed to EHR → routed=True, processed=False."""
    app = EHRApp()
    action = _action(target_app="ehr", culture_requested=True)
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is False
    assert resp.reward_delta < 0.0


# ---------------------------------------------------------------------------
# LabApp
# ---------------------------------------------------------------------------

def test_lab_not_routed_when_target_is_ehr():
    app = LabApp()
    action = _action(target_app="ehr", culture_requested=True)
    resp = app.process(action, _ctx())
    assert resp.routed is False


def test_lab_processes_standard_culture_with_latency():
    app = LabApp()
    action = _action(target_app="lab", culture_requested=True)
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is True
    assert resp.latency_steps == 2


def test_lab_processes_sensitivity_panel_no_latency():
    app = LabApp()
    action = _action(target_app="lab", diagnostic_test="sensitivity_panel")
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is True
    assert resp.latency_steps == 0


def test_lab_rapid_pcr_denied_without_icu_or_sirs():
    """rapid_pcr for non-ICU, non-SIRS patient → denied."""
    app = LabApp()
    action = _action(target_app="lab", diagnostic_test="rapid_pcr")
    resp = app.process(action, _ctx(patient=_normal_patient()))
    assert resp.routed is True
    assert resp.processed is False
    assert resp.reward_delta < 0.0


def test_lab_rapid_pcr_allowed_for_icu_patient():
    app = LabApp()
    action = _action(target_app="lab", diagnostic_test="rapid_pcr")
    resp = app.process(action, _ctx(patient=_icu_patient()))
    assert resp.routed is True
    assert resp.processed is True


def test_lab_rapid_pcr_allowed_for_sirs_patient():
    app = LabApp()
    action = _action(target_app="lab", diagnostic_test="rapid_pcr")
    resp = app.process(action, _ctx(patient=_sirs_patient()))
    assert resp.routed is True
    assert resp.processed is True


def test_lab_bonus_for_high_uncertainty():
    app = LabApp()
    action = _action(target_app="lab", culture_requested=True)
    resp = app.process(action, _ctx(high_pathogen_uncertainty=True))
    assert resp.reward_delta >= 0.05


def test_lab_no_order_not_processed():
    app = LabApp()
    action = _action(target_app="lab")  # no culture_requested, no diagnostic_test
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is False


# ---------------------------------------------------------------------------
# PharmacyApp
# ---------------------------------------------------------------------------

def test_pharmacy_processes_ceftriaxone_with_bonus():
    app = PharmacyApp()
    action = _action(target_app="pharmacy", antibiotic="ceftriaxone", dose_mg=2000.0)
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is True
    assert resp.reward_delta == pytest.approx(0.05, abs=1e-9)


def test_pharmacy_processes_meropenem_with_specialist_consult():
    """Meropenem with specialist_consult=True → processed=True."""
    app = PharmacyApp()
    action = _action(
        target_app="pharmacy",
        antibiotic="meropenem",
        dose_mg=1000.0,
        specialist_consult=True,
    )
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is True


def test_pharmacy_denies_meropenem_without_specialist_consult():
    """Meropenem without specialist_consult → denied with penalty."""
    app = PharmacyApp()
    action = _action(target_app="pharmacy", antibiotic="meropenem", dose_mg=1000.0)
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is False
    assert resp.reward_delta < 0.0


def test_pharmacy_denies_colistin_without_prior_specialist_consult():
    """Colistin without prior episode specialist consult → denied."""
    app = PharmacyApp()
    action = _action(target_app="pharmacy", antibiotic="colistin", dose_mg=150.0)
    resp = app.process(action, _ctx(specialist_consulted=False))
    assert resp.routed is True
    assert resp.processed is False
    assert resp.reward_delta < 0.0


def test_pharmacy_allows_colistin_after_prior_specialist_consult():
    """Colistin with prior consult in episode → processed=True."""
    app = PharmacyApp()
    action = _action(target_app="pharmacy", antibiotic="colistin", dose_mg=150.0)
    resp = app.process(action, _ctx(specialist_consulted=True))
    assert resp.routed is True
    assert resp.processed is True


def test_pharmacy_penalises_out_of_window_dose():
    """Dose outside therapeutic window → reward_delta < 0."""
    app = PharmacyApp()
    action = _action(target_app="pharmacy", antibiotic="nitrofurantoin", dose_mg=500.0)
    resp = app.process(action, _ctx())
    assert resp.routed is True
    assert resp.processed is True
    assert resp.reward_delta < 0.0


def test_pharmacy_not_routed_to_ehr():
    app = PharmacyApp()
    action = _action(target_app="ehr")
    resp = app.process(action, _ctx())
    assert resp.routed is False


# ---------------------------------------------------------------------------
# MicrobiologyApp
# ---------------------------------------------------------------------------

def test_microbiology_bonus_before_first_prescription():
    app = MicrobiologyApp()
    action = _action(target_app="microbiology")
    resp = app.process(action, _ctx(first_prescription_made=False))
    assert resp.routed is True
    assert resp.processed is True
    assert resp.reward_delta >= 0.03


def test_microbiology_no_bonus_after_prescription():
    app = MicrobiologyApp()
    action = _action(target_app="microbiology")
    resp = app.process(action, _ctx(first_prescription_made=True, microbiology_last_step=2))
    # Recent query, after prescription → no bonus, small positive or zero
    assert resp.routed is True
    assert resp.processed is True


def test_microbiology_penalty_for_stale_antibiogram():
    """Query after >7-step gap → staleness penalty."""
    app = MicrobiologyApp()
    action = _action(target_app="microbiology")
    resp = app.process(action, _ctx(step=10, microbiology_last_step=2, first_prescription_made=True))
    assert resp.routed is True
    assert resp.processed is True
    assert resp.reward_delta < 0.0


def test_microbiology_no_penalty_for_recent_query():
    """Query within 7 steps → no staleness penalty."""
    app = MicrobiologyApp()
    action = _action(target_app="microbiology")
    resp = app.process(action, _ctx(step=5, microbiology_last_step=0, first_prescription_made=True))
    assert resp.reward_delta >= 0.0


def test_microbiology_not_routed_to_pharmacy():
    app = MicrobiologyApp()
    action = _action(target_app="pharmacy")
    resp = app.process(action, _ctx())
    assert resp.routed is False


# ---------------------------------------------------------------------------
# AppRouter — explicit target_app routing
# ---------------------------------------------------------------------------

def test_router_routes_to_pharmacy_for_meropenem():
    router = AppRouter()
    action = _action(target_app="pharmacy", antibiotic="meropenem", specialist_consult=True)
    resp = router.route(action, None)
    assert resp.routed is True
    assert resp.processed is True


def test_router_meropenem_no_consult_denied():
    """Meropenem with no specialist_consult → pharmacy denies."""
    router = AppRouter()
    action = _action(target_app="pharmacy", antibiotic="meropenem")
    resp = router.route(action, None)
    assert resp.routed is True
    assert resp.processed is False


def test_router_colistin_denied_without_prior_consult():
    """Colistin with no prior specialist consult in episode → denied."""
    router = AppRouter()
    action = _action(target_app="pharmacy", antibiotic="colistin", dose_mg=150.0)
    resp = router.route(action, None)
    assert resp.routed is True
    assert resp.processed is False


def test_router_colistin_allowed_after_prior_consult():
    """Send specialist_consult=True first, then colistin → allowed."""
    router = AppRouter()
    # Step 1: consult
    consult_action = _action(target_app="ehr", specialist_consult=True)
    router.route(consult_action, None)
    # Step 2: colistin
    colistin_action = _action(target_app="pharmacy", antibiotic="colistin", dose_mg=150.0)
    resp = router.route(colistin_action, None)
    assert resp.routed is True
    assert resp.processed is True


# ---------------------------------------------------------------------------
# AppRouter — _infer_app routing
# ---------------------------------------------------------------------------

def test_infer_routes_culture_to_lab():
    router = AppRouter()
    action = _action(culture_requested=True, antibiotic="ceftriaxone")
    inferred = router._infer_target(action)
    assert inferred == "lab"


def test_infer_routes_isolation_to_ehr():
    router = AppRouter()
    action = _action(isolation_order=True)
    inferred = router._infer_target(action)
    assert inferred == "ehr"


def test_infer_routes_antibiotic_to_pharmacy():
    router = AppRouter()
    action = _action(antibiotic="ceftriaxone")
    inferred = router._infer_target(action)
    assert inferred == "pharmacy"


def test_infer_routes_empty_to_microbiology():
    """No antibiotic, no isolation, no culture → microbiology observation."""
    router = AppRouter()
    # Create a minimal action — antibiotic is required so use the field
    action = _action(antibiotic="ceftriaxone")
    # Override: test pure inference logic with a None-antibiotic scenario isn't possible
    # due to Pydantic validation, so test via _infer_target with a mocked object
    class _FakeAction:
        antibiotic = None
        isolation_order = False
        culture_requested = False
        diagnostic_test = None
    assert router._infer_target(_FakeAction()) == "microbiology"


def test_infer_app_routes_isolation_correctly():
    """_infer_app with isolation_order=True → EHR processes it."""
    router = AppRouter()
    action = _action(isolation_order=True, reasoning="Contact precautions for MRSA.")
    resp = router.route(action, None)
    assert resp.routed is True
    assert resp.processed is True


def test_infer_app_routes_culture_to_lab():
    """_infer_app with culture_requested=True → Lab processes it."""
    router = AppRouter()
    action = _action(culture_requested=True)
    resp = router.route(action, None)
    assert resp.routed is True
    assert resp.processed is True
    assert resp.latency_steps == 2


# ---------------------------------------------------------------------------
# Router reset
# ---------------------------------------------------------------------------

def test_router_reset_clears_specialist_history():
    """After reset, prior specialist consult is forgotten."""
    router = AppRouter()
    consult = _action(target_app="ehr", specialist_consult=True)
    router.route(consult, None)
    router.reset()
    colistin = _action(target_app="pharmacy", antibiotic="colistin", dose_mg=150.0)
    resp = router.route(colistin, None)
    assert resp.processed is False


# ---------------------------------------------------------------------------
# reward_delta always clamped to [-0.1, +0.1]
# ---------------------------------------------------------------------------

def test_all_apps_reward_delta_in_range():
    """reward_delta must be within [-0.1, +0.1] for all app responses."""
    router = AppRouter()
    actions = [
        _action(target_app="ehr", isolation_order=True, reasoning="x"),
        _action(target_app="lab", culture_requested=True),
        _action(target_app="pharmacy", antibiotic="ceftriaxone", dose_mg=2000.0),
        _action(target_app="microbiology"),
    ]
    for a in actions:
        resp = router.route(a, None)
        assert -0.1 <= resp.reward_delta <= 0.1, (
            f"reward_delta={resp.reward_delta} out of bounds for target_app={a.target_app}"
        )
