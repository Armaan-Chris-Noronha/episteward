"""Tests for pure math modules: PK/PD, Wright-Fisher, network, Bayes."""

import json
from pathlib import Path

import numpy as np
import pytest

from episteward.math.pkpd import (
    concentration_profile,
    get_concentration_curve,
    get_pkpd_score,
    hill_effect,
    is_in_therapeutic_window,
    therapeutic_score,
)
from episteward.math.evolution import (
    evolve_resistance,
    evolve_resistance_legacy,
    wright_fisher_step,
    compute_selective_coefficient,
    resistance_emerged,
    get_resistance_trajectory,
)
from episteward.math.network import (
    build_graph,
    transmission_probability,
    compute_transmission_probability,
    get_at_risk_wards,
    simulate_spread_step,
    get_transmission_chain,
)
from episteward.math.bayes import (
    estimate_resistance,
    update_posterior,
    get_resistance_probability,
    get_empiric_recommendation,
)
from episteward.math.bayesian_diagnostics import (
    PATHOGENS,
    init_prior,
    update_posterior as diag_update_posterior,
    shannon_entropy,
    value_of_information,
    get_diagnostic_reward,
)
from episteward.math.pareto_reward import (
    compute_reward_vector,
    compute_wrpi,
    compute_adaptive_weights,
    scalarize,
    is_dominated,
    update_pareto_front,
    compute_hypervolume,
)
from episteward.math.game_theory import (
    ward_utility,
    nash_equilibrium,
    social_optimum,
    price_of_anarchy,
    get_game_reward,
)

_NET = json.loads(
    (Path(__file__).parent.parent / "episteward/data/hospital_network.json").read_text()
)


# ---------------------------------------------------------------------------
# PK/PD — internal helpers
# ---------------------------------------------------------------------------

class TestConcentrationProfile:
    _PK = {"F": 1.0, "Vd_L_kg": 0.3, "CL_L_h_kg": 0.1, "ke": 0.33}

    def test_decays_over_time(self):
        t, C = concentration_profile(1000.0, self._PK, t_span=(0.0, 24.0))
        assert C[0] > C[-1]

    def test_non_negative(self):
        t, C = concentration_profile(1000.0, self._PK)
        assert np.all(C >= 0)

    def test_shape(self):
        t, C = concentration_profile(1000.0, self._PK, n_points=100)
        assert len(t) == 100
        assert len(C) == 100

    def test_therapeutic_score_in_range(self):
        score = therapeutic_score(1000.0, self._PK, mic=2.0, frequency_hours=8.0)
        assert 0.0 <= score <= 1.0


class TestHillEffect:
    def test_zero_concentration(self):
        assert hill_effect(0.0, emax=1.0, ec50=1.0) == 0.0

    def test_at_ec50(self):
        effect = hill_effect(1.0, emax=1.0, ec50=1.0, hill_n=1.0)
        assert effect == pytest.approx(0.5)

    def test_saturates_at_emax(self):
        effect = hill_effect(1e9, emax=0.8, ec50=1.0)
        assert effect == pytest.approx(0.8, abs=1e-4)

    def test_negative_concentration(self):
        assert hill_effect(-5.0, emax=1.0, ec50=1.0) == 0.0


# ---------------------------------------------------------------------------
# PK/PD — public API
# ---------------------------------------------------------------------------

class TestGetConcentrationCurve:
    def test_returns_ndarray(self):
        C = get_concentration_curve("meropenem", 1000.0, 8.0)
        assert isinstance(C, np.ndarray)

    def test_decays(self):
        C = get_concentration_curve("meropenem", 1000.0, 24.0)
        assert C[0] > C[-1]

    def test_non_negative(self):
        C = get_concentration_curve("ciprofloxacin", 500.0, 12.0)
        assert np.all(C >= 0)

    def test_higher_dose_higher_peak(self):
        C_low = get_concentration_curve("meropenem", 500.0, 8.0)
        C_high = get_concentration_curve("meropenem", 1000.0, 8.0)
        assert C_high[0] > C_low[0]

    def test_unknown_drug_raises(self):
        with pytest.raises(ValueError, match="Unknown drug"):
            get_concentration_curve("made_up_drug", 100.0, 8.0)


class TestIsInTherapeuticWindow:
    def test_returns_tuple(self):
        result = is_in_therapeutic_window("meropenem", 1000.0, mic=2.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_bool_and_float(self):
        in_window, score = is_in_therapeutic_window("meropenem", 1000.0, mic=2.0)
        assert isinstance(in_window, bool)
        assert isinstance(score, float)

    def test_score_in_range(self):
        _, score = is_in_therapeutic_window("meropenem", 1000.0, mic=2.0)
        assert 0.0 <= score <= 1.0

    def test_good_drug_in_window(self):
        in_window, score = is_in_therapeutic_window("meropenem", 1000.0, mic=2.0)
        assert in_window is True
        assert score >= 0.4

    def test_resistant_mic_not_in_window(self):
        # Ampicillin vs ESBL E. coli — MIC >> achievable concentration
        in_window, score = is_in_therapeutic_window("ampicillin", 2000.0, mic=256.0)
        assert in_window is False
        assert score <= 0.1


class TestGetPkpdScore:
    def test_score_in_range(self):
        score = get_pkpd_score("meropenem", 1000.0, 8.0, 2.0)
        assert 0.0 <= score <= 1.0

    def test_score_is_float(self):
        score = get_pkpd_score("ciprofloxacin", 500.0, 12.0, 0.5)
        assert isinstance(score, float)

    def test_higher_mic_lower_score(self):
        score_low_mic = get_pkpd_score("meropenem", 1000.0, 8.0, 0.25)
        score_high_mic = get_pkpd_score("meropenem", 1000.0, 8.0, 8.0)
        assert score_low_mic >= score_high_mic

    def test_higher_dose_higher_score(self):
        score_low = get_pkpd_score("meropenem", 500.0, 8.0, 2.0)
        score_high = get_pkpd_score("meropenem", 2000.0, 8.0, 2.0)
        assert score_high >= score_low

    def test_unknown_drug_raises(self):
        with pytest.raises(ValueError):
            get_pkpd_score("fantasy_drug", 100.0, 8.0, 2.0)


# ---------------------------------------------------------------------------
# The mandatory acceptance test
# ---------------------------------------------------------------------------

def test_pkpd():
    """
    Acceptance criteria from CLAUDE.md:
      - meropenem 1000mg q8h vs MIC=2  → score ≥ 0.8
      - ampicillin vs ESBL (MIC=256)   → score ≤ 0.1
    """
    score_mero = get_pkpd_score("meropenem", 1000.0, 8.0, pathogen_mic=2.0)
    assert score_mero >= 0.8, (
        f"meropenem 1000mg q8h vs MIC=2 should score ≥0.8, got {score_mero:.3f}"
    )

    # ESBL E. coli ampicillin MIC >> resistant breakpoint (≥256 mg/L)
    score_amp = get_pkpd_score("ampicillin", 2000.0, 6.0, pathogen_mic=256.0)
    assert score_amp <= 0.1, (
        f"ampicillin vs ESBL (MIC=256) should score ≤0.1, got {score_amp:.3f}"
    )


# ---------------------------------------------------------------------------
# The mandatory evolution acceptance test
# ---------------------------------------------------------------------------

def test_evolution():
    """
    Acceptance criteria:
      1. 72h meropenem on ESBL (MIC=2) drives frequency up sharply.
      2. No drug (dose=0) leaves frequency stable (neutral drift only).
      3. Same seed produces identical trajectory (reproducibility).
    """
    # --- 1. 72h meropenem increases resistant-allele frequency ---
    rng = np.random.default_rng(99)
    # 3 steps × 24h = 72h; meropenem 1000mg q8h vs MIC=2 → s ≈ 0.9
    traj = get_resistance_trajectory(
        "meropenem", dose_mg=1000.0, duration_steps=3,
        initial_freq=0.01, rng=rng, mic=2.0,
    )
    assert len(traj) == 4                          # initial + 3 steps
    assert traj[-1] > traj[0], (
        f"meropenem 72h: expected freq to rise, got {traj[0]:.4f} → {traj[-1]:.4f}"
    )
    assert traj[-1] > 0.1, (
        f"meropenem 72h: expected final freq > 0.1, got {traj[-1]:.4f}"
    )

    # --- 2. No drug → neutral drift, frequency stays near initial value ---
    rng2 = np.random.default_rng(7)
    traj_no_drug = get_resistance_trajectory(
        "meropenem", dose_mg=0.0, duration_steps=10,
        initial_freq=0.1, rng=rng2, mic=2.0,
    )
    # With N=1e8, drift variance is tiny — should stay within ±0.02 of 0.1
    final_nodrug = traj_no_drug[-1]
    assert abs(final_nodrug - 0.1) < 0.02, (
        f"No-drug neutral drift: expected freq ≈ 0.1, got {final_nodrug:.4f}"
    )

    # --- 3. Seed reproducibility ---
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    traj_a = get_resistance_trajectory(
        "ciprofloxacin", dose_mg=500.0, duration_steps=5,
        initial_freq=0.05, rng=rng_a, mic=0.5,
    )
    traj_b = get_resistance_trajectory(
        "ciprofloxacin", dose_mg=500.0, duration_steps=5,
        initial_freq=0.05, rng=rng_b, mic=0.5,
    )
    assert traj_a == traj_b, "Same seed must produce identical trajectory"


# ---------------------------------------------------------------------------
# Wright-Fisher
# ---------------------------------------------------------------------------

class TestEvolution:
    """Unit tests for the new public evolution API."""

    def test_evolve_resistance_returns_float(self):
        rng = np.random.default_rng(1)
        result = evolve_resistance(0.1, 0.5, rng)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_evolve_resistance_bounded(self):
        rng = np.random.default_rng(2)
        for p in [0.0, 0.01, 0.5, 0.99, 1.0]:
            r = evolve_resistance(p, 0.5, rng)
            assert 0.0 <= r <= 1.0

    def test_evolve_resistance_zero_selection_is_neutral(self):
        # With s=0 and large N, result must be very close to p
        rng = np.random.default_rng(3)
        p = 0.3
        result = evolve_resistance(p, 0.0, rng, N=int(1e8))
        assert abs(result - p) < 0.01

    def test_evolve_resistance_high_selection_moves_up(self):
        rng = np.random.default_rng(4)
        p = 0.5
        result = evolve_resistance(p, 0.9, rng)
        assert result > p  # selection should push toward fixation

    def test_compute_selective_coeff_zero_dose(self):
        s = compute_selective_coefficient("meropenem", 0.0, 2.0)
        assert s == 0.0

    def test_compute_selective_coeff_in_range(self):
        s = compute_selective_coefficient("meropenem", 1000.0, 2.0)
        assert 0.0 <= s <= 0.9

    def test_compute_selective_coeff_high_mic_low_s(self):
        # When MIC >> achievable concentration, drug has no effect → s ≈ 0
        s = compute_selective_coefficient("ampicillin", 2000.0, 256.0)
        assert s < 0.1

    def test_compute_selective_coeff_effective_drug_high_s(self):
        # Meropenem 1000mg vs MIC=2 → 100% T>MIC → s near _MAX_S
        s = compute_selective_coefficient("meropenem", 1000.0, 2.0)
        assert s > 0.5

    def test_resistance_emerged_default_threshold(self):
        assert resistance_emerged(0.6) is True
        assert resistance_emerged(0.4) is False
        assert resistance_emerged(0.5) is False  # strictly >

    def test_resistance_emerged_custom_threshold(self):
        assert resistance_emerged(0.3, threshold=0.2) is True
        assert resistance_emerged(0.3, threshold=0.4) is False

    def test_trajectory_length(self):
        rng = np.random.default_rng(5)
        traj = get_resistance_trajectory("meropenem", 1000.0, 5, 0.05, rng)
        assert len(traj) == 6  # initial + 5 steps

    def test_trajectory_first_element_is_initial(self):
        rng = np.random.default_rng(6)
        traj = get_resistance_trajectory("meropenem", 1000.0, 3, 0.07, rng)
        assert traj[0] == pytest.approx(0.07)

    def test_trajectory_all_values_bounded(self):
        rng = np.random.default_rng(7)
        traj = get_resistance_trajectory("ciprofloxacin", 500.0, 8, 0.1, rng)
        assert all(0.0 <= v <= 1.0 for v in traj)

    def test_trajectory_zero_steps(self):
        rng = np.random.default_rng(8)
        traj = get_resistance_trajectory("meropenem", 1000.0, 0, 0.05, rng)
        assert traj == [pytest.approx(0.05)]


class TestWrightFisher:
    def test_stays_bounded(self):
        rng = np.random.default_rng(42)
        freq = 0.1
        for _ in range(20):
            freq = wright_fisher_step(freq, s=0.3, rng=rng)
            assert 0.0 <= freq <= 1.0

    def test_high_selection_increases_frequency(self):
        # With s=0.9 (near-maximum selection), resistant allele should trend up
        rng = np.random.default_rng(0)
        freq = 0.5
        results = []
        for _ in range(30):
            freq = wright_fisher_step(freq, s=0.9, rng=rng)
            results.append(freq)
        # Majority of steps should move toward fixation
        assert max(results) > 0.5

    def test_zero_selection_neutral(self):
        rng = np.random.default_rng(7)
        freq = 0.5
        for _ in range(10):
            new_freq = wright_fisher_step(freq, s=0.0, rng=rng)
            assert 0.0 <= new_freq <= 1.0

    def test_resistance_emerges_under_pressure(self):
        rng = np.random.default_rng(42)
        freq = 0.3
        for _ in range(50):
            freq, _ = evolve_resistance_legacy(
                freq, treatment_hours=96.0, dose_mg=1000.0,
                standard_dose_mg=500.0, rng=rng,
            )
        assert freq > 0.3 or freq < 0.1  # moved significantly


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class TestNetwork:
    def test_loads(self):
        G = build_graph(_NET)
        assert len(G.nodes) == 10
        assert len(G.edges) > 0

    def test_all_edges_have_weight(self):
        G = build_graph(_NET)
        for u, v, d in G.edges(data=True):
            assert "weight" in d, f"Edge ({u},{v}) missing weight"

    def test_node_attributes_present(self):
        G = build_graph(_NET)
        for node in G.nodes:
            assert "ward_capacity" in G.nodes[node], f"{node} missing ward_capacity"
            assert "isolation_beds" in G.nodes[node]
            assert "average_los_days" in G.nodes[node]

    def test_transmission_zero_when_isolated(self):
        G = build_graph(_NET)
        # ICU → StepDownUnit edge exists; immune=True should give 0
        p = transmission_probability(G, "ICU", "StepDownUnit", infected_count=5, immune=True)
        assert p == 0.0

    def test_transmission_positive_when_connected(self):
        G = build_graph(_NET)
        p = transmission_probability(G, "ICU", "StepDownUnit", infected_count=3)
        assert p > 0.0

    def test_beta_values_on_graph(self):
        G = build_graph(_NET)
        betas = G.graph["beta_values"]
        assert betas["CRK"] == pytest.approx(0.08)
        assert betas["ESBL"] == pytest.approx(0.15)
        assert betas["MRSA"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Bayes
# ---------------------------------------------------------------------------

class TestBayes:
    def _profiles(self):
        return json.loads(
            (Path(__file__).parent.parent / "episteward/data/resistance_profiles.json").read_text()
        )

    def test_posterior_increases_on_resistant(self):
        posterior = update_posterior(0.2, "resistant")
        assert posterior > 0.2
        assert 0.0 <= posterior <= 1.0

    def test_posterior_decreases_on_sensitive(self):
        posterior = update_posterior(0.5, "sensitive")
        assert posterior < 0.5

    def test_no_update_on_none(self):
        mean1 = update_posterior(0.3, None)
        mean2 = update_posterior(0.3, None)
        assert mean1 == pytest.approx(mean2)

    def test_estimate_resistance_returns_dict(self):
        result = estimate_resistance(
            "E_coli_ESBL", "ciprofloxacin", "icu", self._profiles(), "resistant"
        )
        assert "posterior" in result
        assert 0.0 <= result["posterior"] <= 1.0

    def test_estimate_has_all_keys(self):
        result = estimate_resistance(
            "K_pneumoniae_CRK", "meropenem", "icu", self._profiles()
        )
        assert {"prior", "posterior", "ci_lower", "ci_upper"} <= result.keys()


# ---------------------------------------------------------------------------
# Mandatory acceptance test — network
# ---------------------------------------------------------------------------

def test_network():
    """
    Acceptance criteria:
      1. Spread propagates over 5 steps (infected set grows).
      2. Isolation reduces transmission probability by 90%.
      3. Chain detection returns correct source from a known scenario.
    """
    G = build_graph(_NET)

    # --- 1. Spread propagates over 5 steps ---
    rng = np.random.default_rng(0)
    infected = {"ICU"}
    # Run 5 steps accumulating newly infected wards
    all_infected = set(infected)
    for _ in range(5):
        newly = simulate_spread_step(all_infected, "ESBL", {}, rng, graph=G)
        all_infected |= newly
    assert len(all_infected) > 1, (
        f"Expected spread beyond ICU after 5 steps, got {all_infected}"
    )

    # --- 2. Isolation reduces P by 90% ---
    p_normal = compute_transmission_probability(
        "ICU", "StepDownUnit", "ESBL", isolation_active=False, graph=G
    )
    p_isolated = compute_transmission_probability(
        "ICU", "StepDownUnit", "ESBL", isolation_active=True, graph=G
    )
    assert p_normal > 0.0, "Expected non-zero transmission ICU→StepDownUnit"
    assert p_isolated > 0.0, "Expected non-zero (just reduced) isolated transmission"
    ratio = p_isolated / p_normal
    assert ratio == pytest.approx(0.1, abs=1e-9), (
        f"Isolation should give 10% of normal P, got ratio={ratio:.4f}"
    )

    # --- 3. Chain detection returns correct source from known scenario ---
    transfer_logs = [
        {"patient_id": "P001", "from_ward": "ICU",      "to_ward": "MedWard_A", "timestamp": "2024-01-01T12:00:00"},
        {"patient_id": "P001", "from_ward": "MedWard_A", "to_ward": "SurgWard",  "timestamp": "2024-01-02T08:00:00"},
        {"patient_id": "P002", "from_ward": "EmergencyDept", "to_ward": "ICU",  "timestamp": "2024-01-01T06:00:00"},
    ]
    culture_results = {
        "P001": {"result": "positive", "timestamp": "2024-01-01T10:00:00"},
        "P002": {"result": "negative", "timestamp": "2024-01-01T07:00:00"},
    }
    chain = get_transmission_chain(transfer_logs, culture_results)
    assert chain == ["ICU", "MedWard_A", "SurgWard"], (
        f"Expected ['ICU', 'MedWard_A', 'SurgWard'], got {chain}"
    )

    # --- Edge cases ---
    # Empty culture results → empty chain
    assert get_transmission_chain(transfer_logs, {}) == []
    # No positive cultures → empty chain
    assert get_transmission_chain(transfer_logs, {"P001": {"result": "negative", "timestamp": "T"}}) == []

    # get_at_risk_wards excludes already-infected
    at_risk = get_at_risk_wards({"ICU"}, graph=G)
    assert isinstance(at_risk, list)
    assert "ICU" not in at_risk
    assert len(at_risk) > 0


# ---------------------------------------------------------------------------
# Mandatory acceptance test — bayes
# ---------------------------------------------------------------------------

def test_bayes():
    """
    Acceptance criteria:
      1. prior=0.3 + positive culture  → posterior > 0.85
      2. prior=0.3 + negative culture  → posterior < 0.15
      3. get_resistance_probability returns valid CI tuple
      4. get_empiric_recommendation returns the least-resistant antibiotic
    """
    # --- 1. Positive culture drives posterior high ---
    posterior_pos = update_posterior(0.3, "resistant")
    assert posterior_pos > 0.85, (
        f"prior=0.3 + positive culture: expected posterior > 0.85, got {posterior_pos:.4f}"
    )

    # --- 2. Negative culture drives posterior low ---
    posterior_neg = update_posterior(0.3, "sensitive")
    assert posterior_neg < 0.15, (
        f"prior=0.3 + negative culture: expected posterior < 0.15, got {posterior_neg:.4f}"
    )

    # --- 3. Return is float in [0, 1] ---
    assert isinstance(posterior_pos, float)
    assert 0.0 <= posterior_pos <= 1.0
    assert 0.0 <= posterior_neg <= 1.0

    # --- 4. sensitivity_accuracy param wired through ---
    # Perfect test (accuracy=1.0): positive culture → near certainty
    p_perfect = update_posterior(0.3, "resistant", sensitivity_accuracy=0.9999)
    assert p_perfect > posterior_pos  # even higher confidence

    # --- 5. get_resistance_probability returns valid (mean, ci_lower, ci_upper) ---
    mean, lo, hi = get_resistance_probability(
        "E_coli_ESBL", "ciprofloxacin", ["resistant"], ward_id="icu"
    )
    assert 0.0 <= lo <= mean <= hi <= 1.0
    assert mean > 0.5  # positive culture should push above 0.5

    mean2, lo2, hi2 = get_resistance_probability(
        "E_coli_ESBL", "meropenem", ["sensitive"], ward_id="icu"
    )
    assert mean2 < mean  # meropenem sensitive → lower resistance probability

    # No cultures → CI around prior
    mean_prior, _, _ = get_resistance_probability("E_coli_ESBL", "meropenem")
    assert 0.0 < mean_prior < 1.0

    # --- 6. get_empiric_recommendation picks lowest resistance ---
    # meropenem=0.1 vs colistin=0.05 → colistin recommended
    rec = get_empiric_recommendation(
        "E_coli_ESBL",
        ["meropenem", "colistin", "ampicillin"],
        {"meropenem": 0.1, "colistin": 0.05, "ampicillin": 0.9},
    )
    assert rec == "colistin"

    # Unknown drug → treated as resistance=1.0 (last resort)
    rec2 = get_empiric_recommendation(
        "E_coli_ESBL",
        ["known_bad", "unknown_drug"],
        {"known_bad": 0.8},  # unknown_drug defaults to 1.0 → known_bad wins
    )
    assert rec2 == "known_bad"

    # Single option always returned
    rec3 = get_empiric_recommendation("E_coli_ESBL", ["meropenem"], {"meropenem": 0.3})
    assert rec3 == "meropenem"


# ---------------------------------------------------------------------------
# Population PK — two-compartment model (population_pk.py)
# ---------------------------------------------------------------------------

from episteward.math.population_pk import (
    PKParams,
    PatientCovariates,
    sample_individual_pk,
    solve_two_compartment,
    bayesian_tdm_update,
    get_pkpd_score as population_get_pkpd_score,
)


class TestSolveTwoCompartment:
    """Unit tests for solve_two_compartment."""

    def test_returns_ndarray(self):
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        C = solve_two_compartment(pk, 1000.0, 8.0)
        assert isinstance(C, np.ndarray)

    def test_non_negative(self):
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        C = solve_two_compartment(pk, 1000.0, 24.0)
        assert np.all(C >= 0.0)

    def test_peak_at_t0_for_iv(self):
        """IV bolus: central compartment peaks at t=0 then declines."""
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        C = solve_two_compartment(pk, 1000.0, 8.0)
        assert C[0] >= C[-1]

    def test_initial_concentration_iv(self):
        """For IV bolus, C1(0) must equal F*D/V1."""
        pk = PKParams(V1=10.0, V2=8.0, CL=5.0, Q=2.0, F=1.0, ka=0.0)
        C = solve_two_compartment(pk, 500.0, 12.0)
        assert C[0] == pytest.approx(500.0 / 10.0, rel=1e-3)

    def test_higher_dose_higher_concentration(self):
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        C_low = solve_two_compartment(pk, 500.0, 8.0)
        C_high = solve_two_compartment(pk, 1000.0, 8.0)
        assert C_high[0] > C_low[0]
        assert np.all(C_high >= C_low - 1e-9)  # linearity

    def test_oral_absorption_peak_after_zero(self):
        """For oral dose (ka>0), C1 starts at 0 and peaks after t=0."""
        pk = PKParams(V1=10.0, V2=5.0, CL=3.0, Q=1.5, F=0.8, ka=0.5)
        C = solve_two_compartment(pk, 500.0, 12.0)
        assert C[0] == pytest.approx(0.0, abs=1e-6)
        assert C.max() > C[0]

    def test_output_length_matches_dt(self):
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        C = solve_two_compartment(pk, 1000.0, 8.0, dt=0.5)
        expected_points = int(round(8.0 / 0.5)) + 1  # 17
        assert len(C) == expected_points


class TestSampleIndividualPK:
    """Unit tests for sample_individual_pk."""

    _PATIENT = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=1.0, is_icu=False)

    def test_returns_pkparams(self):
        rng = np.random.default_rng(0)
        pk = sample_individual_pk("meropenem", self._PATIENT, rng)
        assert isinstance(pk, PKParams)

    def test_positive_params(self):
        rng = np.random.default_rng(1)
        pk = sample_individual_pk("meropenem", self._PATIENT, rng)
        assert pk.V1 > 0 and pk.V2 > 0 and pk.CL > 0 and pk.Q > 0

    def test_reproducible_with_same_seed(self):
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        pk_a = sample_individual_pk("meropenem", self._PATIENT, rng_a)
        pk_b = sample_individual_pk("meropenem", self._PATIENT, rng_b)
        assert pk_a.V1 == pytest.approx(pk_b.V1)
        assert pk_a.CL == pytest.approx(pk_b.CL)

    def test_different_seeds_differ(self):
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        pk_a = sample_individual_pk("meropenem", self._PATIENT, rng_a)
        pk_b = sample_individual_pk("meropenem", self._PATIENT, rng_b)
        assert pk_a.V1 != pytest.approx(pk_b.V1)

    def test_renal_impairment_reduces_cl(self):
        """Higher creatinine → lower estimated CrCl → lower CL (same seed)."""
        rng_normal = np.random.default_rng(7)
        rng_impaired = np.random.default_rng(7)
        patient_normal = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=1.0, is_icu=False)
        patient_impaired = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=4.0, is_icu=False)
        pk_normal = sample_individual_pk("meropenem", patient_normal, rng_normal)
        pk_impaired = sample_individual_pk("meropenem", patient_impaired, rng_impaired)
        assert pk_normal.CL > pk_impaired.CL

    def test_renal_impairment_no_effect_ceftriaxone(self):
        """Ceftriaxone CL is not renally adjusted."""
        rng_normal = np.random.default_rng(9)
        rng_impaired = np.random.default_rng(9)
        patient_normal = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=1.0, is_icu=False)
        patient_impaired = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=4.0, is_icu=False)
        pk_normal = sample_individual_pk("ceftriaxone", patient_normal, rng_normal)
        pk_impaired = sample_individual_pk("ceftriaxone", patient_impaired, rng_impaired)
        # Same seed, same omega → same CL (covariate adjustment disabled)
        assert pk_normal.CL == pytest.approx(pk_impaired.CL)

    def test_unknown_drug_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Unknown drug"):
            sample_individual_pk("fantasy_drug", self._PATIENT, rng)

    def test_all_three_drugs_work(self):
        rng = np.random.default_rng(0)
        patient = self._PATIENT
        for drug in ["meropenem", "vancomycin", "ceftriaxone"]:
            rng_copy = np.random.default_rng(0)
            pk = sample_individual_pk(drug, patient, rng_copy)
            assert pk.V1 > 0

    def test_vancomycin_iv_ka_zero(self):
        rng = np.random.default_rng(0)
        pk = sample_individual_pk("vancomycin", self._PATIENT, rng)
        assert pk.ka == 0.0
        assert pk.F == 1.0


class TestBayesianTDMUpdate:
    """Unit tests for bayesian_tdm_update."""

    _PATIENT = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=1.0, is_icu=False)

    def test_returns_pkparams(self):
        rng = np.random.default_rng(0)
        pk = sample_individual_pk("meropenem", self._PATIENT, rng)
        observations = [(2.0, 25.0), (6.0, 8.0)]
        pk_updated = bayesian_tdm_update(pk, observations, "meropenem", dose_mg=1000.0)
        assert isinstance(pk_updated, PKParams)

    def test_empty_observations_returns_unchanged(self):
        rng = np.random.default_rng(0)
        pk = sample_individual_pk("meropenem", self._PATIENT, rng)
        pk_updated = bayesian_tdm_update(pk, [], "meropenem")
        assert pk_updated.V1 == pytest.approx(pk.V1)
        assert pk_updated.CL == pytest.approx(pk.CL)

    def test_updated_params_positive(self):
        rng = np.random.default_rng(1)
        pk = sample_individual_pk("meropenem", self._PATIENT, rng)
        observations = [(1.0, 40.0), (4.0, 15.0), (8.0, 3.0)]
        pk_updated = bayesian_tdm_update(pk, observations, "meropenem", dose_mg=1000.0)
        assert pk_updated.V1 > 0 and pk_updated.CL > 0

    def test_f_and_ka_preserved(self):
        rng = np.random.default_rng(2)
        pk = sample_individual_pk("meropenem", self._PATIENT, rng)
        observations = [(2.0, 30.0)]
        pk_updated = bayesian_tdm_update(pk, observations, "meropenem")
        assert pk_updated.F == pk.F
        assert pk_updated.ka == pk.ka

    def test_unknown_drug_raises(self):
        rng = np.random.default_rng(0)
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        with pytest.raises(ValueError, match="Unknown drug"):
            bayesian_tdm_update(pk, [(1.0, 30.0)], "fantasy_drug")


class TestPopulationGetPkpdScore:
    """Unit tests for population_pk.get_pkpd_score."""

    def test_score_in_range(self):
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        score = population_get_pkpd_score(pk, 1000.0, 8.0, 7, 2.0, "meropenem")
        assert 0.0 <= score <= 1.0

    def test_score_is_float(self):
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        score = population_get_pkpd_score(pk, 1000.0, 8.0, 7, 2.0, "meropenem")
        assert isinstance(score, float)

    def test_higher_mic_lower_score(self):
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        score_low = population_get_pkpd_score(pk, 1000.0, 8.0, 7, 0.5, "meropenem")
        score_high = population_get_pkpd_score(pk, 1000.0, 8.0, 7, 16.0, "meropenem")
        assert score_low >= score_high

    def test_higher_dose_higher_score(self):
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        score_low = population_get_pkpd_score(pk, 250.0, 8.0, 7, 2.0, "meropenem")
        score_high = population_get_pkpd_score(pk, 2000.0, 8.0, 7, 2.0, "meropenem")
        assert score_high >= score_low

    def test_mic_above_achievable_gives_zero(self):
        """When C_max < MIC, %T>MIC = 0 → score = 0."""
        pk = PKParams(V1=14.0, V2=4.0, CL=13.0, Q=2.0, F=1.0, ka=0.0)
        # dose 250 mg → C1(0) = 250/14 ≈ 17.9 < MIC=32
        score = population_get_pkpd_score(pk, 250.0, 6.0, 5, 32.0, "ampicillin")
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_unknown_drug_uses_defaults(self):
        """Unknown drug falls back to time-dependent with T_target=0.50."""
        pk = PKParams(V1=12.5, V2=9.5, CL=9.0, Q=4.0, F=1.0, ka=0.0)
        score = population_get_pkpd_score(pk, 1000.0, 8.0, 7, 2.0, "mystery_drug")
        assert 0.0 <= score <= 1.0

    def test_creatinine_affects_score(self):
        """Different creatinine → different CL → different score for meropenem."""
        rng1 = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        pt_normal = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=1.0, is_icu=False)
        pt_impaired = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=4.0, is_icu=False)
        pk1 = sample_individual_pk("meropenem", pt_normal, rng1)
        pk2 = sample_individual_pk("meropenem", pt_impaired, rng2)
        score1 = population_get_pkpd_score(pk1, 1000.0, 8.0, 7, 2.0, "meropenem")
        score2 = population_get_pkpd_score(pk2, 1000.0, 8.0, 7, 2.0, "meropenem")
        # Renally impaired patient has lower CL → slower elimination → higher T>MIC → score2 >= score1
        assert score2 >= score1


# ---------------------------------------------------------------------------
# Mandatory acceptance test — population PK
# ---------------------------------------------------------------------------


def test_population_pkpd():
    """
    Acceptance criteria from CLAUDE.md (population_pk module):

    1. meropenem 1000 mg q8h vs MIC=2 (standard patient)  → score ≥ 0.80
    2. ampicillin-like PK, 500 mg q6h vs MIC=32 (ESBL)   → score ≤ 0.10
       (C_max ≈ 35.7 mg/L, rapid biexponential elimination → <4% T>MIC)
    3. Same dose / drug, different creatinine → different C(t) curves
    """
    # ---- 1. meropenem 1000 mg q8h vs MIC=2 → score ≥ 0.80 ----
    rng = np.random.default_rng(0)
    standard_patient = PatientCovariates(age=50, weight_kg=70,
                                         creatinine_mg_dl=1.0, is_icu=False)
    pk_mero = sample_individual_pk("meropenem", standard_patient, rng)
    score_mero = population_get_pkpd_score(pk_mero, 1000.0, 8.0, 7, 2.0, "meropenem")
    assert score_mero >= 0.80, (
        f"meropenem 1000 mg q8h vs MIC=2 should score ≥0.80, got {score_mero:.3f}"
    )

    # ---- 2. Ampicillin-like PK vs MIC=32 (ESBL) → score ≤ 0.10 ----
    # Typical ampicillin IV PK: V1≈14L, CL≈13L/h (very rapid elimination).
    # 500 mg → C1(0)≈35.7 mg/L, biexponential alpha≈1.18/h drops below
    # MIC=32 within ~0.15h; <4% of the 6h interval is above MIC.
    pk_amp = PKParams(V1=14.0, V2=4.0, CL=13.0, Q=2.0, F=1.0, ka=0.0)
    score_amp = population_get_pkpd_score(pk_amp, 500.0, 6.0, 5, 32.0, "ampicillin")
    assert score_amp <= 0.10, (
        f"ampicillin-like 500 mg q6h vs ESBL MIC=32 should score ≤0.10, "
        f"got {score_amp:.3f}"
    )

    # ---- 3. Different creatinine → different C(t) curves ----
    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)
    pt_normal = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=1.0, is_icu=False)
    pt_renal  = PatientCovariates(age=50, weight_kg=70, creatinine_mg_dl=3.5, is_icu=False)
    pk_normal = sample_individual_pk("meropenem", pt_normal, rng_a)
    pk_renal  = sample_individual_pk("meropenem", pt_renal,  rng_b)
    C_normal = solve_two_compartment(pk_normal, 1000.0, 24.0, dt=0.5)
    C_renal  = solve_two_compartment(pk_renal,  1000.0, 24.0, dt=0.5)
    assert not np.allclose(C_normal, C_renal, atol=1e-3), (
        "Different creatinine should produce different C(t) curves for meropenem"
    )


# ---------------------------------------------------------------------------
# MSW — Mutant Selection Window (msw.py)
# ---------------------------------------------------------------------------

from episteward.math.msw import (
    compute_mpc,
    classify_zone,
    compute_msw_risk,
    get_msw_reward_component,
)

_MUTATION_FREQ = 1e-8
_HILL_COEFF = 1.5


class TestComputeMPC:
    def test_mpc_greater_than_mic(self):
        mpc = compute_mpc("meropenem", "E_coli", mic=2.0)
        assert mpc > 2.0

    def test_mpc_scales_linearly_with_mic(self):
        mpc1 = compute_mpc("meropenem", "E_coli", mic=1.0)
        mpc2 = compute_mpc("meropenem", "E_coli", mic=4.0)
        assert mpc2 == pytest.approx(4.0 * mpc1, rel=1e-9)

    def test_mpc_formula(self):
        """MPC = MIC × (1/mutation_freq)^(1/hill_coeff) with hardcoded constants."""
        mic = 2.0
        expected = mic * (1.0 / _MUTATION_FREQ) ** (1.0 / _HILL_COEFF)
        mpc = compute_mpc("meropenem", "E_coli", mic)
        assert mpc == pytest.approx(expected, rel=1e-9)

    def test_different_drugs_same_result(self):
        """All drugs use the same hardcoded constants — same MPC for same MIC."""
        mic = 1.0
        assert compute_mpc("meropenem", "E_coli", mic) == pytest.approx(
            compute_mpc("vancomycin", "S_aureus", mic)
        )

    def test_returns_float(self):
        assert isinstance(compute_mpc("meropenem", "E_coli", 1.0), float)


class TestClassifyZone:
    def _mpc(self, mic):
        return compute_mpc("meropenem", "E_coli", mic)

    def test_below_mic_is_sub_mic(self):
        assert classify_zone(0.5, mic=1.0, mpc=100.0) == "sub_mic"

    def test_zero_is_sub_mic(self):
        assert classify_zone(0.0, mic=1.0, mpc=100.0) == "sub_mic"

    def test_exactly_mic_is_msw(self):
        """Lower boundary of MSW is inclusive."""
        assert classify_zone(1.0, mic=1.0, mpc=100.0) == "msw"

    def test_interior_of_msw(self):
        assert classify_zone(50.0, mic=1.0, mpc=100.0) == "msw"

    def test_exactly_mpc_is_msw(self):
        """Upper boundary of MSW is inclusive."""
        assert classify_zone(100.0, mic=1.0, mpc=100.0) == "msw"

    def test_above_mpc_is_mpc_plus(self):
        assert classify_zone(101.0, mic=1.0, mpc=100.0) == "mpc_plus"

    def test_large_concentration_is_mpc_plus(self):
        mic = 2.0
        mpc = self._mpc(mic)
        assert classify_zone(mpc * 2.0, mic, mpc) == "mpc_plus"


class TestComputeMSWRisk:
    def test_all_above_mpc_zero_risk(self):
        """C entirely above MPC → 0 % in MSW → risk = 0."""
        mic = 0.001
        mpc = compute_mpc("meropenem", "E_coli", mic)
        C = np.full(50, mpc * 2.0)
        assert compute_msw_risk(C, "meropenem", "E_coli", mic) == pytest.approx(0.0)

    def test_all_in_msw_full_risk(self):
        """C entirely within [MIC, MPC] → risk = 1.0."""
        mic = 0.001
        mpc = compute_mpc("meropenem", "E_coli", mic)
        C = np.full(50, (mic + mpc) / 2.0)
        assert compute_msw_risk(C, "meropenem", "E_coli", mic) == pytest.approx(1.0)

    def test_all_below_mic_zero_risk(self):
        """Sub-MIC concentrations are NOT in the MSW → risk = 0."""
        mic = 2.0
        C = np.full(50, mic * 0.1)
        assert compute_msw_risk(C, "meropenem", "E_coli", mic) == pytest.approx(0.0)

    def test_mixed_series_partial_risk(self):
        mic = 1.0
        mpc = compute_mpc("meropenem", "E_coli", mic)
        # 60 points in MSW, 40 points below MIC
        C = np.concatenate([
            np.full(60, (mic + mpc) / 2.0),
            np.full(40, mic * 0.1),
        ])
        risk = compute_msw_risk(C, "meropenem", "E_coli", mic)
        assert risk == pytest.approx(0.60, abs=1e-9)

    def test_empty_series_returns_zero(self):
        assert compute_msw_risk(np.array([]), "meropenem", "E_coli", 1.0) == 0.0

    def test_risk_in_unit_interval(self):
        C = np.linspace(0.0, 1000.0, 200)
        risk = compute_msw_risk(C, "meropenem", "E_coli", mic=1.0)
        assert 0.0 <= risk <= 1.0

    def test_dt_does_not_change_result(self):
        """dt cancels from T_MSW/T_total ratio; result is the same for any dt."""
        mic = 1.0
        mpc = compute_mpc("meropenem", "E_coli", mic)
        C = np.array([mic * 0.5, (mic + mpc) / 2.0, mpc * 2.0])
        risk_05 = compute_msw_risk(C, "meropenem", "E_coli", mic, dt=0.5)
        risk_1  = compute_msw_risk(C, "meropenem", "E_coli", mic, dt=1.0)
        assert risk_05 == pytest.approx(risk_1)


class TestGetMSWRewardComponent:
    def test_all_in_msw_gives_zero(self):
        mic = 0.001
        mpc = compute_mpc("meropenem", "E_coli", mic)
        C = np.full(50, (mic + mpc) / 2.0)
        score = get_msw_reward_component(C, "meropenem", "E_coli", mic)
        assert score == pytest.approx(0.0)

    def test_all_above_mpc_gives_max(self):
        """msw_risk=0 + 0.2 bonus (all above MPC) → clamped to 1.0."""
        mic = 0.001
        mpc = compute_mpc("meropenem", "E_coli", mic)
        C = np.full(50, mpc * 2.0)
        score = get_msw_reward_component(C, "meropenem", "E_coli", mic)
        assert score == pytest.approx(1.0)

    def test_bonus_requires_80pct_above_mpc(self):
        """Bonus (+0.2) only fires when >80% of points are above MPC."""
        mic = 0.001
        mpc = compute_mpc("meropenem", "E_coli", mic)
        # 85 % above MPC, 15 % in MSW → base = 0.85, bonus = +0.2 → 1.0 (clamped)
        C_with_bonus = np.concatenate([
            np.full(85, mpc * 2.0),
            np.full(15, (mic + mpc) / 2.0),
        ])
        # 75 % above MPC, 25 % in MSW → base = 0.75, no bonus → 0.75
        C_no_bonus = np.concatenate([
            np.full(75, mpc * 2.0),
            np.full(25, (mic + mpc) / 2.0),
        ])
        score_with = get_msw_reward_component(C_with_bonus, "meropenem", "E_coli", mic)
        score_no   = get_msw_reward_component(C_no_bonus,   "meropenem", "E_coli", mic)
        assert score_with > score_no

    def test_score_in_unit_interval(self):
        C = np.linspace(0.0, 500.0, 100)
        score = get_msw_reward_component(C, "meropenem", "E_coli", mic=1.0)
        assert 0.0 <= score <= 1.0

    def test_empty_series_returns_zero(self):
        assert get_msw_reward_component(np.array([]), "meropenem", "E_coli", 1.0) == 0.0


# ---------------------------------------------------------------------------
# Mandatory acceptance test — MSW
# ---------------------------------------------------------------------------


def test_msw():
    """
    Acceptance criteria from CLAUDE.md (msw module):

    1. MPC > MIC confirmed for meropenem vs E_coli
    2. C series mostly above MPC → msw_risk < 0.2
    3. C series mostly in MSW → msw_risk > 0.6
    """
    mic = 2.0
    mpc = compute_mpc("meropenem", "E_coli", mic)

    # ---- 1. MPC > MIC ----
    assert mpc > mic, f"Expected MPC > MIC={mic}, got MPC={mpc:.2f}"

    # ---- 2. C mostly above MPC → msw_risk < 0.2 ----
    # 90 % of points above MPC, 10 % in MSW zone
    C_high = np.concatenate([
        np.full(90, mpc * 1.5),            # above MPC
        np.full(10, (mic + mpc) / 2.0),    # in MSW
    ])
    risk_high = compute_msw_risk(C_high, "meropenem", "E_coli", mic)
    assert risk_high < 0.2, (
        f"C mostly above MPC: expected msw_risk < 0.2, got {risk_high:.3f}"
    )

    # ---- 3. C mostly in MSW → msw_risk > 0.6 ----
    # All points within [MIC, MPC]
    C_msw = np.full(100, (mic + mpc) / 2.0)
    risk_msw = compute_msw_risk(C_msw, "meropenem", "E_coli", mic)
    assert risk_msw > 0.6, (
        f"C in MSW: expected msw_risk > 0.6, got {risk_msw:.3f}"
    )


# ---------------------------------------------------------------------------
# Resistance SDE — Itô SDE model (resistance_sde.py)
# ---------------------------------------------------------------------------

from episteward.math.resistance_sde import (
    selection_coefficient,
    euler_maruyama_step,
    simulate_resistance_trajectory,
    resistance_emerged as sde_resistance_emerged,
)
from episteward.math.msw import compute_mpc as _compute_mpc_for_sde


class TestSelectionCoefficient:
    def test_sub_mic_is_negative(self):
        """Below MIC: resistant allele pays a fitness cost → s < 0."""
        mic = 2.0
        mpc = _compute_mpc_for_sde("meropenem", "E_coli", mic)
        s = selection_coefficient(0.5, mic, mpc)
        assert s < 0.0

    def test_sub_mic_equals_fitness_cost(self):
        mic = 2.0
        mpc = _compute_mpc_for_sde("meropenem", "E_coli", mic)
        s = selection_coefficient(0.0, mic, mpc)
        assert s == pytest.approx(-0.05)

    def test_above_mpc_is_zero(self):
        """Above MPC: sterilising concentration — no selection."""
        mic = 2.0
        mpc = _compute_mpc_for_sde("meropenem", "E_coli", mic)
        s = selection_coefficient(mpc * 2.0, mic, mpc)
        assert s == pytest.approx(0.0)

    def test_msw_interior_positive(self):
        """Inside MSW: selection is positive (mutant amplification)."""
        mic = 2.0
        mpc = _compute_mpc_for_sde("meropenem", "E_coli", mic)
        s = selection_coefficient((mic + mpc) / 2.0, mic, mpc)
        assert s > 0.0

    def test_msw_at_mic_is_half_smax(self):
        """At C=MIC, EC50=MIC → Hill term = 0.5 → s = s_max/2."""
        mic = 2.0
        mpc = _compute_mpc_for_sde("meropenem", "E_coli", mic)
        s = selection_coefficient(mic, mic, mpc, s_max=0.3, n=2.0)
        assert s == pytest.approx(0.3 / 2.0, rel=1e-6)

    def test_msw_approaches_smax_near_mpc(self):
        """Near MPC (C >> MIC but still ≤ MPC), s ≈ s_max."""
        mic = 0.001
        mpc = _compute_mpc_for_sde("meropenem", "E_coli", mic)
        s = selection_coefficient(mpc * 0.99, mic, mpc, s_max=0.3, n=2.0)
        assert s == pytest.approx(0.3, abs=1e-3)

    def test_s_in_valid_range(self):
        mic = 2.0
        mpc = _compute_mpc_for_sde("meropenem", "E_coli", mic)
        for C in [0.0, mic, (mic + mpc) / 2.0, mpc, mpc * 2.0]:
            s = selection_coefficient(C, mic, mpc)
            assert -0.1 <= s <= 0.4


class TestEulerMaruyamaStep:
    _SIGMA = float(np.sqrt(1.0 / (2.0 * 1e8)))

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(0)
        for p in [0.0, 0.01, 0.5, 0.99, 1.0]:
            p_new = euler_maruyama_step(p, s=0.3, sigma=self._SIGMA, dt=0.5, rng=rng)
            assert 0.0 <= p_new <= 1.0

    def test_zero_freq_stays_zero(self):
        """p_R = 0: both drift and diffusion terms vanish → stays at 0."""
        rng = np.random.default_rng(1)
        p_new = euler_maruyama_step(0.0, s=0.3, sigma=self._SIGMA, dt=0.5, rng=rng)
        assert p_new == pytest.approx(0.0, abs=1e-12)

    def test_unity_freq_stays_unity(self):
        """p_R = 1: both terms vanish → stays at 1."""
        rng = np.random.default_rng(2)
        p_new = euler_maruyama_step(1.0, s=0.3, sigma=self._SIGMA, dt=0.5, rng=rng)
        assert p_new == pytest.approx(1.0, abs=1e-12)

    def test_positive_selection_tends_upward(self):
        """With strong positive selection, repeated steps should trend up."""
        rng = np.random.default_rng(42)
        p = 0.3
        steps = [p]
        for _ in range(200):
            p = euler_maruyama_step(p, s=0.3, sigma=self._SIGMA, dt=1.0, rng=rng)
            steps.append(p)
        assert steps[-1] > steps[0]

    def test_zero_selection_minimal_drift(self):
        """With s=0 and tiny sigma, frequency barely moves."""
        rng = np.random.default_rng(7)
        p = 0.5
        for _ in range(100):
            p = euler_maruyama_step(p, s=0.0, sigma=self._SIGMA, dt=0.5, rng=rng)
        assert abs(p - 0.5) < 0.001

    def test_reproducible_with_seed(self):
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        result_a = euler_maruyama_step(0.1, 0.2, self._SIGMA, 0.5, rng_a)
        result_b = euler_maruyama_step(0.1, 0.2, self._SIGMA, 0.5, rng_b)
        assert result_a == pytest.approx(result_b)


class TestSimulateResistanceTrajectory:
    def test_length_is_c_series_plus_one(self):
        rng = np.random.default_rng(0)
        C = np.full(10, 50.0)
        traj = simulate_resistance_trajectory(0.01, C, "meropenem", "E_coli", 0.5, rng)
        assert len(traj) == 11  # initial + 10 steps

    def test_first_element_is_p_init(self):
        rng = np.random.default_rng(1)
        C = np.full(5, 30.0)
        traj = simulate_resistance_trajectory(0.07, C, "meropenem", "E_coli", 0.5, rng)
        assert traj[0] == pytest.approx(0.07)

    def test_all_values_in_unit_interval(self):
        rng = np.random.default_rng(2)
        C = np.linspace(0.0, 500.0, 50)
        traj = simulate_resistance_trajectory(0.05, C, "meropenem", "E_coli", 0.5, rng)
        assert all(0.0 <= v <= 1.0 for v in traj)

    def test_reproducible_with_same_seed(self):
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        C = np.full(20, 50.0)
        traj_a = simulate_resistance_trajectory(0.01, C, "meropenem", "E_coli", 0.5, rng_a)
        traj_b = simulate_resistance_trajectory(0.01, C, "meropenem", "E_coli", 0.5, rng_b)
        assert traj_a == pytest.approx(traj_b)

    def test_different_seeds_differ(self):
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        C = np.full(50, 50.0)
        traj_a = simulate_resistance_trajectory(0.1, C, "meropenem", "E_coli", 0.5, rng_a)
        traj_b = simulate_resistance_trajectory(0.1, C, "meropenem", "E_coli", 0.5, rng_b)
        # Trajectories should diverge due to different noise realisations
        assert not np.allclose(traj_a, traj_b)

    def test_empty_c_series_returns_single_value(self):
        rng = np.random.default_rng(0)
        traj = simulate_resistance_trajectory(0.05, np.array([]), "meropenem", "E_coli", 0.5, rng)
        assert traj == [pytest.approx(0.05)]


class TestResistanceEmerged:
    def test_never_above_threshold_returns_false(self):
        trajectory = [0.1, 0.2, 0.3, 0.4]
        assert sde_resistance_emerged(trajectory, threshold=0.5, sustained_steps=1) is False

    def test_single_step_above_threshold(self):
        trajectory = [0.1, 0.6, 0.1]
        assert sde_resistance_emerged(trajectory, threshold=0.5, sustained_steps=1) is True

    def test_sustained_required(self):
        """Must be above threshold for ≥ sustained_steps consecutive steps."""
        trajectory = [0.1, 0.6, 0.6, 0.1, 0.6, 0.6, 0.6]
        assert sde_resistance_emerged(trajectory, threshold=0.5, sustained_steps=3) is True
        assert sde_resistance_emerged(trajectory, threshold=0.5, sustained_steps=4) is False

    def test_resets_on_drop_below(self):
        """Streak resets when p_R drops below threshold."""
        trajectory = [0.6, 0.6, 0.4, 0.6]
        assert sde_resistance_emerged(trajectory, threshold=0.5, sustained_steps=3) is False

    def test_all_above_short_sustained(self):
        trajectory = [0.8] * 10
        assert sde_resistance_emerged(trajectory, threshold=0.5, sustained_steps=5) is True

    def test_empty_trajectory(self):
        assert sde_resistance_emerged([], threshold=0.5) is False


# ---------------------------------------------------------------------------
# Mandatory acceptance test — resistance SDE
# ---------------------------------------------------------------------------


def test_resistance_sde():
    """
    Acceptance criteria from CLAUDE.md (resistance_sde module):

    1. Same seed → identical trajectories (reproducibility)
    2. Different seeds → different trajectories
    3. 72h exposure in MSW zone → resistance frequency increases
    4. C(t) consistently above MPC → resistance stays low
    """
    mic = 2.0
    mpc = _compute_mpc_for_sde("meropenem", "E_coli", mic)
    dt = 0.5
    n_steps = int(72.0 / dt)  # 144 steps = 72 h

    # ---- 1. Same seed → identical trajectories ----
    C_msw = np.full(n_steps, (mic + mpc) / 2.0)
    traj_a = simulate_resistance_trajectory(0.01, C_msw, "meropenem", "E_coli", dt,
                                             np.random.default_rng(42), mic=mic)
    traj_b = simulate_resistance_trajectory(0.01, C_msw, "meropenem", "E_coli", dt,
                                             np.random.default_rng(42), mic=mic)
    assert traj_a == pytest.approx(traj_b), "Same seed must produce identical trajectories"

    # ---- 2. Different seeds → different trajectories ----
    traj_c = simulate_resistance_trajectory(0.01, C_msw, "meropenem", "E_coli", dt,
                                             np.random.default_rng(1), mic=mic)
    traj_d = simulate_resistance_trajectory(0.01, C_msw, "meropenem", "E_coli", dt,
                                             np.random.default_rng(2), mic=mic)
    assert not np.allclose(traj_c, traj_d), "Different seeds should produce different trajectories"

    # ---- 3. 72h in MSW → resistance frequency increases ----
    # C in MSW → s ≈ s_max ≈ 0.3; logistic growth pushes p_R up significantly
    rng = np.random.default_rng(0)
    traj_msw = simulate_resistance_trajectory(0.01, C_msw, "meropenem", "E_coli", dt,
                                               rng, mic=mic)
    assert traj_msw[-1] > traj_msw[0], (
        f"72h in MSW: expected resistance to increase, "
        f"got {traj_msw[0]:.4f} → {traj_msw[-1]:.4f}"
    )
    assert traj_msw[-1] > 0.10, (
        f"72h in MSW: expected final p_R > 0.10, got {traj_msw[-1]:.4f}"
    )

    # ---- 4. C consistently above MPC → resistance stays low ----
    # C > MPC → s = 0; only tiny diffusion remains; p_R barely moves
    rng2 = np.random.default_rng(0)
    C_high = np.full(n_steps, mpc * 2.0)
    traj_high = simulate_resistance_trajectory(0.01, C_high, "meropenem", "E_coli", dt,
                                                rng2, mic=mic)
    assert traj_high[-1] < 0.05, (
        f"C above MPC: expected p_R to stay low (<0.05), got {traj_high[-1]:.6f}"
    )


# ---------------------------------------------------------------------------
# HGT — Horizontal Gene Transfer (hgt.py)
# ---------------------------------------------------------------------------

from episteward.math.hgt import (
    hgt_rate,
    solve_hgt_ode,
    cross_species_transfer,
    get_hgt_pressure_score,
)


class TestHGTRate:
    def test_sos_increases_with_concentration(self):
        """γ increases with C (SOS response)."""
        mic = 2.0
        assert hgt_rate(mic * 2, mic) > hgt_rate(0, mic)

    def test_zero_concentration_equals_baseline(self):
        mic = 2.0
        assert hgt_rate(0.0, mic) == pytest.approx(0.001)

    def test_formula(self):
        """γ(C) = γ_baseline * (1 + α_SOS * C/MIC)."""
        mic = 2.0
        C = 4.0
        expected = 0.001 * (1.0 + 2.0 * C / mic)
        assert hgt_rate(C, mic) == pytest.approx(expected)

    def test_higher_alpha_sos_higher_rate(self):
        mic = 2.0
        low = hgt_rate(mic, mic, alpha_sos=1.0)
        high = hgt_rate(mic, mic, alpha_sos=4.0)
        assert high > low

    def test_returns_float(self):
        assert isinstance(hgt_rate(2.0, 2.0), float)


class TestSolveHGTODE:
    def test_returns_three_arrays(self):
        S, R, D = solve_hgt_ode(1e8, 0, 0, np.full(5, 2.0), "meropenem", "E_coli")
        assert isinstance(S, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert isinstance(D, np.ndarray)

    def test_output_length(self):
        """Returns T+1 values (including initial condition at t=0)."""
        S, R, D = solve_hgt_ode(1e8, 0, 0, np.full(6, 2.0), "meropenem", "E_coli")
        assert len(S) == 7 and len(R) == 7 and len(D) == 7

    def test_initial_conditions_preserved(self):
        S, R, D = solve_hgt_ode(1e8, 500, 1000, np.full(3, 2.0), "meropenem", "E_coli")
        assert S[0] == pytest.approx(1e8, rel=1e-3)
        assert R[0] == pytest.approx(500, rel=1e-3)
        assert D[0] == pytest.approx(1000, rel=1e-3)

    def test_non_negative_populations(self):
        """All population counts must be ≥ 0."""
        S, R, D = solve_hgt_ode(1e8, 100, 1000, np.full(24, 10.0),
                                  "meropenem", "E_coli", dt=1.0)
        assert np.all(S >= 0) and np.all(R >= 0) and np.all(D >= 0)

    def test_empty_c_series_returns_initial(self):
        S, R, D = solve_hgt_ode(1e8, 500, 0, np.array([]), "meropenem", "E_coli")
        assert S[0] == pytest.approx(1e8, rel=1e-3)
        assert len(S) == 1

    def test_d0_zero_r_stays_low(self):
        """Without donors (D0=0) and small population, R stays negligible vs K=1e9.

        μ·S with S=1000 → ~1e-5 donors/h; after 24h R << 1000 << K=1e9.
        """
        S, R, D = solve_hgt_ode(
            S0=1_000, R0=0, D0=0,
            C_series=np.full(24, 2.0),
            drug_name="meropenem", pathogen="E_coli",
            dt=1.0, mic=2.0,
        )
        # R should stay negligible (<<K=1e9); threshold 1e3 = 6 orders below capacity
        assert R[-1] < 1_000, f"D0=0 small S: expected R << 1e3, got R={R[-1]:.2f}"

    def test_donors_present_creates_r(self):
        """With a large donor population, R should grow via plasmid transfer."""
        S, R, D = solve_hgt_ode(
            S0=1e8, R0=0, D0=1e6,
            C_series=np.zeros(12),   # no drug pressure
            drug_name="meropenem", pathogen="E_coli",
            dt=1.0, mic=2.0,
        )
        assert R[-1] > R[0]  # R grows from zero via transfer

    def test_high_c_at_capacity_r_decreases(self):
        """
        With S=0 (no susceptibles → no mutation, no HGT) and R0=K,
        high drug concentration drives R down:
            dR/dt = r_R*R*(1-K/K) - k_kill*R/f - δ*R = -k_kill/f·R - δ·R < 0
        """
        mic = 2.0
        S, R, D = solve_hgt_ode(
            S0=0,       # no susceptibles → no mutation, no donors, no HGT
            R0=1e9,     # at K → logistic term ≈ 0
            D0=0,
            C_series=np.full(6, 100 * mic),  # C >> MIC; k_kill ≈ E_max = 0.8
            drug_name="meropenem", pathogen="E_coli",
            dt=1.0, mic=mic,
        )
        assert R[-1] < R[0], (
            f"High C, S=0, R0=K: expected R to decrease, got R0={R[0]:.1f} → R_final={R[-1]:.1f}"
        )


class TestCrossSpeciesTransfer:
    def test_returns_float(self):
        assert isinstance(cross_species_transfer(1e6, 1e8, 0.1), float)

    def test_zero_donor_gives_zero(self):
        assert cross_species_transfer(0.0, 1e8, 0.5) == pytest.approx(0.0)

    def test_zero_recipient_gives_zero(self):
        assert cross_species_transfer(1e6, 0.0, 0.5) == pytest.approx(0.0)

    def test_scales_with_contact_rate(self):
        low  = cross_species_transfer(1e6, 1e8, contact_rate=0.1)
        high = cross_species_transfer(1e6, 1e8, contact_rate=0.5)
        assert high == pytest.approx(5.0 * low, rel=1e-9)

    def test_formula(self):
        D, S, cr, gc = 1e5, 1e7, 0.2, 0.001
        expected = gc * cr * D * S
        assert cross_species_transfer(D, S, cr, gc) == pytest.approx(expected)


class TestGetHGTPressureScore:
    def test_score_in_unit_interval(self):
        score = get_hgt_pressure_score(1e6, 0.0, 0.0)
        assert 0.0 <= score <= 1.0

    def test_no_increase_no_transfer_gives_zero(self):
        """R_final == R_initial, no cross-transfers → score = 0."""
        assert get_hgt_pressure_score(500.0, 500.0, 0.0) == pytest.approx(0.0)

    def test_decrease_in_r_gives_zero(self):
        """R_final < R_initial (resistance reduced) → delta_R clamped to 0."""
        assert get_hgt_pressure_score(100.0, 500.0, 0.0) == pytest.approx(0.0)

    def test_higher_r_increase_higher_score(self):
        score_low  = get_hgt_pressure_score(1e6,   0.0, 0.0)
        score_high = get_hgt_pressure_score(1e7,   0.0, 0.0)
        assert score_high > score_low

    def test_cross_transfers_contribute(self):
        score_no_cross   = get_hgt_pressure_score(0.0, 0.0, 0.0)
        score_with_cross = get_hgt_pressure_score(0.0, 0.0, 1e7)
        assert score_with_cross > score_no_cross

    def test_returns_float(self):
        assert isinstance(get_hgt_pressure_score(1e6, 0.0, 0.0), float)


# ---------------------------------------------------------------------------
# Mandatory acceptance test — HGT
# ---------------------------------------------------------------------------


def test_hgt():
    """
    Acceptance criteria from CLAUDE.md (hgt module):

    1. hgt_rate(C=MIC*2) > hgt_rate(C=0)       — SOS response confirmed
    2. D0=0 → R near 0 after 24h               — no donor = no significant transfer
    3. C >> MIC at N=K, D0=0 → R_final < R_initial — killing dominates logistic growth
    """
    mic = 2.0

    # ---- 1. SOS: hgt_rate increases with C ----
    assert hgt_rate(mic * 2, mic) > hgt_rate(0, mic), (
        "SOS response: hgt_rate(C=2·MIC) should exceed hgt_rate(C=0)"
    )

    # ---- 2. D0=0 → R stays near 0 (small S so mutation is negligible) ----
    # μ·S = 1e-8 * 1000 = 1e-5 donors/h → HGT ≈ 0
    S, R, D = solve_hgt_ode(
        S0=1_000, R0=0, D0=0,
        C_series=np.full(24, mic),
        drug_name="meropenem", pathogen="E_coli",
        dt=1.0, mic=mic,
    )
    assert R[-1] < 1_000, (
        f"D0=0 small S: expected R << K=1e9 after 24h, got R={R[-1]:.2f}"
    )

    # ---- 3. C >> MIC, S=0 → R decreases (no mutation/HGT; killing > growth at N=K) ----
    # dR/dt = 0 - k_kill/f·R - δ·R < 0  when N = K (logistic = 0) and S = 0
    S2, R2, D2 = solve_hgt_ode(
        S0=0,       # no susceptibles → no mutation, no HGT
        R0=1e9,     # at K → logistic term ≈ 0
        D0=0,
        C_series=np.full(6, 100 * mic),   # high concentration
        drug_name="meropenem", pathogen="E_coli",
        dt=1.0, mic=mic,
    )
    assert R2[-1] < R2[0], (
        f"C >> MIC, S=0, R0=K: expected R to decrease from {R2[0]:.1f}, "
        f"got R_final={R2[-1]:.1f}"
    )


# ===========================================================================
# Bayesian Diagnostics
# ===========================================================================


class TestInitPrior:
    def test_sums_to_one(self):
        prior = init_prior("urinary", [], {})
        assert sum(prior.values()) == pytest.approx(1.0, abs=1e-9)

    def test_all_pathogens_present(self):
        prior = init_prior("bloodstream", [], {})
        assert set(prior.keys()) == set(PATHOGENS)

    def test_all_probabilities_positive(self):
        prior = init_prior("respiratory", ["icu"], {})
        assert all(v > 0 for v in prior.values())

    def test_uti_ecoli_dominant(self):
        """Urinary tract infections are dominated by E. coli empirically."""
        prior = init_prior("uti", [], {})
        assert prior["E_coli"] == max(prior.values())

    def test_icu_risk_increases_kp_pa(self):
        """ICU risk factor should raise K_pneumoniae and P_aeruginosa."""
        base  = init_prior("bloodstream", [], {})
        icu   = init_prior("bloodstream", ["icu"], {})
        assert icu["K_pneumoniae"] > base["K_pneumoniae"]
        assert icu["P_aeruginosa"] > base["P_aeruginosa"]

    def test_unknown_site_returns_uniform(self):
        """Unknown infection site should yield roughly equal probabilities."""
        prior = init_prior("unknown_site_xyz", [], {})
        vals = list(prior.values())
        assert max(vals) - min(vals) < 0.05  # near-uniform

    def test_ward_antibiogram_blends_prior(self):
        """Antibiogram heavily favouring S_aureus should raise its probability."""
        prior_no_ab = init_prior("bloodstream", [], {})
        ab = {"S_aureus": 100.0, "E_coli": 0.0, "K_pneumoniae": 0.0,
              "P_aeruginosa": 0.0, "E_faecalis": 0.0}
        prior_ab = init_prior("bloodstream", [], ab)
        assert prior_ab["S_aureus"] > prior_no_ab["S_aureus"]


class TestUpdatePosterior:
    _UNIFORM = {p: 0.2 for p in PATHOGENS}

    def test_gram_negative_increases_ecoli(self):
        post = diag_update_posterior(self._UNIFORM, "gram_stain", "gram_negative")
        assert post["E_coli"] > self._UNIFORM["E_coli"]

    def test_gram_negative_decreases_saureus(self):
        post = diag_update_posterior(self._UNIFORM, "gram_stain", "gram_negative")
        assert post["S_aureus"] < self._UNIFORM["S_aureus"]

    def test_gram_positive_increases_saureus(self):
        post = diag_update_posterior(self._UNIFORM, "gram_stain", "gram_positive")
        assert post["S_aureus"] > self._UNIFORM["S_aureus"]

    def test_culture_ecoli_posterior_sums_to_one(self):
        post = diag_update_posterior(self._UNIFORM, "culture", "E_coli")
        assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)

    def test_culture_makes_reported_pathogen_dominant(self):
        for pathogen in PATHOGENS:
            post = diag_update_posterior(self._UNIFORM, "culture", pathogen)
            assert post[pathogen] == max(post.values()), (
                f"Culture {pathogen}: expected dominant posterior"
            )

    def test_sensitivity_panel_resistant_raises_pa(self):
        """Resistant result should elevate P. aeruginosa (MDR gram-negative)."""
        post = diag_update_posterior(self._UNIFORM, "sensitivity_panel", "resistant")
        assert post["P_aeruginosa"] > post["E_coli"]

    def test_posterior_always_sums_to_one(self):
        scenarios = [
            ("gram_stain", "gram_negative"),
            ("gram_stain", "gram_positive"),
            ("sensitivity_panel", "susceptible"),
            ("sensitivity_panel", "resistant"),
        ] + [("culture", p) for p in PATHOGENS]
        for obs_type, result in scenarios:
            post = diag_update_posterior(self._UNIFORM, obs_type, result)
            assert sum(post.values()) == pytest.approx(1.0, abs=1e-9), (
                f"Posterior sum ≠ 1 for ({obs_type}, {result})"
            )

    def test_invalid_obs_type_raises(self):
        with pytest.raises(ValueError, match="Unknown observation_type"):
            diag_update_posterior(self._UNIFORM, "x_ray", "positive")

    def test_invalid_gram_result_raises(self):
        with pytest.raises(ValueError, match="Unknown gram_stain result"):
            diag_update_posterior(self._UNIFORM, "gram_stain", "unknown")

    def test_multiple_updates_reduce_entropy(self):
        """Sequential observations should reduce uncertainty."""
        prior = {p: 0.2 for p in PATHOGENS}
        h0 = shannon_entropy(prior)
        post1 = diag_update_posterior(prior, "gram_stain", "gram_negative")
        post2 = diag_update_posterior(post1, "culture", "E_coli")
        h2 = shannon_entropy(post2)
        assert h2 < h0


class TestShannonEntropy:
    def test_uniform_is_max_entropy(self):
        uniform = {p: 0.2 for p in PATHOGENS}
        h = shannon_entropy(uniform)
        import math
        assert h == pytest.approx(math.log2(5), rel=1e-6)

    def test_certain_distribution_has_zero_entropy(self):
        certain = {"E_coli": 1.0, "K_pneumoniae": 0.0, "S_aureus": 0.0,
                   "P_aeruginosa": 0.0, "E_faecalis": 0.0}
        assert shannon_entropy(certain) == pytest.approx(0.0, abs=1e-12)

    def test_entropy_in_valid_range(self):
        import math
        max_h = math.log2(len(PATHOGENS))
        for dist in [
            {p: 0.2 for p in PATHOGENS},
            {"E_coli": 0.9, "K_pneumoniae": 0.025, "S_aureus": 0.025,
             "P_aeruginosa": 0.025, "E_faecalis": 0.025},
        ]:
            h = shannon_entropy(dist)
            assert 0.0 <= h <= max_h + 1e-9

    def test_returns_float(self):
        assert isinstance(shannon_entropy({"E_coli": 0.5, "K_pneumoniae": 0.5,
                                           "S_aureus": 0.0, "P_aeruginosa": 0.0,
                                           "E_faecalis": 0.0}), float)

    def test_handles_zero_probabilities_gracefully(self):
        """0 · log₂(0) should be treated as 0."""
        dist = {"E_coli": 0.5, "K_pneumoniae": 0.5, "S_aureus": 0.0,
                "P_aeruginosa": 0.0, "E_faecalis": 0.0}
        assert shannon_entropy(dist) == pytest.approx(1.0, rel=1e-6)  # H({0.5,0.5}) = 1 bit


class TestValueOfInformation:
    _UNIFORM = {p: 0.2 for p in PATHOGENS}
    _CERTAIN = {"E_coli": 0.999, "K_pneumoniae": 0.0003, "S_aureus": 0.0,
                "P_aeruginosa": 0.0003, "E_faecalis": 0.0004}

    def test_voi_high_when_uncertain(self):
        """Uniform prior → testing should have high VoI."""
        voi = value_of_information(self._UNIFORM, "gram_stain", test_cost=0.0)
        assert voi > 0.3  # large reduction in uncertainty expected

    def test_voi_near_zero_when_certain(self):
        """Near-certain prior → test gives little new information."""
        voi = value_of_information(self._CERTAIN, "gram_stain", test_cost=0.0)
        assert voi < 0.1

    def test_culture_higher_voi_than_gram_stain(self):
        """Culture resolves more uncertainty than gram stain (2 results vs 5)."""
        voi_gram = value_of_information(self._UNIFORM, "gram_stain",    test_cost=0.0)
        voi_cult = value_of_information(self._UNIFORM, "culture", test_cost=0.0)
        assert voi_cult > voi_gram

    def test_high_cost_reduces_voi(self):
        voi_free = value_of_information(self._UNIFORM, "gram_stain", test_cost=0.0)
        voi_paid = value_of_information(self._UNIFORM, "gram_stain", test_cost=0.5)
        assert voi_free > voi_paid

    def test_invalid_test_type_raises(self):
        with pytest.raises(ValueError, match="Unknown test_type"):
            value_of_information(self._UNIFORM, "mri_scan", test_cost=0.1)

    def test_returns_float(self):
        voi = value_of_information(self._UNIFORM, "sensitivity_panel", test_cost=0.05)
        assert isinstance(voi, float)


class TestGetDiagnosticReward:
    def test_large_entropy_reduction_high_reward(self):
        """Going from max entropy to near-zero should give high reward."""
        import math
        h_max = math.log2(5)
        reward = get_diagnostic_reward(
            prior_entropy=h_max, posterior_entropy=0.01,
            test_ordered=True, test_cost=0.0,
        )
        assert reward > 0.9

    def test_no_entropy_reduction_gives_zero(self):
        """If entropy didn't change, reward = 0 (no information gained)."""
        reward = get_diagnostic_reward(
            prior_entropy=1.5, posterior_entropy=1.5,
            test_ordered=True, test_cost=0.0,
        )
        assert reward == pytest.approx(0.0, abs=1e-9)

    def test_cost_reduces_reward(self):
        import math
        h = math.log2(5)
        reward_no_cost = get_diagnostic_reward(h, 0.5, True, test_cost=0.0)
        reward_with_cost = get_diagnostic_reward(h, 0.5, True, test_cost=0.3)
        assert reward_no_cost > reward_with_cost

    def test_reward_in_unit_interval(self):
        import math
        for h_prior, h_post, ordered, cost in [
            (math.log2(5), 0.0, True, 0.0),
            (math.log2(5), 0.0, True, 0.5),
            (1.0, 0.5, False, 0.2),
            (0.1, 0.05, True, 0.9),
        ]:
            r = get_diagnostic_reward(h_prior, h_post, ordered, cost)
            assert 0.0 <= r <= 1.0, f"Reward {r} out of [0,1] for {(h_prior,h_post,ordered,cost)}"

    def test_test_not_ordered_no_cost_penalty(self):
        """If no test was ordered, cost should not reduce reward."""
        import math
        h = math.log2(5)
        r_no_test  = get_diagnostic_reward(h, 0.5, test_ordered=False, test_cost=0.5)
        r_with_test = get_diagnostic_reward(h, 0.5, test_ordered=True,  test_cost=0.5)
        assert r_no_test > r_with_test

    def test_returns_float(self):
        assert isinstance(get_diagnostic_reward(1.0, 0.5, True, 0.1), float)


# ---------------------------------------------------------------------------
# Mandatory acceptance test — Bayesian diagnostics
# ---------------------------------------------------------------------------


def test_bayesian_diagnostics():
    """
    Acceptance criteria from CLAUDE.md (bayesian_diagnostics module):

    1. gram_negative → E_coli posterior increases, S_aureus decreases
    2. Full culture result (exact species) → entropy near 0
    3. VoI high when prior is uncertain, near 0 when already certain
    4. Posterior always sums to 1.0
    """
    uniform = {p: 0.2 for p in PATHOGENS}

    # ---- 1. gram_negative update ----
    post_gn = diag_update_posterior(uniform, "gram_stain", "gram_negative")
    assert post_gn["E_coli"] > uniform["E_coli"], (
        "gram_negative: E_coli posterior should increase"
    )
    assert post_gn["S_aureus"] < uniform["S_aureus"], (
        "gram_negative: S_aureus posterior should decrease"
    )

    # ---- 2. Full culture → entropy near 0 ----
    post_culture = diag_update_posterior(uniform, "culture", "E_coli")
    h_post = shannon_entropy(post_culture)
    assert h_post < 0.5, (
        f"culture(E_coli): expected entropy near 0, got H={h_post:.3f} bits"
    )

    # ---- 3. VoI: uncertain >> certain ----
    certain = {"E_coli": 0.97, "K_pneumoniae": 0.01, "S_aureus": 0.01,
               "P_aeruginosa": 0.005, "E_faecalis": 0.005}
    voi_uncertain = value_of_information(uniform, "gram_stain", test_cost=0.0)
    voi_certain   = value_of_information(certain, "gram_stain", test_cost=0.0)
    assert voi_uncertain > voi_certain, (
        f"VoI should be higher for uncertain prior: "
        f"uncertain={voi_uncertain:.3f}, certain={voi_certain:.3f}"
    )
    assert voi_certain < 0.1, (
        f"VoI near-certain prior should be < 0.1, got {voi_certain:.3f}"
    )

    # ---- 4. Posterior always sums to 1 ----
    for obs_type, result in [
        ("gram_stain", "gram_negative"),
        ("gram_stain", "gram_positive"),
        ("culture",    "K_pneumoniae"),
        ("sensitivity_panel", "resistant"),
    ]:
        post = diag_update_posterior(uniform, obs_type, result)
        assert sum(post.values()) == pytest.approx(1.0, abs=1e-9), (
            f"Posterior must sum to 1 for ({obs_type}, {result})"
        )


# ===========================================================================
# Pareto Reward
# ===========================================================================

# Ward params giving tragedy-of-the-commons Nash (alpha < f_max-f_min = 0.25)
_TOC_PARAMS = {"f_min": 0.70, "f_max": 0.95, "alpha": 0.13, "beta": 0.80}


class TestComputeRewardVector:
    def test_shape(self):
        v = compute_reward_vector(0.8, 0.6, 0.5, 0.7)
        assert v.shape == (4,)

    def test_values_preserved(self):
        v = compute_reward_vector(0.1, 0.2, 0.3, 0.4)
        assert v == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_clamps_above_one(self):
        v = compute_reward_vector(1.5, 0.5, 0.5, 0.5)
        assert v[0] == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        v = compute_reward_vector(-0.3, 0.5, 0.5, 0.5)
        assert v[0] == pytest.approx(0.0)


class TestComputeWRPI:
    def test_uniform_full_resistance_is_one(self):
        prev = {"E_coli": 1.0, "K_pneumoniae": 1.0}
        wts  = {"E_coli": 1.0, "K_pneumoniae": 1.0}
        assert compute_wrpi(prev, wts) == pytest.approx(1.0)

    def test_no_resistance_is_zero(self):
        prev = {"E_coli": 0.0, "K_pneumoniae": 0.0}
        wts  = {"E_coli": 2.0, "K_pneumoniae": 1.0}
        assert compute_wrpi(prev, wts) == pytest.approx(0.0)

    def test_partial_resistance(self):
        prev = {"A": 0.5, "B": 0.5}
        wts  = {"A": 1.0, "B": 1.0}
        assert compute_wrpi(prev, wts) == pytest.approx(0.5)

    def test_empty_weights_returns_zero(self):
        assert compute_wrpi({"E_coli": 0.8}, {}) == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        prev = {"A": 0.3, "B": 0.7, "C": 0.5}
        wts  = {"A": 2.0, "B": 1.0, "C": 3.0}
        wrpi = compute_wrpi(prev, wts)
        assert 0.0 <= wrpi <= 1.0


class TestComputeAdaptiveWeights:
    def test_shape_and_sum_to_one(self):
        w = compute_adaptive_weights(0.5)
        assert w.shape == (4,)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_high_wrpi_ecology_dominates_clinical(self):
        """At WRPI=1 ecology weight should exceed clinical weight."""
        w = compute_adaptive_weights(1.0)
        assert w[1] > w[0], "Ecology weight should exceed clinical at WRPI=1"

    def test_zero_wrpi_base_order_preserved(self):
        """At WRPI=0 clinical should dominate (largest base weight)."""
        w = compute_adaptive_weights(0.0)
        assert w[0] == max(w), "Clinical should be largest at WRPI=0"

    def test_weights_monotone_in_wrpi(self):
        """Ecology weight increases, clinical weight decreases with WRPI."""
        w_low  = compute_adaptive_weights(0.0)
        w_high = compute_adaptive_weights(1.0)
        assert w_high[1] > w_low[1]   # ecology rises
        assert w_high[0] < w_low[0]   # clinical drops


class TestScalarize:
    def test_dot_product(self):
        v = np.array([0.8, 0.6, 0.4, 0.5])
        w = np.array([0.4, 0.3, 0.15, 0.15])
        assert scalarize(v, w) == pytest.approx(float(np.dot(v, w)), rel=1e-6)

    def test_in_unit_interval(self):
        for v, w in [
            (np.ones(4), np.array([0.4, 0.3, 0.15, 0.15])),
            (np.zeros(4), np.ones(4)),
        ]:
            assert 0.0 <= scalarize(v, w) <= 1.0


class TestIsDominated:
    def test_dominated_by_single_better_point(self):
        p = np.array([0.5, 0.5, 0.5, 0.5])
        q = np.array([0.8, 0.8, 0.8, 0.8])
        assert is_dominated(p, [q])

    def test_not_dominated_if_trade_off(self):
        p = np.array([0.9, 0.3, 0.5, 0.5])
        q = np.array([0.3, 0.9, 0.5, 0.5])
        assert not is_dominated(p, [q])

    def test_equal_point_not_dominated(self):
        p = np.array([0.5, 0.5, 0.5, 0.5])
        assert not is_dominated(p, [p.copy()])

    def test_empty_front_not_dominated(self):
        assert not is_dominated(np.array([0.5]*4), [])


class TestUpdateParetoFront:
    def test_non_dominated_point_added(self):
        front = [np.array([0.8, 0.3, 0.5, 0.5])]
        new   = np.array([0.3, 0.9, 0.5, 0.5])
        updated = update_pareto_front(front, new)
        assert len(updated) == 2

    def test_dominated_point_rejected(self):
        q = np.array([0.9, 0.9, 0.9, 0.9])
        p = np.array([0.5, 0.5, 0.5, 0.5])  # dominated by q
        front = [q]
        updated = update_pareto_front(front, p)
        assert len(updated) == 1
        assert np.allclose(updated[0], q)

    def test_new_dominator_prunes_old(self):
        weak = np.array([0.5, 0.5, 0.5, 0.5])
        strong = np.array([0.9, 0.9, 0.9, 0.9])
        front = [weak]
        updated = update_pareto_front(front, strong)
        assert len(updated) == 1
        assert np.allclose(updated[0], strong)

    def test_original_front_not_mutated(self):
        original = [np.array([0.8, 0.3, 0.5, 0.5])]
        update_pareto_front(original, np.array([0.3, 0.9, 0.5, 0.5]))
        assert len(original) == 1  # original unchanged


class TestComputeHypervolume:
    def test_empty_front_is_zero(self):
        assert compute_hypervolume([]) == pytest.approx(0.0)

    def test_single_point(self):
        """HV of a single point = product of coordinates (with zero reference)."""
        p = np.array([0.8, 0.7, 0.6, 0.5])
        expected = 0.8 * 0.7 * 0.6 * 0.5
        assert compute_hypervolume([p]) == pytest.approx(expected, rel=1e-6)

    def test_dominated_point_does_not_increase_hv(self):
        """Adding a dominated point to the front must not increase HV."""
        strong = np.array([0.9, 0.8, 0.7, 0.6])
        weak   = np.array([0.5, 0.5, 0.5, 0.5])
        hv_strong = compute_hypervolume([strong])
        hv_both   = compute_hypervolume([strong, weak])
        assert hv_both == pytest.approx(hv_strong, rel=1e-6)

    def test_non_dominated_point_increases_hv(self):
        """Adding a genuinely non-dominated point must increase HV."""
        p1 = np.array([0.9, 0.2, 0.5, 0.5])
        p2 = np.array([0.2, 0.9, 0.5, 0.5])
        hv1 = compute_hypervolume([p1])
        hv2 = compute_hypervolume([p1, p2])
        assert hv2 > hv1

    def test_custom_reference_point(self):
        """HV with a non-zero reference should be smaller than with zero ref."""
        front = [np.array([0.8, 0.7, 0.6, 0.5])]
        hv_zero = compute_hypervolume(front, np.zeros(4))
        hv_ref  = compute_hypervolume(front, np.array([0.1, 0.1, 0.1, 0.1]))
        assert hv_zero > hv_ref

    def test_two_non_dominated_points_4d(self):
        """Manual inclusion-exclusion check for 2 points."""
        p1 = np.array([0.8, 0.3, 0.6, 0.5])
        p2 = np.array([0.3, 0.8, 0.6, 0.5])
        ref = np.zeros(4)
        # vol(box1) + vol(box2) - vol(intersection)
        box1 = 0.8 * 0.3 * 0.6 * 0.5
        box2 = 0.3 * 0.8 * 0.6 * 0.5
        inter = min(0.8, 0.3) * min(0.3, 0.8) * 0.6 * 0.5  # = 0.3*0.3*0.6*0.5
        expected = box1 + box2 - inter
        assert compute_hypervolume([p1, p2], ref) == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# Game Theory
# ===========================================================================


class TestWardUtility:
    def test_broad_spectrum_at_zero_externality(self):
        """With no others (σ_{-i} = 0), utility of σ=1 vs σ=0."""
        u1 = ward_utility(1.0, np.array([0.0]), _TOC_PARAMS)
        u0 = ward_utility(0.0, np.array([0.0]), _TOC_PARAMS)
        # With alpha=0.13 < 0.25: σ=1 gives higher utility
        assert u1 > u0

    def test_utility_decreases_with_others_broad(self):
        """Higher others' σ increases cost, lowering ward i's utility at σ=1."""
        u_low  = ward_utility(1.0, np.zeros(4),  _TOC_PARAMS)
        u_high = ward_utility(1.0, np.ones(4),   _TOC_PARAMS)
        assert u_low > u_high

    def test_sigma_zero_is_baseline(self):
        """At σ=0, utility = f_min regardless of others."""
        u = ward_utility(0.0, np.random.rand(4), _TOC_PARAMS)
        assert u == pytest.approx(_TOC_PARAMS["f_min"])

    def test_returns_float(self):
        assert isinstance(ward_utility(0.5, np.array([0.5]), _TOC_PARAMS), float)


class TestNashEquilibrium:
    _params_list = [_TOC_PARAMS] * 5   # 5 identical wards

    def test_returns_correct_shape(self):
        ne = nash_equilibrium(5, self._params_list)
        assert ne.shape == (5,)

    def test_values_in_unit_interval(self):
        ne = nash_equilibrium(5, self._params_list)
        assert np.all(ne >= 0.0) and np.all(ne <= 1.0)

    def test_nash_broad_spectrum_dominant(self):
        """With TOC params (α=0.13 < f_max-f_min=0.25), Nash ≈ all σ=1."""
        ne = nash_equilibrium(5, self._params_list)
        assert np.all(ne > 0.8), f"Expected Nash ≈ 1 for all wards, got {ne}"

    def test_single_ward(self):
        """Single ward: best-response is σ=1 with TOC params."""
        ne = nash_equilibrium(1, _TOC_PARAMS)
        assert ne[0] > 0.8


class TestSocialOptimum:
    _params_list = [_TOC_PARAMS] * 5

    def test_returns_correct_shape(self):
        opt = social_optimum(5, self._params_list)
        assert opt.shape == (5,)

    def test_values_in_unit_interval(self):
        opt = social_optimum(5, self._params_list)
        assert np.all(opt >= 0.0) and np.all(opt <= 1.0)

    def test_social_opt_below_nash(self):
        """Social optimum should prescribe lower σ than Nash (stewardship)."""
        ne  = nash_equilibrium(5, self._params_list)
        opt = social_optimum(5, self._params_list)
        assert np.mean(opt) < np.mean(ne), (
            f"Social opt mean={np.mean(opt):.3f} should be < Nash mean={np.mean(ne):.3f}"
        )

    def test_social_welfare_at_opt_geq_nash(self):
        """By definition, social optimum ≥ Nash in total welfare."""
        ne  = nash_equilibrium(5, self._params_list)
        opt = social_optimum(5, self._params_list)

        def total_welfare(sigma):
            return sum(
                ward_utility(sigma[i], np.delete(sigma, i), _TOC_PARAMS)
                for i in range(5)
            )
        assert total_welfare(opt) >= total_welfare(ne) - 1e-4


class TestPriceOfAnarchy:
    _params_list = [_TOC_PARAMS] * 5

    def test_poa_greater_than_one(self):
        """Tragedy of commons: social optimum > Nash → PoA > 1."""
        ne  = nash_equilibrium(5, self._params_list)
        opt = social_optimum(5, self._params_list)
        poa = price_of_anarchy(ne, opt, 5, self._params_list)
        assert poa > 1.0, f"Expected PoA > 1.0, got {poa:.4f}"

    def test_poa_equals_one_when_strategies_equal(self):
        """If ne == opt, PoA = 1."""
        sigma = np.full(5, 0.5)
        poa = price_of_anarchy(sigma, sigma, 5, self._params_list)
        assert poa == pytest.approx(1.0, rel=1e-6)

    def test_returns_float(self):
        ne  = nash_equilibrium(3, _TOC_PARAMS)
        opt = social_optimum(3, _TOC_PARAMS)
        assert isinstance(price_of_anarchy(ne, opt, 3, _TOC_PARAMS), float)


class TestGetGameReward:
    def test_full_improvement_gives_one(self):
        """PoA dropping from poa_before to 1.0 should give reward = 1."""
        assert get_game_reward(2.0, 1.0) == pytest.approx(1.0)

    def test_no_improvement_gives_zero(self):
        assert get_game_reward(2.0, 2.0) == pytest.approx(0.0)

    def test_partial_improvement(self):
        """PoA drops by half the gap → reward = 0.5."""
        assert get_game_reward(3.0, 2.0) == pytest.approx(0.5, rel=1e-6)

    def test_reward_in_unit_interval(self):
        for pb, pa in [(1.5, 1.2), (2.0, 1.0), (1.0, 1.0), (3.0, 0.5)]:
            r = get_game_reward(pb, pa)
            assert 0.0 <= r <= 1.0

    def test_returns_float(self):
        assert isinstance(get_game_reward(2.0, 1.5), float)


# ---------------------------------------------------------------------------
# Mandatory acceptance tests — Pareto and Game Theory
# ---------------------------------------------------------------------------


def test_pareto_reward():
    """
    Acceptance criteria (pareto_reward):
    1. Hypervolume increases when a non-dominated point is added.
    2. Adaptive weights: ecology > clinical at WRPI=1.
    """
    # 1. Hypervolume increases
    p1 = np.array([0.8, 0.3, 0.5, 0.5])
    p2 = np.array([0.3, 0.8, 0.5, 0.5])
    front = [p1]
    hv_before = compute_hypervolume(front)
    front = update_pareto_front(front, p2)
    hv_after = compute_hypervolume(front)
    assert hv_after > hv_before, (
        f"Adding non-dominated point should increase HV: {hv_before:.4f} → {hv_after:.4f}"
    )

    # 2. Adaptive weights ecology > clinical at WRPI=1
    w = compute_adaptive_weights(1.0)
    assert w[1] > w[0], (
        f"At WRPI=1, ecology weight ({w[1]:.3f}) should exceed clinical ({w[0]:.3f})"
    )


def test_game_theory():
    """
    Acceptance criteria (game_theory):
    1. Nash: all σ_i close to 1.0 (broad-spectrum dominant with TOC params).
    2. Social optimum: mean σ < Nash mean σ.
    3. PoA > 1.0.
    """
    n = 5
    params = [_TOC_PARAMS] * n

    ne  = nash_equilibrium(n, params)
    opt = social_optimum(n, params)
    poa = price_of_anarchy(ne, opt, n, params)

    # 1. Nash ≈ 1
    assert np.all(ne > 0.8), f"Nash should be ≈ 1 for all wards, got {ne}"

    # 2. Social opt < Nash
    assert np.mean(opt) < np.mean(ne), (
        f"Social opt mean ({np.mean(opt):.3f}) should be < Nash mean ({np.mean(ne):.3f})"
    )

    # 3. PoA > 1
    assert poa > 1.0, f"PoA should be > 1.0 (tragedy of commons), got {poa:.4f}"
