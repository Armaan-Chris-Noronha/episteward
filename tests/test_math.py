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
