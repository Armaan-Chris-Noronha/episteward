"""Tests for task reset/step and observation validity."""

import pytest
from episteward.models import EpiAction
from episteward.tasks import PrescriptionTriage, ResistanceContainment, NetworkOutbreakResponse
from episteward.graders import TriageGrader, ContainmentGrader, OutbreakGrader


_BASIC_ACTION = EpiAction(
    antibiotic="meropenem",
    dose_mg=1000.0,
    frequency_hours=8.0,
    duration_days=7,
    route="IV",
    isolation_order=False,
    culture_requested=True,
)

_OPTIMAL_ACTION = EpiAction(
    antibiotic="nitrofurantoin",
    dose_mg=100.0,
    frequency_hours=6.0,
    duration_days=5,
    route="PO",
)

_WORST_ACTION = EpiAction(
    antibiotic="meropenem",
    dose_mg=1000.0,
    frequency_hours=8.0,
    duration_days=7,
    route="IV",
)


def test_triage_reset_returns_observation():
    task = PrescriptionTriage()
    obs = task.reset(seed=0)
    assert obs.step_number == 1
    assert obs.patient_id != ""


def test_triage_step_advances():
    task = PrescriptionTriage()
    task.reset(seed=0)
    obs, done = task.step(_BASIC_ACTION)
    assert obs.step_number == 2


def test_triage_terminates_at_max_steps():
    task = PrescriptionTriage()
    task.reset(seed=0)
    done = False
    for _ in range(10):
        _, done = task.step(_BASIC_ACTION)
        if done:
            break
    assert done


def test_containment_reset():
    task = ResistanceContainment()
    obs = task.reset(seed=1)
    assert obs.ward_id == "MedWard_A"


def test_outbreak_reset():
    task = NetworkOutbreakResponse()
    obs = task.reset(seed=2)
    assert obs.network_alert is not None


def test_ground_truth_not_in_observation():
    task = PrescriptionTriage()
    obs = task.reset(seed=0)
    # Ground truth fields must not bleed into observation
    obs_dict = obs.model_dump()
    assert "correct_drug_class" not in obs_dict
    assert "optimal_dose_mg" not in obs_dict


# ---------------------------------------------------------------------------
# Mandatory acceptance test — task1
# ---------------------------------------------------------------------------

def test_task1():
    """
    Acceptance criteria:
      1. Optimal action (nitrofurantoin 100mg q6h PO 5d) → score ≥ 0.85
      2. Worst action (meropenem) → score ≤ 0.25
      3. reward ∈ [0, 1] on every step
      4. Deterministic given seed
    """
    grader = TriageGrader()

    # --- 1. Optimal action scores ≥ 0.85 ---
    task = PrescriptionTriage()
    task.reset(seed=0)
    gt = task.ground_truth
    result_opt = grader.grade(_OPTIMAL_ACTION, task.state, gt, step_number=1)
    assert result_opt["reward"] >= 0.85, (
        f"Optimal nitrofurantoin should score ≥0.85, got {result_opt['reward']:.3f}\n"
        f"components={result_opt['components']}"
    )

    # --- 2. Worst action scores ≤ 0.25 ---
    task2 = PrescriptionTriage()
    task2.reset(seed=0)
    gt2 = task2.ground_truth
    result_worst = grader.grade(_WORST_ACTION, task2.state, gt2, step_number=1)
    assert result_worst["reward"] <= 0.25, (
        f"Worst meropenem should score ≤0.25, got {result_worst['reward']:.3f}\n"
        f"components={result_worst['components']}"
    )

    # --- 3. reward ∈ [0, 1] on every step ---
    task3 = PrescriptionTriage()
    task3.reset(seed=0)
    gt3 = task3.ground_truth
    done = False
    step = 1
    while not done:
        r_opt = grader.grade(_OPTIMAL_ACTION, task3.state, gt3, step_number=step)
        assert 0.0 <= r_opt["reward"] <= 1.0, f"Reward out of range at step {step}"
        r_bad = grader.grade(_WORST_ACTION, task3.state, gt3, step_number=step)
        assert 0.0 <= r_bad["reward"] <= 1.0, f"Reward out of range at step {step}"
        _, done = task3.step(_OPTIMAL_ACTION)
        step += 1
        if step > 10:
            break  # safety valve

    # --- 4. Deterministic: same seed → same ground truth ---
    t_a = PrescriptionTriage()
    t_b = PrescriptionTriage()
    t_a.reset(seed=42)
    t_b.reset(seed=42)
    assert t_a.ground_truth == t_b.ground_truth

    # Different seeds should still produce [0,1] rewards
    t_c = PrescriptionTriage()
    t_c.reset(seed=7)
    r_c = grader.grade(_OPTIMAL_ACTION, t_c.state, t_c.ground_truth, step_number=1)
    assert 0.0 <= r_c["reward"] <= 1.0

    # --- 5. Episode terminates within max_steps ---
    task4 = PrescriptionTriage()
    task4.reset(seed=0)
    done = False
    for _ in range(10):
        _, done = task4.step(_OPTIMAL_ACTION)
        if done:
            break
    assert done, "Episode should have terminated by step 10"


# ---------------------------------------------------------------------------
# Mandatory acceptance test — task2
# ---------------------------------------------------------------------------

_CORRECT_ACTION = EpiAction(
    antibiotic="piperacillin-tazobactam",
    dose_mg=4500.0,
    frequency_hours=8.0,
    duration_days=7,
    route="IV",
    isolation_order=True,
    culture_requested=True,
)

_RANDOM_ACTION = EpiAction(
    antibiotic="ceftriaxone",
    dose_mg=2000.0,
    frequency_hours=24.0,
    duration_days=5,
    route="IV",
    isolation_order=False,
    culture_requested=False,
)


def test_task2():
    """
    Acceptance criteria:
      1. Correct agent (pip-tazo + isolate + culture every step) → score ≥ 0.65
      2. Random agent (ceftriaxone, no isolation, no culture) → score ≤ 0.30
      3. reward ∈ [0, 1] on every step
      4. Deterministic given seed (same seed → same ground truth)
      5. Episode terminates at step 15
    """
    grader = ContainmentGrader()
    MAX_TOTAL_REWARD = 15.0  # max 1.0 per step × 15 steps

    # --- 1. Correct agent scores ≥ 0.65 ---
    task_opt = ResistanceContainment()
    task_opt.reset(seed=0)
    rewards_opt = []
    prev_cases = 0
    done = False
    step = 1
    while not done:
        _, done = task_opt.step(_CORRECT_ACTION)
        gt = task_opt.ground_truth
        result = grader.grade(_CORRECT_ACTION, task_opt.state, gt, step, prev_cases)
        assert 0.0 <= result["reward"] <= 1.0, f"Reward OOB at step {step}: {result['reward']}"
        rewards_opt.append(result["reward"])
        prev_cases = gt["new_cases_total"]
        step += 1
        if step > 20:
            break  # safety valve
    score_opt = sum(rewards_opt) / MAX_TOTAL_REWARD
    assert score_opt >= 0.65, (
        f"Correct agent should score ≥0.65, got {score_opt:.3f}\n"
        f"step_rewards={[round(r, 3) for r in rewards_opt]}"
    )

    # --- 2. Random agent scores ≤ 0.30 ---
    task_rand = ResistanceContainment()
    task_rand.reset(seed=0)
    rewards_rand = []
    prev_cases = 0
    done = False
    step = 1
    while not done:
        _, done = task_rand.step(_RANDOM_ACTION)
        gt = task_rand.ground_truth
        result = grader.grade(_RANDOM_ACTION, task_rand.state, gt, step, prev_cases)
        assert 0.0 <= result["reward"] <= 1.0, f"Reward OOB at step {step}: {result['reward']}"
        rewards_rand.append(result["reward"])
        prev_cases = gt["new_cases_total"]
        step += 1
        if step > 20:
            break
    score_rand = sum(rewards_rand) / MAX_TOTAL_REWARD
    assert score_rand <= 0.30, (
        f"Random agent should score ≤0.30, got {score_rand:.3f}\n"
        f"step_rewards={[round(r, 3) for r in rewards_rand]}"
    )

    # --- 3. Deterministic: same seed → same ground truth ---
    t_a = ResistanceContainment()
    t_b = ResistanceContainment()
    t_a.reset(seed=42)
    t_b.reset(seed=42)
    assert t_a.ground_truth["index_patient_id"] == t_b.ground_truth["index_patient_id"]

    # --- 4. Episode terminates at max_steps (15) ---
    task_term = ResistanceContainment()
    task_term.reset(seed=0)
    done = False
    for _ in range(20):
        _, done = task_term.step(_CORRECT_ACTION)
        if done:
            break
    assert done, "Task 2 episode should terminate within 20 steps"
    assert task_term.state.step_number > 15, "Should exceed 15 steps before done"


# ---------------------------------------------------------------------------
# Mandatory acceptance test — task3
# ---------------------------------------------------------------------------

_COLISTIN_ACTION = EpiAction(
    antibiotic="colistin",
    dose_mg=150.0,
    frequency_hours=12.0,
    duration_days=7,
    route="IV",
    isolation_order=True,
    culture_requested=True,
)

_NO_COLISTIN_ACTION = EpiAction(
    antibiotic="meropenem",
    dose_mg=1000.0,
    frequency_hours=8.0,
    duration_days=7,
    route="IV",
    isolation_order=False,
    culture_requested=False,
)


def test_task3():
    """
    Acceptance criteria:
      1. Episode runs exactly 30 steps
      2. Every per-step reward is in [0, 1]
      3. Colistin overspend at unit 11 triggers a negative penalty component
      4. Deterministic: same seed → same ground truth
      5. network_alert is populated from the first observation
    """
    grader = OutbreakGrader()
    MAX_STEPS = 30

    # --- 1 & 2: Full 30 steps, all rewards in [0, 1] ---
    task = NetworkOutbreakResponse()
    task.reset(seed=0)
    rewards = []
    done = False
    step = 1
    while not done:
        _, done = task.step(_COLISTIN_ACTION)
        gt = task.ground_truth
        result = grader.grade(_COLISTIN_ACTION, task.state, gt, step_number=step)
        assert 0.0 <= result["reward"] <= 1.0, (
            f"Reward out of [0,1] at step {step}: {result['reward']:.4f}\n"
            f"components={result['components']}"
        )
        assert "components" in result and result["components"], "grader must expose components dict"
        rewards.append(result["reward"])
        step += 1
        if step > MAX_STEPS + 5:
            break  # safety valve
    assert done, "Episode should terminate"
    assert len(rewards) == MAX_STEPS, f"Expected {MAX_STEPS} steps, got {len(rewards)}"

    # --- 3. Colistin overspend at unit 11 triggers penalty ---
    task_over = NetworkOutbreakResponse()
    task_over.reset(seed=0)
    # Force 11 colistin uses (budget = 10 → unit 11 is overspend)
    for i in range(11):
        _, _ = task_over.step(_COLISTIN_ACTION)
    gt_over = task_over.ground_truth
    assert gt_over["colistin_overspend"] >= 1, (
        f"Expected colistin_overspend ≥ 1 after 11 uses, got {gt_over['colistin_overspend']}"
    )
    result_over = grader.grade(
        _COLISTIN_ACTION, task_over.state, gt_over, step_number=11
    )
    colistin_pen = result_over["components"]["colistin_penalty"]
    assert colistin_pen < 0.0, (
        f"Expected negative colistin_penalty at unit 11, got {colistin_pen}"
    )

    # --- 4. Deterministic: same seed → same initial ground truth ---
    t_a = NetworkOutbreakResponse()
    t_b = NetworkOutbreakResponse()
    t_a.reset(seed=42)
    t_b.reset(seed=42)
    assert t_a.ground_truth["source_hospitals"] == t_b.ground_truth["source_hospitals"]
    assert t_a.ground_truth["initial_crk_count"] == t_b.ground_truth["initial_crk_count"]

    # --- 5. network_alert present from first observation ---
    task_obs = NetworkOutbreakResponse()
    obs = task_obs.reset(seed=0)
    assert obs.network_alert is not None and len(obs.network_alert) > 0, (
        "network_alert must be populated with outbreak context"
    )
    assert "CRK" in obs.network_alert, "network_alert should mention CRK"
    assert "colistin" in obs.network_alert.lower(), "network_alert should mention colistin budget"

    # --- 6. Initial state: 2 infected hospitals, correct patient counts ---
    task_init = NetworkOutbreakResponse()
    task_init.reset(seed=0)
    gt_init = task_init.ground_truth
    assert gt_init["initial_crk_count"] == 6, (
        f"Expected 6 initial CRK patients (2 hospitals × 3), got {gt_init['initial_crk_count']}"
    )
    assert gt_init["total_crk_patients"] == 6, (
        f"Expected 6 CRK patients at reset, got {gt_init['total_crk_patients']}"
    )
    assert gt_init["colistin_budget"] == 10
    assert gt_init["colistin_overspend"] == 0
