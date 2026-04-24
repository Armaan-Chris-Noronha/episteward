# CLAUDE.md — EpiSteward R2
# AI Antibiotic Stewardship Environment — Grand Finale Build

## Project identity

- Name: EpiSteward
- Tagline: Self-Improving Multi-Agent World Model for Antimicrobial Resistance
- Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler — Grand Finale, Bangalore, 25–26 April 2026
- Themes covered: Theme 3.1 (World Modeling), Theme 1 (Multi-Agent), Theme 4 (Self-Improvement)
- Bonus prizes targeted: Scaler AI Labs (multi-app enterprise), Fleet AI (oversight agents), Snorkel AI (experts-in-the-loop)
- Stack: Python 3.11, FastAPI, OpenEnv, Pydantic v2, scipy, numpy, networkx, Docker, HuggingFace Spaces
- Training: Unsloth + HF TRL GRPO, Qwen2.5-3B-Instruct, 4-bit QLoRA

---

## R1 baseline — DO NOT break any of these

These files are complete and tested. All R2 work is additive only.

```
episteward/
  __init__.py          — exports EpiStewardEnv, EpiAction, EpiObservation, EpiReward, StepResult
  env.py               — EpiStewardEnv with in_process() + from_docker_image() factory modes
  models.py            — EpiObservation, EpiAction, EpiReward, StepResult, StateResult, ResetRequest
  graders.py           — TriageGrader, ContainmentGrader, OutbreakGrader
  state.py             — HospitalState with .clone() and .to_dict()
  tasks/               — TASK_REGISTRY, BaseTask, task1_triage, task2_containment, task3_outbreak
inference.py           — async LLM agent, exact [START]/[STEP]/[END] log format
openenv.yaml           — 3 tasks registered
Dockerfile             — FROM python:3.11-slim, EXPOSE 7860, uvicorn on 0.0.0.0:7860
pyproject.toml         — episteward package, deps: fastapi, uvicorn, pydantic>=2, scipy, numpy, networkx
requirements.txt       — same deps as pyproject.toml
```

### R1 invariants — never violate these

- `EpiAction.frequency_hours` must be in `{4.0, 6.0, 8.0, 12.0, 24.0}`
- `EpiAction.route` must be in `{"IV", "PO", "IM"}`
- `EpiReward.value` is hard-clamped to `[0.0, 1.0]` by model_validator
- All rewards returned by graders must be in `[0.0, 1.0]` — checked in `env.py step()`
- `[START]/[STEP]/[END]` log format in `inference.py` must not change
- `/health` endpoint must return 200 for Docker mode
- `in_process()` mode must work without Docker or network
- `TASK_REGISTRY` must contain all 4 task IDs after R2 is complete
- R1 grader score thresholds must still hold: meropenem triage >= 0.80, random baseline ~0.10

### R1 existing models (do not rename or remove fields)

```python
# EpiObservation existing fields:
patient_id, ward_id, infection_site, symptoms, vitals, culture_results,
resistance_flags, transfer_history, antibiotic_history, network_alert, step_number

# EpiAction existing fields:
antibiotic, dose_mg, frequency_hours, duration_days, route,
isolation_order, culture_requested, specialist_consult, reasoning

# EpiReward existing components: pkpd, stewardship, resistance, coverage
```

### R1 antibiotics list (valid values for EpiAction.antibiotic)

colistin, meropenem, ertapenem, piperacillin-tazobactam, ceftriaxone, cefazolin,
ampicillin, vancomycin, linezolid, azithromycin, ciprofloxacin, nitrofurantoin,
trimethoprim-sulfamethoxazole

---

## R2 new file structure

```
episteward/
  math/
    __init__.py
    population_pk.py       — two-compartment PK, IIV, Bayesian MAP (Ticket 1)
    msw.py                 — Mutant Selection Window (Ticket 2)
    resistance_sde.py      — Itô SDE Euler-Maruyama (Ticket 3)
    hgt.py                 — Horizontal Gene Transfer ODE (Ticket 4)
    bayesian_diagnostics.py — pathogen posterior, VoI (Ticket 5)
    pareto_reward.py       — 4-objective Pareto, adaptive weights, hypervolume (Ticket 6)
    game_theory.py         — N-ward game, Nash equilibrium, Price of Anarchy (Ticket 6)
  oversight/
    __init__.py
    oversight_agent.py     — Fleet AI safety monitor (Ticket 10)
  curriculum/
    __init__.py
    generator.py           — adaptive curriculum from failure modes (Ticket 11)
  experts/
    __init__.py
    id_specialist.py       — simulated ID specialist, evolving stance (Ticket 12)
  apps/
    __init__.py
    enterprise_apps.py     — EHR, Lab, Pharmacy, Microbiology apps (Ticket 13)
    app_router.py          — routes EpiAction to correct enterprise app (Ticket 13)
  tasks/
    task4_multiagent.py    — 5 ward agents + coordinator (Ticket 9)
  graders/
    task4_grader.py        — MultiAgentGrader using game_theory (Ticket 9)
notebooks/
  train_grpo.ipynb         — Colab training script, Unsloth + HF TRL GRPO (Ticket 15)
scripts/
  demo_assets.py           — generates 5 pitch demo PNGs (Ticket 15)
assets/                    — generated PNGs go here
```

---

## R2 model upgrades (surgical additions to models.py)

All new fields are Optional with defaults — existing code that creates
EpiObservation/EpiAction without them must continue to work.

### EpiObservation — add after step_number

```python
pathogen_posterior: Dict[str, float] = Field(default_factory=dict)
# {"E_coli": 0.62, "K_pneumoniae": 0.28, ...} sums to 1.0

msw_zone: Optional[str] = None
# "sub_mic" | "msw" | "mpc_plus" | None

ward_resistance_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
# normalised Ward Resistance Pressure Index [0,1]

app_context: Optional[str] = None
# "ehr" | "lab" | "pharmacy" | "microbiology" | None
```

### EpiAction — add after specialist_consult

```python
diagnostic_test: Optional[str] = Field(default=None)
# None | "rapid_pcr" | "standard_culture" | "sensitivity_panel"

target_app: Optional[str] = Field(default=None)
# None | "ehr" | "lab" | "pharmacy" | "microbiology"
```

### Validators to add

```python
@field_validator("diagnostic_test")
def validate_diagnostic_test(cls, v):
    valid = {None, "rapid_pcr", "standard_culture", "sensitivity_panel"}
    if v not in valid:
        raise ValueError(f"diagnostic_test must be one of {valid}")
    return v

@field_validator("target_app")
def validate_target_app(cls, v):
    valid = {None, "ehr", "lab", "pharmacy", "microbiology"}
    if v not in valid:
        raise ValueError(f"target_app must be one of {valid}")
    return v
```

### EpiReward new component keys (added by R2 graders, logged in components dict)

```
"msw_risk"              — float [0,1], penalty for time spent in MSW
"hgt_pressure"          — float [0,1], penalty for elevated plasmid transfer
"bayesian_entropy"      — float [0,1], reward for reducing pathogen uncertainty
"ecological_footprint"  — float [0,1], penalty for excess antibiotic-days
"oversight_flags"       — int, count of safety flags fired this step
"specialist_alignment"  — float [0,1], alignment with ID specialist current stance
"app_routing_score"     — float [0,1], bonus for correct enterprise app routing
"pareto_vector"         — List[float] len 4, logged only, not averaged into scalar
```

---

## Math module public APIs

### episteward/math/population_pk.py

```python
@dataclass
class PKParams:
    V1: float   # central volume (L)
    V2: float   # peripheral volume (L)
    CL: float   # clearance (L/h)
    Q: float    # inter-compartmental clearance (L/h)
    F: float    # bioavailability (0-1)
    ka: float   # absorption rate (1/h) — set 0 for IV

@dataclass
class PatientCovariates:
    age: float           # years
    weight_kg: float
    creatinine_mg_dl: float
    is_icu: bool

def sample_individual_pk(drug_name: str, patient: PatientCovariates,
                          rng: np.random.Generator) -> PKParams
    # theta_i = theta_pop * exp(eta_i), eta_i ~ N(0, omega^2)

def solve_two_compartment(pk: PKParams, dose_mg: float, duration_h: float,
                           dt: float = 0.5) -> np.ndarray
    # ODE: dC1/dt = (F*D*ka*exp(-ka*t))/V1 - (k12+k10)*C1 + k21*(V2/V1)*C2
    # Use scipy.integrate.solve_ivp method='RK45'

def bayesian_tdm_update(pk: PKParams, observed_levels: List[Tuple[float,float]],
                         drug_name: str) -> PKParams
    # MAP: minimise OFV = sum((C_obs-C_model)^2/sigma^2) + sum((ln(theta_i)-ln(theta_pop))^2/omega^2)
    # Use scipy.optimize.minimize method='L-BFGS-B'

def get_pkpd_score(pk: PKParams, dose_mg: float, frequency_h: float,
                   duration_days: int, mic: float, drug_name: str) -> float
    # Returns [0,1]. AUC/MIC for time-dependent, Cmax/MIC for concentration-dependent.
```

Hardcoded population params (theta_pop, omega, sigma) for: meropenem, vancomycin, ceftriaxone.

Done when: meropenem 1000mg q8h vs MIC=2 → score >= 0.80 | ampicillin vs MIC=32 → score <= 0.10

---

### episteward/math/msw.py

```python
def compute_mpc(drug_name: str, pathogen: str, mic: float) -> float
    # MPC = MIC * (1/mutation_freq)^(1/hill_coeff)
    # mutation_freq=1e-8, hill_coeff=1.5

def classify_zone(C: float, mic: float, mpc: float) -> str
    # "sub_mic" | "msw" | "mpc_plus"

def compute_msw_risk(C_series: np.ndarray, drug_name: str, pathogen: str,
                      mic: float, dt: float = 0.5) -> float
    # T_MSW/T_total — fraction of time in [MIC, MPC]. Returns [0,1].

def get_msw_reward_component(C_series: np.ndarray, drug_name: str,
                              pathogen: str, mic: float) -> float
    # 1 - msw_risk, +0.2 bonus if C(t) > MPC for >80% of exposure. Returns [0,1].
```

Done when: MPC > MIC confirmed | C above MPC → risk < 0.2 | C in MSW → risk > 0.6

---

### episteward/math/resistance_sde.py

```python
def selection_coefficient(C: float, mic: float, mpc: float,
                            s_max: float = 0.3, n: float = 2.0) -> float
    # s(C) = s_max*C^n/(EC50^n+C^n)  when MIC <= C <= MPC
    # s(C) = 0                         when C > MPC
    # s(C) = -fitness_cost (0.05)      when C < MIC

def euler_maruyama_step(p_R: float, s: float, sigma: float,
                         dt: float, rng: np.random.Generator) -> float
    # dp_R = s*p_R*(1-p_R)*dt + sigma*sqrt(p_R*(1-p_R))*sqrt(dt)*xi
    # xi ~ N(0,1). Clamp to [0,1].
    # sigma = sqrt(1/(2*N_eff)), N_eff=1e8

def simulate_resistance_trajectory(p_init: float, C_series: np.ndarray,
                                    drug_name: str, pathogen: str,
                                    dt: float, rng: np.random.Generator) -> List[float]

def resistance_emerged(trajectory: List[float], threshold: float = 0.5,
                        sustained_steps: int = 48) -> bool
```

Done when: seed reproducibility confirmed | 72h in MSW → resistance increases | C > MPC → stays low

---

### episteward/math/hgt.py

```python
def hgt_rate(C: float, mic: float, gamma_baseline: float = 0.001,
              alpha_sos: float = 2.0) -> float
    # gamma(C) = gamma_baseline * (1 + alpha_sos * C/MIC)
    # SOS response: antibiotic stress INCREASES plasmid transfer

def solve_hgt_ode(S0: float, R0: float, D0: float,
                   C_series: np.ndarray, drug_name: str, pathogen: str,
                   dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    # dS/dt = r_S*S*(1-N/K) - k_kill(C)*S - gamma(C)*D*S + delta*R
    # dR/dt = r_R*R*(1-N/K) - k_kill(C)*R/f + gamma(C)*D*S - delta*R
    # dD/dt = r_D*D*(1-N/K) - k_kill(C)*D/f + mu*S
    # Params: r_S=0.5, r_R=0.45, r_D=0.45, K=1e9, f=2.0, delta=0.01, mu=1e-8
    # k_kill(C) = E_max*C^n/(EC50^n+C^n), E_max=0.8, EC50=MIC, n=1.5
    # Use scipy.integrate.solve_ivp

def cross_species_transfer(D_donor: float, S_recipient: float,
                             contact_rate: float, gamma_cross: float = 0.001) -> float

def get_hgt_pressure_score(R_final: float, R_initial: float,
                             cross_transfers: float) -> float  # [0,1]
```

Done when: hgt_rate(C=MIC*2) > hgt_rate(C=0) | D0=0 → R near 0 | C >> MPC → R decreases

---

### episteward/math/bayesian_diagnostics.py

```python
PATHOGENS = ["E_coli", "K_pneumoniae", "S_aureus", "P_aeruginosa", "E_faecalis"]

def init_prior(infection_site: str, risk_factors: List[str],
                ward_antibiogram: Dict[str, float]) -> Dict[str, float]
    # P(pathogen=k | site, risk_factors). Normalise to sum=1.

def update_posterior(prior: Dict[str, float], observation_type: str,
                      result: str) -> Dict[str, float]
    # obs_type: "gram_stain" | "culture" | "sensitivity_panel"
    # Bayesian update: posterior(k) ∝ L(result|k) * prior(k). Normalise.
    # Hardcoded likelihoods:
    # gram_negative: E_coli=0.95, K_pneumoniae=0.95, S_aureus=0.02, P_aeruginosa=0.95, E_faecalis=0.02
    # gram_positive: reverse

def shannon_entropy(posterior: Dict[str, float]) -> float
    # H(pi) = -sum(pi(k) * log2(pi(k))). In bits.

def value_of_information(posterior: Dict[str, float],
                          test_type: str, test_cost: float) -> float
    # VoI = H(prior) - E[H(posterior | test_result)]

def get_diagnostic_reward(prior_entropy: float, posterior_entropy: float,
                           test_ordered: bool, test_cost: float) -> float  # [0,1]
```

Done when: gram_negative → E_coli increases | full culture → entropy near 0 | posterior always sums to 1

---

### episteward/math/pareto_reward.py

```python
def compute_reward_vector(clinical: float, ecology: float,
                           economics: float, stewardship: float) -> np.ndarray
    # shape (4,), all components in [0,1]

def compute_wrpi(ward_resistance_prevalences: Dict[str, float],
                  severity_weights: Dict[str, float]) -> float
    # WRPI = sum(prev_R(k) * severity_weight(k)), normalised to [0,1]

def compute_adaptive_weights(wrpi: float,
                              base_weights: np.ndarray = np.array([0.4, 0.3, 0.15, 0.15])
                              ) -> np.ndarray
    # lambda_1 *= (1 - 0.3*WRPI)  — clinical drops in AMR crisis
    # lambda_2 *= (1 + 0.6*WRPI)  — ecology rises with WRPI
    # Apply softmax. Returns shape (4,) summing to 1.

def scalarize(reward_vector: np.ndarray, weights: np.ndarray) -> float  # [0,1]

def is_dominated(point: np.ndarray, front: List[np.ndarray]) -> bool

def update_pareto_front(front: List[np.ndarray],
                         new_point: np.ndarray) -> List[np.ndarray]

def compute_hypervolume(front: List[np.ndarray],
                         reference: np.ndarray = np.zeros(4)) -> float
    # Iterative inclusion-exclusion for 4D. Fronts will be < 50 points.
```

---

### episteward/math/game_theory.py

```python
def ward_utility(sigma_i: float, sigma_others: np.ndarray,
                  ward_params: Dict) -> float
    # U_i = (f_min + (f_max-f_min)*sigma_i) - alpha_i*sigma_i*(1 + beta*mean(sigma_others))
    # f_min=0.7, f_max=0.95, alpha_i=0.3, beta=0.8

def nash_equilibrium(n_wards: int, ward_params: List[Dict],
                      max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray
    # Best-response iteration. Returns sigma* shape (n_wards,).

def social_optimum(n_wards: int, ward_params: List[Dict]) -> np.ndarray
    # argmax sum_i U_i. Use scipy.optimize.minimize L-BFGS-B.

def price_of_anarchy(ne_strategies: np.ndarray, opt_strategies: np.ndarray,
                      n_wards: int, ward_params: List[Dict]) -> float
    # PoA = sum_i U_i(sigma_hat) / sum_i U_i(sigma*)
    # PoA > 1.0 = tragedy of commons confirmed

def get_game_reward(poa_before: float, poa_after: float) -> float  # [0,1]
    # clip((poa_before - poa_after) / (poa_before - 1.0 + 1e-8), 0, 1)
```

Done when: Nash ~ sigma=1 for all wards | social opt < Nash | PoA > 1.0

---

## New system components

### episteward/oversight/oversight_agent.py

```python
@dataclass
class OversightReport:
    step: int
    flags: List[str]
    explanations: List[str]
    severity: str        # "safe" | "warning" | "critical"
    recommended_action: Optional[str]

class OversightAgent:
    def observe(self, step_log: List[EpiAction], ward_states: List[dict]) -> OversightReport
    def flag_unsafe(self, action: EpiAction, context: dict) -> List[str]
    def explain(self, flags: List[str], action: EpiAction) -> str
    def oversight_reward(self, flags_before: List[str], flags_after: List[str]) -> float
```

Safety flags:
- `CARBAPENEM_WITHOUT_JUSTIFICATION` — meropenem/colistin with no resistance_flags
- `MISSED_DEESCALATION` — culture shows narrow sufficiency but broad prescribed
- `DURATION_EXCEEDS_GUIDELINE` — duration_days > evidence-based max for infection_site
- `MSW_RISK_HIGH` — msw_zone == "msw" for >3 consecutive steps
- `HGT_CASCADE_RISK` — ward resistance_prevalence rising across 3+ consecutive steps

Integration in env.py: run after grading each step. If severity=="critical" → -0.15 reward penalty.
Attach report to StepResult.info["oversight"].

---

### episteward/curriculum/generator.py

```python
@dataclass
class ScenarioParams:
    pathogen_complexity: int      # 1-4 resistance mechanisms
    culture_delay_days: float     # 1.0-5.0
    initial_wrpi: float           # 0.0-0.8
    patient_charlson: int         # 0-6
    n_network_edges: int          # 5-20

class CurriculumGenerator:
    difficulty_level: float = 0.3    # rises to 1.0

    def record_episode(self, task_id, actions, rewards, final_state) -> None
    def get_failure_distribution(self) -> Dict[str, float]
    def generate_scenario(self, task_id: str, seed: int) -> ScenarioParams
    def should_increase_difficulty(self) -> bool
        # True if rolling avg reward over last 10 episodes > 0.65
```

Failure types tracked: missed_deescalation, msw_exposure, hgt_cascade,
diagnostic_underuse, carbapenem_overuse

Integration in env.py: curriculum.generate_scenario() called in reset() when curriculum enabled.
Expose stats in StepResult.info["curriculum"].

---

### episteward/experts/id_specialist.py

```python
@dataclass
class SpecialistFeedback:
    approved: bool
    comment: str
    stance: str                      # "conservative" | "moderate" | "aggressive"
    suggested_alternative: Optional[str]
    reward_signal: float             # [0,1]

class IDSpecialist:
    stance: str = "moderate"
    preference_weights: Dict[str, float]

    def update_stance(self, wrpi: float, recent_failures: int,
                       outbreak_active: bool) -> None
        # outbreak + wrpi>0.6 → aggressive | wrpi<0.2 + no failures → conservative

    def evaluate_action(self, action: EpiAction, context: dict) -> SpecialistFeedback
        # Feedback CHANGES with stance — same action, opposite evaluation
        # This is what the agent must learn to adapt to

    def get_feedback_reward(self, feedback: SpecialistFeedback) -> float
```

Integration: 0.1 * specialist_feedback.reward_signal added to total reward.
Attach to StepResult.info["specialist_feedback"].

---

### episteward/apps/enterprise_apps.py + app_router.py

```python
@dataclass
class AppResponse:
    routed: bool
    processed: bool
    reward_delta: float     # [-0.1, +0.1]
    message: str
    latency_steps: int      # 0 for most, 2 for standard_culture

class EHRApp:       # handles: isolation_order, specialist_consult, history queries
class LabApp:       # handles: culture_requested, diagnostic_test orders
class PharmacyApp:  # handles: antibiotic, dose, frequency, duration, route
class MicrobiologyApp:  # handles: antibiogram queries, resistance surveillance

class AppRouter:
    def route(self, action: EpiAction, state) -> AppResponse
    def _infer_app(self, action, state) -> AppResponse
        # culture_requested=True → lab
        # isolation_order=True → ehr
        # antibiotic is not None → pharmacy
        # no clear signal → microbiology
```

Business rules:
- meropenem requires pharmacy pre-authorisation flag
- colistin requires prior specialist_consult=True
- rapid_pcr only available in ICU or SIRS criteria met
- antibiogram data stale if >7 steps since last microbiology query

Integration: route() called after grading each step.
reward_delta added to total. Attach to StepResult.info["app_response"].

---

### episteward/tasks/task4_multiagent.py

20-step episode. Agent is stewardship coordinator for 5-ward hospital.

State each step: per-ward sigma_i [0,1], resistance_prevalence_i, cure_rate_i,
recent_failures_i, current PoA, WRPI, network_alert, step_number

Ward agent update each step:
sigma_new = 0.7*sigma_old + 0.2*peer_avg + 0.1*coordinator_signal

Coordinator signal decoded from EpiAction.antibiotic:
- narrow (cefalexin/amoxicillin/nitrofurantoin) → signal = 0.1
- moderate (ceftriaxone/pip-tazo) → signal = 0.5
- broad (meropenem/colistin) → signal = 0.9
- isolation_order=True → +0.1 stewardship signal all wards
- specialist_consult=True → reduces sigma by 0.1
- target_app="ehr" → observation step (no sigma change)

Grader: get_game_reward(poa_before, poa_after) + bonus/penalty
- +0.05 if all sigma_i < 0.5 (full cooperative de-escalation)
- -0.08 per ward treatment failure this step

Expected: random baseline ~0.10 | good coordinator ~0.60+

---

## env.py integration checklist for R2

In _InProcessBackend.__init__():
```python
self._oversight = OversightAgent()
self._curriculum = CurriculumGenerator()
self._specialist = IDSpecialist()
self._router = AppRouter()
self._poa_before = None
```

In _InProcessBackend.step() after base grading:
```python
1. oversight_report = self._oversight.flag_unsafe(action, context)
2. self._specialist.update_stance(wrpi, failures, outbreak)
3. specialist_feedback = self._specialist.evaluate_action(action, context)
4. app_response = self._router.route(action, state)
5. final_reward = base_reward
              + (-0.15 if oversight_report.severity == "critical" else 0)
              + 0.1 * specialist_feedback.reward_signal
              + app_response.reward_delta
6. final_reward = float(min(max(final_reward, 0.0), 1.0))
7. attach all sub-reports to StepResult.info
```

_GRADERS must include:
```python
"task4_multiagent": MultiAgentGrader()
```

---

## openenv.yaml R2 spec

```yaml
name: episteward
version: "2.0.0"
description: >
  Self-improving multi-agent world model for antimicrobial resistance.
  Covers Themes 1 (Multi-Agent + Fleet AI), 3.1 (World Modeling + Scaler enterprise),
  and 4 (Self-Improvement + Snorkel experts-in-the-loop). Math grounded in population
  PK/PD, Itô SDE resistance dynamics, Bayesian diagnostics, Pareto reward optimization,
  and game-theoretic prescribing equilibrium analysis.
tags:
  - openenv
  - healthcare
  - antimicrobial-resistance
  - epidemiology
  - multi-agent
  - fleet-ai
  - oversight
  - curriculum-learning
  - self-improving
  - enterprise
  - multi-app
  - workflow
  - experts-in-the-loop
  - adaptive-preferences
tasks:
  - id: task1_triage
    name: Prescription Triage
    difficulty: easy
    max_steps: 5
    reward_range: [0.0, 1.0]
  - id: task2_containment
    name: Resistance Containment
    difficulty: medium
    max_steps: 15
    reward_range: [0.0, 1.0]
  - id: task3_outbreak
    name: Network Outbreak Response
    difficulty: hard
    max_steps: 30
    reward_range: [0.0, 1.0]
  - id: task4_multiagent
    name: Multi-Ward Stewardship Game
    difficulty: expert
    max_steps: 20
    reward_range: [0.0, 1.0]
```

---

## inference.py updates for R2

- Add "task4_multiagent" to TASKS list
- Add MAX_STEPS["task4_multiagent"] = 20
- Update system prompt to mention:
  - target_app field: route actions to correct enterprise system
  - diagnostic_test field: order tests to reduce pathogen uncertainty
  - specialist_feedback available in StepResult.info
  - oversight flags in StepResult.info — agent should avoid triggering them
  - For task4: antibiotic field encodes stewardship signal to ward agents

---

## Build order for Claude Code sessions

Session 1:  episteward/math/__init__.py + population_pk.py + tests  (Ticket 1)
Session 2:  episteward/math/msw.py + resistance_sde.py + tests       (Tickets 2-3)
Session 3:  episteward/math/hgt.py + tests                           (Ticket 4)
Session 4:  episteward/math/bayesian_diagnostics.py + tests          (Ticket 5)
Session 5:  episteward/math/pareto_reward.py + game_theory.py + tests (Ticket 6)
Session 6:  models.py surgical upgrades + regression tests            (Ticket 7)
Session 7:  graders.py upgrade to use new math modules                (Ticket 8)
Session 8:  task4_multiagent.py + task4_grader.py + register          (Ticket 9)
Session 9:  oversight_agent.py + curriculum/generator.py              (Tickets 10-11)
Session 10: experts/id_specialist.py + apps/enterprise_apps.py        (Tickets 12-13)
Session 11: env.py final integration + openenv.yaml + inference.py    (Ticket 14)
Session 12: notebooks/train_grpo.ipynb + scripts/demo_assets.py       (Ticket 15)

Start each session with:
"We are working on EpiSteward R2. Here is my CLAUDE.md: [paste this file].
Working on Session N / Ticket N: [ticket title].
Relevant existing files: [only list files needed for this ticket]."

---

## Demo assets to generate onsite (Session 12, with compute credits)

After training:

1. demo_reward_curves.png
   4 subplots: total reward, de-escalation rate, broad-spectrum %, PoA delta
   3 lines: task1, task2, task4_multiagent
   Must show visible improvement over 500 steps

2. demo_pareto_front.png
   Scatter: r_clinical vs r_ecology at steps 0, 100, 500
   Arrow showing Pareto front expanding = policy improving on both objectives

3. demo_poa.png
   Bar chart: Price of Anarchy before training (~2.4) vs after (~1.3)
   Dashed line at PoA=1.0 labelled "Social optimum"
   Caption: "Value recovered from tragedy of the commons"

4. demo_before_after.png
   Table: untrained vs trained agent on 3 patient vignettes
   Show: antibiotic choice, route, target_app, diagnostic_test ordered

5. demo_oversight.png
   Timeline of oversight flags per episode
   CARBAPENEM_WITHOUT_JUSTIFICATION drops ~70% after 200 steps

---

## Pitch structure (3 minutes)

0:00-0:25  Hook — AMR kills 700k/year, projected 10M by 2050
0:25-0:55  Patient vignette — the hidden cost of the "safe" choice
0:55-1:30  Environment — partial observability, 7-day episode, dual objective
1:30-2:15  Demo — untrained vs trained, reward curves, PoA improvement
2:15-3:00  Vision — "We're training an agent that understands every action has an ecology"

Q&A prep:
- "Different from CDSS?" → CDSS retrieves guidelines. EpiSteward learns causal policy.
- "Why RL not SL?" → No ground truth label. Outcomes delayed, counterfactual, multi-objective.
- "Clinical accuracy?" → PK/PD from published pharmacology, EUCAST breakpoint tables.
- "Which model?" → Qwen2.5-3B-Instruct, GRPO via HF TRL, Unsloth 4-bit QLoRA.

---

## New dependencies to add

```
# Already in requirements.txt: scipy, numpy, networkx, fastapi, pydantic, openai
# Add:
openenv>=0.1.0   # already present
# pygmo>=2.19.0  # optional for hypervolume — use manual impl if install fails
```

---

## Minimum viable submission checklist

Before leaving for Bangalore, confirm:
- [ ] All 4 tasks run end-to-end via env.in_process()
- [ ] All rewards in [0,1] — no exceptions
- [ ] Full test suite green
- [ ] Docker build succeeds: docker build -t episteward .
- [ ] Container starts: docker run -p 7860:7860 episteward
- [ ] /reset returns 200, /step returns valid StepResult
- [ ] openenv.yaml validates
- [ ] inference.py runs all 4 tasks sequentially without error
- [ ] train_grpo.ipynb runs on CPU (slow is fine — just confirm no crashes)
- [ ] demo_assets.py runs and produces 5 PNGs in assets/

Onsite with compute credits:
- [ ] Run train_grpo.ipynb on GPU for 500+ steps
- [ ] Export reward curves showing visible improvement
- [ ] Generate final demo_assets.py PNGs with real training data
- [ ] Record <2 min video for HuggingFace blog post
