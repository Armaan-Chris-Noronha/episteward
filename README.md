---
title: EpiSteward
emoji: 🏥
colorFrom: green
colorTo: teal
sdk: docker
app_port: 7860
license: mit
---

# EpiSteward

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-brightgreen)](https://github.com/openenv)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](https://hub.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**AI Antibiotic Stewardship Environment for Reinforcement Learning**

Antimicrobial resistance (AMR) kills ~700,000 people per year today — and is projected to reach **10 million deaths per year by 2050**, surpassing cancer. A key driver is inappropriate antibiotic prescribing inside hospital networks: wrong drug, wrong dose, wrong duration. EpiSteward places an RL agent in that role, forcing it to make evidence-based prescribing decisions under uncertainty, contain resistance transmission chains, and allocate last-resort antibiotics across a multi-hospital network.

---

## Math Models

The reward signal is grounded in real pharmacology and epidemiology, not heuristics.

**PK/PD — one-compartment model**

```
C(t) = (F · D / Vd) · exp(−ke · t)     ke = CL / Vd
Effect = Emax · Cⁿ / (EC50ⁿ + Cⁿ)     Therapeutic window: [4×MIC, 64×MIC]
```

**Resistance Evolution — Wright-Fisher process**

```
p(t+1) ~ Binomial(2N, p̃) / 2N
p̃ = p·wR / (p·wR + (1−p)·wS)          wS = 1 − s  (under drug pressure)
```

Resistance emerges when allele frequency > 0.5 sustained > 48 h.

**Network Transmission**

```
P(i→j) = β · w(i,j) · I_i(t) · (1 − immune_j)
β = 0.15 (ESBL)   β = 0.08 (CRK)
```

**Bayesian Resistance Estimation**

```
P(resistant | result) ∝ P(result | resistant) · P(resistant)
```

Prior from local antibiogram; posterior mean + credible interval returned with each observation.

---

## Spaces

### Observation — `EpiObservation`

| Field | Type | Description |
|---|---|---|
| `patient_id` | `str` | Unique patient identifier |
| `ward_id` | `str` | Current ward |
| `infection_site` | `str` | e.g. `urinary_tract`, `bloodstream` |
| `symptoms` | `list[str]` | Clinical presentation |
| `vitals` | `dict[str, float]` | temp, HR, WBC, CRP, procalcitonin |
| `culture_results` | `dict` | Status + sensitivities (may be partial) |
| `resistance_flags` | `list[str]` | ESBL, MRSA, CRE, CRK |
| `transfer_history` | `list[str]` | Ward movement chain |
| `antibiotic_history` | `list[dict]` | Prior prescriptions this episode |
| `network_alert` | `str \| null` | Outbreak broadcast (Task 3 only) |

### Action — `EpiAction`

| Field | Type | Constraints |
|---|---|---|
| `antibiotic` | `str` | 13 agents from colistin to TMP-SMX |
| `dose_mg` | `float` | > 0 |
| `frequency_hours` | `float` | e.g. `8.0` = q8h |
| `duration_days` | `int` | 1–14 |
| `route` | `str` | `IV`, `PO`, `IM` |
| `isolation_order` | `bool` | Contact precautions |
| `culture_requested` | `bool` | Blood/urine/wound culture |
| `specialist_consult` | `bool` | ID consult flag |
| `reasoning` | `str \| null` | Agent explanation (logged, not graded) |

---

## Tasks

### Task 1 — Prescription Triage `[easy]` · 5 steps

Single patient, complete culture data. Agent selects antibiotic + dose; grader checks drug class match, PK/PD therapeutic window, and narrow-spectrum preference. De-escalation on new sensitivity data is rewarded.

| Agent | Score |
|---|---|
| Random baseline | ~0.10 |
| Target (strong) | ≥ 0.85 |

### Task 2 — Resistance Containment `[medium]` · 15 steps

6-patient ESBL *E. coli* cluster in MedWard_A. Agent must identify the index patient, issue isolation orders, and adjust empiric therapy. New resistance cases penalise −0.05/step; correct isolation within 3 steps gives +0.10 bonus.

| Agent | Score |
|---|---|
| Random baseline | ~0.10 |
| Target (strong) | ≥ 0.65 |

Optimal: piperacillin-tazobactam 4500 mg q8h IV + isolation order + cultures.

### Task 3 — Network Outbreak Response `[hard]` · 30 steps

10-hospital CRK network, 2 infected hospitals at start, **finite colistin budget (10 uses)**. Agent traces phylogenetic spread, issues hospital-level containment, and allocates last-resort therapy.

```
reward = 0.6 · lives_saved_ratio
       − 0.25 · colistin_overspend_fraction
       − 0.15 · resistance_amplification_fraction
```

| Agent | Score |
|---|---|
| Random baseline | ~0.10 |
| Target (strong) | ≥ 0.65 |

---

## Setup

**Local (in-process, no Docker)**

```bash
pip install -e ".[dev]"
python -c "
import asyncio
from episteward import EpiStewardEnv

async def run():
    env = EpiStewardEnv.in_process()
    obs = await env.reset('task1_triage')
    print(obs.observation.model_dump())

asyncio.run(run())
"
```

**Docker**

```bash
docker build -t episteward .
docker run -p 7860:7860 episteward
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" -d '{}'
```

**Baseline inference (LLM agent)**

```bash
export HF_TOKEN=hf_...
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | `{"status":"ok","env":"episteward"}` |
| `/health` | GET | Liveness probe |
| `/tasks` | GET | List task IDs |
| `/reset` | POST | New episode — empty body `{}` defaults to `task1_triage` |
| `/step` | POST | Submit `EpiAction`, get `StepResult` |
| `/state` | GET | Read-only episode snapshot |
