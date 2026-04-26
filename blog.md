# EpiSteward: Training an AI agent to understand restraint

## Why this particular problem?

A couple of months ago my grandma was diagnosed with a Urinary Tract Infection (UTI).

UTIs are very common. Antibiotics are common. I believed that once the doctors started the right injections, the infection would come under control. For a while it even looked that way. Her condition improved, and I felt extremely relieved. But then after a few weeks the infection would return. The medicines had to be changed because the bacteria was resistant to the previous ones. The injections continued. Every new round came with the same hope, maybe this one would finally work.

But eventually came a point when no drug seemed to be working. The doctors called it drug resistant infection. Sounds very clinical doesn't it? But as someone who has seen the effects first hand, its haunting. It feels like watching modern medicine fight with all its strength and slowly run out of useful weapons. The medicine can exist, doctors can prescribe it, nurses can inject it and still the infection may no longer respond.

A few weeks later, my grandma passed away.

Antibiotic resistance is projected to kill 10 million people a year by 2050 — more than cancer. And the worst part is that resistance doesn't grow because of obviously wrong decisions. It grows because of decisions that seem completely reasonable at the moment.

A patient comes in with a serious infection. The exact bacteria is unknown. The lab report takes 48 hours. The doctor obviously has to act immediately. In that moment choosing a broad-spectrum antibiotic is the safest option — it covers more possibilities and protects the patient in front of you.

But a series of these 'safe' decisions can create a much larger danger. Every bacteria that survives builds tolerance and comes back stronger. What looks like a good decision for one patient today quietly reduces the treatment options for the next.

**This is not a knowledge problem. Every doctor knows antibiotics should be used carefully. It is a decision-making problem under uncertainty, with delayed consequences, across a system of competing incentives.**

That is exactly the kind of problem reinforcement learning was built for. EpiSteward models this very scenario.

---

## What the agent sees, does, and gets rewarded for

EpiSteward is an OpenEnv reinforcement learning environment where an AI agent acts as an antibiotic steward in a 5-ward hospital.

**What the agent sees:** Patient vitals, infection site, resistance flags, a Bayesian posterior over five possible pathogens, culture results that arrive incrementally across days, ward-level resistance pressure, and the current prescribing patterns of neighbouring wards.

**What the agent does:** Each step, the agent prescribes an antibiotic (or changes it), sets dose and route, orders diagnostics, requests isolation, consults specialists, and routes its action to the correct enterprise system (EHR, Lab, Pharmacy, Microbiology).

**What the agent gets rewarded for:** Curing the patient — but not at any ecological cost. The reward signal is a Pareto-weighted combination of four objectives:

- **Clinical** — was the drug effective? (PK/PD therapeutic window)
- **Ecological** — did it breed resistant mutants? (Mutant Selection Window)
- **Economic** — was the course duration appropriate?
- **Stewardship** — did it de-escalate when culture results allowed it?

These objectives conflict. Meropenem cures the patient (↑ clinical) but breeds resistance (↓ ecology). The agent must learn where to trade off — and that trade-off changes depending on how bad the resistance crisis is across the ward.

There are four tasks of increasing difficulty — from a single UTI patient (5 steps) to a 10-hospital CRK outbreak (30 steps) to a multi-ward prescribing game (20 steps). A ward-level resistance pressure index accumulates across episodes. Future episodes get harder as the agent's past decisions catch up with it.

**EpiSteward makes the agent live with delayed consequences.**

### The Four Tasks

**Task 1 — Prescription Triage** `[easy · 5 steps]`
Single patient, E. coli ESBL urinary tract infection. Culture data is revealed incrementally across steps. The agent selects antibiotic, dose, route, and duration. The grader checks drug class correctness, PK/PD therapeutic window, spectrum appropriateness, and de-escalation timing. Optimal action: `nitrofurantoin 100mg q6h PO 5 days` — the narrowest effective drug for a UTI, oral not IV, short course.

**Task 2 — Resistance Containment** `[medium · 15 steps]`
Six-patient ESBL cluster in a medical ward. The agent must identify the index patient, order isolation, and prescribe appropriately across all exposed patients. New resistance cases incur a penalty each step; early isolation gives a bonus. This is where the agent learns that prescribing and infection control are inseparable decisions.

**Task 3 — Network Outbreak Response** `[hard · 30 steps]`
Ten hospitals, two already infected with carbapenem-resistant Klebsiella. The agent has a finite budget of colistin — the last-resort antibiotic. It must trace spread through the network, issue containment orders, and allocate last-resort therapy only where necessary. Every carbapenem use risks amplifying resistance further. This task cannot be solved by prescribing the same thing every step.

**Task 4 — Multi-Ward Stewardship Game** `[expert · 20 steps]`
The agent plays coordinator across 5 wards simultaneously. Each ward has its own prescribing intensity σᵢ ∈ [0,1] that shifts based on the coordinator's signal. The antibiotic the agent prescribes encodes that signal — narrow drugs push wards toward conservative prescribing, broad drugs push them toward overuse. The agent must learn to steer a hospital-wide equilibrium toward the social optimum, not just treat the patient in front of it.

---

## The Science

I didn't want a toy environment. I wanted something a pharmacology PhD would recognize.

### Pharmacokinetics — Different Patients, Different Drugs

The same antibiotic does not behave the same way in every patient. Age, kidney function, and individual variability all affect how drug concentrations change inside the body.

I modeled this with a two-compartment ODE system with inter-individual variability:

```
dC₁/dt = (F·D·kₐ·exp(−kₐ·t))/V₁ − (k₁₂ + k₁₀)·C₁ + k₂₁·(V₂/V₁)·C₂
dC₂/dt = k₁₂·C₁ − k₂₁·C₂

θᵢ = θ_pop · exp(ηᵢ)     ηᵢ ~ N(0, ω²)
```

An elderly patient with kidney disease metabolizes meropenem differently from a healthy 30-year-old. The agent sees creatinine and age, and must learn to adjust dosing.

### The Mutant Selection Window — The Piece No Other RL Environment Models

Between MIC (the concentration that kills susceptible bacteria) and MPC (the concentration that kills resistant mutants too), there's a danger zone. Drugs sitting in this window kill susceptibles but selectively amplify resistant mutants.

```
MPC = MIC × (1 / mutation_freq)^(1 / hill_coeff)
MSW risk = T_MSW / T_total
```

The agent learns to dose above MPC when possible — collapsing the window — or pick antibiotics with a higher MPC/MIC ratio. No other RL environment captures this.

### Stochastic Resistance Evolution

Biology is not perfectly predictable. Two similar treatment decisions may not produce identical outcomes. I replaced the deterministic Wright-Fisher model with an Itô stochastic differential equation:

```
dp_R = s(C) · p_R · (1 − p_R) · dt + σ · √(p_R · (1 − p_R)) · dW
```

This forces the agent to learn robust policies, not memorize deterministic answers.

### Horizontal Gene Transfer

Resistance doesn't stay in one species. Plasmids carry it between bacteria — and antibiotic stress paradoxically accelerates the transfer through the SOS response:

```
γ(C) = γ_baseline · (1 + α_SOS · C/MIC)
```

A well-intentioned prescription can accelerate resistance spread between species. This is the mechanism behind carbapenem-resistant Klebsiella spreading from E. coli in real hospital wards.

### Bayesian Pathogen Inference

The agent maintains a probability distribution over five pathogens, updated as diagnostic results arrive:

```
P(pathogen=k | obs) ∝ L(obs | k) · P(pathogen=k)
```

Ordering tests reduces Shannon entropy. Value of Information determines whether a test is worth its cost. The agent learns when to commit and when to gather more evidence.

### Game Theory — The Heart of the Problem

Antibiotic resistance is not only a biological problem. It is a coordination problem. Each doctor wants to protect the patient in front of them. That is completely understandable. But if everyone always uses the strongest antibiotic as their default safety strategy, the shared antibiotic ecosystem deteriorates.

Each ward's prescribing intensity σᵢ ∈ [0,1] enters a strategic game:

```
Uᵢ = (f_min + (f_max − f_min)·σᵢ) − αᵢ·σᵢ·(1 + β·mean(σ_{−ᵢ}))

PoA = ΣUᵢ(Nash) / ΣUᵢ(Social Optimum)
```

The Nash equilibrium of every ward acting selfishly is everyone using broad-spectrum antibiotics. The Price of Anarchy measures how much value is lost. The agent's goal is to push PoA toward 1.0 — recovering value lost to the tragedy of the commons.

---

## Training Evidence

I trained Qwen2.5-3B-Instruct using GRPO (same algorithm as DeepSeek-R1) for 200 steps on a T4 GPU via Unsloth + HF TRL. The training loop connects directly to EpiSteward in-process — the reward function is a live environment step, not a static dataset.

![Training curves — total reward, de-escalation rate, broad-spectrum usage, and PoA improvement across 200 GRPO steps](https://raw.githubusercontent.com/Armaan-Chris-Noronha/episteward/main/assets/demo_reward_curves.png)
*Total reward, de-escalation rate, broad-spectrum usage, and PoA improvement across 200 GRPO training steps on a T4 GPU.*

| Metric | Start | End | Change |
|---|---|---|---|
| **Total reward** | 0.44 | 0.70 | +59% |
| **Broad-spectrum usage** | 71% | 50% | −21pp |
| **De-escalation rate** | 20% | 36% | +80% |

Broad-spectrum usage fell by 21 percentage points. The agent was learning **restraint**.

### Before vs After — 3 Patient Vignettes

| Patient | Untrained agent | Trained agent |
|---|---|---|
| Elderly UTI (E. coli) | meropenem IV → **0.039** | nitrofurantoin PO → **1.000** |
| ICU Sepsis (ESBL) | meropenem IV → **0.070** | piperacillin-tazo + sensitivity panel → **0.809** |
| Multi-Ward AMR | meropenem IV → **0.005** | ceftriaxone + EHR routing → **0.035** |

The untrained agent prescribed meropenem for everything — the last-resort carbapenem — regardless of patient, infection site, or resistance profile. It ordered no diagnostics and routed to no enterprise system.

The trained agent learned to prescribe nitrofurantoin for a low-risk UTI, de-escalate to piperacillin-tazobactam when culture confirmed ESBL, order diagnostics to reduce pathogen uncertainty, and route actions to the correct enterprise system.

Nobody told it to do these things. The reward function — grounded in real pharmacology and game theory — was enough.

---

## Why This Matters Beyond Antibiotics

A lot of AI systems today are built to answer questions. The real world needs AI systems that can act responsibly in complex environments.

In healthcare, finance, public policy and many other fields failure comes from the same pattern. A decision may look good immediately but can be damaging later on.

EpiSteward isn't only about antibiotics. It's about training AI agents to understand delayed consequences. The strongest choice is not always the best choice.

Sometimes the most intelligent thing an AI can learn is restraint.

When I think about my grandmother now, I don't think of antibiotic resistance just as an abstract global problem. I think of repeated injections. Temporary improvement. Returning infections. And overall the helplessness of watching medicine lose power.

That is why this project matters to me. The future of medicine cannot depend only on stronger antibiotics. It must depend on smarter decisions.

That is the key lesson EpiSteward teaches.

---

## Try It

- **🤗 Space:** [armaancn/episteward_openenv](https://huggingface.co/spaces/armaancn/episteward_openenv)
- **📓 Training Notebook:** [Colab](https://colab.research.google.com/drive/1t1JMl2Iqc5T7w5noUS1C-kwoTr-PnHh4?usp=sharing)
- **💻 Code:** [github.com/Armaan-Chris-Noronha/episteward](https://github.com/Armaan-Chris-Noronha/episteward)
