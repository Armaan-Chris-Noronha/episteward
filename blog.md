# EpiSteward: A reinforcement learning environment for antibiotic stewardship, built to train AI agents to understand restraint

## Why this particular problem statement?

A couple of months ago my grandma was diagnosed with a Urinary Tract Infection (UTI).

UTIs are very common. Antibiotics are common. I believed that once the doctors started the right injections, the infection would come under control. For a while it even looked that way. Her condition improved, and I felt extremely relieved. But then after a few weeks the infection would return. The medicines had to be changed because the bacteria was resistant to the previous ones. The injections continued. Every new round came with the same hope, maybe this one would finally work.

But eventually came a point when no drug seemed to be working. The doctors called it drug resistant infection. Sounds very clinical doesn't it? But as someone who has seen the effects first hand, its haunting. It feels like watching modern medicine fight with all its strength and slowly run out of useful weapons. The medicine can exist, doctors can prescribe it, nurses can inject it and still the infection may no longer respond.

A few weeks later, my grandma passed away.

Antibiotic resistance is projected to kill 10 million people a year by 2050 — more than cancer. And the worst part is that resistance doesn't grow because of obviously wrong decisions. It grows because of decisions that seem completely reasonable at the moment.

A patient comes in with a serious infection. The exact bacteria is unknown. The lab report takes 48 hours. The doctor obviously has to act immediately. In that moment choosing a broad spectrum antibiotic is the safest option — it covers more possibilities and protects the patient in front of you.

But a series of these 'safe' decisions can create a much larger danger. Every bacteria that survives builds tolerance and comes back stronger. What looks like a good decision for one patient today quietly reduces the treatment options for the next.

EpiSteward models this very scenario.

## What I built

EpiSteward is an OpenEnv reinforcement learning environment where an AI agent acts as an antibiotic steward in a 5-ward hospital. Each episode runs for 7 simulated days. The agent prescribes empirically on Day 0 without knowing the pathogen. Day 2 the agent gets information on the gram stain. Day 3 it gets the full culture report. Day 5 is the IV-to-oral decision. Day 7 is the outcome.

The agent is rewarded for curing patients but penalised for unnecessary broad-spectrum antibiotic use. A ward-level resistance pressure index accumulates across episodes and future episodes get harder.

EpiSteward makes the agent live with delayed consequences.

## The Science

We didn't want a toy environment. We wanted something a pharmacology PhD would recognize.

**Two-compartment population PK** with inter-individual variability — different patients metabolize the same drug differently. **Itô SDE for resistance evolution** replacing deterministic Wright-Fisher with stochastic dynamics so the agent learns robust policies. **Mutant Selection Window** — between MIC and MPC, drugs paradoxically amplify resistant mutants. No other RL environment models this.

**Horizontal Gene Transfer** — antibiotic stress increases plasmid transfer rate via the SOS response, so a "safe" prescription can accelerate resistance spread between species. **Bayesian pathogen inference** — the agent maintains a posterior over five pathogens and decides whether ordering tests is worth the cost.

And **game theory**:

```
Uᵢ = (f_min + (f_max − f_min)·σᵢ) − αᵢ·σᵢ·(1 + β·mean(σ_{−ᵢ}))

PoA = ΣUᵢ(Nash) / ΣUᵢ(Social Optimum)
```

Each ward acting selfishly converges on broad-spectrum prescribing. The Price of Anarchy measures the cost. The agent's goal is to push PoA toward 1.0 — recovering value lost to the tragedy of the commons.

## What Training Showed

We trained Qwen2.5-3B-Instruct using GRPO for 200 steps on a T4.

| Metric | Start | End |
|---|---|---|
| **Total reward** | 0.44 | 0.70 |
| **Broad-spectrum usage** | 71% | 50% |
| **De-escalation rate** | 20% | 36% |

Broad-spectrum usage fell by 21 percentage points. The agent was learning **restraint**.

### Before vs After

| Patient | Untrained | Trained |
|---|---|---|
| Elderly UTI (E. coli) | meropenem IV → 0.039 | **nitrofurantoin PO** → 1.000 |
| ICU Sepsis (ESBL) | meropenem IV → 0.070 | **piperacillin-tazo + sensitivity panel** → 0.809 |
| Multi-Ward AMR | meropenem IV → 0.005 | **ceftriaxone + EHR routing** → 0.035 |

The untrained agent prescribed meropenem for everything. The trained agent learned to prescribe nitrofurantoin for a low-risk UTI, de-escalate to piperacillin-tazobactam when culture confirmed ESBL, order diagnostics to reduce uncertainty, and route actions to the correct enterprise system.

Nobody told it to do these things. The reward function — grounded in real pharmacology and game theory — was enough.

## Why This Matters Beyond Antibiotics

A lot of AI systems today are built to answer questions. The real world needs AI systems that can act responsibly in complex environments.

In healthcare, finance, public policy and many other fields failure comes from the same pattern. A decision may look good immediately but can be damaging later on.

EpiSteward isn't only about antibiotics. It's about training AI agents to understand delayed consequences. The strongest choice is not always the best choice.

Sometimes the most intelligent thing an AI can learn is restraint.

When I think about my grandmother now, I don't think of antibiotic resistance just as an abstract global problem. I think of repeated injections. Temporary improvement. Returning infections. And overall the helplessness of watching medicine lose power.

That is why this project matters to me. The future of medicine cannot depend only on stronger antibiotics. It must depend on smarter decisions.

That is the key lesson EpiSteward teaches.

## Try It

- **🤗 Space:** [armaancn/episteward_openenv](https://huggingface.co/spaces/armaancn/episteward_openenv)
- **📓 Notebook:** [Colab](https://colab.research.google.com/drive/1t1JMl2Iqc5T7w5noUS1C-kwoTr-PnHh4?usp=sharing)
- **💻 Code:** [github.com/Armaan-Chris-Noronha/episteward](https://github.com/Armaan-Chris-Noronha/episteward)
