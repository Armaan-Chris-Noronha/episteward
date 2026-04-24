#!/usr/bin/env python3
"""
EpiSteward demo asset generator.

Produces 5 pitch-ready PNGs in assets/ using synthetic placeholder data.
Replace the synthetic arrays with real training logs from train_grpo.ipynb
after the onsite GPU run.

Usage:
    python scripts/demo_assets.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless; safe for scripted runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "..", "assets")
FIG_W, FIG_H = 12, 8      # inches (150 dpi → 1800×1200 px)
DPI = 150
BG = "white"

RNG = np.random.default_rng(42)
STEPS = np.arange(0, 501, 5)   # training steps 0..500
N = len(STEPS)

# Palette
C = {
    "task1":      "#2196F3",   # blue
    "task2":      "#4CAF50",   # green
    "task4":      "#FF9800",   # orange
    "early":      "#EF5350",   # red  (step 0)
    "mid":        "#FFC107",   # amber (step 100)
    "late":       "#66BB6A",   # light green (step 500)
    "bad":        "#EF5350",
    "good":       "#66BB6A",
    "carb":       "#EF5350",
    "deesc":      "#FF9800",
    "dur":        "#9C27B0",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, w: int = 9) -> np.ndarray:
    pad = np.pad(arr, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")[: len(arr)]


def _lc(
    start: float,
    end: float,
    noise: float,
    knee: float = 180,
    rng: np.random.Generator = RNG,
) -> np.ndarray:
    """Realistic RL learning curve with exponential rise and noise."""
    t = STEPS / STEPS[-1]
    base = start + (end - start) * (1.0 - np.exp(-t * STEPS[-1] / knee))
    return _smooth(np.clip(base + rng.normal(0, noise, N), 0.0, 1.0))


def _save(filename: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def _spine_off(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# 1. demo_reward_curves.png
# ---------------------------------------------------------------------------

def make_reward_curves() -> None:
    """4 subplots × 3 task lines — total reward, de-escalation rate,
    broad-spectrum %, PoA delta."""
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.suptitle("EpiSteward GRPO Training Curves", fontsize=16, fontweight="bold", y=0.99)

    task_data = {
        "task1": {
            "reward":   _lc(0.08, 0.72, 0.030),
            "deesc":    _lc(0.14, 0.81, 0.040),
            "broad":    _lc(0.76, 0.11, 0.035),
            "poa":      _lc(0.00, 0.56, 0.025, knee=220),
        },
        "task2": {
            "reward":   _lc(0.10, 0.64, 0.040),
            "deesc":    _lc(0.12, 0.73, 0.045),
            "broad":    _lc(0.81, 0.17, 0.040),
            "poa":      _lc(0.00, 0.47, 0.030, knee=230),
        },
        "task4": {
            "reward":   _lc(0.06, 0.57, 0.050),
            "deesc":    _lc(0.09, 0.65, 0.055),
            "broad":    _lc(0.86, 0.23, 0.050),
            "poa":      _lc(0.00, 0.61, 0.035, knee=200),
        },
    }

    labels = {
        "task1": "Task 1: Triage",
        "task2": "Task 2: Containment",
        "task4": "Task 4: Multi-Ward",
    }

    subplot_meta = [
        (axes[0, 0], "reward", "Total Reward",          "Reward  [0, 1]"),
        (axes[0, 1], "deesc",  "De-escalation Rate",    "Fraction of episodes"),
        (axes[1, 0], "broad",  "Broad-spectrum %",      "Fraction of prescriptions"),
        (axes[1, 1], "poa",    "PoA Delta (Game Theory)","Improvement vs. naive"),
    ]

    for ax, key, title, ylabel in subplot_meta:
        for task in ("task1", "task2", "task4"):
            ax.plot(STEPS, task_data[task][key], color=C[task],
                    lw=2, label=labels[task], alpha=0.92)
        ax.axhline(0.10, ls=":", color="#9E9E9E", lw=1.5, zorder=1,
                   label="Random baseline")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, 500)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        _spine_off(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save("demo_reward_curves.png")


# ---------------------------------------------------------------------------
# 2. demo_pareto_front.png
# ---------------------------------------------------------------------------

def make_pareto_front() -> None:
    """Scatter: r_clinical vs r_ecology at steps 0 / 100 / 500.
    Arrow shows Pareto front expanding — both objectives improve jointly."""
    fig, ax = plt.subplots(figsize=(FIG_W * 0.72, FIG_H * 0.92), facecolor=BG)

    def _cloud(cx: float, cy: float, spread: float, n: int = 45) -> tuple:
        x = np.clip(RNG.normal(cx, spread, n), 0, 1)
        y = np.clip(RNG.normal(cy, spread, n), 0, 1)
        return x, y

    clouds = [
        (_cloud(0.32, 0.28, 0.07), C["early"], "Step 0 — untrained"),
        (_cloud(0.53, 0.49, 0.06), C["mid"],   "Step 100"),
        (_cloud(0.73, 0.69, 0.05), C["late"],  "Step 500 — trained"),
    ]

    for (x, y), color, label in clouds:
        ax.scatter(x, y, c=color, s=45, alpha=0.72, label=label, zorder=3)

    # Arrow indicating Pareto front expansion direction
    ax.annotate(
        "Pareto front\nexpanding",
        xy=(0.70, 0.67),
        xytext=(0.42, 0.38),
        fontsize=10,
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8),
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFFDE7", ec="#BDBDBD"),
    )

    ax.set_xlabel("Clinical Outcome Score  (r_clinical)", fontsize=12)
    ax.set_ylabel("Ecological Footprint Score  (r_ecology)", fontsize=12)
    ax.set_title(
        "Pareto Front Expansion During GRPO Training\n"
        "Clinical outcomes and ecological stewardship improve simultaneously",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    _spine_off(ax)
    _save("demo_pareto_front.png")


# ---------------------------------------------------------------------------
# 3. demo_poa.png
# ---------------------------------------------------------------------------

def make_poa() -> None:
    """Bar chart: Price of Anarchy before (~2.4) vs after (~1.3).
    Dashed line at 1.0 = social optimum."""
    fig, ax = plt.subplots(figsize=(FIG_W * 0.58, FIG_H * 0.88), facecolor=BG)

    x_labels = ["Before Training", "After Training\n(500 steps GRPO)"]
    poa_values = [2.38, 1.31]

    bars = ax.bar(
        x_labels, poa_values,
        color=[C["bad"], C["good"]],
        width=0.50,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    for bar, val in zip(bars, poa_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}×",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
        )

    # Social optimum reference line
    ax.axhline(1.0, ls="--", color="#1565C0", lw=2.0, zorder=4)
    ax.text(1.48, 1.07, "Social Optimum\n(PoA = 1.0)",
            fontsize=9, color="#1565C0", ha="right", va="bottom")

    # Value recovered annotation
    pct = (poa_values[0] - poa_values[1]) / (poa_values[0] - 1.0) * 100
    ax.annotate(
        f"{pct:.0f}% of tragedy-of-the-\ncommons value recovered",
        xy=(1, poa_values[1]),
        xytext=(0.18, 1.95),
        fontsize=9, color="#1B5E20",
        arrowprops=dict(arrowstyle="-|>", color="#1B5E20", lw=1.4),
    )

    ax.set_ylabel("Price of Anarchy (PoA)", fontsize=12)
    ax.set_title(
        "Price of Anarchy: Before vs After Training\n"
        "Lower PoA = more cooperative prescribing, less antibiotic overuse",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 3.0)
    ax.grid(axis="y", alpha=0.25)
    _spine_off(ax)
    _save("demo_poa.png")


# ---------------------------------------------------------------------------
# 4. demo_before_after.png
# ---------------------------------------------------------------------------

def make_before_after() -> None:
    """Table: untrained vs trained agent for 3 patient vignettes."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.52), facecolor=BG)
    ax.axis("off")

    columns = [
        "Patient Scenario",
        "Untrained Antibiotic",
        "Trained Antibiotic",
        "App Routed",
    ]
    rows = [
        [
            "UTI — Elderly 80F  (E. coli)",
            "Meropenem IV  (broad, last-resort)",
            "Nitrofurantoin PO  (narrow, targeted)",
            "lab",
        ],
        [
            "Sepsis — ICU, ESBL flags",
            "Meropenem IV  (last-resort, no consult)",
            "Pip-Tazo IV + sensitivity panel",
            "pharmacy",
        ],
        [
            "Multi-Ward AMR Cascade  (PoA 2.4)",
            "Meropenem all wards  (broad signal)",
            "Ceftriaxone + isolation order",
            "ehr",
        ],
    ]

    col_w = [0.33, 0.24, 0.26, 0.11]
    row_h = 0.175
    hdr_h = 0.175
    y_top = 0.82

    # X starts for each column
    x0 = [0.02]
    for w in col_w[:-1]:
        x0.append(x0[-1] + w)

    # Header row
    for col, xi, cw in zip(columns, x0, col_w):
        r = mpatches.FancyBboxPatch(
            (xi, y_top), cw - 0.01, hdr_h,
            boxstyle="round,pad=0.01", fc="#1565C0", ec="white", lw=1,
        )
        ax.add_patch(r)
        ax.text(xi + cw / 2 - 0.005, y_top + hdr_h / 2, col,
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    row_bg = ["#FFF9C4", "#E8F5E9", "#FFF9C4"]
    for ri, (row, bg) in enumerate(zip(rows, row_bg)):
        y = y_top - (ri + 1) * row_h
        for ci, (cell, xi, cw) in enumerate(zip(row, x0, col_w)):
            r = mpatches.FancyBboxPatch(
                (xi, y), cw - 0.01, row_h - 0.01,
                boxstyle="round,pad=0.01", fc=bg, ec="#BDBDBD", lw=0.5,
            )
            ax.add_patch(r)
            # Colour bad choices red, good choices dark-green
            lo = cell.lower()
            color = (
                "#B71C1C" if "meropenem" in lo and ri != 99 and ci == 1
                else "#1B5E20" if ci == 2
                else "black"
            )
            ax.text(xi + cw / 2 - 0.005, y + row_h / 2, cell,
                    ha="center", va="center", fontsize=8.5, color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 1.08)
    ax.set_title(
        "EpiSteward: Prescribing Behaviour Before vs After GRPO Training",
        fontsize=13, fontweight="bold", pad=8,
    )
    _save("demo_before_after.png")


# ---------------------------------------------------------------------------
# 5. demo_oversight.png
# ---------------------------------------------------------------------------

def make_oversight() -> None:
    """Timeline of oversight flags per episode.
    CARBAPENEM_WITHOUT_JUSTIFICATION drops ~70% after episode 200."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.75), facecolor=BG)

    eps = np.arange(1, 251)

    def _flag_series(
        peak: float, decay: float, drop_ep: int, drop_frac: float, noise_amp: float
    ) -> np.ndarray:
        base = peak * np.exp(-eps / decay)
        post = np.where(eps > drop_ep, base * drop_frac, base)
        return _smooth(np.clip(post + RNG.uniform(-noise_amp, noise_amp, len(eps)), 0, None), w=7)

    carb  = _flag_series(peak=5.0, decay=110, drop_ep=200, drop_frac=0.30, noise_amp=0.30)
    deesc = _flag_series(peak=3.2, decay=140, drop_ep=200, drop_frac=0.45, noise_amp=0.22)
    dur   = _flag_series(peak=2.6, decay=95,  drop_ep=200, drop_frac=0.40, noise_amp=0.20)

    for series, color, label in [
        (carb,  C["carb"],  "CARBAPENEM_WITHOUT_JUSTIFICATION"),
        (deesc, C["deesc"], "MISSED_DEESCALATION"),
        (dur,   C["dur"],   "DURATION_EXCEEDS_GUIDELINE"),
    ]:
        ax.fill_between(eps, 0, series, alpha=0.18, color=color)
        ax.plot(eps, series, color=color, lw=2.0, label=label)

    # Mark the stabilisation point
    ax.axvline(200, ls="--", color="#616161", lw=1.5, alpha=0.70)
    ax.text(202, carb.max() * 0.92, "Training\nstabilises",
            fontsize=9, color="#616161", va="top")

    # Annotate the ~70% carbapenem drop
    idx200 = int(np.searchsorted(eps, 200))
    before = float(carb[max(0, idx200 - 12): idx200].mean())
    after  = float(carb[idx200 + 8: idx200 + 28].mean())
    drop_pct = max(0.0, (before - after) / (before + 1e-8) * 100)
    ax.annotate(
        f"-{drop_pct:.0f}% carbapenem\noveruse flags",
        xy=(230, after + 0.08),
        xytext=(180, before * 0.65),
        fontsize=9, color=C["carb"],
        arrowprops=dict(arrowstyle="-|>", color=C["carb"], lw=1.4),
    )

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Oversight Flags per Episode", fontsize=12)
    ax.set_title(
        "Oversight Agent Safety Flags During Training\n"
        "Fleet AI monitor drives progressive elimination of unsafe prescribing",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(1, 250)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25)
    _spine_off(ax)
    _save("demo_oversight.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating EpiSteward pitch demo assets...")
    make_reward_curves()
    make_pareto_front()
    make_poa()
    make_before_after()
    make_oversight()
    print("\nAll 5 assets generated successfully.")
