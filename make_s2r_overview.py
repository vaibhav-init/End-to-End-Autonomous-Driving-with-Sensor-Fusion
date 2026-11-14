#!/usr/bin/env python3
"""One-page overview figure for the matched synthetic-to-real study."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np


FIG_W, FIG_H = 14.0, 8.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.text(
    0.035,
    0.955,
    "Synthetic → Real Radar Ghost Detection",
    fontsize=23,
    fontweight="bold",
    ha="left",
    va="top",
    color="#111111",
)
fig.text(
    0.035,
    0.915,
    "Shared backbone + dataset-specific head, adapted head-first on real RGD",
    fontsize=13,
    ha="left",
    va="top",
    color="#555555",
)

# ---------------------------------------------------------------- pipeline
ax = fig.add_axes([0.03, 0.26, 0.55, 0.60])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

boxes = [
    (
        "1  Matched synthetic data (CARLA)",
        "20 sequences · pedestrian + cyclist · 10 Hz\n"
        "330–356 points/scan (real RGD: 336) · 4–10 m targets",
        "#e8f1fb",
        "#2f6fb0",
    ),
    (
        "2  Shared backbone",
        "TemporalPointNet per-point encoder\n"
        "pretrained on synthetic ghosts (10 epochs)",
        "#eef7ee",
        "#2e7d32",
    ),
    (
        "3  Dataset-specific head (head-first)",
        "same ghost / real classifier, adapted on real data\n"
        "backbone frozen for 3 epochs (head-only ablation below)",
        "#fff4e5",
        "#b26a00",
    ),
    (
        "4  Two-stage real adaptation",
        "unfreeze backbone at 0.1× learning rate (27 epochs)\n"
        "real RGD train split · select on real val AUROC",
        "#f3e8fd",
        "#6a1b9a",
    ),
    (
        "5  Evaluation",
        "official RGD v1.1 test · 28 sequences · 5 scenarios\n"
        "5 seeds · threshold fixed at 1% val FPR",
        "#fdeaea",
        "#b71c1c",
    ),
]

top = 0.94
height = 0.145
gap = 0.035
for index, (title, detail, face, edge) in enumerate(boxes):
    y = top - index * (height + gap) - height
    patch = FancyBboxPatch(
        (0.02, y),
        0.96,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        0.05,
        y + height * 0.66,
        title,
        fontsize=11.5,
        fontweight="bold",
        va="center",
        color=edge,
    )
    ax.text(
        0.05,
        y + height * 0.28,
        detail,
        fontsize=9.2,
        va="center",
        color="#333333",
        linespacing=1.25,
    )
    if index < len(boxes) - 1:
        next_y = y - gap
        arrow = FancyArrowPatch(
            (0.5, y),
            (0.5, next_y),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.4,
            color="#888888",
        )
        ax.add_patch(arrow)

# ---------------------------------------------------------------- results
ax_bar = fig.add_axes([0.64, 0.34, 0.33, 0.52])
models = ["Real-only", "v7s synthetic", "Matched synthetic"]
auroc = [0.8611, 0.8713, 0.8849]
auprc = [0.4486, 0.4830, 0.5536]
x = np.arange(len(models))
width = 0.34
colors = ["#9e9e9e", "#5c8dc9", "#2e7d32"]

bars_auroc = ax_bar.bar(
    x - width / 2, auroc, width, label="AUROC", color=colors, alpha=0.95
)
bars_auprc = ax_bar.bar(
    x + width / 2,
    auprc,
    width,
    label="AUPRC",
    color=colors,
    alpha=0.45,
    hatch="//",
    edgecolor="white",
    linewidth=0.0,
)
for bars in (bars_auroc, bars_auprc):
    for bar in bars:
        ax_bar.annotate(
            f"{bar.get_height():.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(models, fontsize=9.5)
ax_bar.set_ylim(0.0, 1.0)
ax_bar.set_ylabel("score", fontsize=9.5)
ax_bar.set_title("Official RGD test — 5-seed mean", fontsize=12, pad=10)
ax_bar.grid(axis="y", color="#dddddd", linewidth=0.7)
ax_bar.set_axisbelow(True)
for spine in ("top", "right"):
    ax_bar.spines[spine].set_visible(False)
ax_bar.legend(
    handles=[
        Patch(facecolor="#777777", label="AUROC (solid)"),
        Patch(facecolor="#bbbbbb", hatch="//", edgecolor="white", label="AUPRC (hatched)"),
    ],
    loc="upper left",
    fontsize=8.5,
    frameon=False,
    ncol=2,
)

fig.text(
    0.64,
    0.285,
    "5-seed ensemble: 0.892 AUROC / 0.607 AUPRC",
    fontsize=10.5,
    fontweight="bold",
    color="#2e7d32",
    ha="left",
)
fig.text(
    0.64,
    0.255,
    "vs real-only: +0.024 AUROC (p=0.017), +0.105 AUPRC (p=0.019)",
    fontsize=9.0,
    color="#333333",
    ha="left",
)
fig.text(
    0.64,
    0.228,
    "vs v7s: +0.014 AUROC (p=0.19), +0.071 AUPRC (p=0.054)",
    fontsize=9.0,
    color="#333333",
    ha="left",
)

# ---------------------------------------------------------------- bottom
fig.text(
    0.035,
    0.175,
    "What made the difference: matching the scene/event regime — not the object class or sensor physics alone.",
    fontsize=10.5,
    fontweight="bold",
    color="#111111",
)
fig.text(
    0.035,
    0.142,
    "v7s already used pedestrian/cyclist targets; the matched set fixes point density, target range, reflector gap, motion and calibrated amplitude.",
    fontsize=9.0,
    color="#444444",
)
fig.text(
    0.035,
    0.112,
    "Backbone/head ablation (v7s, seed 42, AUROC): full low-LR 0.863 · head-only 0.869 · reset-head 0.875 · two-stage 0.891.",
    fontsize=9.0,
    color="#444444",
)
fig.text(
    0.035,
    0.070,
    "At 5% FPR — the filter operating range: recall 46.8% (matched) vs 41.8% (v7s) vs 37.7% (real-only).",
    fontsize=9.5,
    fontweight="bold",
    color="#b71c1c",
)
fig.text(
    0.035,
    0.038,
    "Caveat: five test scenarios; gain is significant vs real-only and on AUPRC vs v7s. Zero-shot and label-scarce transfer do not improve.",
    fontsize=8.5,
    color="#666666",
)

fig.savefig("/home/vaibhav/carla-claude/S2R_OVERVIEW.png", dpi=200, bbox_inches="tight")
fig.savefig("/home/vaibhav/carla-claude/S2R_OVERVIEW.svg", bbox_inches="tight")
print("wrote S2R_OVERVIEW.png and .svg")
