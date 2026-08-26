#!/usr/bin/env python3
"""Figures for the cross-abstraction-type generalisation matrix.
Reads results/matrix4/*.jsonl and invert_matched_probe.jsonl directly.
No hardcoded numbers. Outputs PDF (LaTeX) + PNG (slides) to figs/.
Usage: python make_matrix_figs.py"""
import json, glob, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPES = ["invert", "swap", "tower", "equal_towers"]
LABEL = {"invert": "invert", "swap": "swap", "tower": "tower",
         "equal_towers": "equal\ntowers"}
R = Path("results/matrix4")
OUT = Path("figs"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def rate(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        raise ValueError(f"empty file: {path}")
    return 100.0 * sum(r["equivalent"] for r in rows) / len(rows)


def cell(pattern):
    """mean, sd, n_seeds over files matching pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None, 0
    v = [rate(f) for f in files]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/{name}.pdf and .png")


# ---------- Figure 1: 4x4 matrix heatmap + base row ----------
def fig_matrix():
    M = np.full((len(TYPES) + 1, len(TYPES)), np.nan)
    S = np.zeros_like(M)
    for i, tr in enumerate(TYPES):
        for j, te in enumerate(TYPES):
            m, s, n = cell(str(R / f"{tr}_on_{te}_s*.jsonl"))
            if m is not None:
                M[i, j], S[i, j] = m, s
    for j, te in enumerate(TYPES):
        m, s, n = cell(str(R / f"base_on_{te}.jsonl"))
        if m is not None:
            M[len(TYPES), j], S[len(TYPES), j] = m, s

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels([LABEL[t] for t in TYPES])
    ax.set_yticks(range(len(TYPES) + 1))
    ax.set_yticklabels([LABEL[t] for t in TYPES] + ["base\n(no FT)"])
    ax.set_xlabel("evaluated on (test type)")
    ax.set_ylabel("fine-tuned on (train type)")
    ax.set_title("Goal equivalence (%), mean over seeds")
    # separate the base control row
    ax.axhline(len(TYPES) - 0.5, color="white", lw=2.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            txt = f"{M[i, j]:.1f}" + (f"\n±{S[i, j]:.1f}" if S[i, j] > 0 else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] < 55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="equivalence (%)")
    save(fig, "fig_matrix_heatmap")


# ---------- Figure 2: transfer asymmetry ----------
def fig_asymmetry():
    pairs = [("invert", "tower"), ("swap", "tower"), ("invert", "equal_towers")]
    labels, fwd, rev, fe, re_ = [], [], [], [], []
    for a, b in pairs:
        ma, sa, _ = cell(str(R / f"{a}_on_{b}_s*.jsonl"))
        mb, sb, _ = cell(str(R / f"{b}_on_{a}_s*.jsonl"))
        if ma is None or mb is None:
            continue
        labels.append(f"{a}\n{b}")
        fwd.append(ma); fe.append(sa)
        rev.append(mb); re_.append(sb)

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(x - w/2, fwd, w, yerr=fe, capsize=3, label="A → B", color="#3b7dd8")
    ax.bar(x + w/2, rev, w, yerr=re_, capsize=3, label="B → A", color="#d8843b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"A={l.splitlines()[0]}\nB={l.splitlines()[1]}" for l in labels])
    ax.set_ylabel("equivalence (%)")
    ax.set_title("Cross-type transfer is asymmetric")
    ax.legend(frameon=False)
    for xi, v in zip(x - w/2, fwd):
        ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, rev):
        ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=8)
    save(fig, "fig_transfer_asymmetry")


# ---------- Figure 3: in-domain vs held-out size ----------
def fig_heldsize():
    ind, inde, held, helde, labels = [], [], [], [], []
    for t in TYPES:
        m1, s1, _ = cell(str(R / f"{t}_on_{t}_s*.jsonl"))
        m2, s2, _ = cell(str(R / f"{t}_heldsize_s*.jsonl"))
        if m1 is None or m2 is None:
            continue
        labels.append(LABEL[t])
        ind.append(m1); inde.append(s1)
        held.append(m2); helde.append(s2)

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(x - w/2, ind, w, yerr=inde, capsize=3,
           label="in-domain (seen sizes)", color="#3b7dd8")
    ax.bar(x + w/2, held, w, yerr=helde, capsize=3,
           label="held-out sizes", color="#7bb0a0")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("equivalence (%)")
    ax.set_title("In-domain performance partly survives unseen problem sizes")
    ax.legend(frameon=False)
    save(fig, "fig_heldout_size")


# ---------- Figure 4: mechanism comparison on invert ----------
def fig_mechanism():
    base_m, _, _ = cell(str(R / "base_on_invert.jsonl"))
    ft_m, ft_s, _ = cell(str(R / "invert_on_invert_s*.jsonl"))
    probe = Path("invert_matched_probe.jsonl")
    if not probe.exists():
        print("skip fig_mechanism: invert_matched_probe.jsonl missing")
        return
    pr_m = rate(probe)

    names = ["base\n(0-shot)", "4-shot,\ntype-matched", "fine-tuned\n(500 ex.)"]
    vals = [base_m, pr_m, ft_m]
    errs = [0, 0, ft_s]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(names, vals, yerr=errs, capsize=3,
           color=["#999999", "#d8843b", "#3b7dd8"])
    ax.set_ylabel("equivalence (%)")
    ax.set_title("Invert goals: in-context learning vs weight updates")
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=9)
    save(fig, "fig_mechanism_invert")


if __name__ == "__main__":
    fig_matrix()
    fig_asymmetry()
    fig_heldsize()
    fig_mechanism()
    print("\ndone -> figs/")
