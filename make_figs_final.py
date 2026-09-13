"""
Publication-quality figures for the cross-abstraction-type generalisation matrix.

Reads results/matrix4/*.jsonl at runtime; the only hardcoded number is the
4-shot type-matched probe (0.0%, n = 50), noted in Figure 4.

Outputs four PDF + PNG pairs (300 dpi, bbox_inches="tight") to figs/.

    fig1_matrix_heatmap       4x4 train->test equivalence heatmap + base row
    fig2_indomain_vs_heldout  in-domain vs held-out-size grouped bars
    fig3_transfer_asymmetry   three paired A->B / B->A bars
    fig4_mechanism_invert     base vs 4-shot vs fine-tuned on invert

Usage: python make_figs_final.py
"""

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- style: seaborn whitegrid + Okabe-Ito colourblind palette ----
# Use matplotlib's built-in seaborn-compatible style (no seaborn package required).
plt.style.use("seaborn-v0_8-whitegrid")
# Okabe-Ito colourblind-safe hex codes (the seaborn 'colorblind' palette).
CB = ["#0072B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC"]

TYPES = ["invert", "swap", "tower", "equal_towers"]
LABEL = {"invert": "invert", "swap": "swap", "tower": "tower",
         "equal_towers": "equal\ntowers"}
FLAT  = {"invert": "invert", "swap": "swap", "tower": "tower",
         "equal_towers": "equal towers"}

R   = Path("results/matrix4_v2")
OUT = Path("figs")
OUT.mkdir(exist_ok=True)

# ---- typography (min 12 pt, axes 13 pt, titles 13 pt bold) -------
plt.rcParams.update({
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
})

# ===================================================================
#  Data helpers
# ===================================================================

def load_rows(path: str):
    """Return list of json dicts, or None if file missing."""
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return None
    return rows


def rate(path: str):
    """Equivalence % for one jsonl file, or None if the file is missing/empty."""
    rows = load_rows(path)
    if rows is None:
        print(f"warning: missing file {path}")
        return None
    if not rows:
        print(f"warning: empty file {path}")
        return None
    eq = [bool(r.get("equivalent")) for r in rows]
    return 100.0 * sum(eq) / len(eq)


def cell(pattern: str):
    """(mean, sd, n_seeds) across files matching pattern.
    Returns (None, None, 0) when nothing matches; prints a warning."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"warning: no files match {pattern}")
        return None, None, 0
    vals = [v for v in (rate(f) for f in files) if v is not None]
    if not vals:
        print(f"warning: no rates computed for {pattern}")
        return None, None, 0
    mean = st.mean(vals)
    sd   = st.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd, len(vals)


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/{name}.pdf and figs/{name}.png")


def finish_bars(ax, xs_list, vals_list, errs_list):
    """Set y-limits with headroom and annotate each bar with its value."""
    top = max((v + e
               for vals, errs in zip(vals_list, errs_list)
               for v, e in zip(vals, errs)), default=0.0)
    ax.set_ylim(0, top * 1.28 if top > 0 else 1.0)
    ymax = ax.get_ylim()[1]
    for xs, vals, errs in zip(xs_list, vals_list, errs_list):
        for x, v, e in zip(xs, vals, errs):
            ax.text(x, v + e + 0.02 * ymax, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=12)


# ===================================================================
#  Figure 1: 4×4 matrix heatmap + base row
# ===================================================================

def fig_matrix():
    M = np.full((len(TYPES) + 1, len(TYPES)), np.nan)
    S = np.zeros_like(M)

    for i, tr in enumerate(TYPES):
        for j, te in enumerate(TYPES):
            m, s, _ = cell(str(R / f"{tr}_on_{te}_s*.jsonl"))
            if m is not None:
                M[i, j], S[i, j] = m, s

    for j, te in enumerate(TYPES):
        m, s, _ = cell(str(R / f"base_on_{te}.jsonl"))
        if m is not None:
            M[len(TYPES), j], S[len(TYPES), j] = m, s

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.grid(False)  # remove whitegrid dashes through cells

    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels([LABEL[t] for t in TYPES])
    ax.set_yticks(range(len(TYPES) + 1))
    ax.set_yticklabels([LABEL[t] for t in TYPES] + ["base\n(no FT)"])
    ax.set_xlabel("Evaluated on (test type)")
    ax.set_ylabel("Fine-tuned on (train type)")
    ax.set_title("Goal equivalence (%), mean over seeds")

    # white cell separators + emphasised base-model divider
    for x in range(1, len(TYPES)):
        ax.axvline(x - 0.5, color="white", lw=1.2, zorder=3)
    for y in range(1, len(TYPES) + 1):
        ax.axhline(y - 0.5, color="white", lw=1.2, zorder=3)
    ax.axhline(len(TYPES) - 0.5, color="white", lw=3.0, zorder=3)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:.1f}±{S[i, j]:.1f}",
                    ha="center", va="center", fontsize=12,
                    color="white" if M[i, j] < 55 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Equivalence (%)")
    save(fig, "fig1_matrix_heatmap")


# ===================================================================
#  Figure 2: in-domain vs held-out size
# ===================================================================

def fig_heldsize():
    labels, ind, inde, held, helde = [], [], [], [], []
    for t in TYPES:
        m1, s1, _ = cell(str(R / f"{t}_on_{t}_s*.jsonl"))
        m2, s2, _ = cell(str(R / f"{t}_heldsize_s*.jsonl"))
        if m1 is None or m2 is None:
            continue
        labels.append(LABEL[t])
        ind.append(m1);   inde.append(s1)
        held.append(m2);  helde.append(s2)

    if not labels:
        print("warning: no data for fig2 (in-domain vs held-out)")
        return

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - w / 2, ind, w, yerr=inde, capsize=3,
           label="in-domain (seen sizes)", color=CB[0])
    ax.bar(x + w / 2, held, w, yerr=helde, capsize=3,
           label="held-out sizes", color=CB[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Equivalence (%)")
    ax.set_title("In-domain vs held-out-size performance")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    finish_bars(ax, [x - w / 2, x + w / 2], [ind, held], [inde, helde])
    save(fig, "fig2_indomain_vs_heldout")


# ===================================================================
#  Figure 3: transfer asymmetry
# ===================================================================

def fig_asymmetry():
    pairs = [("invert", "tower"), ("swap", "tower"),
             ("invert", "equal_towers")]
    labels, fwd, rev, fe, re_ = [], [], [], [], []
    for a, b in pairs:
        ma, sa, _ = cell(str(R / f"{a}_on_{b}_s*.jsonl"))
        mb, sb, _ = cell(str(R / f"{b}_on_{a}_s*.jsonl"))
        if ma is None or mb is None:
            continue
        labels.append(f"{FLAT[a]}\n\u2194 {FLAT[b]}")
        fwd.append(ma);  fe.append(sa)
        rev.append(mb);  re_.append(sb)

    if not labels:
        print("warning: no data for fig3 (asymmetry)")
        return

    x   = np.arange(len(labels))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.bar(x - w / 2, fwd, w, yerr=fe, capsize=3,
           label="train A \u2192 test B", color=CB[0], zorder=3)
    ax.bar(x + w / 2, rev, w, yerr=re_, capsize=3,
           label="train B \u2192 test A", color=CB[1], zorder=3)

    # symlog: linear 0-0.5 so zero bars render, log above
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_ylim(bottom=0, top=40)
    ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20, 30])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{v:.1f}" if v < 1 else f"{int(round(v))}"))
    ax.set_ylabel("Equivalence (%, symlog scale)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Cross-type transfer is asymmetric")
    ax.legend(frameon=False)

    # annotate each bar (handle near-zero carefully on log axis)
    for xi, vals, errs in [(x - w / 2, fwd, fe), (x + w / 2, rev, re_)]:
        for px, v, e in zip(xi, vals, errs):
            top = v + e
            if top <= 0.05:
                y_ann = 0.55   # just above zero in linear region
            elif top < 0.5:
                y_ann = top * 1.25
            else:
                y_ann = min(top * 1.18, 33)
            ax.text(px, y_ann, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=11, zorder=4)

    save(fig, "fig3_transfer_asymmetry")

def fig_mechanism():
    base_m, _, _  = cell(str(R / "base_on_invert.jsonl"))
    ft_m, ft_s, _ = cell(str(R / "invert_on_invert_s*.jsonl"))
    if base_m is None or ft_m is None:
        print("warning: skipping fig4: required files missing")
        return

    names  = ["base\n(0-shot)", "4-shot\ntype-matched\n(n=50)",
              "fine-tuned\n(in-domain)"]
    vals   = [base_m, 0.0, ft_m]
    errs   = [0.0,    0.0, ft_s]
    # grey for zero conditions, blue for fine-tuned
    colors = ["#aaaaaa", "#aaaaaa", CB[0]]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.bar(names, vals, yerr=errs, capsize=3, color=colors,
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_ylabel("Equivalence (%)")
    ax.set_title("Invert goals: in-context learning vs weight updates")
    ax.set_ylim(0, ft_m * 1.3)

    ymax = ax.get_ylim()[1]
    for i, (v, e) in enumerate(zip(vals, errs)):
        y_ann = v + e + 0.03 * ymax
        ax.text(i, max(y_ann, 0.025 * ymax), f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12,
                color="#777777" if v == 0 else "black")

    ax.annotate(
        "Grey bars score 0 % equivalence: the model produces\n"
        "syntactically valid PDDL with the wrong goal in every case.",
        xy=(0.03, 0.97), xycoords="axes fraction",
        va="top", fontsize=11, style="italic", color="#666666")

    save(fig, "fig4_mechanism_invert")

def print_summary():
    print("\ncomputed equivalence % (mean±sd over seeds):")
    header = ("train\\test".ljust(14) + " | "
              + " | ".join(t.rjust(12) for t in TYPES)
              + " | " + "heldsize".rjust(12))
    print(header)
    for tr in TYPES:
        cells = []
        for te in TYPES:
            m, s, _ = cell(str(R / f"{tr}_on_{te}_s*.jsonl"))
            cells.append("    ---     " if m is None
                         else f"{m:6.1f}±{s:<4.1f}")
        m, s, _ = cell(str(R / f"{tr}_heldsize_s*.jsonl"))
        cells.append("    ---     " if m is None
                     else f"{m:6.1f}±{s:<4.1f}")
        print(tr.ljust(14) + " | " + " | ".join(cells))
    base_cells = []
    for te in TYPES:
        m, s, _ = cell(str(R / f"base_on_{te}.jsonl"))
        base_cells.append("    ---     " if m is None
                          else f"{m:6.1f}±{s:<4.1f}")
    base_cells.append("    ---     ")
    print("base".ljust(14) + " | " + " | ".join(base_cells))


def main():
    print(f"reading results from {R}/")
    fig_matrix()
    fig_heldsize()

    fig_mechanism()
    print_summary()
    print("\ndone -> figs/")


if __name__ == "__main__":
    main()

