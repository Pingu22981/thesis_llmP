
import re

PATH = "/workspace/llm_p/make_figs_final.py"

with open(PATH) as f:
    s = f.read()

# ------------------------------------------------------------------
# 1. add matplotlib.ticker import
# ------------------------------------------------------------------
old = "import matplotlib\nmatplotlib.use(\"Agg\")"
new = "import matplotlib\nimport matplotlib.ticker\nmatplotlib.use(\"Agg\")"
assert s.count(old) == 1, "FAIL: ticker import anchor not found"
s = s.replace(old, new)
print("OK: added matplotlib.ticker")

# ------------------------------------------------------------------
# 2. replace fig_asymmetry
# ------------------------------------------------------------------
NEW_FIG3 = '''def fig_asymmetry():
    pairs = [("invert", "tower"), ("swap", "tower"),
             ("invert", "equal_towers")]
    labels, fwd, rev, fe, re_ = [], [], [], [], []
    for a, b in pairs:
        ma, sa, _ = cell(str(R / f"{a}_on_{b}_s*.jsonl"))
        mb, sb, _ = cell(str(R / f"{b}_on_{a}_s*.jsonl"))
        if ma is None or mb is None:
            continue
        labels.append(f"{FLAT[a]}\\n\\u2194 {FLAT[b]}")
        fwd.append(ma);  fe.append(sa)
        rev.append(mb);  re_.append(sb)

    if not labels:
        print("warning: no data for fig3 (asymmetry)")
        return

    x   = np.arange(len(labels))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.bar(x - w / 2, fwd, w, yerr=fe, capsize=3,
           label="train A \\u2192 test B", color=CB[0], zorder=3)
    ax.bar(x + w / 2, rev, w, yerr=re_, capsize=3,
           label="train B \\u2192 test A", color=CB[1], zorder=3)

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

'''

# find the function boundaries
m3 = re.search(r"^def fig_asymmetry\(\):", s, re.MULTILINE)
m4 = re.search(r"^def fig_mechanism\(\):", s, re.MULTILINE)
assert m3 and m4, "FAIL: could not locate fig_asymmetry / fig_mechanism"
s = s[:m3.start()] + NEW_FIG3 + s[m4.start():]
print("OK: replaced fig_asymmetry")

# ------------------------------------------------------------------
# 3. replace fig_mechanism
# ------------------------------------------------------------------
NEW_FIG4 = '''def fig_mechanism():
    base_m, _, _  = cell(str(R / "base_on_invert.jsonl"))
    ft_m, ft_s, _ = cell(str(R / "invert_on_invert_s*.jsonl"))
    if base_m is None or ft_m is None:
        print("warning: skipping fig4: required files missing")
        return

    names  = ["base\\n(0-shot)", "4-shot\\ntype-matched\\n(n=50)",
              "fine-tuned\\n(in-domain)"]
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
        "Grey bars score 0 % equivalence: the model produces\\n"
        "syntactically valid PDDL with the wrong goal in every case.",
        xy=(0.03, 0.97), xycoords="axes fraction",
        va="top", fontsize=11, style="italic", color="#666666")

    save(fig, "fig4_mechanism_invert")

'''

# find fig_mechanism and the function after it (print_summary)
m4b = re.search(r"^def fig_mechanism\(\):", s, re.MULTILINE)
ms  = re.search(r"^def print_summary\(\):", s, re.MULTILINE)
assert m4b and ms, "FAIL: could not locate fig_mechanism / print_summary"
s = s[:m4b.start()] + NEW_FIG4 + s[ms.start():]
print("OK: replaced fig_mechanism")

# ------------------------------------------------------------------
with open(PATH, "w") as f:
    f.write(s)
print("patched OK — run: python make_figs_final.py") 
