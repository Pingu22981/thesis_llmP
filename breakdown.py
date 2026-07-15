import json, glob
import pandas as pd

for path in sorted(glob.glob("baseline_*.jsonl")):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        continue
    df = pd.DataFrame(rows)
    name = path.replace("baseline_", "").replace(".jsonl", "")
    print(f"\n===== {name}  (n={len(df)}) =====")
    print(f"overall: parseable {df.parseable.mean():.0%}  equivalent {df.equivalent.mean():.0%}")
    df["obj_bin"] = pd.cut(df["num_objects"], [0, 3, 5, 8, 100])
    print("\nby object count:")
    print(df.groupby("obj_bin", observed=False)[["parseable", "equivalent"]].mean().round(2))
    print("\nby goal abstraction:")
    print(df.groupby("goal_is_abstract")[["parseable", "equivalent"]].mean().round(2))
