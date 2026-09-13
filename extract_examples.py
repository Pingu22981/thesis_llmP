#!/usr/bin/env python3
"""Pull one short, representative natural language example per goal type."""
import json
from pathlib import Path

TYPES = ["invert", "swap", "tower", "equal_towers"]

def shortest(path, k=3):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=lambda r: len(r["natural_language"]))
    return rows[:k]

print("=" * 70)
print("ABSTRACT TYPES (from generated data)")
print("=" * 70)
for t in TYPES:
    p = Path(f"data_gen4/{t}_test.jsonl")
    if not p.exists():
        print(f"\n[{t}] missing {p}")
        continue
    print(f"\n--- {t} ---")
    for r in shortest(p):
        print(f"  objects={r.get('num_objects','?'):>3}  {r['natural_language']}")

print("\n" + "=" * 70)
print("CONCRETE (from Planetarium, for contrast)")
print("=" * 70)
try:
    from datasets import load_dataset
    ds = load_dataset("BatsResearch/planetarium")
    rows = [r for r in ds["test"]
            if r["domain"] == "blocksworld" and r["goal_is_abstract"] == 0]
    rows.sort(key=lambda r: len(r["natural_language"]))
    for r in rows[:3]:
        print(f"  objects={r['num_objects']:>3}  {r['natural_language']}")
except Exception as e:
    print(f"  could not load Planetarium: {e}")
