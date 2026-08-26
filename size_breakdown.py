#!/usr/bin/env python3
"""Committed source for the size-band figure. Reads only saved result files."""
import json, re, sys
from collections import defaultdict

def band(n):
    return "1-5" if n <= 5 else "6-10" if n <= 10 else "11-15" if n <= 15 else "16+"

def report(path, label, nfield="num_objects", nre=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    b = defaultdict(lambda: [0, 0])
    for r in rows:
        n = r.get(nfield)
        if n is None and nre:
            m = re.search(nre, r.get("resolved_nl", ""))
            n = int(m.group(1)) if m else None
        if n is None:
            continue
        e = r.get("equivalent", r.get("equivalent_final"))
        k = band(n)
        b[k][0] += 1
        b[k][1] += int(bool(e))
    print(f"\n{label}\n  {path}  n={len(rows)}")
    for k in ["1-5", "6-10", "11-15", "16+"]:
        if k in b:
            n, e = b[k]
            print(f"  {k:<7} n={n:<4} equiv {e}/{n} ({100*e/n:.1f}%)")

report("results/spread_fix/spread_bw_concrete_500.jsonl",
       "Concrete blocksworld, spread examples")
report("results/finetuned/pipeline_invert_v2.jsonl",
       "Resolver pipeline, invert", nre=r"You have (\d+) blocks")
report("results/gripper/gripper_retry_heuristic.jsonl",
       "Gripper concrete, after retry")
