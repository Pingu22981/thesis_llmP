#!/usr/bin/env python3
"""Corrected equal_towers generator.
For n blocks and T target towers (T | n, tower height n/T >= 2), emits a
real goal of exactly T towers of height n/T. Initial state is a random
partition of n, matching Planetarium's tower_to_equal_towers style."""
import argparse, json, random

IS_PLACEHOLDER = 0   # corrected ground truth is a REAL goal, not a placeholder

def divisors(n):
    return [d for d in range(2, n // 2 + 1) if n % d == 0]

def random_composition(n, k, rng):
    cuts = sorted(rng.sample(range(1, n), k - 1))
    prev = 0
    parts = []
    for c in cuts + [n]:
        parts.append(c - prev)
        prev = c
    return parts

def state_predicates(heights):
    preds = ["(arm-empty)"]
    b = 1
    for h in heights:
        preds.append(f"(on-table b{b})")
        for _ in range(h - 1):
            preds.append(f"(on b{b+1} b{b})")
            b += 1
        preds.append(f"(clear b{b})")
        b += 1
    return preds

def make_problem(n, init_heights, T):
    h = n // T
    init = state_predicates(init_heights)
    goal = state_predicates([h] * T)
    objects = " ".join(f"b{i}" for i in range(1, n + 1))
    hkey = "_".join(str(x) for x in init_heights)
    pname = f"tower_to_equal_towers_{hkey}"
    pddl = (
        f"(define (problem {pname})\n"
        f"    (:domain blocksworld)\n"
        f"    (:requirements :strips)\n"
        f"    (:objects {objects})\n"
        f"    (:init " + "\n    ".join(init) + ")\n"
        f"    (:goal (and " + "\n    ".join(goal) + ")))\n"
    )
    nl = (f"You have {n} blocks, stacked into {len(init_heights)} towers of heights "
          f"{', '.join(str(x) for x in init_heights)}, and your arm is empty.\n"
          f"Your goal is to stack the blocks into {T} towers of equal heights.")
    return {
        "name": f"blocksworld_tower_to_equal_towers_blocks_list_{hkey}",
        "problem_pddl": pddl,
        "natural_language": nl,
        "is_placeholder": IS_PLACEHOLDER,
        "num_objects": n,
        "type": "equal_towers",
        "goal_is_abstract": 1,
    }

def generate(count, n_lo, n_hi, seed):
    rng = random.Random(seed)
    cands = [(n, T) for n in range(n_lo, n_hi + 1)
             for T in divisors(n) if n // T >= 2]
    out, seen = [], set()
    while len(out) < count:
        n, T = rng.choice(cands)
        h = n // T
        k = rng.randint(2, min(n, 8))          # 2..8 towers in the initial state
        init = sorted(random_composition(n, k, rng))
        if tuple(init) == tuple([h] * T):       # skip already-solved state
            continue
        key = (n, T, tuple(init))
        if key in seen:
            continue
        seen.add(key)
        out.append(make_problem(n, init, T))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--lo", type=int, default=4)
    ap.add_argument("--hi", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = generate(a.count, a.lo, a.hi, a.seed)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {a.out}")
