#!/usr/bin/env python3
"""CoT training data for abstract blocksworld.
Reasoning is derived FROM the ground-truth goal so it always matches the target."""
import json, re, argparse, random
from pathlib import Path
from datasets import load_dataset

DOMAINS = Path("/workspace/planetarium/planetarium/domains")

def parse_goal_towers(pddl):
    """Read the goal, reconstruct each tower bottom->top from on-table + on relations."""
    g = pddl[pddl.find("(:goal"):]
    ontable = set(re.findall(r"\(on-table (b\d+)\)", g))
    on = re.findall(r"\(on (b\d+) (b\d+)\)", g)   # (upper, lower): upper on lower
    up_of = {lower: upper for upper, lower in on}  # what's on top of each block
    towers = []
    for base in sorted(ontable, key=lambda b: int(b[1:])):
        tower = [base]
        cur = base
        while cur in up_of:
            cur = up_of[cur]
            tower.append(cur)
        towers.append(tower)   # bottom -> top
    return towers

def describe(towers):
    lines = [f"The goal has {len(towers)} tower(s)."]
    for i, t in enumerate(towers, 1):
        if len(t) == 1:
            lines.append(f"Tower {i}: {t[0]} on the table, clear.")
        else:
            steps = ", ".join(f"{u} on {l}" for l, u in zip(t, t[1:]))
            lines.append(f"Tower {i} (height {len(t)}): {t[0]} on the table, then {steps}; {t[-1]} is clear.")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--output", default="train_cot.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("BatsResearch/planetarium")
    domain_str = (DOMAINS/"blocksworld.pddl").read_text()
    sub = ds["train"].filter(lambda r: r["domain"]=="blocksworld" and r["goal_is_abstract"]==1)

    system = ("You convert natural language planning problems into PDDL for a fixed domain.\n\n"
              f"The domain is:\n{domain_str}\n\n"
              "First reason step by step about the goal configuration, then output the PDDL "
              "problem definition starting with (define (problem ...)). Use only predicates from the domain.")

    examples = []
    for r in sub:
        nl = r["natural_language"].lower()
        # only single_stack and towers-of-heights, the deterministic types
        if not ("single stack" in nl or "towers of height" in nl):
            continue
        towers = parse_goal_towers(r["problem_pddl"])
        if not towers:
            continue
        reasoning = describe(towers)
        pddl = r["problem_pddl"].strip()
        assistant = f"{reasoning}\n\nPDDL:\n{pddl}"
        examples.append({"messages":[
            {"role":"system","content":system},
            {"role":"user","content":f"Natural language:\n{r['natural_language']}"},
            {"role":"assistant","content":assistant}]})
        if len(examples) >= args.n:
            break

    random.Random(args.seed).shuffle(examples)
    with open(args.output,"w") as f:
        for e in examples: f.write(json.dumps(e)+"\n")
    print(f"wrote {len(examples)} CoT examples -> {args.output}")

if __name__=="__main__":
    main()
