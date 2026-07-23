#!/usr/bin/env python3
"""Format Planetarium blocksworld pairs into chat-format training examples for QLoRA.
Matches the inference prompt structure so train and eval are consistent."""
import json, argparse
from pathlib import Path
from datasets import load_dataset

DOMAINS_DIR = Path("/workspace/planetarium/planetarium/domains")

def build_example(domain_str, nl, pddl):
    """One training example as messages, same shape as inference (system + user + assistant)."""
    system = ("You convert natural language descriptions of planning problems into "
              "PDDL problem files for a fixed domain.\n\n"
              f"The domain is:\n{domain_str}\n\n"
              "Write only the PDDL problem definition. It must start with "
              "(define (problem ...)) and use only predicates from the domain above. "
              "Do not include markdown fences, comments, or any explanation.")
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Natural language:\n{nl}"},
        {"role": "assistant", "content": pddl},
    ]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="blocksworld")
    ap.add_argument("--n-abstract", type=int, default=1400)
    ap.add_argument("--n-concrete", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="train_data.jsonl")
    args = ap.parse_args()

    ds = load_dataset("BatsResearch/planetarium")
    domain_str = (DOMAINS_DIR / f"{args.domain}.pddl").read_text()
    train = ds["train"].filter(lambda r: r["domain"] == args.domain)

    abstract = train.filter(lambda r: r["goal_is_abstract"] == 1).shuffle(seed=args.seed).select(range(args.n_abstract))
    concrete = train.filter(lambda r: r["goal_is_abstract"] == 0).shuffle(seed=args.seed).select(range(args.n_concrete))

    examples = []
    for r in abstract:
        examples.append(build_example(domain_str, r["natural_language"], r["problem_pddl"]))
    for r in concrete:
        examples.append(build_example(domain_str, r["natural_language"], r["problem_pddl"]))

    import random
    random.Random(args.seed).shuffle(examples)

    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"wrote {len(examples)} examples ({args.n_abstract} abstract, {args.n_concrete} concrete) -> {args.output}")

if __name__ == "__main__":
    main()
