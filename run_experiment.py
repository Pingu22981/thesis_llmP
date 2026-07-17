#!/usr/bin/env python3
"""k-shot NL->PDDL on Planetarium via Ollama, split by goal abstraction.
Crash-safe and resumable (one jsonl line per item, skips done ids)."""
import argparse, json, sys, requests
from pathlib import Path
from datasets import load_dataset
import planetarium

OLLAMA_URL = "http://localhost:11434/api/chat"
DOMAINS_DIR = Path("/workspace/planetarium/planetarium/domains")


def extract_pddl(text):
    start = text.find("(define")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def load_domain(domain):
    path = DOMAINS_DIR / f"{domain}.pddl"
    if not path.exists():
        print(f"Domain not at {path}. Present:", [p.name for p in DOMAINS_DIR.glob("*")])
        sys.exit(1)
    return path.read_text()


def pick_examples(ds, domain, k, exclude_ids, example_abstract=0):
    """k fixed examples from train, smallest by object count, no overlap with test slice.
    example_abstract selects concrete (0) or abstract (1) worked examples."""
    if k == 0:
        return []
    split = "train" if "train" in ds else "test"
    pool = ds[split].filter(lambda r: r["domain"] == domain and r["goal_is_abstract"] == example_abstract)
    pool = pool.sort("num_objects")
    cands = [r for r in pool if r["id"] not in exclude_ids]
    if not cands:
        return []
    # spread examples across the size range rather than taking the k smallest,
    # so the model sees the clear/on convention demonstrated on tall stacks too
    idxs = [int(i * (len(cands) - 1) / max(1, k - 1)) for i in range(k)]
    picked, seen = [], set()
    for i in idxs:
        r = cands[i]
        if r["id"] in seen:
            continue
        picked.append(r); seen.add(r["id"])
    return picked


def build_messages(domain_str, examples, query_nl):
    system = (
        "You convert natural language descriptions of planning problems into "
        "PDDL problem files for a fixed domain.\n\n"
        f"The domain is:\n{domain_str}\n\n"
        "Write only the PDDL problem definition. It must start with "
        "(define (problem ...)) and use only predicates from the domain above. "
        "Do not include markdown fences, comments, or any explanation. "
        "Always state the robot's location with exactly one (at-robby ...) fact."
    )
    msgs = [{"role": "system", "content": system}]
    for ex in examples:
        msgs.append({"role": "user", "content": f"Natural language:\n{ex['natural_language']}"})
        msgs.append({"role": "assistant", "content": ex["problem_pddl"]})
    msgs.append({"role": "user", "content": f"Natural language:\n{query_nl}"})
    return msgs


def query_ollama(messages, model):
    last = None
    for _ in range(2):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "num_predict": 2048, "num_ctx": 8192},
            }, timeout=(10, 120))
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            last = e
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["blocksworld", "gripper"])
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--goal-type", required=True, choices=["abstract", "concrete", "all"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--match-examples", action="store_true",
                    help="use worked examples matching the query goal type")
    ap.add_argument("--num-objects", type=int, default=None,
                    help="restrict test items to exactly this many objects")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ds = load_dataset("BatsResearch/planetarium")
    domain_str = load_domain(args.domain)

    test = ds["test"].filter(lambda r: r["domain"] == args.domain)
    if args.goal_type == "abstract":
        test = test.filter(lambda r: r["goal_is_abstract"] == 1)
    elif args.goal_type == "concrete":
        test = test.filter(lambda r: r["goal_is_abstract"] == 0)
    if args.num_objects is not None:
        test = test.filter(lambda r: r["num_objects"] == args.num_objects)
    test = test.shuffle(seed=args.seed).select(range(min(args.n, len(test))))
    test_ids = set(test["id"])

    ex_abstract = 1 if (args.match_examples and args.goal_type == "abstract") else 0
    examples = pick_examples(ds, args.domain, args.shots, exclude_ids=test_ids, example_abstract=ex_abstract)
    print(f"{args.domain} {args.shots}-shot {args.goal_type}: {len(test)} items, {len(examples)} examples in prompt", flush=True)

    out_path = Path(args.output or f"{args.domain}_{args.shots}shot_{args.goal_type}.jsonl")
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"resuming, {len(done)} already done", flush=True)

    f = out_path.open("a")
    for i, r in enumerate(test):
        if r["id"] in done:
            continue
        msgs = build_messages(domain_str, examples, r["natural_language"])
        try:
            raw = query_ollama(msgs, args.model)
        except Exception as e:
            print(f"[{i+1}/{len(test)}] request failed: {e}", flush=True)
            continue
        pred = extract_pddl(raw)
        parseable = solveable = equivalent = False
        if pred:
            try:
                parseable, solveable, equivalent = planetarium.evaluate(
                    r["problem_pddl"], pred, domain_str=domain_str, check_solveable=False)
            except Exception as e:
                raw += f"\n\n[evaluate error: {e}]"
        f.write(json.dumps({
            "id": r["id"], "num_objects": r.get("num_objects"),
            "goal_is_abstract": r.get("goal_is_abstract"),
            "goal_num_propositions": r.get("goal_num_propositions"),
            "extracted_ok": pred is not None,
            "parseable": bool(parseable), "solveable": bool(solveable), "equivalent": bool(equivalent),
            "raw_response": raw, "extracted": pred,
        }) + "\n")
        f.flush()
        print(f"[{i+1}/{len(test)}] parseable={bool(parseable)} equivalent={bool(equivalent)}", flush=True)
    f.close()

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    p = sum(x["parseable"] for x in rows); e = sum(x["equivalent"] for x in rows)
    print(f"\n{out_path.name}: n={len(rows)}  parseable {p}/{len(rows)} ({p/len(rows):.0%})  "
          f"equivalent {e}/{len(rows)} ({e/len(rows):.0%})", flush=True)


if __name__ == "__main__":
    main()
