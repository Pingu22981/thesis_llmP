#!/usr/bin/env python3
"""Rewording study: same abstract blocksworld goals, different phrasings, same PDDL truth."""
import argparse, json, requests
from pathlib import Path
from datasets import load_dataset
import planetarium

OLLAMA_URL = "http://localhost:11434/api/chat"
DOMAINS_DIR = Path("/workspace/planetarium/planetarium/domains")

# the phrase the dataset uses, and our rewordings. All mean the same thing.
ORIGINAL = ("invert each individual stack of blocks, such that the block that in each "
            "tower that was originally on the bottom will be on the top")
VARIANTS = {
    "original": ORIGINAL,
    "clean": ("reverse each stack so that the block originally at the bottom ends up on "
              "top and the block originally on top ends up at the bottom"),
    "stepwise": ("for each stack, flip its order completely so the bottom block becomes the "
                 "top block, the next one up becomes second from the top, and so on"),
    "terse": "invert each stack",
}


def extract_pddl(text):
    s = text.find("(define")
    if s == -1: return None
    d = 0
    for i in range(s, len(text)):
        if text[i] == "(": d += 1
        elif text[i] == ")":
            d -= 1
            if d == 0: return text[s:i+1]
    return None


def build_messages(domain_str, examples, query_nl):
    system = ("You convert natural language descriptions of planning problems into PDDL "
              f"problem files for a fixed domain.\n\nThe domain is:\n{domain_str}\n\n"
              "Write only the PDDL problem definition. It must start with (define (problem ...)) "
              "and use only predicates from the domain above. Do not include markdown fences, "
              "comments, or any explanation.")
    msgs = [{"role": "system", "content": system}]
    for ex in examples:
        msgs.append({"role": "user", "content": f"Natural language:\n{ex['natural_language']}"})
        msgs.append({"role": "assistant", "content": ex["problem_pddl"]})
    msgs.append({"role": "user", "content": f"Natural language:\n{query_nl}"})
    return msgs


def query(messages, model):
    last = None
    for _ in range(2):
        try:
            r = requests.post(OLLAMA_URL, json={"model": model, "messages": messages,
                "stream": False, "options": {"temperature": 0, "num_predict": 2048, "num_ctx": 8192}},
                timeout=(10, 120))
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            last = e
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--model", default="llama3.1:8b")
    args = ap.parse_args()

    ds = load_dataset("BatsResearch/planetarium")
    domain_str = (DOMAINS_DIR / "blocksworld.pddl").read_text()

    # abstract blocksworld problems that use the standard invert template
    pool = ds["test"].filter(lambda r: r["domain"] == "blocksworld"
                             and r["goal_is_abstract"] == 1
                             and ORIGINAL in r["natural_language"])
    pool = pool.shuffle(seed=42).select(range(min(args.n, len(pool))))
    ids = set(pool["id"])

    # concrete examples, same as main matrix
    ex_pool = ds["train"].filter(lambda r: r["domain"] == "blocksworld" and r["goal_is_abstract"] == 0).sort("num_objects")
    examples = [e for e in ex_pool if e["id"] not in ids][:args.shots]

    phrase = VARIANTS[args.variant]
    out = Path(f"reword_blocksworld_{args.variant}.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()}
        print(f"resuming, {len(done)} done", flush=True)

    print(f"variant={args.variant}  n={len(pool)}  examples={len(examples)}", flush=True)
    f = out.open("a")
    for i, r in enumerate(pool):
        if r["id"] in done: continue
        nl = r["natural_language"].replace(ORIGINAL, phrase)  # swap only the instruction
        try:
            raw = query(build_messages(domain_str, examples, nl), args.model)
        except Exception as e:
            print(f"[{i+1}] failed: {e}", flush=True); continue
        pred = extract_pddl(raw)
        par = sol = eq = False
        if pred:
            try:
                par, sol, eq = planetarium.evaluate(r["problem_pddl"], pred, domain_str=domain_str, check_solveable=False)
            except Exception as e:
                raw += f"\n[eval error: {e}]"
        f.write(json.dumps({"id": r["id"], "variant": args.variant, "nl_used": nl,
            "parseable": bool(par), "equivalent": bool(eq), "extracted": pred, "raw_response": raw}) + "\n")
        f.flush()
        print(f"[{i+1}/{len(pool)}] parseable={bool(par)} equivalent={bool(eq)}", flush=True)
    f.close()
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    p = sum(x["parseable"] for x in rows); e = sum(x["equivalent"] for x in rows)
    print(f"\n{args.variant}: n={len(rows)} parseable {p/len(rows):.0%} equivalent {e/len(rows):.0%}", flush=True)


if __name__ == "__main__":
    main()
