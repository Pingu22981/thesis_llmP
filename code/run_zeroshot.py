#!/usr/bin/env python3
"""One-shot NL->PDDL baseline on Planetarium via Ollama.
Writes each result to a .jsonl as it goes, so it is crash-safe and resumable."""
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
        print(f"Domain file not at {path}. Files present:")
        for p in sorted(DOMAINS_DIR.glob("*")):
            print("  ", p.name)
        sys.exit(1)
    return path.read_text()


def pick_example(ds, domain):
    split = "train" if "train" in ds else "test"
    pool = ds[split].filter(lambda r: r["domain"] == domain)
    cand = pool.select(range(min(200, len(pool))))
    return min(cand, key=lambda r: r.get("num_objects", 99))


def build_messages(domain_str, ex_nl, ex_pddl, query_nl):
    system = (
        "You convert natural language descriptions of planning problems into "
        "PDDL problem files for a fixed domain.\n\n"
        f"The domain is:\n{domain_str}\n\n"
        "Write only the PDDL problem definition. It must start with "
        "(define (problem ...)) and use only predicates from the domain above. "
        "Do not include markdown fences, comments, or any explanation."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Natural language:\n{query_nl}"},
    ]


def query_ollama(messages, model):
    last = None
    for attempt in range(2):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "num_predict": 2048, "num_ctx": 8192},
            }, timeout=(10, 120))
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            last = e
            continue
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="blocksworld", choices=["blocksworld", "gripper"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_path = Path(args.output or f"baseline_{args.domain}.jsonl")

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"resuming, {len(done)} already done", flush=True)

    ds = load_dataset("BatsResearch/planetarium")
    domain_str = load_domain(args.domain)
    ex = pick_example(ds, args.domain)

    test = ds["test"].filter(
        lambda r: r["domain"] == args.domain and r["id"] != ex["id"]
    ).select(range(args.n))

    f = out_path.open("a")
    for i, r in enumerate(test):
        if r["id"] in done:
            continue
        msgs = build_messages(domain_str, ex["natural_language"], ex["problem_pddl"], r["natural_language"])
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
        row = {
            "id": r["id"], "num_objects": r.get("num_objects"),
            "init_is_abstract": r.get("init_is_abstract"), "goal_is_abstract": r.get("goal_is_abstract"),
            "extracted_ok": pred is not None,
            "parseable": bool(parseable), "solveable": bool(solveable), "equivalent": bool(equivalent),
            "raw_response": raw, "extracted": pred,
        }
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"[{i+1}/{len(test)}] parseable={bool(parseable)} equivalent={bool(equivalent)}", flush=True)
    f.close()

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    p = sum(x["parseable"] for x in rows)
    e = sum(x["equivalent"] for x in rows)
    print(f"\n{args.domain} TOTAL in {out_path.name}: {len(rows)} items, "
          f"parseable {p}/{len(rows)} ({p/len(rows):.0%}), "
          f"equivalent {e}/{len(rows)} ({e/len(rows):.0%})", flush=True)


if __name__ == "__main__":
    main()
