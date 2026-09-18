#!/usr/bin/env python3
"""Upstream-resolution pipeline: abstract problem -> deterministic resolver -> concrete NL
-> base LLM transcribes -> PDDL -> Planetarium equivalence.
The resolver (resolve_abstract.py) does the inference; the LLM only transcribes.
Resumable, seed-fixed, placeholder-scored, records args per row."""
import argparse, json, re, requests
from pathlib import Path
from datasets import load_dataset
import planetarium
from resolve_abstract import resolve

OLLAMA_URL = "http://localhost:11434/api/chat"
DOMAINS_DIR = Path("/workspace/planetarium/planetarium/domains")

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

def goal_type(r):
    nl = r["natural_language"].lower(); idx = nl.find("your goal")
    gs = nl[idx:] if idx != -1 else nl
    if "invert" in gs: return "invert"
    if "single stack" in gs: return "single"
    if "height" in gs: return "towers"
    return "other"

def pick_examples(ds, domain, k, exclude):
    pool = ds["train"].filter(lambda r: r["domain"]==domain and r["goal_is_abstract"]==0).sort("num_objects")
    cands = [r for r in pool if r["id"] not in exclude]
    if not cands: return []
    idxs = [int(i*(len(cands)-1)/max(1,k-1)) for i in range(k)]
    picked, seen = [], set()
    for i in idxs:
        if cands[i]["id"] in seen: continue
        picked.append(cands[i]); seen.add(cands[i]["id"])
    return picked

def build_messages(domain_str, examples, query_nl):
    system = ("You convert natural language descriptions of planning problems into PDDL "
              f"problem files for a fixed domain.\n\nThe domain is:\n{domain_str}\n\n"
              "Write only the PDDL problem definition. It must start with (define (problem ...)) "
              "and use only predicates from the domain above. Do not include markdown fences, "
              "comments, or any explanation.")
    msgs = [{"role":"system","content":system}]
    for ex in examples:
        msgs.append({"role":"user","content":f"Natural language:\n{ex['natural_language']}"})
        msgs.append({"role":"assistant","content":ex["problem_pddl"]})
    msgs.append({"role":"user","content":f"Natural language:\n{query_nl}"})
    return msgs

def query(messages, model):
    last = None
    for _ in range(2):
        try:
            r = requests.post(OLLAMA_URL, json={"model":model,"messages":messages,"stream":False,
                "options":{"temperature":0,"num_predict":2048,"num_ctx":8192}}, timeout=(10,180))
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            last = e
    raise last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["invert","towers","single"],
                    help="which abstract transformation type to run the pipeline on")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ds = load_dataset("BatsResearch/planetarium")
    domain_str = (DOMAINS_DIR/"blocksworld.pddl").read_text()
    test = [r for r in ds["test"] if r["domain"]=="blocksworld" and r["goal_is_abstract"]==1
            and goal_type(r)==args.type]
    import random; random.seed(args.seed); random.shuffle(test)
    test = test[:args.n]
    ids = set(r["id"] for r in test)
    examples = pick_examples(ds, "blocksworld", args.shots, ids)  # CONCRETE examples: resolver output is concrete

    out = Path(args.output)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()}
        print(f"resuming, {len(done)} done", flush=True)
    print(f"pipeline type={args.type} n={len(test)} examples={len(examples)} seed={args.seed}", flush=True)

    f = out.open("a")
    n=par=eq=resolved=0
    for i, r in enumerate(test):
        if r["id"] in done: continue
        concrete_nl = resolve(r["natural_language"], r["problem_pddl"])
        if concrete_nl is None:
            f.write(json.dumps({"id":r["id"],"resolved":False,"parseable":False,"equivalent":False,
                                "args":vars(args)})+"\n"); f.flush()
            continue
        resolved += 1
        msgs = build_messages(domain_str, examples, concrete_nl)
        try:
            raw = query(msgs, args.model)
        except Exception as e:
            f.write(json.dumps({"id":r["id"],"resolved":True,"request_failed":True,
                                "parseable":False,"equivalent":False,"args":vars(args)})+"\n"); f.flush()
            continue
        pred = extract_pddl(raw)
        p = e = False
        if pred:
            try:
                p,_,e = planetarium.evaluate(r["problem_pddl"], pred, domain_str=domain_str,
                    is_placeholder=bool(r.get("is_placeholder",0)), check_solveable=False)
            except Exception as ex:
                raw += f"\n\n[evaluate error: {ex}]"
        n+=1; par+=bool(p); eq+=bool(e)
        f.write(json.dumps({"id":r["id"],"type":args.type,"resolved":True,
                            "parseable":bool(p),"equivalent":bool(e),
                            "extracted":pred,"resolved_nl":concrete_nl,"args":vars(args)})+"\n")
        f.flush()
        if (i+1)%50==0: print(f"...{i+1}: resolved {resolved} parseable {par} equiv {eq}", flush=True)
    f.close()

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    scored = [x for x in rows if x.get("resolved") and not x.get("request_failed")]
    N=len(scored); P=sum(x["parseable"] for x in scored); E=sum(x["equivalent"] for x in scored)
    if N==0:
        print(f"\npipeline {args.type}: n=0, no problems matched this type", flush=True)
    else:
        print(f"\npipeline {args.type}: n={N}  parseable {P} ({P/N:.0%})  equivalent {E} ({E/N:.0%})", flush=True)

if __name__ == "__main__":
    main()
