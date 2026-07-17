#!/usr/bin/env python3
"""Gripper concrete with detect-and-retry for the dropped at-robby / free-gripper facts."""
import argparse, json, re, requests
from pathlib import Path
from datasets import load_dataset
import planetarium

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


def find_omissions(pred):
    """Return a list of human-readable problems with the generated PDDL."""
    probs = []
    if "(at-robby" not in pred:
        probs.append("you did not state the robot's location; add exactly one (at-robby roomN) fact")
    # grippers that are declared but neither carrying nor free
    grippers = set(re.findall(r"\(gripper (gripper\d+)\)", pred))
    carrying = set(re.findall(r"\(carry \w+ (gripper\d+)\)", pred))
    free = set(re.findall(r"\(free (gripper\d+)\)", pred))
    undefined = grippers - carrying - free
    if undefined:
        probs.append(f"these grippers have no state, mark each free or carrying: {', '.join(sorted(undefined))}")
    # every ball mentioned must be declared with (ball ballN)
    mentioned = set(re.findall(r"\((?:at|carry) (ball\d+)", pred))
    declared = set(re.findall(r"\(ball (ball\d+)\)", pred))
    undeclared = mentioned - declared
    if undeclared:
        probs.append(f"these balls are used but not declared, add (ball ballN) for each: {', '.join(sorted(undeclared))}")
    return probs


def query(messages, model):
    last = None
    for _ in range(2):
        try:
            r = requests.post(OLLAMA_URL, json={"model": model, "messages": messages,
                "stream": False, "options": {"temperature": 0, "num_predict": 2048, "num_ctx": 8192}},
                timeout=(10, 180))
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            last = e
    raise last


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


def score(truth, pred, domain_str):
    if not pred: return False, False
    try:
        par, _, eq = planetarium.evaluate(truth, pred, domain_str=domain_str, check_solveable=False)
        return bool(par), bool(eq)
    except Exception:
        return False, False


def pick_examples(ds, k, exclude):
    pool = ds["train"].filter(lambda r: r["domain"]=="gripper" and r["goal_is_abstract"]==0).sort("num_objects")
    cands = [r for r in pool if r["id"] not in exclude]
    if not cands: return []
    idxs = [int(i*(len(cands)-1)/max(1,k-1)) for i in range(k)]
    return [cands[i] for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ds = load_dataset("BatsResearch/planetarium")
    domain_str = (DOMAINS_DIR / "gripper.pddl").read_text()
    test = ds["test"].filter(lambda r: r["domain"]=="gripper" and r["goal_is_abstract"]==0)
    test = test.shuffle(seed=args.seed).select(range(min(args.n, len(test))))
    ids = set(test["id"])
    examples = pick_examples(ds, args.shots, ids)

    tag = "retry" if args.retry else "noretry"
    out = Path(args.output or f"results/gripper/gripper_concrete_{tag}.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()}
        print(f"resuming, {len(done)} done", flush=True)

    print(f"gripper concrete retry={args.retry} n={len(test)} examples={len(examples)}", flush=True)
    f = out.open("a")
    for i, r in enumerate(test):
        if r["id"] in done: continue
        msgs = build_messages(domain_str, examples, r["natural_language"])
        try:
            raw1 = query(msgs, args.model)
        except Exception as e:
            print(f"[{i+1}] failed: {e}", flush=True); continue
        pred1 = extract_pddl(raw1)
        par1, eq1 = score(r["problem_pddl"], pred1, domain_str)

        pred_final, par_f, eq_f, retried = pred1, par1, eq1, False
        if args.retry and pred1 and not eq1:
            probs = find_omissions(pred1)
            if probs:
                retried = True
                msg = "; ".join(probs)
                msgs2 = msgs + [
                    {"role": "assistant", "content": pred1},
                    {"role": "user", "content":
                     f"That problem file has issues: {msg}. Rewrite the complete PDDL problem file "
                     "with these fixed. Output only the PDDL."},
                ]
                try:
                    raw2 = query(msgs2, args.model)
                    p2 = extract_pddl(raw2)
                    if p2:
                        par_f, eq_f = score(r["problem_pddl"], p2, domain_str)
                        pred_final = p2
                except Exception as e:
                    print(f"[{i+1}] retry failed: {e}", flush=True)

        f.write(json.dumps({
            "id": r["id"], "num_objects": r["num_objects"],
            "parseable_1": par1, "equivalent_1": eq1, "retried": retried,
            "parseable_final": par_f, "equivalent_final": eq_f,
            "extracted_final": pred_final,
        }) + "\n")
        f.flush()
        fix = " FIXED" if (retried and eq_f and not eq1) else ""
        print(f"[{i+1}/{len(test)}] eq1={eq1}{' RETRY' if retried else ''} -> eq_final={eq_f}{fix}", flush=True)
    f.close()

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    n=len(rows); e1=sum(x["equivalent_1"] for x in rows); ef=sum(x["equivalent_final"] for x in rows)
    ret=sum(x["retried"] for x in rows); fix=sum(1 for x in rows if x["retried"] and x["equivalent_final"] and not x["equivalent_1"])
    print(f"\nn={n}  first-pass eq {e1/n:.0%}  retried {ret}  fixed-by-retry {fix}  final eq {ef/n:.0%}", flush=True)


if __name__ == "__main__":
    main()
