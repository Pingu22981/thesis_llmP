import json, re, requests, random
from pathlib import Path
from datasets import load_dataset
import planetarium

ds = load_dataset("BatsResearch/planetarium")
domain = open("/workspace/planetarium/planetarium/domains/blocksworld.pddl").read()

def is_invert(r):
    nl = r["natural_language"].lower()
    return "invert" in nl[nl.find("your goal"):]

def extract_after_think(text):
    if "</think>" in text: text = text.split("</think>")[-1]
    s = text.find("(define")
    if s == -1: return None
    d = 0
    for i in range(s, len(text)):
        if text[i] == "(": d += 1
        elif text[i] == ")":
            d -= 1
            if d == 0: return text[s:i+1]
    return None

def run_condition(items, out_path, label, n=50):
    random.seed(42)
    items = random.sample(items, min(n, len(items)))
    done = set()
    results = []
    if Path(out_path).exists():
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add(r["id"]); results.append(r)
        print(f"resuming {label}: {len(done)} done")
    f = open(out_path, "a")
    eq = par = 0
    for r in results:
        eq += r["equivalent"]; par += r["parseable"]
    for r in items:
        if r["id"] in done: continue
        try:
            raw_msg = requests.post("http://localhost:11434/api/chat", json={
                "model": "deepseek-r1:8b",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Natural language:\n{r['natural_language']}"}
                ],
                "stream": False,
                "options": {"temperature": 0.6, "num_predict": 8192, "num_ctx": 16384}
            }, timeout=(10, 600)).json()["message"]
            resp = raw_msg.get("content","") or raw_msg.get("thinking","")
        except Exception as e:
            print(f"request failed: {e}", flush=True); continue
        pred = extract_after_think(resp); p = e = False
        if pred:
            try:
                p, _, e = planetarium.evaluate(r["problem_pddl"], pred,
                    domain_str=domain,
                    is_placeholder=bool(r["is_placeholder"]),
                    check_solveable=False)
            except: pass
        row = {"id": r["id"], "parseable": bool(p), "equivalent": bool(e)}
        f.write(json.dumps(row) + "\n"); f.flush()
        par += bool(p); eq += bool(e)
        n_done = len(done) + sum(1 for _ in open(out_path) if _.strip())
        print(f"[{n_done}/{len(items)}] parseable={p} equiv={e}", flush=True)
        done.add(r["id"])
    f.close()
    total = len(list(open(out_path)))
    print(f"\n{label}: parseable {par}/{total} equivalent {eq}/{total} ({eq/total:.0%})\n")

system = ("You convert natural language planning problems into PDDL for a fixed domain.\n\n"
          f"The domain is:\n{domain}\n\nOutput a PDDL problem definition starting with "
          "(define (problem ...)) using only predicates from the domain.")

# condition 1: abstract invert (cross-type, matches your original R1 probe)
abstract = [r for r in ds["test"]
            if r["domain"] == "blocksworld" and r["goal_is_abstract"] == 1 and is_invert(r)]
run_condition(abstract, "results/r1_abstract_invert.jsonl",
              "R1-8B abstract invert 0-shot", n=50)

# condition 2: concrete (controlled baseline at same scale)
concrete = [r for r in ds["test"]
            if r["domain"] == "blocksworld" and r["goal_is_abstract"] == 0]
run_condition(concrete, "results/r1_concrete.jsonl",
              "R1-8B concrete 0-shot", n=50)

# condition 3: abstract mixed (Planetarium test split, invert+swap)
abstract_mixed = [r for r in ds["test"]
                  if r["domain"] == "blocksworld" and r["goal_is_abstract"] == 1]
run_condition(abstract_mixed, "results/r1_abstract_mixed.jsonl",
              "R1-8B abstract mixed 0-shot (Planetarium test split)", n=50)

print("ALL DONE")
print("compare: Llama-8B concrete 4-shot 83%, abstract invert ~0%")
print("compare: Sonnet abstract 34%")
