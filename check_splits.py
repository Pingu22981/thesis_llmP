import json, os, collections

candidates = []
for root in ["/workspace/planetarium", "/workspace/llm_p/data_gen4"]:
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".jsonl") and ("abstract" in fn.lower() or "split" in fn.lower()):
                p = os.path.join(dirpath, fn)
                if "/results/" not in p and "lora_" not in p:
                    candidates.append(p)

candidates = sorted(set(candidates))
print("Candidate files:")
for p in candidates:
    print("  ", p)

def get_type(r):
    for k in ("type", "goal_type", "task", "abstraction"):
        if k in r:
            return r[k]
    return str(r.get("name", ""))

for p in candidates:
    try:
        counts = collections.Counter()
        keys = None
        n = 0
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if keys is None:
                    keys = list(r.keys())
                counts[get_type(r)] += 1
                n += 1
        print("")
        print("===", p, "n=", n, "===")
        print("  keys:", keys)
        for t, c in counts.most_common():
            print("   ", repr(t), c)
    except Exception as e:
        print("")
        print("===", p, "=== ERROR", e)
