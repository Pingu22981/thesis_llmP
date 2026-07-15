import json, re
def stats(path):
    try:
        rows=[json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return None
    if not rows: return None
    def goal(t):
        i=t.find("(:goal"); return t[i:] if i!=-1 else ""
    c=0
    for r in rows:
        ext = r.get("extracted") or r.get("extracted_1")
        if not ext: continue
        g=goal(ext)
        cl=set(re.findall(r"\(clear (b\d+)\)",g)); cov={y for x,y in re.findall(r"\(on (b\d+) (b\d+)\)",g)}
        if cl&cov: c+=1
    key = "equivalent" if "equivalent" in rows[0] else "equivalent_1"
    eq=sum(r[key] for r in rows)/len(rows)
    return len(rows), eq, c/len(rows)

print(f"{'n':>3}  {'OLD eq':>7} {'OLD con':>8}   {'NEW eq':>7} {'NEW con':>8}")
for n in [3,4,5,6,7,8,10,12,15,20]:
    o=stats(f"concrete_bw_n{n}.jsonl")
    w=stats(f"spread_bw_n{n}.jsonl")
    if o and w:
        print(f"{n:>3}  {o[1]:>6.0%} {o[2]:>7.0%}   {w[1]:>6.0%} {w[2]:>7.0%}")
    elif o:
        print(f"{n:>3}  {o[1]:>6.0%} {o[2]:>7.0%}   (pending)")
