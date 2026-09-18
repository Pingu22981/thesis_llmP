import os, re, random, json
from datasets import load_dataset
import planetarium
from anthropic import Anthropic
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-5"
BUDGET_USD = 4.50
IN_PRICE, OUT_PRICE = 3.0/1e6, 15.0/1e6
ds = load_dataset("BatsResearch/planetarium")
domain = open("/workspace/planetarium/planetarium/domains/blocksworld.pddl").read()
def gtype(r):
    nl=r["natural_language"].lower(); gs=nl[nl.find("your goal"):]
    if "invert" in gs: return "invert"
    if "single stack" in gs: return "single"
    if "height" in gs: return "towers"
    return "other"
test=[r for r in ds["test"] if r["domain"]=="blocksworld" and r["goal_is_abstract"]==1]
ex_pool=sorted([r for r in ds["train"] if r["domain"]=="blocksworld" and r["goal_is_abstract"]==1],key=lambda r:r["num_objects"])
k=4; examples=[ex_pool[int(i*(len(ex_pool)-1)/(k-1))] for i in range(k)]
ex_ids={e["id"] for e in examples}; ex_nl={e["natural_language"] for e in examples}
random.seed(42); buckets={}
for r in test:
    if r["id"] in ex_ids or r["natural_language"] in ex_nl: continue
    buckets.setdefault(gtype(r),[]).append(r)
sample=[]
for t,rows in buckets.items():
    random.shuffle(rows); sample+=rows[:25]
random.shuffle(sample)
print(f"{len(sample)} problems: "+", ".join(f"{t}={min(len(v),25)}" for t,v in buckets.items()),flush=True)
def extract(t):
    s=t.find("(define")
    if s==-1: return None
    d=0
    for i in range(s,len(t)):
        if t[i]=="(":d+=1
        elif t[i]==")":
            d-=1
            if d==0: return t[s:i+1]
    return None
system=("You convert natural language descriptions of planning problems into PDDL problem files "
    f"for a fixed domain.\n\nThe domain is:\n{domain}\n\nWrite only the PDDL problem definition. "
    "It must start with (define (problem ...)) and use only predicates from the domain above. "
    "Do not include markdown fences, comments, or any explanation.")
def prompt_for(nl):
    s=""
    for ex in examples: s+=f"Natural language:\n{ex['natural_language']}\n\nPDDL:\n{ex['problem_pddl']}\n\n"
    return s+f"Natural language:\n{nl}\n\nPDDL:\n"
from collections import Counter
spend=0.0;n=eq=par=0;tt=Counter();te=Counter()
out=open("results/finetuned/sonnet_abstract.jsonl","w")
for r in sample:
    if spend>BUDGET_USD: print(f"budget stop ${spend:.2f}",flush=True); break
    m=client.messages.create(model=MODEL,max_tokens=2048,system=system,messages=[{"role":"user","content":prompt_for(r["natural_language"])}])
    spend+=m.usage.input_tokens*IN_PRICE+m.usage.output_tokens*OUT_PRICE
    pred=extract(m.content[0].text); p=e=False
    if pred:
        try: p,_,e=planetarium.evaluate(r["problem_pddl"],pred,domain_str=domain,is_placeholder=bool(r["is_placeholder"]),check_solveable=False)
        except: pass
    t=gtype(r); n+=1; par+=p; eq+=e; tt[t]+=1; te[t]+=int(e)
    out.write(json.dumps({"id":r["id"],"type":t,"parseable":bool(p),"equivalent":bool(e)})+"\n"); out.flush()
    if n%10==0: print(f"...{n} | equiv {eq} | ${spend:.2f}",flush=True)
out.close()
print(f"\n{MODEL} abstract 4-shot matched, placeholder: n={n} parseable {par}({par/n:.0%}) equiv {eq}({eq/n:.0%}) spend ${spend:.2f}")
for t in tt: print(f"  {t}: {te[t]}/{tt[t]} ({te[t]/tt[t]:.0%})")
