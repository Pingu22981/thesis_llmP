import json, re, requests, random
from datasets import load_dataset
import planetarium
ds=load_dataset("BatsResearch/planetarium")
domain=open("/workspace/planetarium/planetarium/domains/blocksworld.pddl").read()
def is_invert(r):
    nl=r["natural_language"].lower(); return "invert" in nl[nl.find("your goal"):]
test=[r for r in ds["test"] if r["domain"]=="blocksworld" and r["goal_is_abstract"]==1 and is_invert(r)]
random.seed(42); test=random.sample(test,20)

def extract_after_think(text):
    if "</think>" in text: text=text.split("</think>")[-1]
    s=text.find("(define")
    if s==-1: return None
    d=0
    for i in range(s,len(text)):
        if text[i]=="(":d+=1
        elif text[i]==")":
            d-=1
            if d==0: return text[s:i+1]
    return None

system=("You convert natural language planning problems into PDDL for a fixed domain.\n\n"
        f"The domain is:\n{domain}\n\nOutput a PDDL problem definition starting with "
        "(define (problem ...)) using only predicates from the domain.")
eq=par=n=0
for r in test:
    resp=requests.post("http://localhost:11434/api/chat",json={"model":"deepseek-r1:8b",
        "messages":[{"role":"system","content":system},
                    {"role":"user","content":f"Natural language:\n{r['natural_language']}"}],
        "stream":False,"options":{"temperature":0.6,"num_predict":8192,"num_ctx":16384}},
        timeout=(10,600)).json()["message"]["content"]
    pred=extract_after_think(resp); p=e=False
    if pred:
        try:p,_,e=planetarium.evaluate(r["problem_pddl"],pred,domain_str=domain,
            is_placeholder=bool(r["is_placeholder"]),check_solveable=False)
        except:pass
    n+=1;par+=p;eq+=e
    print(f"[{n}/20] parseable={p} equiv={e}",flush=True)
print(f"\nR1-Distill-8B invert zero-shot: parseable {par}/{n} equiv {eq}/{n} ({eq/n:.0%})",flush=True)
print("compare: your Llama 8B invert ~0%, Sonnet abstract 34%",flush=True)
