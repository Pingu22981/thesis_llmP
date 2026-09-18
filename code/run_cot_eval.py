#!/usr/bin/env python3
"""Evaluate the CoT-trained model (llama3.1-cot) on the trained abstract types.
Records parseable, whether reasoning appeared, and equivalence (placeholder-scored)."""
import json, re, requests, random, argparse
from pathlib import Path
from datasets import load_dataset
import planetarium

OLLAMA="http://localhost:11434/api/chat"
DOM=Path("/workspace/planetarium/planetarium/domains/blocksworld.pddl").read_text()

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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--model",default="llama3.1-cot")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--output",default="results/finetuned/cot_eval.jsonl")
    a=ap.parse_args()

    ds=load_dataset("BatsResearch/planetarium")
    def trained_type(r):
        nl=r["natural_language"].lower(); gs=nl[nl.find("your goal"):]
        return "invert" in gs
    test=[r for r in ds["test"] if r["domain"]=="blocksworld" and r["goal_is_abstract"]==1 and trained_type(r)]
    random.seed(a.seed); random.shuffle(test); test=test[:a.n]
    print(f"{len(test)} trained-type problems", flush=True)

    system=("You convert natural language planning problems into PDDL for a fixed domain.\n\n"
            f"The domain is:\n{DOM}\n\n"
            "First reason step by step about the goal configuration, then output the PDDL "
            "problem definition starting with (define (problem ...)). Use only predicates from the domain.")

    out=Path(a.output); f=out.open("w")
    n=par=eq=reasoned=0
    for i,r in enumerate(test):
        msgs=[{"role":"system","content":system},
              {"role":"user","content":f"Natural language:\n{r['natural_language']}"}]
        try:
            resp=requests.post(OLLAMA,json={"model":a.model,"messages":msgs,"stream":False,
                "options":{"temperature":0,"num_predict":2048,"num_ctx":8192}},timeout=(10,180)).json()["message"]["content"]
        except Exception as e:
            f.write(json.dumps({"id":r["id"],"request_failed":True})+"\n"); f.flush(); continue
        pred=extract(resp)
        # reasoning = any "tower"/"Tower" text before the (define block
        pre = resp[:resp.find("(define")] if "(define" in resp else resp
        has_reason = bool(re.search(r"tower|on the table|clear|stack", pre, re.I)) and len(pre.strip())>0
        p=e=False
        if pred:
            try: p,_,e=planetarium.evaluate(r["problem_pddl"],pred,domain_str=DOM,
                    is_placeholder=bool(r["is_placeholder"]),check_solveable=False)
            except: pass
        n+=1; par+=bool(p); eq+=bool(e); reasoned+=int(has_reason)
        f.write(json.dumps({"id":r["id"],"parseable":bool(p),"equivalent":bool(e),
                            "reasoned":bool(has_reason)})+"\n"); f.flush()
        if (i+1)%25==0: print(f"...{i+1}: parseable {par} reasoned {reasoned} equiv {eq}",flush=True)
    f.close()
    print(f"\nCoT eval n={n}: parseable {par}({par/n:.0%})  reasoned {reasoned}({reasoned/n:.0%})  equivalent {eq}({eq/n:.0%})",flush=True)

if __name__=="__main__": main()
