#!/usr/bin/env python3
"""Evaluate a LoRA adapter (or base) on a test jsonl. Scores Planetarium equivalence.
Usage: python eval_matrix.py <adapter_dir_or_'base'> <test_jsonl> <output_jsonl>"""
import json, sys, torch, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import planetarium

BASE="meta-llama/Llama-3.1-8B-Instruct"
DOM=Path("/workspace/planetarium/planetarium/domains/blocksworld.pddl").read_text()
SYSTEM=("You convert natural language descriptions of planning problems into "
        "PDDL problem files for a fixed domain.\n\n"
        f"The domain is:\n{DOM}\n\n"
        "Write only the PDDL problem definition. It must start with "
        "(define (problem ...)) and use only predicates from the domain above. "
        "Do not include markdown fences, comments, or any explanation.")

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
    adapter, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16)
    if adapter!="base":
        model=PeftModel.from_pretrained(model, adapter)
    model.eval()

    rows=[json.loads(l) for l in open(test_path) if l.strip()]
    f=open(out_path,"w"); n=par=eq=0
    for r in rows:
        msgs=[{"role":"system","content":SYSTEM},
              {"role":"user","content":f"Natural language:\n{r['natural_language']}"}]
        inp=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out=model.generate(inp, max_new_tokens=2048, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        pred=extract(text); p=e=False
        if pred:
            try: p,_,e=planetarium.evaluate(r["problem_pddl"], pred, domain_str=DOM,
                    is_placeholder=bool(r.get("is_placeholder",1)), check_solveable=False)
            except: pass
        n+=1; par+=bool(p); eq+=bool(e)
        f.write(json.dumps({"name":r.get("name"),"parseable":bool(p),"equivalent":bool(e)})+"\n"); f.flush()
    f.close()
    print(f"\n{adapter} on {test_path}: n={n} parseable {par}({par/n:.0%}) equivalent {eq}({eq/n:.0%})")

if __name__=="__main__": main()
