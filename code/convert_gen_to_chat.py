#!/usr/bin/env python3
import json, sys
from pathlib import Path
domain_str = Path("/workspace/planetarium/planetarium/domains/blocksworld.pddl").read_text()
SYSTEM = ("You convert natural language descriptions of planning problems into "
          "PDDL problem files for a fixed domain.\n\n"
          f"The domain is:\n{domain_str}\n\n"
          "Write only the PDDL problem definition. It must start with "
          "(define (problem ...)) and use only predicates from the domain above. "
          "Do not include markdown fences, comments, or any explanation.")
def convert(inp, outp):
    n=0
    with open(inp) as f, open(outp,"w") as g:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            g.write(json.dumps({"messages":[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":f"Natural language:\n{r['natural_language']}"},
                {"role":"assistant","content":r["problem_pddl"]}]})+"\n"); n+=1
    print(f"{inp} -> {outp}: {n}")
if __name__=="__main__":
    convert(sys.argv[1], sys.argv[2])
