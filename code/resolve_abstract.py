#!/usr/bin/env python3
"""Upstream resolver: abstract blocksworld problem -> fully concrete NL (init + goal spelled out).
The resolver does all inference; the LLM only transcribes concrete NL -> PDDL."""
import re

def stacks_from(block_text):
    ot=set(re.findall(r"\(on-table (b\d+)\)",block_text))
    on=re.findall(r"\(on (b\d+) (b\d+)\)",block_text)
    up={l:u for u,l in on}
    res=[]
    for base in sorted(ot,key=lambda b:int(b[1:])):
        s=[base];c=base
        while c in up:c=up[c];s.append(c)
        res.append(s)  # bottom->top
    return res

def all_blocks(pddl):
    m=re.search(r"\(:objects ([^)]+)\)",pddl)
    return sorted(m.group(1).split(),key=lambda b:int(b[1:])) if m else []

def init_stacks(pddl):
    i=pddl.find("(:init"); j=pddl.find("(:goal")
    return stacks_from(pddl[i:j])

def compute_goal_stacks(nl, pddl):
    nll=nl.lower(); blocks=all_blocks(pddl); n=len(blocks)
    # decide type from the GOAL sentence only (after "your goal is")
    goal_sent = nll[nll.find("your goal"):]
    if "invert" in goal_sent:
        return [list(reversed(s)) for s in init_stacks(pddl)]
    if "single stack" in goal_sent:
        return [blocks]
    m=re.search(r"heights? ([\d, and]+)", goal_sent)
    if m:
        heights=[int(x) for x in re.findall(r"\d+",m.group(1))]
        if sum(heights)!=n: return None
        out=[];idx=0
        for h in heights:
            grp=blocks[idx:idx+h];idx+=h
            out.append(list(reversed(grp)))
        return out
    return None

def stacks_to_nl(stacks, verb):
    """verb: 'is'/'is on' for init, 'should be'/'should be on' for goal."""
    be = "is" if verb=="init" else "should be"
    lines=[]
    for t in stacks:
        lines.append(f"{t[0]} {be} on the table.")
        for lower,upper in zip(t,t[1:]):
            lines.append(f"{upper} {be} on {lower}.")
        lines.append(f"{t[-1]} {be} clear.")
    return lines

def resolve(nl, pddl):
    goal=compute_goal_stacks(nl,pddl)
    if goal is None: return None
    ist=init_stacks(pddl)
    n=len(all_blocks(pddl))
    parts=[f"You have {n} blocks.", "Your arm is empty."]
    parts += stacks_to_nl(ist,"init")
    parts += ["Your goal is to have the following:","Your arm should be empty."]
    parts += stacks_to_nl(goal,"goal")
    return "\n".join(parts)

if __name__=="__main__":
    import os; os.environ["HF_HOME"]="/scratch/hf_cache"
    from datasets import load_dataset
    ds=load_dataset("BatsResearch/planetarium")
    for r in ds["test"]:
        if r["domain"]=="blocksworld" and r["goal_is_abstract"]==1 and "invert" in r["natural_language"].lower() and r["num_objects"]>=6:
            print("RESOLVED:\n", resolve(r["natural_language"],r["problem_pddl"]))
            p=r["problem_pddl"]
            print("\nGT INIT:", p[p.find("(:init"):p.find("(:goal")][:200])
            print("GT GOAL:", p[p.find("(:goal"):][:200])
            break
