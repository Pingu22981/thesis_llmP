#!/usr/bin/env python3
"""Evaluate a LoRA adapter (or base) on a test jsonl. Scores Planetarium equivalence.
Usage: python eval_matrix.py <adapter_dir_or_'base'> <test_jsonl> <output_jsonl> [batch_size]"""
import json, sys, torch
from pathlib import Path
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList)
from peft import PeftModel
import planetarium

BASE = "meta-llama/Llama-3.1-8B-Instruct"
DOM = Path("/workspace/planetarium/planetarium/domains/blocksworld.pddl").read_text()
SYSTEM = ("You convert natural language descriptions of planning problems into "
          "PDDL problem files for a fixed domain.\n\n"
          f"The domain is:\n{DOM}\n\n"
          "Write only the PDDL problem definition. It must start with "
          "(define (problem ...)) and use only predicates from the domain above. "
          "Do not include markdown fences, comments, or any explanation.")


class BatchPddlDone(StoppingCriteria):
    """Stop once EVERY sequence in the batch has emitted a balanced (define ...).
    Tracks paren depth incrementally per sequence: O(1) per step instead of
    re-decoding the whole string every token."""
    def __init__(self, tok, prompt_len, batch_size):
        self.tok = tok
        self.prompt_len = prompt_len
        self.started = [False] * batch_size
        self.depth = [0] * batch_size
        self.done = [False] * batch_size
        self.buf = [""] * batch_size

    def __call__(self, input_ids, scores, **kw):
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            piece = self.tok.decode(input_ids[i][-1:], skip_special_tokens=True)
            for ch in piece:
                if not self.started[i]:
                    self.buf[i] += ch
                    if self.buf[i].endswith("(define"):
                        self.started[i] = True
                        self.depth[i] = 1
                    elif len(self.buf[i]) > 4096:
                        self.buf[i] = self.buf[i][-16:]
                    continue
                if ch == "(":
                    self.depth[i] += 1
                elif ch == ")":
                    self.depth[i] -= 1
                    if self.depth[i] == 0:
                        self.done[i] = True
                        break
        return all(self.done)


def extract(t):
    s = t.find("(define")
    if s == -1:
        return None
    d = 0
    for i in range(s, len(t)):
        if t[i] == "(":
            d += 1
        elif t[i] == ")":
            d -= 1
            if d == 0:
                return t[s:i + 1]
    return None


def main():
    adapter, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    bs = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for decoder-only batched generation
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    if adapter != "base":
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    rows = [json.loads(l) for l in open(test_path) if l.strip()]
    f = open(out_path, "w")
    n = par = eq = 0

    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]
        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"Natural language:\n{r['natural_language']}"}],
                add_generation_prompt=True, tokenize=False)
            for r in chunk
        ]
        # add_special_tokens=False: the chat template already emits BOS
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        plen = enc.input_ids.shape[1]

        sc = StoppingCriteriaList([BatchPddlDone(tok, plen, len(chunk))])
        with torch.no_grad():
            out = model.generate(input_ids=enc.input_ids,
                                 attention_mask=enc.attention_mask,
                                 max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.eos_token_id,
                                 stopping_criteria=sc)

        for j, r in enumerate(chunk):
            text = tok.decode(out[j][plen:], skip_special_tokens=True)
            pred = extract(text)
            p = e = False
            if pred:
                try:
                    p, _, e = planetarium.evaluate(
                        r["problem_pddl"], pred, domain_str=DOM,
                        is_placeholder=bool(r.get("is_placeholder", 1)),
                        check_solveable=False)
                except Exception:
                    pass
            n += 1; par += bool(p); eq += bool(e)
            f.write(json.dumps({"name": r.get("name"),
                                "parseable": bool(p),
                                "equivalent": bool(e)}) + "\n")
        f.flush()
        print(f"  [{n}/{len(rows)}] parseable {par} equivalent {eq}", flush=True)

    f.close()
    print(f"\n{adapter} on {test_path}: n={n} parseable {par}({par/n:.0%}) equivalent {eq}({eq/n:.0%})")


if __name__ == "__main__":
    main()
