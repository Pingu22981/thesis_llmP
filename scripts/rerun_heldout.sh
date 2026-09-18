#!/bin/bash
export HF_HOME=/scratch/hf_cache
cd /workspace/llm_p
for s in 1 2 3; do
  for t in invert swap tower equal_towers; do
    OUT="results/matrix4_v2/${t}_heldsize_s${s}.jsonl"
    [ -f "$OUT" ] || python eval_matrix.py lora_${t}_s$s data_gen4/${t}_heldtest.jsonl "$OUT" 1
  done
done
echo "HELDOUT DONE"
