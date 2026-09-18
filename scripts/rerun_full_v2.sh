#!/bin/bash
export HF_HOME=/scratch/hf_cache
cd /workspace/llm_p
mkdir -p results/matrix4_v2
T="invert swap tower equal_towers"
for s in 1 2 3; do
  for tr in $T; do
    for te in $T; do
      OUT="results/matrix4_v2/${tr}_on_${te}_s${s}.jsonl"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      python eval_matrix.py lora_${tr}_s$s data_gen4/${te}_test.jsonl "$OUT" 1
    done
    OUT="results/matrix4_v2/${tr}_heldsize_s${s}.jsonl"
    [ -f "$OUT" ] || python eval_matrix.py lora_${tr}_s$s data_gen4/${tr}_heldtest.jsonl "$OUT" 1
  done
done
for te in $T; do
  OUT="results/matrix4_v2/base_on_${te}.jsonl"
  [ -f "$OUT" ] || python eval_matrix.py base data_gen4/${te}_test.jsonl "$OUT" 1
done
echo "FULL V2 DONE"
