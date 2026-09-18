#!/bin/bash
export HF_HOME=/scratch/hf_cache
cd /workspace/llm_p
mkdir -p results/matrix4_v2
T="invert swap tower equal_towers"

# 1. DIAGONAL — in-domain, all 3 seeds (headline numbers, must verify)
for s in 1 2 3; do
  for t in $T; do
    OUT="results/matrix4_v2/${t}_on_${t}_s${s}.jsonl"
    [ -f "$OUT" ] || python eval_matrix.py lora_${t}_s$s data_gen4/${t}_test.jsonl "$OUT" 1
  done
done

# 2. HELD-OUT diagonal — memorization check, all 3 seeds
for s in 1 2 3; do
  for t in $T; do
    OUT="results/matrix4_v2/${t}_heldsize_s${s}.jsonl"
    [ -f "$OUT" ] || python eval_matrix.py lora_${t}_s$s data_gen4/${t}_heldtest.jsonl "$OUT" 1
  done
done

# 3. BASE row — reference, once
for t in $T; do
  OUT="results/matrix4_v2/base_on_${t}.jsonl"
  [ -f "$OUT" ] || python eval_matrix.py base data_gen4/${t}_test.jsonl "$OUT" 1
done

# 4. TRANSFER — seed 1 only, to document near-zero (off-diagonal)
for tr in $T; do
  for te in $T; do
    [ "$tr" = "$te" ] && continue
    OUT="results/matrix4_v2/${tr}_on_${te}_s1.jsonl"
    [ -f "$OUT" ] || python eval_matrix.py lora_${tr}_s1 data_gen4/${te}_test.jsonl "$OUT" 1
  done
done
echo "CORE DONE"
