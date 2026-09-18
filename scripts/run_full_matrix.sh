#!/bin/bash
set -e
export HF_HOME=/scratch/hf_cache
cd /workspace/llm_p
TYPES="invert swap tower equal_towers"
SEEDS="1 2 3"

# ---- train 12 adapters (skip if already done) ----
for t in $TYPES; do
  for s in $SEEDS; do
    OUT="lora_${t}_s${s}"
    if [ -f "$OUT/adapter_model.safetensors" ]; then
      echo "skip $OUT (exists)"
    else
      echo "=== training $OUT ==="
      python train_qlora_fixed.py --data data_gen4/${t}_train_chat.jsonl \
        --output "$OUT" --epochs 3 --lr 2e-4 --seed $s || echo "FAILED $OUT"
    fi
  done
done

# ---- evaluate: 4x4 in-distribution matrix per seed ----
for s in $SEEDS; do
  for tr in $TYPES; do
    for te in $TYPES; do
      OUT="results/matrix4/${tr}_on_${te}_s${s}.jsonl"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      echo "=== eval lora_${tr}_s${s} on ${te}_test ==="
      python eval_matrix.py "lora_${tr}_s${s}" "data_gen4/${te}_test.jsonl" "$OUT" || echo "FAILED $OUT"
    done
    # held-out-size diagonal (memorization check): adapter on its OWN type's unseen sizes
    OUT="results/matrix4/${tr}_heldsize_s${s}.jsonl"
    [ -f "$OUT" ] || python eval_matrix.py "lora_${tr}_s${s}" "data_gen4/${tr}_heldtest.jsonl" "$OUT" || echo "FAILED heldsize $tr s$s"
  done
done

# ---- base model (no fine-tuning) on each test set, once ----
for te in $TYPES; do
  OUT="results/matrix4/base_on_${te}.jsonl"
  [ -f "$OUT" ] || python eval_matrix.py base "data_gen4/${te}_test.jsonl" "$OUT" || echo "FAILED base $te"
done

echo "ALL DONE"
