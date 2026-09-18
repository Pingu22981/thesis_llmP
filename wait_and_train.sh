#!/bin/bash
while :; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date '+%H:%M:%S') free=${free_mb}MiB"
  [ "${free_mb:-0}" -ge 16000 ] && break
  sleep 60
done
for s in 1 2 3; do
  echo "=== training seed $s ==="
  python train_qlora_fixed.py --data data_gen4/equal_towers_train_chat.jsonl \
    --output lora_equal_towers_s$s --epochs 3 --lr 2e-4 --seed $s
done
echo ALL SEEDS DONE
