#!/bin/bash
export HF_HOME=/scratch/hf_cache
cd /workspace/llm_p
# clear stale tower cells
rm -f results/matrix4/tower_on_*_s*.jsonl results/matrix4/*_on_tower_s*.jsonl
rm -f results/matrix4/tower_heldsize_s*.jsonl results/matrix4/base_on_tower.jsonl
for s in 1 2 3; do
  for te in invert swap tower equal_towers; do
    python eval_matrix.py lora_tower_s$s data_gen4/${te}_test.jsonl results/matrix4/tower_on_${te}_s$s.jsonl
  done
  for tr in invert swap equal_towers; do
    python eval_matrix.py lora_${tr}_s$s data_gen4/tower_test.jsonl results/matrix4/${tr}_on_tower_s$s.jsonl
  done
  python eval_matrix.py lora_tower_s$s data_gen4/tower_heldtest.jsonl results/matrix4/tower_heldsize_s$s.jsonl
done
python eval_matrix.py base data_gen4/tower_test.jsonl results/matrix4/base_on_tower.jsonl
echo "TOWER RERUN DONE"
