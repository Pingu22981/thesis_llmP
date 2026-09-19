#!/bin/bash
cd /workspace/llm_p
export HF_HOME=/workspace/hf_cache2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/planetarium:$PYTHONPATH

for s in 1 2 3; do
  echo "=== equal_towers row, seed $s ==="
  python code/eval_matrix.py "lora_equal_towers_s$s" "data_gen4/invert_test.jsonl" "results/matrix4_clean/equal_towers_on_invert_s${s}.jsonl" 1
  python code/eval_matrix.py "lora_equal_towers_s$s" "data_gen4/swap_test.jsonl"  "results/matrix4_clean/equal_towers_on_swap_s${s}.jsonl" 1
  python code/eval_matrix.py "lora_equal_towers_s$s" "data_gen4/tower_test.jsonl" "results/matrix4_clean/equal_towers_on_tower_s${s}.jsonl" 1
done

echo "=== base on corrected equal_towers ==="
python code/eval_matrix.py base "data_gen4/equal_towers_test.jsonl" "results/matrix4_clean/base_on_equal_towers.jsonl" 1

echo DONE
