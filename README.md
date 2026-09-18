# Fine-Tuning Language Models for Natural Language to PDDL Translation

Repository accompanying the MSc dissertation *Fine-Tuning Language Models for
Natural Language to PDDL Translation: Cross-Type Generalisation of Abstract
Goal Inference* (Benjamin Fletcher, University of Bath, 2026).

## Purpose

Automated planning requires a problem to be written as a formal specification.
The LLM+P architecture uses a language model for one step only: turning a
natural-language description into a PDDL problem file, leaving the search to a
classical planner. This project asks what supervised fine-tuning acquires for
that translation step, with the focus on abstract goals, where the model has to
compute a target configuration rather than transcribe one.

The central finding is that fine-tuning an 8B model does learn abstract goal
translation, but as a procedure tied to a specific goal type, not as a general
capability. In-domain performance is high for all four goal types while
cross-type transfer stays near zero. The type-disjoint train/test splits in the
Planetarium benchmark make this appear as complete failure under the standard
evaluation.

## Main results

Goal equivalence (%), mean over three seeds. Diagonal cells are in-domain,
off-diagonal cells are cross-type transfer.

| Fine-tuned on / evaluated on | invert | swap | tower | equal towers | held-out sizes |
|---|---|---|---|---|---|
| invert       | 59.6 +- 0.4 | 0.0 | 0.7 | 0.0 | 15.0 +- 4.4 |
| swap         | 0.0 | 94.4 +- 1.9 | 0.7 | 0.0 | 42.2 +- 5.1 |
| tower        | 0.0 | 0.0 | 95.8 +- 3.7 | 0.0 | 54.7 +- 19.2 |
| equal towers | 0.0 | 0.0 | 0.0 | 96.4 +- 2.1 | 22.0 +- 2.6 |
| base (no FT) | 0.0 | 0.0 | 0.0 | 0.0 | - |

Type-matched prompting reaches 0% equivalence on invert, against 59.5% for
in-domain fine-tuning on the same type.

## Note on the equal-towers oracle

During a data audit I found two defects in the Planetarium equal-towers
generator. When the block count is not divisible by the requested tower count,
it emits a degenerate goal with every block on the table. When the count is
divisible, it often emits a partition with the wrong number of towers. About
70-77% of the original equal-towers splits were affected. The corrected
generator in `code/gen_equal_towers_fixed.py` regenerates the splits so that
every instance has a real solution with exactly the requested number of towers.
The equal-towers figures above use the corrected data. The original defective
results are kept in `results/matrix4_v2_buggy/`.

## Repository layout

- `code/` - Python scripts for data generation, training, evaluation, and figures
- `scripts/` - shell wrappers used to run the full matrix
- `data_gen4/` - generated splits for all four abstract goal types, including the
  corrected equal-towers data
- `results/` - raw JSONL outputs, including the final matrix (`results/matrix4_v2/`)
  and the corrected equal-towers re-run (`results/matrix4_clean/`)
- `figs/` - generated figures (PDF and PNG)
- `logs/` - training and evaluation logs

Model checkpoints and quantised weights are not tracked; they can be regenerated
with the scripts below.

## Dependencies

The base model is `meta-llama/Llama-3.1-8B-Instruct`, loaded with 4-bit NF4
quantisation. Access to the weights requires a Hugging Face account with access
to the model.

Python packages, with the versions used:

- torch 2.2.0 (CUDA 11.8)
- transformers 4.44.2
- peft 0.12.0
- trl 0.9.6
- datasets 5.0.0
- accelerate 0.33.0
- bitsandbytes 0.43.3
- protobuf 4.25.9
- tokenizers 0.19.1

The evaluation also needs the Planetarium metric and its `pddl` dependency,
which is a fork rather than the PyPI release:

    pip install "git+https://github.com/maxzuo/pddl.git"

with `PYTHONPATH` set to the Planetarium checkout so that `import planetarium`
resolves.

## Reproduction

Generate the corrected equal-towers data:

    python code/gen_equal_towers_fixed.py --out data_gen4/equal_towers_train.jsonl --count 500 --lo 4 --hi 20 --seed 100
    python code/gen_equal_towers_fixed.py --out data_gen4/equal_towers_test.jsonl --count 150 --lo 4 --hi 20 --seed 200
    python code/gen_equal_towers_fixed.py --out data_gen4/equal_towers_heldtest.jsonl --count 100 --lo 21 --hi 40 --seed 300

Convert to the chat format used by the trainer:

    python code/convert_gen_to_chat.py data_gen4/equal_towers_train.jsonl data_gen4/equal_towers_train_chat.jsonl

Fine-tune an adapter:

    python code/train_qlora_fixed.py --data data_gen4/equal_towers_train_chat.jsonl \
      --output lora_equal_towers_s1 --epochs 3 --lr 2e-4 --seed 1

Evaluate the adapter:

    python code/eval_matrix.py lora_equal_towers_s1 data_gen4/equal_towers_test.jsonl \
      results/equal_towers_on_equal_towers_s1.jsonl 1

Regenerate the figures:

    python code/make_figs_final.py

The figures read from `results/matrix4_v2/`, which holds the final numbers.
