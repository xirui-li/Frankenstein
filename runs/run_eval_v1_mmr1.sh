#!/usr/bin/env bash
# MMR1 v1 Evaluation
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_API}"

python scripts/transferability_test.py \
  --sft_model "MMR1/MMR1-7B-SFT" \
  --rl_model "MMR1/MMR1-7B-RL" \
  --output_dir "./results/transferability_test" \
  --run_name "MMR1" \
  --sample_size 100 \
  --sample_seed 123 \
  --max_new_tokens 8192 \
  --inference_timeout 1200 \
  --openai_model "gpt-4o-mini" \
  --experiment_type v1 \
  --use_gpt_descriptions \
  --use_gpt_grader \
  --models \
    "MMR1/MMR1-7B-SFT" \
    "MMR1/MMR1-7B-RL" \
    "SFT early 1/3 layers + RL late 2/3 layers" \
    "RL early 1/3 layers + SFT late 2/3 layers" \
    "SFT early 2/3 layers + RL late 1/3 layers" \
    "RL early 2/3 layers + SFT late 1/3 layers" \
    "SFT early 1/3 + RL middle 1/3 + SFT late 1/3 layers" \
    "RL early 1/3 + SFT middle 1/3 + RL late 1/3 layers"
