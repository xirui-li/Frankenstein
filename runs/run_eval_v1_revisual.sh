#!/usr/bin/env bash
# Revisual-R1 v1 Evaluation
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_API}"

python scripts/transferability_test.py \
  --sft_model "csfufu/Revisual-R1-Coldstart" \
  --rl_model "csfufu/Revisual-R1-final" \
  --output_dir "./results/transferability_test" \
  --run_name "Revisual-R1" \
  --sample_size 100 \
  --sample_seed 123 \
  --max_new_tokens 8192 \
  --inference_timeout 1200 \
  --openai_model "gpt-4o-mini" \
  --experiment_type v1 \
  --use_gpt_descriptions \
  --use_gpt_grader \
  --models \
    "csfufu/Revisual-R1-Coldstart" \
    "csfufu/Revisual-R1-final" \
    "Coldstart early 1/3 layers + final late 2/3 layers" \
    "final early 1/3 layers + Coldstart late 2/3 layers" \
    "Coldstart early 2/3 layers + final late 1/3 layers" \
    "final early 2/3 layers + Coldstart late 1/3 layers" \
    "Coldstart early 1/3 + final middle 1/3 + Coldstart late 1/3 layers" \
    "final early 1/3 + Coldstart middle 1/3 + final late 1/3 layers"
