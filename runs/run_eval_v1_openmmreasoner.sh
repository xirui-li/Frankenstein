#!/usr/bin/env bash
# OpenMMReasoner v1 Evaluation
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_API}"

python scripts/transferability_test.py \
  --sft_model "OpenMMReasoner/OpenMMReasoner-ColdStart" \
  --rl_model "OpenMMReasoner/OpenMMReasoner-RL" \
  --output_dir "./results/transferability_test" \
  --run_name "OpenMMReasoner" \
  --sample_size 100 \
  --sample_seed 123 \
  --max_new_tokens 8192 \
  --inference_timeout 1200 \
  --openai_model "gpt-4o-mini" \
  --experiment_type v1 \
  --use_gpt_descriptions \
  --use_gpt_grader \
  --models \
    "Qwen/Qwen2.5-VL-3B-Instruct" \
    "OpenMMReasoner/OpenMMReasoner-ColdStart" \
    "OpenMMReasoner/OpenMMReasoner-RL" \
    "ColdStart early 1/3 layers + RL late 2/3 layers" \
    "RL early 1/3 layers + ColdStart late 2/3 layers" \
    "ColdStart early 2/3 layers + RL late 1/3 layers" \
    "RL early 2/3 layers + ColdStart late 1/3 layers" \
    "ColdStart early 1/3 + RL middle 1/3 + ColdStart late 1/3 layers" \
    "RL early 1/3 + ColdStart middle 1/3 + RL late 1/3 layers"
