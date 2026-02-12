export CUDA_VISIBLE_DEVICES=0

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Run 1: OpenMMReasoner
python scripts/characterisation_energy.py \
  --base $BASE_MODEL \
  --sft OpenMMReasoner/OpenMMReasoner-ColdStart \
  --rl OpenMMReasoner/OpenMMReasoner-RL \
  --output results/characterisation_energy/OpenMMReasoner/fro_norm.json

# Run 2: Revisual-R1
python scripts/characterisation_energy.py \
  --base $BASE_MODEL \
  --sft csfufu/Revisual-R1-Coldstart \
  --rl csfufu/Revisual-R1-final \
  --output results/characterisation_energy/Revisual-R1/fro_norm.json

# Run 3: MMR1
python scripts/characterisation_energy.py \
  --base $BASE_MODEL \
  --sft MMR1/MMR1-7B-SFT \
  --rl MMR1/MMR1-7B-RL \
  --output results/characterisation_energy/MMR1/fro_norm.json
