#!/bin/bash
# Large-scale SFT training: 200 samples per character

CHARACTER="LinDaiyu"
NUM_SAMPLES=200
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME="${CHARACTER}_sft_${NUM_SAMPLES}"
OUTPUT_DIR="saves/qwen_${DATASET_NAME}"

echo "Starting LARGE-SCALE SFT training for ${CHARACTER}"
echo "Samples: ${NUM_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"

# Register dataset
if ! grep -q "\"${DATASET_NAME}\"" LLaMA-Factory/data/dataset_info.json 2>/dev/null; then
    echo "Registering dataset..."
    python3 << PYEOF
import json
import os
data = {}
info_path = "LLaMA-Factory/data/dataset_info.json"
if os.path.exists(info_path):
    with open(info_path) as f:
        data = json.load(f)
data["${DATASET_NAME}"] = {"file_name": "../../experiments/sft/data_large/${CHARACTER}_${NUM_SAMPLES}.json"}
with open(info_path, "w") as f:
    json.dump(data, f, indent=2)
PYEOF
fi

# Training
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL_PATH} \
    --dataset ${DATASET_NAME} \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --bf16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --warmup_ratio 0.1 \
    --ddp_find_unused_parameters false \
    --max_length 1024

echo "Training complete! Model saved to ${OUTPUT_DIR}"
