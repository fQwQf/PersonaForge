#!/bin/bash

# Configuration
CHARACTER="${1:-LinDaiyu}"  # Default to LinDaiyu if no argument provided
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATA_FILENAME="${CHARACTER}_sft.json"
DATASET_NAME="${CHARACTER}_sft"
OUTPUT_DIR="saves/qwen_${DATASET_NAME}"

# Path to the generated data file (relative to project root)
PROJECT_DATA_PATH="experiments/sft/data/${DATA_FILENAME}"

echo "Preparing to train for character: $CHARACTER"
echo "Data file should be at: $PROJECT_DATA_PATH"

# Ensure we are in the root or experiments folder
# Try to find LLaMA-Factory
if [ -d "LLaMA-Factory" ]; then
    cd LLaMA-Factory
elif [ -d "../LLaMA-Factory" ]; then
    cd ../LLaMA-Factory
else
    echo "LLaMA-Factory not found. Cloning..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e .[metrics]
    pip install bitsandbytes deepspeed
fi

# Ensure data is registered
# We check if the dataset entry exists in dataset_info.json
# If not, we append it using python for safety

if ! grep -q "\"$DATASET_NAME\"" data/dataset_info.json; then
    echo "Registering dataset $DATASET_NAME..."
    
    # We use a python one-liner to inject the config
    # The file path must be relative to LLaMA-Factory/data directory
    # So ../../experiments/sft/data/Filename
    RELATIVE_DATA_PATH="../../experiments/sft/data/${DATA_FILENAME}"
    
    python3 -c "
import json
import os
try:
    with open('data/dataset_info.json', 'r') as f:
        data = json.load(f)
    
    # Add or update the dataset entry
    data['$DATASET_NAME'] = {'file_name': '$RELATIVE_DATA_PATH'} 
    
    with open('data/dataset_info.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Successfully registered $DATASET_NAME pointing to $RELATIVE_DATA_PATH')
except Exception as e:
    print(f'Error registering dataset: {e}')
"
fi

echo "Starting Training on GPU 0..."

echo "Starting Training on GPU 0..."

# Force add src to PYTHONPATH to ensure we can import llamafactory
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Use direct python execution for maximum refreshing reliability
CMD="python3 src/train.py"
echo "Using command: $CMD"

CUDA_VISIBLE_DEVICES=0 $CMD \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --quantization_bit 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --fp16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --warmup_ratio 0.1

echo "Training complete. Adapter saved to $OUTPUT_DIR"
