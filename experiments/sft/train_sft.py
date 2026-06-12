#!/usr/bin/env python3
"""
ACTUAL SFT Training Script - Using Real Transformers + PEFT
Train on 8x RTX 3090
"""

import os
import sys
import json
import time
import torch
from datetime import datetime

# Check GPU availability
print("="*80)
print("ACTUAL SFT TRAINING - Qwen2.5-7B with LoRA")
print("="*80)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# Install required packages if missing
try:
    import transformers
    import peft
    from datasets import Dataset
    print(f"\n✓ Transformers: {transformers.__version__}")
    print(f"✓ PEFT: {peft.__version__}")
except ImportError as e:
    print(f"\n✗ Missing package: {e}")
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "transformers", "peft", "datasets", "accelerate"])
    print("✓ Packages installed")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATHS = {
    100: "experiments/sft/data_large/LinDaiyu_200.json",  # Use 200 as base for 100
    200: "experiments/sft/data_large/LinDaiyu_200.json",
    500: "experiments/sft/data_large/LinDaiyu_500.json",
    1000: "experiments/sft/data_large/LinDaiyu_1000.json"
}

OUTPUT_BASE = "./sft_outputs_actual"

def load_data(n_samples):
    """Load training data"""
    data_path = DATA_PATHS.get(n_samples, DATA_PATHS[200])
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit to n_samples if needed
    if n_samples < len(data):
        data = data[:n_samples]
    
    return data

def train_sft(n_samples=200, output_dir=None):
    """Run actual SFT training"""
    
    if output_dir is None:
        output_dir = f"{OUTPUT_BASE}/lindaiyu_{n_samples}samples"
    
    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION: {n_samples} samples")
    print(f"{'='*80}")
    
    # Load data
    print(f"\n[1/7] Loading data...")
    raw_data = load_data(n_samples)
    print(f"  ✓ Loaded {len(raw_data)} samples")
    
    # Load model and tokenizer
    print(f"\n[2/7] Loading model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Will use all available GPUs
        trust_remote_code=True
        # Flash attention disabled for compatibility
    )
    
    print(f"  ✓ Model loaded on {torch.cuda.device_count()} GPUs")
    print(f"  ✓ Model dtype: {model.dtype}")
    
    # LoRA config
    print(f"\n[3/7] Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print(f"\n[4/7] Preparing dataset...")
    
    system_prompt = "你是林黛玉，性格敏感多愁，说话尖酸刻薄但富有才情。说话要符合《红楼梦》中林黛玉的形象，善用诗词和典故。"
    
    def format_example(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_example)
    
    print(f"  ✓ Dataset prepared: {len(dataset)} examples")
    
    # Training arguments for multi-GPU
    print(f"\n[5/7] Configuring training...")
    
    # Calculate batch size per device
    n_gpus = torch.cuda.device_count()
    per_device_batch = max(1, 8 // n_gpus)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
    )
    
    print(f"  ✓ Batch size per device: {per_device_batch}")
    print(f"  ✓ Total batch size: {per_device_batch * n_gpus * 4} (with grad accum)")
    
    # Tokenize dataset
    print(f"  Tokenizing dataset...")
    def tokenize_function(examples):
        # Tokenize the text field
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(f"  ✓ Dataset tokenized")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True
    )
    
    # Initialize trainer
    print(f"\n[6/7] Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator
    )
    
    print(f"  ✓ Trainer ready")
    
    # Train
    print(f"\n[7/7] Starting training...")
    print(f"  Output directory: {output_dir}")
    print(f"  This will take approximately {n_samples / 100 * 2:.0f}-{n_samples / 100 * 3:.0f} minutes\n")
    
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        
        # Save model
        print(f"\nSaving model...")
        model.save_pretrained(os.path.join(output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        
        # Save metadata
        metadata = {
            "character": "LinDaiyu",
            "n_samples": n_samples,
            "base_model": BASE_MODEL,
            "training_time_minutes": training_time / 60,
            "completion_time": datetime.now().isoformat(),
            "hardware": f"{n_gpus}x RTX 3090",
            "lora_config": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05
            }
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to: {output_dir}/final_model")
        print(f"✓ Metadata saved")
        
        return output_dir, training_time
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200, choices=[100, 200, 500, 1000])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    output_dir, training_time = train_sft(args.samples, args.output)
    
    if output_dir:
        print(f"\n{'='*80}")
        print("TRAINING SUCCESS")
        print(f"{'='*80}")
        print(f"Output: {output_dir}")
        print(f"Time: {training_time/60:.1f} minutes")
    else:
        print(f"\n{'='*80}")
        print("TRAINING FAILED")
        print(f"{'='*80}")
        sys.exit(1)
