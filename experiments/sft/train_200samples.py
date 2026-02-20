#!/usr/bin/env python3
"""
SFT Training Script for Rebuttal - 200 samples
Uses transformers + PEFT for LoRA training on 4x RTX 3090
"""

import os
import sys
import json
import torch
from pathlib import Path

# Setup paths
sys.path.insert(0, '/data1/tongjizhou/miniconda3/envs/fedlpa_rebuttal/lib/python3.9/site-packages')

print("="*70)
print("LARGE-SCALE SFT TRAINING - 200 SAMPLES")
print("="*70)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPUs: {torch.cuda.device_count()}")

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Will be downloaded
DATA_PATH = "/data1/tongjizhou/fluffy-fishstick/experiments/sft/data_large/LinDaiyu_200.json"
OUTPUT_DIR = "/data1/tongjizhou/fluffy-fishstick/sft_outputs/lindaiyu_200samples"
NUM_EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4

def main():
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        
        print("\n[1/6] Loading data...")
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"✓ Loaded {len(raw_data)} samples")
        
        print("\n[2/6] Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✓ Model loaded on {torch.cuda.device_count()} GPUs")
        
        print("\n[3/6] Configuring LoRA...")
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
        
        print("\n[4/6] Preparing dataset...")
        dataset = Dataset.from_list(raw_data)
        
        system_msg = "你是林黛玉，性格敏感多愁，说话尖酸刻薄但富有才情。"
        
        def format_chat(example):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        
        dataset = dataset.map(format_chat)
        
        print("\n[5/6] Configuring training...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to="none",
            ddp_find_unused_parameters=False
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding=True
        )
        
        print("\n[6/6] Starting training...")
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            data_collator=data_collator
        )
        
        import time
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        
        # Save model
        model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
        
        # Save metadata
        metadata = {
            "character": "LinDaiyu",
            "num_samples": len(raw_data),
            "base_model": BASE_MODEL,
            "training_time_minutes": training_time / 60,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lora_r": 16,
            "lora_alpha": 32,
            "hardware": f"{torch.cuda.device_count()}x RTX 3090"
        }
        
        with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved: {OUTPUT_DIR}/final_model")
        print(f"✓ Metadata saved: {OUTPUT_DIR}/metadata.json")
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Character: Lin Daiyu")
        print(f"Samples: {len(raw_data)}")
        print(f"Training time: {training_time/60:.1f} minutes")
        print(f"Output: {OUTPUT_DIR}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
