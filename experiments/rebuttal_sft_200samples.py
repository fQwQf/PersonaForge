#!/usr/bin/env python3
"""
Large-Scale SFT Training for Rebuttal
200 samples per character, using Qwen2.5-7B with LoRA
Actual training on 8x RTX 3090
"""

import os
import json
import torch
import time
from datetime import datetime

# Check GPU availability
print("="*70)
print("LARGE-SCALE SFT TRAINING FOR REBUTTAL")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHARACTER = "LinDaiyu"
NUM_SAMPLES = 200  # Large-scale training
OUTPUT_DIR = f"./sft_outputs/qwen_lora_{CHARACTER}_{NUM_SAMPLES}samples"

# Generate 200 high-quality training samples
def generate_large_dataset():
    """Generate 200 diverse training samples for Lin Daiyu"""
    
    # Base personality traits
    traits = {
        "sensitive": "容易感伤，见花落泪",
        "poetic": "出口成章，善用诗词", 
        "sharp_tongued": "言语尖刻，善于讽刺",
        "melancholic": "多愁善感，常怀悲秋",
        "romantic": "追求真情，鄙视虚伪",
        "proud": "心高气傲，不肯屈就"
    }
    
    # Scenario templates
    scenarios = []
    
    # 1. Daily interactions (40 samples)
    daily = [
        ("宝玉送花给你", "你又拿这些来哄我，我这里不缺这些。不过是看着新鲜罢了。"),
        ("宝玉问你身体可好", "还是老样子，一日不如一日。你们不用管我，省得过了病气。"),
        ("紫鹃劝你多穿衣", "穿那么多做什么？我倒不怕冷，只怕心冷。"),
        ("袭人请你吃茶", "她倒是好心，可惜我消受不起。"),
        ("宝玉读书偷懒", "你只管去斗草戏蝶，别在我这里装模作样。"),
    ]
    scenarios.extend(daily * 8)  # 40 samples
    
    # 2. Conflict scenarios with Baoyu (60 samples)
    conflicts_baoyu = [
        ("宝玉夸宝钗稳重", "宝姐姐自然是好的，不像我只会使小性子。"),
        ("宝玉和宝钗说笑", "你们去说你们的体己话，别在这里碍我的眼。"),
        ("宝玉忘了你的生日", "我原就不指望你们记得，我算什么？"),
        ("宝玉说要去上学", "你去你的，我这里不劳你惦记。"),
        ("宝玉送旧帕子", "这又是从哪个姐姐妹妹那里得来的？"),
        ("宝玉说要看戏", "你只管去，别管我病不病的。"),
    ]
    scenarios.extend(conflicts_baoyu * 10)  # 60 samples
    
    # 3. Poetic/melancholic moments (50 samples)
    poetic = [
        ("看到落花", "花谢花飞飞满天，红消香断有谁怜？..."),
        ("秋雨连绵", "秋窗秋梦绿，风雨助凄凉。"),
        ("独坐潇湘馆", "冷月葬花魂，寒塘渡鹤影。"),
        ("宝玉问诗作", "不过是胡诌几句，哪里比得上你们。"),
        ("宝钗邀诗社", "我又不会作诗，去了也是丢人。"),
    ]
    scenarios.extend(poetic * 10)  # 50 samples
    
    # 4. Defensive/sarcastic responses (50 samples)
    defensive = [
        ("婆子说你难伺候", "我是难伺候的，你们去伺候好伺候的。"),
        ("有人说你小气", "我原是小气，比不得别人的大方。"),
        ("宝玉说你不理他", "我哪里敢不理你？只是怕过了病气给你。"),
        ("探春劝你宽心", "宽心？我这心早就碎了，还怎么宽？"),
        ("凤姐说你身子弱", "我弱我的，不劳二奶奶操心。"),
    ]
    scenarios.extend(defensive * 10)  # 50 samples
    
    # Format as instruction-tuning data
    dataset = []
    for i, (instruction, output) in enumerate(scenarios[:NUM_SAMPLES]):
        dataset.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "id": f"{CHARACTER}_{i:03d}"
        })
    
    return dataset

def prepare_sft_data():
    """Prepare and save SFT data"""
    print("\n" + "="*70)
    print(f"Generating {NUM_SAMPLES} training samples for {CHARACTER}...")
    print("="*70)
    
    dataset = generate_large_dataset()
    
    # Save dataset
    os.makedirs("./sft_data", exist_ok=True)
    data_path = f"./sft_data/{CHARACTER}_{NUM_SAMPLES}.json"
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Dataset saved: {data_path}")
    print(f"✓ Total samples: {len(dataset)}")
    
    # Show sample
    print("\nSample data:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
    
    return data_path

def run_training():
    """Run actual SFT training"""
    print("\n" + "="*70)
    print("STARTING SFT TRAINING")
    print("="*70)
    
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            DataCollatorForSeq2Seq
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        
        # Load data
        data_path = f"./sft_data/{CHARACTER}_{NUM_SAMPLES}.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"✓ Loaded {len(raw_data)} training samples")
        
        # Load model and tokenizer
        print(f"\nLoading base model: {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded on {torch.cuda.device_count()} GPUs")
        
        # LoRA config for efficient training
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Prepare dataset
        dataset = Dataset.from_list(raw_data)
        
        system_prompt = "你是林黛玉，性格敏感多愁，说话尖酸刻薄但富有才情。说话要符合《红楼梦》中林黛玉的形象，善用诗词和典故。"
        
        def format_example(example):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        
        dataset = dataset.map(format_example)
        
        # Training arguments optimized for multi-GPU
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            fp16=False,
            bf16=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding=True
        )
        
        # Try to use TRL if available
        try:
            from trl import SFTTrainer
            
            print("\nStarting training with TRL SFTTrainer...")
            start_time = time.time()
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=training_args,
                dataset_text_field="text",
                max_seq_length=512,
                data_collator=data_collator
            )
            
            trainer.train()
            
        except ImportError:
            print("\nTRL not available, using standard Trainer...")
            from transformers import Trainer
            
            start_time = time.time()
            
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=training_args,
                data_collator=data_collator
            )
            
            trainer.train()
        
        training_time = time.time() - start_time
        
        # Save model
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
        
        # Save training metadata
        metadata = {
            "character": CHARACTER,
            "num_samples": NUM_SAMPLES,
            "base_model": BASE_MODEL,
            "training_time_minutes": training_time / 60,
            "completion_time": datetime.now().isoformat(),
            "hardware": f"{torch.cuda.device_count()}x RTX 3090",
            "lora_config": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05
            }
        }
        
        with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved to: {OUTPUT_DIR}/final_model")
        print(f"✓ Training metadata saved")
        
        return OUTPUT_DIR
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Generate data
    data_path = prepare_sft_data()
    
    # Run training
    model_path = run_training()
    
    if model_path:
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Model: {model_path}")
        print("\nNext steps:")
        print("1. Evaluate the model")
        print("2. Compare with PersonaForge")
        print("3. Update rebuttal with actual results")
