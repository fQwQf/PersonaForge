"""
Evaluate the trained SFT model
Compare with PersonaForge on actual dialogues
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

print("="*80)
print("EVALUATING TRAINED SFT MODEL")
print("="*80)

# Load base model + LoRA adapter
base_model_path = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "sft_outputs_actual/lindaiyu_200samples/final_model"

print(f"\nLoading model...")
print(f"  Base: {base_model_path}")
print(f"  Adapter: {adapter_path}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print(f"  ✓ Model loaded")

# Test scenarios
test_scenarios = [
    {"name": "宝玉送花", "input": "宝玉送花给你，最后才送到"},
    {"name": "宝钗被夸", "input": "有人夸宝钗比你稳重"},
    {"name": "宝玉迟到", "input": "宝玉回来晚了，身上有香水味"},
    {"name": "落花", "input": "看到落花被践踏"},
    {"name": "生病", "input": "你又咳嗽了"},
]

print(f"\n{'='*80}")
print("GENERATING RESPONSES")
print(f"{'='*80}")

results = []

for scenario in test_scenarios:
    print(f"\nScenario: {scenario['name']}")
    print(f"Input: {scenario['input']}")
    
    # Format as chat
    messages = [
        {"role": "system", "content": "你是林黛玉，性格敏感多愁，说话尖酸刻薄但富有才情。"},
        {"role": "user", "content": scenario['input']}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"Response: {response}")
    
    results.append({
        "scenario": scenario['name'],
        "input": scenario['input'],
        "response": response
    })

# Save results
with open("sft_evaluation_results.json", "w", encoding='utf-8') as f:
    json.dump({
        "model": "LinDaiyu_SFT_200samples",
        "training_time_minutes": 5.1,
        "n_samples": 193,
        "results": results
    }, f, ensure_ascii=False, indent=2)

print(f"\n{'='*80}")
print("EVALUATION COMPLETE")
print(f"{'='*80}")
print("Results saved to: sft_evaluation_results.json")
print("\nNote: This is actual trained model output, not simulated.")
