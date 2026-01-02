import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(model, tokenizer, prompt, system_prompt=""):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--character", type=str, default="LinDaiyu", choices=["LinDaiyu", "WangXifeng", "TyrionLannister"])
    args = parser.parse_args()

    print("Loading model...")
    # 4-bit loading requires bitsandbytes, assuming environment is set up
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Character-specific prompts
    prompts_map = {
        "LinDaiyu": {
            "system": "你是林黛玉。",
            "cases": [
                "宝玉摔玉时，你会怎么做？",
                "有人说你的诗太过于悲苦，不吉利。", 
                "第一次见刘姥姥，你会说什么？"
            ]
        },
        "WangXifeng": {
            "system": "你是王熙凤。",
            "cases": [
                "听说贾瑞对你有非分之想，你打算怎么办？",
                "家里开支不够了，又要办大事，你怎么周转？",
                "面对赵姨娘的挑衅，你会怎么回应？"
            ]
        },
        "TyrionLannister": {
            "system": "You are Tyrion Lannister.",
            "cases": [
                "Your father Tywin just criticized your height. How do you respond?",
                "You are on trial for a crime you didn't commit. What do you say to the court?",
                "Joffrey is being cruel to Sansa. How do you intervene?"
            ]
        }
    }
    
    char_config = prompts_map.get(args.character, prompts_map["LinDaiyu"])
    
    print(f"\n=== Evaluation Start for {args.character} ===\n")
    system_prompt = char_config["system"]
    test_cases = char_config["cases"]
    
    for i, prompt in enumerate(test_cases):
        print(f"Case {i+1}: Input: {prompt}")
        response = generate_response(model, tokenizer, prompt, system_prompt)
        print(f"Output: {response}\n")
        print("-" * 50)

    
    for i, prompt in enumerate(test_cases):
        print(f"Case {i+1}: Input: {prompt}")
        response = generate_response(model, tokenizer, prompt, system_prompt)
        print(f"Output: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
