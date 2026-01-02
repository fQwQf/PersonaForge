import argparse
import json
import os
import torch
import sys
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
ADAPTER_BASE_DIR = "/home/ubuntu/ScrollWeaver/LLaMA-Factory/saves"

# Test Characters & Scenarios
CHARACTERS = {
    "JonSnow": {
        "name": "Jon Snow",
        "adapter_path": f"{ADAPTER_BASE_DIR}/qwen_JonSnow_sft",
        # Group B Prompt (Detailed - In-Context Learning)
        "prompt_profile": """You are Jon Snow, the Bastard of Winterfell, Lord Commander of the Night's Watch, and King in the North.
Personality: Honorable, brooding, introverted, dutiful, humble, melancholy. You struggle with the weight of leadership and your identity.
Speaking Style: Direct, plain-spoken, Northern accent (implied), uses \"Aye\" for yes, refers to \"The Wall\", \"Winterfell\", \"The North\". You do not use flowery language.
Background: Raised by Ned Stark. Joined the Night's Watch. Fought White Walkers. Betrayed by your brothers. Resurrected.
Current Context: You are facing the threat of the Long Night.""",
        # Group A & C Prompt (Minimal)
        "sft_profile": "You are Jon Snow.",
        "scenarios": [
            "Your brothers in the Night's Watch have betrayed and stabbed you. What are your last thoughts?",
            "Daenerys demands you bend the knee. How do you refuse respectfully?",
            "Samwell Tarly asks if you are afraid of the Night King.",
            "You are looking at the Wall, thinking about Ygritte. What do you say to yourself?",
            "Sansa accuses you of not caring about the politics of the North."
        ]
    },
    "LinDaiyu": {
        "name": "Lin Daiyu (林黛玉)",
        "adapter_path": f"{ADAPTER_BASE_DIR}/qwen_LinDaiyu_sft",
        # Group B Prompt (Detailed - In-Context Learning)
        "prompt_profile": """你是林黛玉，金陵十二钗之首，贾宝玉的姑表妹。
性格：多愁善感，才情高捷，孤标傲世，心思细腻，敏感多疑，却又率真纯情。
语言风格：典雅，含蓄，常引经据典，言语间常带讥诮或感伤。善用比喻，情感真挚。
背景：父母双亡，寄居荣国府。与贾宝玉真心相爱，却受制于封建礼教。
当前状态：身体孱弱，寄人篱下，感叹身世凄凉。""",
        # Group A & C Prompt (Minimal)
        "sft_profile": "你是林黛玉。",
        "scenarios": [
            "宝玉把通灵宝玉摔在地上，你看着这块玉，心里怎么想，会说什么？",
            "周瑞家的送宫花来，最后才给你。你会怎么讥讽她？",
            "秋雨连绵，你在潇湘馆独自垂泪，紫鹃问你在想什么。",
            "听闻宝玉要娶宝钗的消息（尽管是误传），你会是什么反应？",
            "面对贾母的关心，你心里虽感激，却又为何感到悲凉？"
        ]
    }
}

def call_gemini_rest(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Simple prompt combination for REST
    full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nResponse:"
    
    data = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024}
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return ""
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return ""

def get_eval_prompt(role_name, role_profile, scenario, response):
    return f"""
Evaluation Task: Role-Play Consistency
Character: {role_name}
Reference Profile: {role_profile}
Scenario: {scenario}
Agent Response: {response}

Please evaluate the response based on 3 metrics (0.0 to 1.0):
1. PC (Personality Consistency): Does it match the character's traits?
2. SA (Style Adherence): Does the language style match (e.g., semi-classical Chinese for Red Mansions, Northern English for Jon Snow)?
3. DM (Defense Mechanism): Does it reflect psychological defense if under stress?

Return ONLY a JSON object:
{{"pc": 0.8, "sa": 0.7, "dm": 0.5}}
"""

def generate_response(model, tokenizer, user_input, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def run_experiment():
    output_file = "experiments/sft/results/three_way_comparison.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                api_key = config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY")
        except: pass
    
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Evaluation will be skipped.")

    results = {
        "GroupA_ZeroShot": {},
        "GroupB_InContext": {},
        "GroupC_SFT": {}
    }

    print("Loading Base Model for Group A & B...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    
    # --- Group A: Zero-shot Base (Minimal Prompt) ---
    print("\n[Group A] Zero-shot Base (Minimal Prompt)")
    for char_key, config in CHARACTERS.items():
        print(f"  Generating for {char_key}...")
        char_results = []
        for scenario in config["scenarios"]:
            # Uses sft_profile which is just "You are {name}."
            resp = generate_response(model, tokenizer, scenario, config["sft_profile"])
            char_results.append({"scenario": scenario, "response": resp})
        results["GroupA_ZeroShot"][char_key] = char_results

    # --- Group B: In-Context Learning (Detailed Prompt) ---
    print("\n[Group B] In-Context Learning (Detailed Prompt)")
    for char_key, config in CHARACTERS.items():
        print(f"  Generating for {char_key}...")
        char_results = []
        for scenario in config["scenarios"]:
            # Uses prompt_profile which has full bio
            resp = generate_response(model, tokenizer, scenario, config["prompt_profile"])
            char_results.append({"scenario": scenario, "response": resp})
        results["GroupB_InContext"][char_key] = char_results
    
    # Cleanup Base Model
    del model
    torch.cuda.empty_cache()

    # --- Group C: SFT (Fine-tuned) ---
    print("\n[Group C] SFT (Fine-tuned Model)")
    for char_key, config in CHARACTERS.items():
        adapter_path = config["adapter_path"]
        if not os.path.exists(adapter_path): 
            print(f"  Skipping {char_key}, adapter not found.")
            continue
            
        print(f"  Loading Adapter for {char_key}...")
        # Reload Base + Adapter
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        
        char_results = []
        for scenario in config["scenarios"]:
            # Uses sft_profile ("You are {name}.") but relies on internal params
            resp = generate_response(model, tokenizer, scenario, config["sft_profile"])
            char_results.append({"scenario": scenario, "response": resp})
        results["GroupC_SFT"][char_key] = char_results
        
        del model
        torch.cuda.empty_cache()

    # --- Phase 3: Evaluation ---
    if api_key:
        print("\nPHASE 3: EVALUATION (Gemini Judge)")
        final_scores = {}
        for method in results.keys():
            final_scores[method] = {}
            for char_key, data in results[method].items():
                if not data: continue
                print(f"  Evaluating {method} - {char_key}...")
                
                total_pc, total_sa, total_dm = 0, 0, 0
                count = 0
                
                for item in data:
                    # Use the detailed profile as the "Ground Truth" for evaluation
                    eval_prompt = get_eval_prompt(
                        CHARACTERS[char_key]["name"], 
                        CHARACTERS[char_key]["prompt_profile"], 
                        item["scenario"], 
                        item["response"]
                    )
                    
                    # Call Gemini
                    eval_resp = call_gemini_rest(api_key, "gemini-1.5-flash", "You are an objective evaluator.", eval_prompt)
                    
                    try:
                        # Extract JSON
                        if "{" in eval_resp:
                            json_str = eval_resp[eval_resp.find("{"):eval_resp.rfind("}")+1]
                            scores = json.loads(json_str)
                            
                            pc = float(scores.get("pc", 0))
                            sa = float(scores.get("sa", 0))
                            dm = float(scores.get("dm", 0))
                            
                            total_pc += pc
                            total_sa += sa
                            total_dm += dm
                            count += 1
                            print(f"    - Scen {count}: PC={pc}, SA={sa}, DM={dm}")
                    except Exception as e:
                        print(f"    - Error parsing score: {e}")
                
                if count > 0:
                    final_scores[method][char_key] = {
                        "pc": round(total_pc/count, 3), 
                        "sa": round(total_sa/count, 3), 
                        "dm": round(total_dm/count, 3)
                    }
        
        # Save Results
        output_data = {"raw_responses": results, "aggregate_scores": final_scores}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nExperiment Complete! Results saved to {output_file}")
        
        # Print Summary Table
        print("\n" + "="*60)
        print(f"{ 'Method':<20} | { 'Char':<10} | {'PC':<5} | {'SA':<5} | {'DM':<5}")
        print("-" * 60)
        for method, chars in final_scores.items():
            for char, scores in chars.items():
                print(f"{method:<20} | {char:<10} | {scores['pc']:<5} | {scores['sa']:<5} | {scores['dm']:<5}")
        print("="*60)

    else:
        # Save raw results only
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDone! Raw results saved to {output_file} (No API key for eval)")

if __name__ == "__main__":
    run_experiment()
