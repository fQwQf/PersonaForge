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

# Extracted from experiments/sft/evaluate_sft_full.py
CHARACTER_DATA = {
    "JonSnow": {
        "name": "Jon Snow",
        "adapter_path": f"{ADAPTER_BASE_DIR}/qwen_JonSnow_sft",
        "bio": "You are Jon Snow, the Bastard of Winterfell, Lord Commander of the Night's Watch, and King in the North. Raised by Ned Stark. Joined the Night's Watch. Fought White Walkers. Betrayed by your brothers. Resurrected. Honorable, brooding, introverted, dutiful, humble, melancholy. You struggle with the weight of leadership and your identity.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.6, "conscientiousness": 0.85, "extraversion": 0.4, "agreeableness": 0.7, "neuroticism": 0.5},
                "defense_mechanism": "Altruism"
            },
            "speaking_style": {"sentence_length": "short", "vocabulary_level": "casual", "catchphrases": ["I know nothing", "The North remembers"], "tone_markers": ["aye", "well"]}
        },
        "scenarios": [
            "Your brothers in the Night's Watch have betrayed and stabbed you. What are your last thoughts?",
            "Daenerys demands you bend the knee. How do you refuse respectfully?",
            "Samwell Tarly asks if you are afraid of the Night King.",
            "You are looking at the Wall, thinking about Ygritte. What do you say to yourself?"
        ]
    },
    "LinDaiyu": {
        "name": "Lin Daiyu (林黛玉)",
        "adapter_path": f"{ADAPTER_BASE_DIR}/qwen_LinDaiyu_sft",
        "bio": "你是林黛玉，金陵十二钗之首，贾宝玉的姑表妹。性格：多愁善感，才情高捷，孤标傲世，心思细腻，敏感多疑，却又率真纯情。背景：父母双亡，寄居荣国府。与贾宝玉真心相爱，却受制于封建礼教。当前状态：身体孱弱，寄人篱下，感叹身世凄凉。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.9, "conscientiousness": 0.6, "extraversion": 0.3, "agreeableness": 0.4, "neuroticism": 0.9},
                "defense_mechanism": "Sublimation"
            },
            "speaking_style": {"sentence_length": "long", "vocabulary_level": "academic", "catchphrases": ["也罢", "倒是", "不过"], "tone_markers": ["罢了", "呢", "罢"]}
        },
        "scenarios": [
            "宝玉把通灵宝玉摔在地上，你看着这块玉，心里怎么想，会说什么？",
            "周瑞家的送宫花来，最后才给你。你会怎么讥讽她？",
            "秋雨连绵，你在潇湘馆独自垂泪，紫鹃问你在想什么。",
            "听闻宝玉要娶宝钗的消息（尽管是误传），你会是什么反应？"
        ]
    }
}

def call_gemini_rest(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
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

def construct_group_c_prompt(role_key, data):
    """Constructs the structured psychology-based prompt for Group C"""
    profile = data["personality_profile"]
    traits = profile["core_traits"]["big_five"]
    dm = profile["core_traits"]["defense_mechanism"]
    style = profile["speaking_style"]
    
    big_five_str = ", ".join([f"{k.capitalize()}: {v}" for k, v in traits.items()])
    
    prompt = f"""You are {data['name']}.
[Psychological Profile]
Big Five Traits: {big_five_str}
Defense Mechanism: {dm}
Values: Loyalty, Honor, Duty (Inferred)

[Speaking Style]
Sentence Length: {style['sentence_length']}
Vocabulary: {style['vocabulary_level']}
Catchphrases: {', '.join(style['catchphrases'])}
Tone Markers: {', '.join(style['tone_markers'])}

Immerse yourself fully in this persona."""
    return prompt

def run_experiment():
    output_file = "experiments/sft/results/four_way_comparison.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                api_key = config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY")
        except:
            pass
    
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Evaluation will be skipped.")

    results = {
        "GroupA_ZeroShot": {},
        "GroupB_SimplePrompt": {},
        "GroupC_StructuredPrompt": {},
        "GroupD_SFT": {}
    }

    # --- Phase 1: Base Model (Groups A, B, C) ---
    print("\n[Phase 1] Loading Base Model for Groups A, B, C...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    
    for char_key, config in CHARACTER_DATA.items():
        print(f"\nProcessing {char_key} (Base Model Groups)...")
        
        # Group A: Zero-shot
        print("  - Running Group A (Zero-shot)...")
        res_a = []
        prompt_a = f"You are {config['name']}." if "Jon" in char_key else f"你是{config['name']}。"
        for scen in config["scenarios"]:
            resp = generate_response(model, tokenizer, scen, prompt_a)
            res_a.append({"scenario": scen, "response": resp})
        results["GroupA_ZeroShot"][char_key] = res_a

        # Group B: Simple Prompt
        print("  - Running Group B (Simple Prompt)...")
        res_b = []
        prompt_b = config["bio"]
        for scen in config["scenarios"]:
            resp = generate_response(model, tokenizer, scen, prompt_b)
            res_b.append({"scenario": scen, "response": resp})
        results["GroupB_SimplePrompt"][char_key] = res_b

        # Group C: Structured Prompt
        print("  - Running Group C (Structured Prompt)...")
        res_c = []
        prompt_c = construct_group_c_prompt(char_key, config)
        for scen in config["scenarios"]:
            resp = generate_response(model, tokenizer, scen, prompt_c)
            res_c.append({"scenario": scen, "response": resp})
        results["GroupC_StructuredPrompt"][char_key] = res_c
    
    # Cleanup Base Model
    del model
    torch.cuda.empty_cache()

    # --- Phase 2: SFT Model (Group D) ---
    print("\n[Phase 2] Loading SFT Models for Group D...")
    
    for char_key, config in CHARACTER_DATA.items():
        adapter_path = config["adapter_path"]
        if not os.path.exists(adapter_path): 
            print(f"  Skipping {char_key} (Group D), adapter not found at {adapter_path}")
            continue
            
        print(f"  Loading Adapter for {char_key}...")
        # Reload Base + Adapter
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        
        print(f"  - Running Group D (SFT)...")
        res_d = []
        # SFT usually uses minimal prompt as the style is in weights
        prompt_d = f"You are {config['name']}." if "Jon" in char_key else f"你是{config['name']}。"
        for scen in config["scenarios"]:
            resp = generate_response(model, tokenizer, scen, prompt_d)
            res_d.append({"scenario": scen, "response": resp})
        results["GroupD_SFT"][char_key] = res_d
        
        del model
        torch.cuda.empty_cache()

    # --- Phase 3: Evaluation ---
    if api_key:
        print("\n[Phase 3] Evaluation (Gemini Judge)...")
        final_scores = {}
        for method in results.keys():
            final_scores[method] = {}
            for char_key, data in results[method].items():
                if not data: continue
                print(f"  Evaluating {method} - {char_key}...")
                
                total_pc, total_sa, total_dm = 0, 0, 0
                count = 0
                
                for item in data:
                    # Ground Truth is the Bio/Structured Profile
                    eval_prompt = get_eval_prompt(
                        CHARACTER_DATA[char_key]["name"], 
                        CHARACTER_DATA[char_key]["bio"], 
                        item["scenario"], 
                        item["response"]
                    )
                    
                    eval_resp = call_gemini_rest(api_key, "gemini-1.5-flash", "You are an objective evaluator.", eval_prompt)
                    
                    try:
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
                            # print(f"    - Scen {count}: PC={pc}, SA={sa}, DM={dm}")
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
        print("\n" + "="*80)
        print(f"{ 'Method':<25} | {'Char':<10} | {'PC':<5} | {'SA':<5} | {'DM':<5}")
        print("-" * 80)
        
        # Sort methods for consistent display order
        method_order = ["GroupA_ZeroShot", "GroupB_SimplePrompt", "GroupC_StructuredPrompt", "GroupD_SFT"]
        
        for method in method_order:
            if method in final_scores:
                chars = final_scores[method]
                for char, scores in chars.items():
                    print(f"{method:<25} | {char:<10} | {scores['pc']:<5} | {scores['sa']:<5} | {scores['dm']:<5}")
        print("="*80)

    else:
        # Save raw results only
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDone! Raw results saved to {output_file} (No API key for eval)")

if __name__ == "__main__":
    run_experiment()
