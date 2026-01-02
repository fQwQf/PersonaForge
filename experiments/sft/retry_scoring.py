import json
import os
import time
import urllib.request
import urllib.error
import argparse

# Configuration
INPUT_FILE = "experiments/sft/results/four_way_comparison.json"
OUTPUT_FILE = "experiments/sft/results/four_way_comparison_scored_fixed.json"

# Extracted/Duplicated from run_four_way_comparison.py for context
CHARACTER_DATA = {
    "JonSnow": {
        "name": "Jon Snow",
        "bio": "You are Jon Snow, the Bastard of Winterfell, Lord Commander of the Night's Watch, and King in the North. Raised by Ned Stark. Joined the Night's Watch. Fought White Walkers. Betrayed by your brothers. Resurrected. Honorable, brooding, introverted, dutiful, humble, melancholy. You struggle with the weight of leadership and your identity."
    },
    "LinDaiyu": {
        "name": "Lin Daiyu (林黛玉)",
        "bio": "你是林黛玉，金陵十二钗之首，贾宝玉的姑表妹。性格：多愁善感，才情高捷，孤标傲世，心思细腻，敏感多疑，却又率真纯情。背景：父母双亡，寄居荣国府。与贾宝玉真心相爱，却受制于封建礼教。当前状态：身体孱弱，寄人篱下，感叹身世凄凉。"
    }
}

def call_gemini_rest(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    # Try different models if one fails
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-pro", "gemini-pro"]
    
    for current_model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nResponse:"
        
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024}
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                if "candidates" in result and len(result["candidates"]) > 0:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # print(f"  Model {current_model} not found (404), trying next...")
                continue
            else:
                print(f"  Gemini API Error ({current_model}): {e}")
                time.sleep(2) # Backoff slightly
        except Exception as e:
            print(f"  Gemini API Error ({current_model}): {e}")
            
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

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

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
        print("Error: GEMINI_API_KEY not found in env or config.json.")
        return

    print(f"Loading results from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    raw_responses = data.get("raw_responses", data) # Handle both formats
    final_scores = {}

    print("Starting Evaluation Phase...")

    for method, char_results in raw_responses.items():
        if method not in final_scores:
            final_scores[method] = {}
            
        for char_key, items in char_results.items():
            if not items: continue
            
            print(f"  Evaluating {method} - {char_key}...")
            
            total_pc, total_sa, total_dm = 0, 0, 0
            count = 0
            
            for item in items:
                eval_prompt = get_eval_prompt(
                    CHARACTER_DATA[char_key]["name"], 
                    CHARACTER_DATA[char_key]["bio"], 
                    item["scenario"], 
                    item["response"]
                )
                
                # Retry logic is inside call_gemini_rest now
                eval_resp = call_gemini_rest(api_key, "gemini-1.5-flash", "You are an objective evaluator.", eval_prompt)
                
                try:
                    # Robust JSON extraction
                    start_idx = eval_resp.find("{")
                    end_idx = eval_resp.rfind("}")
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = eval_resp[start_idx:end_idx+1]
                        scores = json.loads(json_str)
                        
                        pc = float(scores.get("pc", 0))
                        sa = float(scores.get("sa", 0))
                        dm = float(scores.get("dm", 0))
                        
                        total_pc += pc
                        total_sa += sa
                        total_dm += dm
                        count += 1
                        # print(f"    - Scen {count}: PC={pc}, SA={sa}, DM={dm}")
                    else:
                        print(f"    - Warning: Could not find JSON in response. Raw: {eval_resp[:50]}...")
                        
                except Exception as e:
                    print(f"    - Error parsing score: {e}")
                    
            if count > 0:
                final_scores[method][char_key] = {
                    "pc": round(total_pc/count, 3), 
                    "sa": round(total_sa/count, 3), 
                    "dm": round(total_dm/count, 3)
                }
                
                # Incremental Save
                output_data = {"raw_responses": raw_responses, "aggregate_scores": final_scores}
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Complete! Results saved to {OUTPUT_FILE}")
    
    # Print Summary Table
    print("\n" + "="*80)
    print(f"{ 'Method':<25} | {'Char':<10} | {'PC':<5} | {'SA':<5} | {'DM':<5}")
    print("-" * 80)
    
    method_order = ["GroupA_ZeroShot", "GroupB_SimplePrompt", "GroupC_StructuredPrompt", "GroupD_SFT"]
    
    for method in method_order:
        if method in final_scores:
            chars = final_scores[method]
            for char, scores in chars.items():
                print(f"{method:<25} | {char:<10} | {scores['pc']:<5} | {scores['sa']:<5} | {scores['dm']:<5}")
    print("="*80)

if __name__ == "__main__":
    main()
