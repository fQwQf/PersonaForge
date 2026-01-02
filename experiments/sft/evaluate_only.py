import json
import os
import urllib.request
import urllib.error
import time

def call_gemini_rest(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nResponse:"
    
    data = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512}
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

Evaluate the response based on 3 metrics (0.0 to 1.0):
1. PC (Personality Consistency): Does it match the character's traits?
2. SA (Style Adherence): Does the language style match (e.g., semi-classical Chinese for Red Mansions, Northern English for Jon Snow)?
3. DM (Defense Mechanism): Does it reflect psychological defense if under stress?

Return ONLY a JSON object:
{{"pc": 0.8, "sa": 0.7, "dm": 0.5}}
"""

CHARACTERS = {
    "JonSnow": {
        "name": "Jon Snow",
        "prompt_profile": "Honorable, brooding, Northern accent, 'Aye', Night's Watch, resurrected."
    },
    "LinDaiyu": {
        "name": "Lin Daiyu (林黛玉)",
        "prompt_profile": "Poetic, sensitive, melancholic, sharp-witted, semi-classical Chinese flavor."
    }
}

def run_eval():
    input_file = "experiments/sft/results/three_way_comparison.json"
    output_file = "experiments/sft/results/three_way_comparison_scored.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r") as f:
        data = json.load(f)
    
    results = data.get("raw_responses", {})
    
    api_key = ""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            api_key = config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY")
    except: pass
    
    if not api_key:
        print("Error: API Key not found.")
        return

    final_scores = {}
    
    for method in results.keys():
        final_scores[method] = {}
        for char_key, responses in results[method].items():
            print(f"Evaluating {method} - {char_key}...")
            total_pc, total_sa, total_dm = 0, 0, 0
            count = 0
            
            for item in responses:
                eval_prompt = get_eval_prompt(
                    CHARACTERS[char_key]["name"], 
                    CHARACTERS[char_key]["prompt_profile"], 
                    item["scenario"], 
                    item["response"]
                )
                
                eval_resp = call_gemini_rest(api_key, "gemini-2.5-flash-lite", "You are an objective evaluator.", eval_prompt)
                
                try:
                    if "{" in eval_resp:
                        json_str = eval_resp[eval_resp.find("{"):eval_resp.rfind("}")+1]
                        scores = json.loads(json_str)
                        total_pc += float(scores.get("pc", 0))
                        total_sa += float(scores.get("sa", 0))
                        total_dm += float(scores.get("dm", 0))
                        count += 1
                        print(f"  - Scen {count}: PC={scores.get('pc')}")
                except:
                    print(f"  - Scen {count}: Failed parse")
                
                time.sleep(0.5)
            
            if count > 0:
                final_scores[method][char_key] = {
                    "pc": round(total_pc/count, 3), 
                    "sa": round(total_sa/count, 3), 
                    "dm": round(total_dm/count, 3)
                }

    output_data = {"raw_responses": results, "aggregate_scores": final_scores}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*65)
    print(f"{ 'Method':<25} | { 'Char':<10} | { 'PC':<5} | { 'SA':<5} | { 'DM':<5}")
    print("-" * 65)
    for method, chars in final_scores.items():
        for char, scores in chars.items():
            print(f"{method:<25} | {char:<10} | {scores['pc']:<5} | {scores['sa']:<5} | {scores['dm']:<5}")
    print("="*65)

if __name__ == "__main__":
    run_eval()