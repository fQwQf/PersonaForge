import json
import os
import time
import argparse
import sys
import urllib.request
import urllib.error
from typing import List, Dict

# Character Definitions (15 characters for scaled comparison)
CHARACTERS = {
    # ========== 红楼梦 (A Dream in Red Mansions) ==========
    "LinDaiyu": {"name": "Lin Daiyu (林黛玉)", "source": "Dream of the Red Chamber (红楼梦)", "style": "Sensitive, poetic, sharp-witted, slightly melancholic, defensiveness via sublimation.", "language": "zh"},
    "WangXifeng": {"name": "Wang Xifeng (王熙凤)", "source": "Dream of the Red Chamber (红楼梦)", "style": "Loud, authoritative, cunning, humorous, manipulative, uses rationalization as defense.", "language": "zh"},
    "JiaBaoyu": {"name": "Jia Baoyu (贾宝玉)", "source": "Dream of the Red Chamber (红楼梦)", "style": "Romantic, rebellious against tradition, sensitive to beauty, uses denial as defense.", "language": "zh"},
    "XueBaochai": {"name": "Xue Baochai (薛宝钗)", "source": "Dream of the Red Chamber (红楼梦)", "style": "Calm, composed, pragmatic, diplomatic, uses suppression as defense.", "language": "zh"},
    # ========== 三国演义 (Romance of Three Kingdoms) ==========
    "ZhugeLiang": {"name": "Zhuge Liang (诸葛亮)", "source": "Romance of the Three Kingdoms (三国演义)", "style": "Wise, strategic, loyal, uses classical Chinese, intellectualization as defense.", "language": "zh"},
    "CaoCao": {"name": "Cao Cao (曹操)", "source": "Romance of the Three Kingdoms (三国演义)", "style": "Ambitious, cunning, poetic, ruthless, uses projection as defense.", "language": "zh"},
    "GuanYu": {"name": "Guan Yu (关羽)", "source": "Romance of the Three Kingdoms (三国演义)", "style": "Loyal, proud, righteous, stoic, uses reaction formation as defense.", "language": "zh"},
    "ZhouYu": {"name": "Zhou Yu (周瑜)", "source": "Romance of the Three Kingdoms (三国演义)", "style": "Brilliant strategist, proud, musical talent, uses displacement as defense.", "language": "zh"},
    # ========== A Song of Ice and Fire ==========
    "TyrionLannister": {"name": "Tyrion Lannister", "source": "A Song of Ice and Fire", "style": "Witty, cynical, intellectual, self-deprecating humor as defense.", "language": "en"},
    "DaenerysTargaryen": {"name": "Daenerys Targaryen", "source": "A Song of Ice and Fire", "style": "Regal, determined, idealistic, uses rationalization as defense.", "language": "en"},
    "JonSnow": {"name": "Jon Snow", "source": "A Song of Ice and Fire", "style": "Honorable, brooding, reluctant leader, uses altruism as defense.", "language": "en"},
    "CerseiLannister": {"name": "Cersei Lannister", "source": "A Song of Ice and Fire", "style": "Cunning, ruthless, protective of family, uses projection as defense.", "language": "en"},
    "AryaStark": {"name": "Arya Stark", "source": "A Song of Ice and Fire", "style": "Fierce, independent, vengeful, uses identification as defense.", "language": "en"},
    "SansaStark": {"name": "Sansa Stark", "source": "A Song of Ice and Fire", "style": "Politically savvy, survivor, refined speech, uses intellectualization as defense.", "language": "en"},
    "JaimeLannister": {"name": "Jaime Lannister", "source": "A Song of Ice and Fire", "style": "Arrogant yet honorable underneath, conflicted, uses undoing as defense.", "language": "en"},
}


def generate_prompt(character_key: str, n_samples: int) -> str:
    """
    Generates the prompt for the teacher LLM.
    """
    char = CHARACTERS[character_key]
    lang_instruction = "Use semi-classical Chinese/Baihua flavor." if char["language"] == "zh" else "Use rich, distinct English fitting the fantasy setting."
    
    return f"""
You are an expert in literature, specifically "{char['source']}".
Your task is to generate high-quality instruction-tuning data for fine-tuning an LLM to role-play **{char['name']}**.

The data should be in alpaca format:
- `instruction`: A scenario or a conversational prompt.
- `input`: The context or what the other person said (can be empty).
- `output`: {char['name']}'s response. MUST be authentic to their character:
  - {char['style']}
  - {lang_instruction}

Generate {n_samples} diverse examples covering:
1. Daily interactions.
2. Conflict scenarios (triggering defense mechanisms).
3. Emotional moments.

Format your output strictly as a JSON list of objects. Do not wrap in markdown code blocks if possible, or just plain text.

Example format:
[
  {{
    "instruction": "...",
    "input": "...",
    "output": "..."
  }}
]
"""

def call_gemini_api(api_key: str, model: str, prompt: str) -> str:
    """
    Calls Gemini API via REST using urllib (standard library).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json"
        }
    }
    
    json_data = json.dumps(data).encode("utf-8")
    
    req = urllib.request.Request(url, data=json_data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                result = json.loads(response.read().decode("utf-8"))
                # Extract text from Gemini response structure
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                except KeyError:
                    return "{}" # Empty JSON on failure
            else:
                print(f"API Error: HTTP {response.status}")
                return "{}"
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        # print body for debugging
        print(e.read().decode('utf-8'))
        return "{}"
    except Exception as e:
        print(f"Request Error: {e}")
        return "{}"

def load_config(start_path):
    """
    Search for config.json starting from start_path and going up directories.
    """
    current_dir = os.path.abspath(start_path)
    while True:
        config_path = os.path.join(current_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config.json: {e}")
                return {}
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir: # Reached root
            return {}
        current_dir = parent_dir

def main():
    parser = argparse.ArgumentParser(description="Generate SFT data using Gemini REST API")
    parser.add_argument("--output_dir", type=str, default="experiments/sft/data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples per character")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", help="Teacher model name")
    parser.add_argument("--characters", type=str, default="all", help="Comma-separated list of characters to generate (or 'all')")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API Key")
    
    args = parser.parse_args()
    
    api_key = args.api_key
    
    # improved config loading
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config = load_config(script_dir)
        api_key = config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY")
        
    if not api_key:
        print("Error: GEMINI_API_KEY not found in args, env, or config.json.")
        return

    # Use the discovered key
    print(f"Using API Key: {api_key[:5]}...{api_key[-5:] if len(api_key)>10 else ''}")

    targets = CHARACTERS.keys() if args.characters == "all" else args.characters.split(",")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for char_key in targets:
        if char_key not in CHARACTERS:
            print(f"Skipping unknown character: {char_key}")
            continue
            
        print(f"Generating data for {char_key}...")
        all_data = []
        batch_size = 5
        
        for i in range(0, args.num_samples, batch_size):
            current_batch_size = min(batch_size, args.num_samples - i)
            prompt = generate_prompt(char_key, current_batch_size)
            
            response_text = call_gemini_api(api_key, args.model, prompt)
            
            if not response_text or response_text == "{}":
                print("  Failed to get response, sleeping and retrying...")
                time.sleep(2)
                continue

            try:
                # Clean potential markdown
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                batch_data = json.loads(clean_text)
                
                if isinstance(batch_data, list):
                    all_data.extend(batch_data)
                    print(f"  Generated {len(batch_data)} samples.")
                elif isinstance(batch_data, dict):
                    # Sometimes it wraps in {"data": [...]}
                    if "data" in batch_data and isinstance(batch_data["data"], list):
                         all_data.extend(batch_data["data"])
                         print(f"  Generated {len(batch_data['data'])} samples.")
                    else:
                        # Or maybe just a single object?
                        all_data.append(batch_data)
                        print(f"  Generated 1 sample.")
                else:
                    print(f"  Warning: Expected list, got {type(batch_data)}")
            except Exception as e:
                print(f"  JSON parse error: {e}")
                print(f"  Raw text: {response_text[:100]}...")
                
            time.sleep(1) # Rate limit
            
        # Save to file
        output_path = os.path.join(args.output_dir, f"{char_key}_sft.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_data)} total samples to {output_path}")

if __name__ == "__main__":
    main()
