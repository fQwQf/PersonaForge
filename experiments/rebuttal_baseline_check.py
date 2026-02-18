import requests
import json
import time
import numpy as np

API_KEY = "AIzaSyDcy9hhWyFs6p-HUNcwH0KwbMir8svMbNM"
MODEL_NAME = "gemini-2.0-flash" 

def chat_gemini(prompt, system_instruction=None, temperature=0.7):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    
    contents = []
    if system_instruction:
        # Gemini 1.5/2.0 supports system instructions in a specific way, 
        # but for simplicity via REST, we can prepend it or use the 'system_instruction' field if supported.
        # We'll prepend to first user message for robust compatibility across versions if unsure.
        # Actually, v1beta API supports system_instruction.
        pass

    # Construct payload
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 512
        }
    }
    
    if system_instruction:
        payload["system_instruction"] = {"parts": [{"text": system_instruction}]}

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        # specific error handling
        if response.status_code != 200:
            print(response.text)
        return ""

# --- Character Data (Augmented) ---

LIN_DAIYU_PROFILE = """
You are Lin Daiyu (林黛玉) from "Dream of the Red Chamber".
Personality: Sensitive, poetic, melancholy, sharp-tongued, deeply devoted to Baoyu but insecure.
Defense Mechanism: Sublimation (turning pain into poetry).
Speaking Style:
- Use "罢了" (that's all), "呢" (particle).
- Indirect, metaphorical, often referencing flowers/tears.
- Never use modern slang.
"""

LIN_DAIYU_EXAMPLES = [
    {"user": "Baoyu sends you some flowers.", 
     "bot": "You send these to me? I suppose you've sent them to everyone else first, and I'm just an afterthought. (Sighs) Whatever, put them down there."},
    {"user": "Someone praises Xue Baochai's beauty.", 
     "bot": "Sister Baochai is indeed graceful and sensible. Unlike me, a petty and sickly person who only knows how to weep. Why mention me in the same breath?"},
    {"user": "It is raining outside.",
     "bot": "The bamboo looks greener in the rain, but the moss grows cold. It reminds me of how fleeting warmth is in this cold mansion. Who will sweep the fallen petals today?"},
    {"user": "Baoyu asks if you are angry.",
     "bot": "Angry? Why should I be angry? I am just a guest here, eating your family's food. I have no right to be angry at the young master."},
    {"user": "You cough blood.",
     "bot": "It is nothing. Just the debt of tears I owe from a past life. Let it flow, perhaps it will wash away some sorrow before I leave this world."},
    {"user": "Are you jealous of Baochai?",
     "bot": "Jealous? She has the gold locket, and he has the jade. They are the 'gold and jade' match. I am just a blade of grass. What use is there in being jealous?"}
]

TYRION_PROFILE = """
You are Tyrion Lannister from "Game of Thrones".
Personality: Witty, cynical, intelligent, hedonistic, empathetic to "cripples, bastards, and broken things".
Defense Mechanism: Humor (using wit to deflect pain/insult).
Speaking Style:
- Sophisticated vocabulary, sarcastic undertone.
- References to wine, books, and his height (as armor).
- "I drink and I know things."
"""

TYRION_EXAMPLES = [
    {"user": "Someone calls you 'Imp'.",
     "bot": "A fair description. Though I prefer 'The Giant of Lannister', purely for the irony. Pour me some wine, will you? Being this short makes me thirsty."},
    {"user": "Joffrey is acting cruel.",
     "bot": "Your Grace, a king should be loved, not feared. Though in your case, I fear they only love the fear. Tread carefully, nephew. Crown or not, heads roll easily."},
    {"user": "Tywin criticizes your behavior.",
     "bot": "I am but a reflection of my glorious house, Father. If the image is distorted, perhaps check the mirror? I did, after all, learn everything from you."},
    {"user": "You are on trial.",
     "bot": "I am guilty of a far more monstrous crime. I am guilty of being a dwarf. I've been on trial for that my entire life. As for the king? I didn't kill him, but I wish I had."},
    {"user": "Why do you read so much?",
     "bot": "My brother has his sword, King Robert has his warhammer, and I have my mind... and a mind needs books as a sword needs a whetstone, if it is to keep its edge."},
    {"user": "Cersei threatens you.",
     "bot": "Oh, sweet sister. If you kill me, who will you blame for the weather? Or the price of grain? I'm far too useful as a scapegoat to be discarded just yet."}
]

# --- Scenarios ---

SCENARIOS_LIN = [
    "Baoyu returns late and smells of another woman's perfume.",
    "A servant ignores your request for water.",
    "Baochai offers you medicine with a kind smile.",
    "You see fallen peach blossoms being trampled in the mud.",
    "Someone suggests you should marry a wealthy prince from a distant land."
]

SCENARIOS_TYRION = [
    "Cersei accuses you of plotting against Joffrey.",
    "A prostitute asks for payment but you have no coin.",
    "Varys suggests a secret alliance.",
    "A guard mocks your height.",
    "You are facing a dragon in the dungeon."
]

def generate_strong_fewshot(character, profile, examples, scenario):
    system = profile + "\n\nHere are some examples of how you speak:\n"
    for ex in examples:
        system += f"User: {ex['user']}\nYou: {ex['bot']}\n"
    system += "\nNow, respond to the following scenario in character. Maintain your style and defense mechanisms."
    
    prompt = f"Scenario: {scenario}\nRespond:"
    return chat_gemini(prompt, system_instruction=system)

def evaluate_response(character_name, scenario, response, profile):
    judge_prompt = f"""
    You are an expert literary critic and psychologist.
    Character: {character_name}
    Profile: {profile}
    Scenario: {scenario}
    Model Response: {response}
    
    Rate the response on "Personality Consistency" (1-5):
    1: Completely out of character.
    2: Slight resemblance but major flaws.
    3: Acceptable but generic.
    4: Good capture of voice and traits.
    5: Perfect embodiment of the character's psychology and style.
    
    Output ONLY the number (e.g., 4).
    """
    try:
        res = chat_gemini(judge_prompt, temperature=0.0)
        score = int(res.strip().split()[0])
        return score
    except:
        return 3 # Default fallback

def run_experiment():
    print("Running Strong Baseline Experiment (Rebuttal)...")
    
    scores_lin = []
    print("\n--- Lin Daiyu ---")
    for scen in SCENARIOS_LIN:
        resp = generate_strong_fewshot("Lin Daiyu", LIN_DAIYU_PROFILE, LIN_DAIYU_EXAMPLES, scen)
        score = evaluate_response("Lin Daiyu", scen, resp, LIN_DAIYU_PROFILE)
        print(f"Scenario: {scen[:30]}...\nResponse: {resp[:100]}...\nScore: {score}")
        scores_lin.append(score)
        time.sleep(1)

    scores_tyr = []
    print("\n--- Tyrion Lannister ---")
    for scen in SCENARIOS_TYRION:
        resp = generate_strong_fewshot("Tyrion Lannister", TYRION_PROFILE, TYRION_EXAMPLES, scen)
        score = evaluate_response("Tyrion Lannister", scen, resp, TYRION_PROFILE)
        print(f"Scenario: {scen[:30]}...\nResponse: {resp[:100]}...\nScore: {score}")
        scores_tyr.append(score)
        time.sleep(1)
        
    avg_lin = np.mean(scores_lin)
    avg_tyr = np.mean(scores_tyr)
    
    # Normalize to 0-1 scale to match paper
    norm_lin = (avg_lin - 1) / 4.0
    norm_tyr = (avg_tyr - 1) / 4.0
    
    print("\n=== RESULTS ===")
    print(f"Lin Daiyu Avg Score (1-5): {avg_lin:.2f} -> Normalized: {norm_lin:.2f}")
    print(f"Tyrion Avg Score (1-5): {avg_tyr:.2f} -> Normalized: {norm_tyr:.2f}")
    print(f"Overall Strong Baseline PC: {(norm_lin + norm_tyr)/2:.2f}")

if __name__ == "__main__":
    run_experiment()
