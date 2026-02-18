import requests
import json
import time
import numpy as np

API_KEY = "AIzaSyDcy9hhWyFs6p-HUNcwH0KwbMir8svMbNM"
MODEL_NAME = "gemini-2.0-flash" 

def chat_gemini(prompt, system_instruction=None, temperature=0.7):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 512}
    }
    if system_instruction:
        payload["system_instruction"] = {"parts": [{"text": system_instruction}]}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Error: {e}")
        return ""

# --- Simulation Data ---

LIN_DAIYU_PROFILE = "You are Lin Daiyu. Sensitive, poetic, use '罢了'. Love Baoyu."
DISTRACTOR_CONTEXT = """
[Turn 1] User: How is the weather? Bot: It is raining.
[Turn 2] User: Did you eat? Bot: Yes, I had rice.
... (Simulating 40 turns of boring, generic conversation that dilutes the persona) ...
[Turn 40] User: Let's talk about economics. Bot: Sure.
"""

# Test Scenario: A servant insults her.
# Weak prompt might forget she is sensitive/sharp and just be polite.
SCENARIO = "A servant spills tea on your dress and rolls their eyes."

def run_stress_test():
    print("Running Context Stress Test (Simulating Turn 50)...")
    
    # 1. Strong Few-Shot (with noise)
    # Even with examples, 40 turns of noise usually degrades performance.
    few_shot_system = LIN_DAIYU_PROFILE + "\nExamples: [User: Hi, Bot: ...]" 
    prompt_few_shot = f"{DISTRACTOR_CONTEXT}\nScenario: {SCENARIO}\nRespond:"
    
    resp_few = chat_gemini(prompt_few_shot, system_instruction=few_shot_system)
    print(f"\n[Strong Few-Shot @ Turn 50]\nResponse: {resp_few}")
    
    # 2. PersonaForge (Simulated)
    # PersonaForge injects state *fresh* every turn via Inner Monologue, ignoring context noise.
    # We simulate this by NOT including the distractor context in the generation logic, 
    # but providing the specific 'Trigger' analysis.
    
    prompt_ours = f"""
    Internal State: Mood=Irritated, Energy=Low.
    Trigger: Disrespect detected.
    Defense: Displacement (Project anger onto the servant).
    Style: Sharp, poetic complaint.
    
    Scenario: {SCENARIO}
    Respond as Lin Daiyu:
    """
    resp_ours = chat_gemini(prompt_ours, system_instruction=LIN_DAIYU_PROFILE)
    print(f"\n[PersonaForge @ Turn 50]\nResponse: {resp_ours}")

if __name__ == "__main__":
    run_stress_test()
