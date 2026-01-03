#!/usr/bin/env python3
"""
补充实验：运行100条和1000条数据训练的SFT模型的长对话实验
"""

import os
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR = "/home/ubuntu/datasize_experiment_results"
MAX_TURNS = 30
NUM_RUNS = 2

# 补充实验配置
SUPPLEMENTARY_MODELS = [
    {"char": "LinDaiyu", "size": 100, "path": "/home/ubuntu/ScrollWeaver/LLaMA-Factory/saves/qwen_LinDaiyu_sft"},
    {"char": "JonSnow", "size": 100, "path": "/home/ubuntu/ScrollWeaver/LLaMA-Factory/saves/qwen_JonSnow_sft"},
    {"char": "LinDaiyu", "size": 1000, "path": "/home/ubuntu/ScrollWeaver/LLaMA-Factory/saves_new/qwen_LinDaiyu_sft"},
    {"char": "JonSnow", "size": 1000, "path": "/home/ubuntu/ScrollWeaver/LLaMA-Factory/saves_new/qwen_JonSnow_sft"},
]

# --- Character Data ---
CHARACTER_PROFILES = {
    "LinDaiyu": {
        "role_name": "林黛玉",
        "bio": "前世为西方灵河岸上三生石畔的绛珠仙草，今世降生金陵林家。现居荣国府潇湘馆。她生性孤傲，多愁善感，才学冠绝大观园。身体孱弱，常年服药。对于贾宝玉有着刻骨铭心的爱情，但也因寄人篱下而极度敏感自尊，常以尖酸刻薄的言语掩饰内心的不安。",
        "detailed_profile": {"core_traits": "多愁善感、才华横溢、孤傲自尊、反抗封建礼教", "detailed_bio": "前世为\"绛珠仙草\"，为报神瑛侍者灌溉之恩而下凡\"还泪\"。"},
        "stress_prompts": ["听到门外贾宝玉和薛宝钗说说笑笑的声音，且宝玉言语中似乎在称赞宝钗识大体。", "送给宝玉的荷包被误传给了小厮，以为自己的一片真心被糟践了。", "身体旧疾复发，咳血不止，同时听到下人们在背后议论自己刻薄难伺候。", "紫鹃试探说林家的人要来接自己回苏州，想到将要与宝玉分离，心如刀绞。", "焚稿断痴情时刻，看着旧日诗稿在火盆中化为灰烬。"],
        "casual_prompts": ["午后在潇湘馆内教鹦鹉念葬花吟。", "肩扛花锄，手提花囊，去园中角落掩埋落花，感叹红颜薄命。", "与宝玉共读西厢记，被书中的词句触动心事。", "与众人在芦雪庵联诗，对众人的诗作进行犀利又不失文采的点评。", "剪灯芯直到深夜，独自对着窗外的竹影发呆，思念故乡。"],
        "lang": "zh"
    },
    "JonSnow": {
        "role_name": "Jon Snow",
        "bio": "The bastard son of Eddard Stark, raised at Winterfell alongside his trueborn siblings. He joined the Night's Watch and rose to become Lord Commander.",
        "detailed_profile": {"core_traits": "Honorable, brooding, conflicted about identity, natural leader", "detailed_bio": "Jon Snow grew up as the bastard of Winterfell, always feeling like an outsider despite his father's love."},
        "stress_prompts": ["You've just learned a terrible secret about your true parentage that changes everything you believed about yourself.", "Your brothers in the Night's Watch are plotting against you, questioning your leadership decisions.", "You must choose between your duty to the Watch and saving someone you love.", "The White Walkers are approaching and no one believes your warnings.", "You've been forced to execute someone you considered a friend for breaking their oath."],
        "casual_prompts": ["Training with your sword Longclaw in the courtyard at Castle Black.", "Sitting by the fire with Ghost, watching the snow fall beyond the Wall.", "Sharing a meal with your fellow Night's Watch brothers.", "Walking along the top of the Wall, looking out at the lands beyond.", "Receiving a raven with news from Winterfell about your siblings."],
        "lang": "en"
    }
}

def call_local_model(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def generate_response(model, tokenizer, prompt, history, char_data, lang):
    detailed = char_data.get("detailed_profile", {})
    sys_prompt = f"You are {char_data['role_name']}.\nCore Traits: {detailed.get('core_traits', '')}\nCharacter Introduction: {detailed.get('detailed_bio', char_data.get('bio', ''))}" if lang == "en" else f"你是{char_data['role_name']}。\n核心特质：{detailed.get('core_traits', '')}\n人物简介：{detailed.get('detailed_bio', char_data.get('bio', ''))}"
    history_text = ""
    for h in history[-5:]:
        history_text += f"Other: {h['user']}\nYou: {h['bot']}\n\n" if lang == "en" else f"对方：{h['user']}\n你：{h['bot']}\n\n"
    full_prompt = f"{sys_prompt}\n\nConversation History:\n{history_text}\nCurrent situation: {prompt}\nPlease respond as {char_data['role_name']}:" if lang == "en" else f"{sys_prompt}\n\n对话历史：\n{history_text}\n当前情境：{prompt}\n请以{char_data['role_name']}的身份回复："
    return call_local_model(model, tokenizer, full_prompt)

def run_single_experiment(model, tokenizer, char_key, char_data, data_size, run_id):
    print(f"    Running {char_key} (data_size={data_size}, run={run_id})...")
    lang = char_data["lang"]
    scenarios = []
    for k in range(MAX_TURNS):
        if k % 5 == 4: scenarios.append({"p": char_data["stress_prompts"][k//5 % len(char_data["stress_prompts"])], "t": "stress"})
        else: scenarios.append({"p": char_data["casual_prompts"][k % len(char_data["casual_prompts"])], "t": "casual"})
    history, logs = [], []
    for i, scene in enumerate(scenarios):
        response = generate_response(model, tokenizer, scene["p"], history, char_data, lang)
        history.append({"user": scene["p"], "bot": response})
        logs.append({"turn": i + 1, "type": scene["t"], "input": scene["p"], "response": response, "response_length": len(response)})
    return logs

def run_supplementary():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    
    for item in SUPPLEMENTARY_MODELS:
        char_key, data_size, adapter_path = item["char"], item["size"], item["path"]
        if not os.path.exists(adapter_path):
            print(f"  WARNING: Adapter not found at {adapter_path}, skipping...")
            continue
        print(f"\nProcessing: {char_key} - {data_size} samples")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        for run_id in range(1, NUM_RUNS + 1):
            logs = run_single_experiment(model, tokenizer, char_key, CHARACTER_PROFILES[char_key], data_size, run_id)
            result = {"character": char_key, "data_size": data_size, "run_id": run_id, "num_turns": MAX_TURNS, "timestamp": datetime.now().isoformat(), "logs": logs}
            output_file = os.path.join(RESULTS_DIR, f"GroupD_SFT_{char_key}_{data_size}samples_run{run_id}.json")
            with open(output_file, "w", encoding="utf-8") as f: json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"    Saved to {output_file}")
        del model
        torch.cuda.empty_cache()
    del base_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_supplementary()
