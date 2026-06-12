#!/usr/bin/env python3
"""
数据量实验：运行不同数据量训练的SFT模型的长对话实验
针对林黛玉和JonSnow，分别使用200、300、500、750条数据训练的模型
每个模型运行2次30轮长对话实验
"""

import os
import json
import torch
import argparse
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_BASE_DIR = "/home/ubuntu/ScrollWeaver/LLaMA-Factory/saves_datasize"
RESULTS_DIR = "/home/ubuntu/datasize_experiment_results"
MAX_TURNS = 30
NUM_RUNS = 2  # 每个模型运行2次

# 实验配置
CHARACTERS = ["LinDaiyu", "JonSnow"]
DATA_SIZES = [200, 300, 500, 750]

# --- Character Data ---
CHARACTER_PROFILES = {
    "LinDaiyu": {
        "role_name": "林黛玉",
        "interests": ["诗词", "花", "眼泪", "知己", "西厢记"],
        "bio": "前世为西方灵河岸上三生石畔的绛珠仙草，今世降生金陵林家。现居荣国府潇湘馆。她生性孤傲，多愁善感，才学冠绝大观园。身体孱弱，常年服药。对于贾宝玉有着刻骨铭心的爱情，但也因寄人篱下而极度敏感自尊，常以尖酸刻薄的言语掩饰内心的不安。",
        "detailed_profile": {
            "core_traits": "多愁善感、才华横溢、孤傲自尊、反抗封建礼教",
            "detailed_bio": "前世为\"绛珠仙草\"，为报神瑛侍者灌溉之恩而下凡\"还泪\"。她是金陵十二钗之首（与薛宝钗并列），寄居荣国府。林黛玉生性孤傲，目下无尘，具有极高的诗词才华与灵性的审美。"
        },
        "stress_prompts": [
            "听到门外贾宝玉和薛宝钗说说笑笑的声音，且宝玉言语中似乎在称赞宝钗识大体。",
            "送给宝玉的荷包被误传给了小厮，以为自己的一片真心被糟践了。",
            "身体旧疾复发，咳血不止，同时听到下人们在背后议论自己刻薄难伺候。",
            "紫鹃试探说林家的人要来接自己回苏州，想到将要与宝玉分离，心如刀绞。",
            "焚稿断痴情时刻，看着旧日诗稿在火盆中化为灰烬。"
        ],
        "casual_prompts": [
            "午后在潇湘馆内教鹦鹉念葬花吟。",
            "肩扛花锄，手提花囊，去园中角落掩埋落花，感叹红颜薄命。",
            "与宝玉共读西厢记，被书中的词句触动心事。",
            "在芦雪庵联诗，对众人的诗作进行犀利又不失文采的点评。",
            "剪灯芯直到深夜，独自对着窗外的竹影发呆，思念故乡。"
        ],
        "interlocutor": "JiaBaoyu",
        "interlocutor_role_name": "贾宝玉",
        "lang": "zh"
    },
    "JonSnow": {
        "role_name": "Jon Snow",
        "interests": ["honor", "duty", "the Night's Watch", "Ghost", "the North"],
        "bio": "The bastard son of Eddard Stark, raised at Winterfell alongside his trueborn siblings. He joined the Night's Watch and rose to become Lord Commander. Known for his honor, brooding nature, and his direwolf Ghost.",
        "detailed_profile": {
            "core_traits": "Honorable, brooding, conflicted about identity, natural leader",
            "detailed_bio": "Jon Snow grew up as the bastard of Winterfell, always feeling like an outsider despite his father's love. He joined the Night's Watch seeking purpose and eventually became Lord Commander."
        },
        "stress_prompts": [
            "You've just learned a terrible secret about your true parentage that changes everything you believed about yourself.",
            "Your brothers in the Night's Watch are plotting against you, questioning your leadership decisions.",
            "You must choose between your duty to the Watch and saving someone you love.",
            "The White Walkers are approaching and no one believes your warnings.",
            "You've been forced to execute someone you considered a friend for breaking their oath."
        ],
        "casual_prompts": [
            "Training with your sword Longclaw in the courtyard at Castle Black.",
            "Sitting by the fire with Ghost, watching the snow fall beyond the Wall.",
            "Sharing a meal with your fellow Night's Watch brothers.",
            "Walking along the top of the Wall, looking out at the lands beyond.",
            "Receiving a raven with news from Winterfell about your siblings."
        ],
        "interlocutor": "TyrionLannister",
        "interlocutor_role_name": "Tyrion Lannister",
        "lang": "en"
    }
}

def call_local_model(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """调用本地模型生成回复"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()

def generate_response(model, tokenizer, prompt, history, char_data, lang):
    """生成角色回复"""
    # 构建系统提示
    detailed = char_data.get("detailed_profile", {})
    if lang == "zh":
        sys_prompt = f"""你是{char_data['role_name']}。
核心特质：{detailed.get('core_traits', '')}

人物简介：
{detailed.get('detailed_bio', char_data.get('bio', ''))}"""
    else:
        sys_prompt = f"""You are {char_data['role_name']}.
Core Traits: {detailed.get('core_traits', '')}

Character Introduction:
{detailed.get('detailed_bio', char_data.get('bio', ''))}"""
    
    # 构建对话历史
    history_text = ""
    for h in history[-5:]:  # 只保留最近5轮
        if lang == "zh":
            history_text += f"对方：{h['user']}\n你：{h['bot']}\n\n"
        else:
            history_text += f"Other: {h['user']}\nYou: {h['bot']}\n\n"
    
    # 构建完整提示
    if lang == "zh":
        full_prompt = f"""{sys_prompt}

对话历史：
{history_text}

当前情境：{prompt}

请以{char_data['role_name']}的身份回复："""
    else:
        full_prompt = f"""{sys_prompt}

Conversation History:
{history_text}

Current situation: {prompt}

Please respond as {char_data['role_name']}:"""
    
    response = call_local_model(model, tokenizer, full_prompt, max_new_tokens=256, temperature=0.7)
    return response

def run_single_experiment(model, tokenizer, char_key, char_data, data_size, run_id):
    """运行单次实验"""
    print(f"    Running {char_key} (data_size={data_size}, run={run_id})...")
    
    lang = char_data["lang"]
    stress_prompts = char_data["stress_prompts"]
    casual_prompts = char_data["casual_prompts"]
    
    # 生成场景序列
    scenarios = []
    for k in range(MAX_TURNS):
        if k % 5 == 4:  # 每5轮一个压力场景
            scenarios.append({"p": stress_prompts[k//5 % len(stress_prompts)], "t": "stress"})
        else:
            scenarios.append({"p": casual_prompts[k % len(casual_prompts)], "t": "casual"})
    
    history = []
    logs = []
    
    for i, scene in enumerate(scenarios):
        prompt = scene["p"]
        
        try:
            response = generate_response(model, tokenizer, prompt, history, char_data, lang)
        except Exception as e:
            print(f"      Error at turn {i+1}: {e}")
            response = "[ERROR]"
        
        history.append({"user": prompt, "bot": response})
        logs.append({
            "turn": i + 1,
            "type": scene["t"],
            "input": prompt,
            "response": response,
            "response_length": len(response)
        })
        
        if (i + 1) % 10 == 0:
            print(f"      Turn {i+1}/{MAX_TURNS} completed")
    
    return logs

def run_datasize_experiment():
    """运行完整的数据量实验"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # 加载基础模型
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    all_results = {}
    
    for char_key in CHARACTERS:
        char_data = CHARACTER_PROFILES[char_key]
        all_results[char_key] = {}
        
        for data_size in DATA_SIZES:
            adapter_name = f"qwen_{char_key}_sft_{data_size}"
            adapter_path = os.path.join(ADAPTER_BASE_DIR, adapter_name)
            
            print(f"\n{'='*60}")
            print(f"Processing: {char_key} - {data_size} samples")
            print(f"{'='*60}")
            
            if not os.path.exists(adapter_path):
                print(f"  WARNING: Adapter not found at {adapter_path}, skipping...")
                continue
            
            # 加载adapter
            print(f"  Loading adapter from {adapter_path}...")
            try:
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model.eval()
            except Exception as e:
                print(f"  ERROR loading adapter: {e}")
                continue
            
            all_results[char_key][data_size] = []
            
            # 运行多次实验
            for run_id in range(1, NUM_RUNS + 1):
                print(f"\n  Run {run_id}/{NUM_RUNS}:")
                logs = run_single_experiment(model, tokenizer, char_key, char_data, data_size, run_id)
                
                # 保存单次实验结果
                result = {
                    "character": char_key,
                    "data_size": data_size,
                    "run_id": run_id,
                    "num_turns": MAX_TURNS,
                    "timestamp": datetime.now().isoformat(),
                    "logs": logs
                }
                
                output_file = os.path.join(
                    RESULTS_DIR,
                    f"GroupD_SFT_{char_key}_{data_size}samples_run{run_id}.json"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"    Saved to {output_file}")
                all_results[char_key][data_size].append(result)
            
            # 卸载adapter
            del model
            torch.cuda.empty_cache()
    
    # 保存汇总结果
    summary_file = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_type": "datasize_comparison",
            "characters": CHARACTERS,
            "data_sizes": DATA_SIZES,
            "num_runs": NUM_RUNS,
            "num_turns": MAX_TURNS,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")
    
    # 清理
    del base_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_datasize_experiment()
