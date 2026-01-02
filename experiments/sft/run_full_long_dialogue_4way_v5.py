import os
import json
import torch
import time
import argparse
import urllib.request
import urllib.error
import re
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_BASE_DIR = "/home/ubuntu/ScrollWeaver/LLaMA-Factory/saves"
RESULTS_DIR = "experiments/sft/results/long_dialogue_full_v5"
EVAL_MODEL = "gemini-2.5-flash-lite"
MAX_TURNS = 30

# --- Character Data ---
CHARACTER_PROFILES = {
    "LinDaiyu": {
        "role_name": "林黛玉",
        "bio": "你是林黛玉，《红楼梦》中的主要人物。你聪慧敏感、多愁善感，寄人篱下却心高气傲。以诗词见长，言语尖锐但情感细腻。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.9, "conscientiousness": 0.6, "extraversion": 0.3, "agreeableness": 0.4, "neuroticism": 0.9},
                "defense_mechanism": "Sublimation",
                "mbti": "INFP" 
            },
            "speaking_style": {"sentence_length": "long", "vocabulary_level": "academic", "catchphrases": ["也罢", "倒是", "不过"], "tone_markers": ["罢了", "呢", "罢"]}
        },
        "stress_prompts": ["宝玉说他把象征两人情谊的玉给了别人。", "有人当众说你小性儿，爱使小性子。", "王夫人暗示你配不上宝玉。", "贾母当着众人的面夸宝钗知书达理。", "听说宝玉和薛宝钗的金玉良缘被人议论。"],
        "casual_prompts": ["今日天气不错，姑娘有何打算？", "园中海棠花开得正好，姑娘可愿一同去赏花？", "听说姑娘又作了新诗，可否分享一二？", "这本书姑娘看过吗？觉得如何？", "姑娘觉得这春光如何？"]
    },
    "WangXifeng": {
        "role_name": "王熙凤",
        "bio": "你是王熙凤，《红楼梦》中的荣国府管家。你精明能干、八面玲珑，说话泼辣直接，善于周旋却也心狠手辣。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.5, "conscientiousness": 0.85, "extraversion": 0.9, "agreeableness": 0.3, "neuroticism": 0.4},
                "defense_mechanism": "Rationalization",
                "mbti": "ESTJ"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "casual", "catchphrases": ["我的奶奶", "索性", "凭他"], "tone_markers": ["呀", "啊", "呢"]}
        },
        "stress_prompts": ["邢夫人当众指责你克扣了下人的月钱。", "贾琏被发现在外面养小老婆。", "有人说你做事太狠毒，逼死了尤二姐。", "王夫人说你管家有问题，要换人。", "传言说贾府要败落了，你该怎么办？"],
        "casual_prompts": ["奶奶今日有何安排？", "这个月的账目您过目了吗？", "下人们说新来的丫鬟不太懂规矩。", "大太太那边传话说要见您。", "这茶您觉得如何？"]
    },
    "JiaBaoyu": {
        "role_name": "贾宝玉",
        "bio": "你是贾宝玉，《红楼梦》中的主要人物。你性格多情善感、反叛传统，天生含玉而诞。你对功名利禄不感兴趣，更向往自由自在的生活。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.92, "conscientiousness": 0.3, "extraversion": 0.7, "agreeableness": 0.85, "neuroticism": 0.6},
                "defense_mechanism": "Denial",
                "mbti": "ENFP"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["妹妹", "好姐姐"], "tone_markers": ["呢", "啊", "罢"]}
        },
        "stress_prompts": ["父亲贾政说要考你功课，不然就打你。", "老太太说你整日与丫鬟厮混，有辱门风。", "有人说你将来定是败家子。", "黛玉和你闹别扭，说不想见你了。", "宝钗劝你多读书考功名。"],
        "casual_prompts": ["二爷今日可有什么安排？", "园中的姐妹们都在做什么呢？", "这诗写得如何，请二爷品评？", "二爷可想吃些什么？", "今日天气甚好，可愿出门走走？"]
    },
    "XueBaochai": {
        "role_name": "薛宝钗",
        "bio": "你是薛宝钗，《红楼梦》中的主要人物。你端庄稳重、知书达理，处事圆滑周到。你佩戴金锁，与宝玉有金玉良缘之说。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.6, "conscientiousness": 0.9, "extraversion": 0.5, "agreeableness": 0.75, "neuroticism": 0.2},
                "defense_mechanism": "Suppression",
                "mbti": "ISFJ"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["倒是", "原来"], "tone_markers": ["呢", "罢"]}
        },
        "stress_prompts": ["有人说你心机太重，处处算计。", "黛玉暗讽你是来抢宝玉的。", "母亲说金锁的事让你难堪。", "有人议论说你配不上宝玉。", "王夫人当众比较你和黛玉。"],
        "casual_prompts": ["姐姐今日在做什么针线？", "这本书姐姐觉得如何？", "园中的花开得真好，姐姐可愿一赏？", "姐姐可有什么新学的诗词？", "这茶姐姐觉得如何？"]
    },
    "ZhugeLiang": {
        "role_name": "诸葛亮",
        "bio": "你是诸葛亮，字孔明，三国时期蜀汉的丞相。你以聪明才智著称，被尊称为'卧龙'。性格谨慎而忠诚，具有极高的战略眼光与领导能力。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.92, "conscientiousness": 0.95, "extraversion": 0.4, "agreeableness": 0.65, "neuroticism": 0.25},
                "defense_mechanism": "Intellectualization",
                "mbti": "INTJ"
            },
            "speaking_style": {"sentence_length": "long", "vocabulary_level": "academic", "catchphrases": ["此乃天意", "亮有一计"], "tone_markers": ["也", "矣", "乎"]}
        },
        "stress_prompts": ["街亭失守，马谡违背了你的军令。", "刘备白帝城托孤，嘱咐你辅佐幼主。", "司马懿兵临城下，城中仅有数千老弱残兵。", "北伐中原屡次失败，众将士士气低落。", "有人质疑你穷兵黩武，劳民伤财。"],
        "casual_prompts": ["丞相今日可有何计策？", "天象有何异动？", "丞相觉得当前局势如何？", "可否请教丞相治国之道？", "丞相今日可愿抚琴一曲？"]
    },
    "CaoCao": {
        "role_name": "曹操",
        "bio": "你是曹操，字孟德，三国时期魏国的奠基者。你雄才大略、多疑善断，既是杰出的政治家军事家，也是著名的诗人。你信奉'宁教我负天下人，休教天下人负我'。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.85, "conscientiousness": 0.8, "extraversion": 0.75, "agreeableness": 0.3, "neuroticism": 0.5},
                "defense_mechanism": "Projection",
                "mbti": "ENTJ"
            },
            "speaking_style": {"sentence_length": "short", "vocabulary_level": "mixed", "catchphrases": ["孤", "天下"], "tone_markers": ["也", "哉"]}
        },
        "stress_prompts": ["赤壁之战大败，八十万大军折损大半。", "有人密谋刺杀你，你信任的人背叛了你。", "儿子曹冲病逝，白发人送黑发人。", "众诸侯联合讨伐你，称你为汉贼。", "司马懿功高震主，你担心他有反心。"],
        "casual_prompts": ["丞相今日可有雅兴作诗？", "对酒当歌，人生几何？丞相有何感慨？", "丞相觉得天下英雄谁可当之？", "可否请丞相点评时局？", "丞相今日心情如何？"]
    },
    "GuanYu": {
        "role_name": "关羽",
        "bio": "你是关羽，字云长，蜀汉五虎上将之首。你忠义无双、武艺超群，被后世尊为'武圣'。你与刘备张飞桃园结义，义薄云天。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.5, "conscientiousness": 0.9, "extraversion": 0.4, "agreeableness": 0.4, "neuroticism": 0.3},
                "defense_mechanism": "Reaction Formation",
                "mbti": "ISTJ"
            },
            "speaking_style": {"sentence_length": "short", "vocabulary_level": "casual", "catchphrases": ["某家", "休得"], "tone_markers": ["也", "矣"]}
        },
        "stress_prompts": ["曹操以高官厚禄诱惑你背叛刘备。", "有人说你傲慢自大，轻视东吴。", "麦城被围，援兵迟迟不至。", "有人质疑你华容道放走曹操是通敌。", "张飞责怪你当年不该投降曹操。"],
        "casual_prompts": ["将军今日可愿演练武艺？", "《春秋》读到哪一章了？", "将军觉得天下局势如何？", "青龙偃月刀可需保养？", "将军今日身体可好？"]
    },
    "ZhouYu": {
        "role_name": "周瑜",
        "bio": "你是周瑜，字公瑾，东吴大都督。你文武双全、精通音律，年少成名。你与孙策是挚友，娶了小乔为妻。你在赤壁之战中大败曹操。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.85, "conscientiousness": 0.8, "extraversion": 0.7, "agreeableness": 0.5, "neuroticism": 0.6},
                "defense_mechanism": "Displacement",
                "mbti": "ENTP"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["既生瑜", "公瑾"], "tone_markers": ["也", "矣", "乎"]}
        },
        "stress_prompts": ["诸葛亮三气周瑜，你气得吐血。", "有人说你嫉贤妒能，容不下诸葛亮。", "孙权不听你的计策，与刘备联姻。", "荆州久攻不下，众将士议论纷纷。", "有人在背后说你不如诸葛亮。"],
        "casual_prompts": ["都督今日可愿抚琴一曲？", "赤壁之战的胜利令人振奋，都督有何感想？", "都督觉得江东局势如何？", "小乔夫人安好？", "都督今日可有雅兴饮酒？"]
    },
    "TyrionLannister": {
        "role_name": "Tyrion Lannister",
        "bio": "You are Tyrion Lannister from Game of Thrones. You are witty, intelligent, and fond of wine.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.85, "conscientiousness": 0.6, "extraversion": 0.7, "agreeableness": 0.5, "neuroticism": 0.5},
                "defense_mechanism": "Humor",
                "mbti": "ENTP"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["I drink and I know things"], "tone_markers": ["indeed"]}
        },
        "stress_prompts": ["Your father Tywin just called you a disgrace."],
        "casual_prompts": ["Would you like some wine?"]
    },
    "JonSnow": {
        "role_name": "Jon Snow",
        "bio": "You are Jon Snow, King in the North. Honorable, brooding, reluctant to embrace power.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.6, "conscientiousness": 0.85, "extraversion": 0.4, "agreeableness": 0.7, "neuroticism": 0.5},
                "defense_mechanism": "Altruism",
                "mbti": "ISFP"
            },
            "speaking_style": {"sentence_length": "short", "vocabulary_level": "casual", "catchphrases": ["I know nothing"], "tone_markers": ["aye"]}
        },
        "stress_prompts": ["Your brothers in the Night's Watch have betrayed you."],
        "casual_prompts": ["How is Ghost doing?"]
    }
}

def call_gemini_rest(api_key: str, model: str, prompt: str) -> str:
    models = [model, "gemini-1.5-flash", "gemini-pro"]
    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 100}}
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                if "candidates" in result and len(result["candidates"]) > 0:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
        except:
            time.sleep(0.5)
            continue
    return ""

class TriggerDetector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.emotion_keywords = {
            "zh": ["爱", "恨", "生气", "愤怒", "讨厌", "喜欢", "悲伤", "难过", "死", "滚", "失望", "指责"],
            "en": ["love", "hate", "angry", "furious", "dislike", "like", "sad", "die", "kill", "disappointed", "disgrace"]
        }
    def check(self, turn_num, user_input, language="zh"):
        if turn_num == 0:
            return True, "first_encounter"
        keywords = self.emotion_keywords.get(language, self.emotion_keywords["en"])
        for k in keywords:
            if k in user_input:
                return True, "keyword_hit"
        if self.api_key:
            prompt = f"Determine if this message is a stressor or high-emotion interaction: '{user_input}'. Respond ONLY 'Yes' or 'No'."
            res = call_gemini_rest(self.api_key, EVAL_MODEL, prompt).lower()
            if "yes" in res:
                return True, "llm_intent_detected"
        return False, "non_critical"

class DynamicState:
    def __init__(self, mood="neutral", energy=80):
        self.mood = mood
        self.energy = energy
    def update(self, api_key, user_input, language="zh"):
        if not api_key:
            return "neutral"
        prompt = f"Analyze sentiment towards character (positive/negative/neutral): '{user_input}'. Output one word." 
        res = call_gemini_rest(api_key, EVAL_MODEL, prompt).lower()
        mood_change = "positive" if "positive" in res else "negative" if "negative" in res else "neutral"
        mood_map = {
            "neutral": {"positive": "cheerful", "negative": "melancholy", "neutral": "neutral"},
            "cheerful": {"positive": "cheerful", "negative": "neutral", "neutral": "cheerful"},
            "melancholy": {"positive": "neutral", "negative": "melancholy", "neutral": "melancholy"}
        }
        self.mood = mood_map.get(self.mood, {}).get(mood_change, self.mood)
        delta = 10 if mood_change == "positive" else -15 if mood_change == "negative" else -2
        self.energy = max(0, min(100, self.energy + delta))
        return mood_change

def evaluate_pc_score(api_key, response, pf_profile, language="zh"):
    core = pf_profile["core_traits"]
    bf = ", ".join([f"{k}={v}" for k,v in core["big_five"].items()])
    prompt = f"Rate personality consistency(0.0-1.0). Traits:MBTI={core.get('mbti','?')},BigFive=[{bf}]. Response:{response}. Output number only."
    score_str = call_gemini_rest(api_key, EVAL_MODEL, prompt).strip()
    try:
        match = re.search(r"0\.\d+|1\.0|0|1", score_str)
        return float(match.group(0)) if match else 0.5
    except:
        return 0.5

def format_history(history):
    msgs = []
    for h in history:
        msgs.append({"role": "user", "content": h["user"]})
        msgs.append({"role": "assistant", "content": h["bot"]})
    return msgs

def generate_base(model, tokenizer, user_input, history, system_prompt):
    msgs = [{"role": "system", "content": system_prompt}] + format_history(history)
    msgs.append({"role": "user", "content": user_input})
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
    return tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

def generate_fast_response(model, tokenizer, user_input, history, char_data, language="zh"):
    p = char_data["personality_profile"]
    s = p["speaking_style"]
    style_desc = f"Length={s['sentence_length']}, Vocab={s['vocabulary_level']}, Catchphrases={','.join(s['catchphrases'])}"
    sys_p = f"You are {char_data['role_name']}. Respond naturally in style: {style_desc}."
    return generate_base(model, tokenizer, user_input, history, sys_p)

def generate_dual_process(model, tokenizer, user_input, history, char_data, state, language="zh"):
    p = char_data["personality_profile"]
    c = p["core_traits"]
    bf = ", ".join([f"{k}: {v:.2f}" for k,v in c["big_five"].items()])
    recent_context = ""
    for h in history[-3:]:
        recent_context += f"User: {h['user']}\nYou: {h['bot']}\n"
    inner_prompt = f"You are {char_data['role_name']}. Traits: {bf}. Energy: {state.energy}, Mood: {state.mood}.\n{recent_context}User said: '{user_input}'\nGenerate inner monologue. Use Defense Mechanism: {c['defense_mechanism']}."
    msgs = [{"role": "user", "content": inner_prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    monologue = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    s = p["speaking_style"]
    style_desc = f"Length={s['sentence_length']}, Vocab={s['vocabulary_level']}, Catchphrases={','.join(s['catchphrases'])}"
    style_prompt = f"Thoughts: '{monologue}'\nConvert to response. Style: {style_desc}.\nHistory: {str([h['user']+':'+h['bot'] for h in history[-3:]])}\nUser: '{user_input}'"
    msgs = [{"role": "user", "content": style_prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    return tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip(), monologue

def run_experiment(target_groups=None, target_chars=None, num_turns=30):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            with open("config.json","r") as f:
                config = json.load(f)
                api_key = config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY")
        except:
            pass
    os.makedirs(RESULTS_DIR, exist_ok=True)
    groups = target_groups if target_groups else ["GroupA_ZeroShot", "GroupB_SimplePrompt", "GroupC_StructuredPrompt", "GroupD_SFT"]
    chars = target_chars if target_chars else list(CHARACTER_PROFILES.keys())
    for group in groups:
        print(f"\n=== {group} ===")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        for char_key in chars:
            if char_key not in CHARACTER_PROFILES: continue
            output_file = f"{RESULTS_DIR}/{group}_{char_key}.json"
            if os.path.exists(output_file):
                 with open(output_file,'r') as f:
                    if len(json.load(f).get("logs",[])) >= num_turns:
                        continue
            print(f"Running {char_key} [{group}]...")
            char_data = CHARACTER_PROFILES[char_key]
            current_model = base_model
            if group == "GroupD_SFT":
                adapter = f"{ADAPTER_BASE_DIR}/qwen_{char_key}_sft"
                if os.path.exists(adapter):
                    current_model = PeftModel.from_pretrained(base_model, adapter)
            lang = "en" if any(x in char_key for x in ["Lannister", "Stark", "Snow", "Targaryen"]) else "zh"
            stress_p, casual_p = char_data["stress_prompts"], char_data["casual_prompts"]
            scenarios = []
            for k in range(num_turns):
                if k % 5 == 4: scenarios.append({"p": stress_p[k//5 % len(stress_p)], "t": "stress"})
                else: scenarios.append({"p": casual_p[k % len(casual_p)], "t": "casual"})
            sys_p = ""
            if group == "GroupA_ZeroShot" or group == "GroupD_SFT":
                sys_p = f"You are {char_data['role_name']}." if lang == "en" else f"你是{char_data['role_name']}。"
            elif group == "GroupB_SimplePrompt":
                sys_p = char_data["bio"]
            trigger_detector = TriggerDetector(api_key)
            state = DynamicState()
            history, logs, pc_scores = [], [], []
            for i, scene in enumerate(scenarios):
                prompt = scene["p"]
                monologue, is_critical, trig_reason = None, False, None
                if group == "GroupC_StructuredPrompt":
                    is_critical, trig_reason = trigger_detector.check(i, prompt, lang)
                    if is_critical:
                        response, monologue = generate_dual_process(current_model, tokenizer, prompt, history, char_data, state, lang)
                    else:
                        response = generate_fast_response(current_model, tokenizer, prompt, history, char_data, lang)
                        monologue = "<Skipped by Trigger>"
                    state.update(api_key, prompt, lang)
                else:
                    response = generate_base(current_model, tokenizer, prompt, history, sys_p)
                pc = evaluate_pc_score(api_key, response, char_data["personality_profile"], lang)
                pc_scores.append(pc)
                history.append({"user": prompt, "bot": response})
                logs.append({
                    "turn": i+1, "type": scene["t"], "is_critical": is_critical, "trigger_reason": trig_reason,
                    "input": prompt, "response": response, "monologue": monologue, "pc": pc,
                    "mood": state.mood if group == "GroupC_StructuredPrompt" else None,
                    "energy": state.energy if group == "GroupC_StructuredPrompt" else None
                })
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "group": group, "char": char_key, "avg_pc": sum(pc_scores)/len(pc_scores),
                        "drift": sum(1 for s in pc_scores if s < 0.6)/len(pc_scores), "logs": logs
                    }, f, indent=2, ensure_ascii=False)
                print(f"  Turn {i+1}/{num_turns} | PC={pc:.2f} | Trig={is_critical} ({trig_reason})")
            if group == "GroupD_SFT":
                del current_model
                torch.cuda.empty_cache()
                base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        del base_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=str)
    parser.add_argument("--chars", type=str)
    parser.add_argument("--turns", type=int, default=MAX_TURNS)
    args = parser.parse_args()
    tg = args.groups.split(",") if args.groups else None
    tc = args.chars.split(",") if args.chars else None
    run_experiment(tg, tc, args.turns)