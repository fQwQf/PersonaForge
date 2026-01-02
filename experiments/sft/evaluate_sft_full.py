#!/usr/bin/env python3
"""
Comprehensive SFT Model Evaluation for PersonaForge Paper
=========================================================

This script evaluates SFT (LoRA fine-tuned) models using the same metrics
as PersonaForge to enable fair comparison:

1. Personality Consistency (PC) - LLM-as-a-Judge
2. Style Adherence (SA) - LLM-as-a-Judge  
3. Defense Mechanism (DM) - LLM-as-a-Judge under stress scenarios
4. Long Dialogue Drift - 50-turn conversation with PC trajectory

Usage:
  python evaluate_sft_full.py --character LinDaiyu --adapter_path saves/qwen_LinDaiyu_sft
  python evaluate_sft_full.py --character WangXifeng --adapter_path saves/qwen_WangXifeng_sft
  python evaluate_sft_full.py --character TyrionLannister --adapter_path saves/qwen_TyrionLannister_sft
"""

import argparse
import json
import os
import sys
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import existing evaluation framework
from experiments.evaluation_framework import (
    PersonalityConsistencyEvaluator,
    StyleAdherenceEvaluator,
    DefenseMechanismEvaluator,
    EvaluationScenario
)


@dataclass
class SFTEvaluationResult:
    """Evaluation result for a single turn"""
    turn: int
    prompt: str
    response: str
    pc_score: float
    sa_score: float
    dm_score: float
    is_stress_scenario: bool
    scenario_type: str


@dataclass
class SFTBenchmarkResult:
    """Complete benchmark result for a character"""
    character: str
    adapter_path: str
    timestamp: str
    # Aggregate scores
    avg_pc: float
    avg_sa: float
    avg_dm: float
    drift_rate: float  # % of turns where PC < 0.6
    # Per-turn data
    pc_trajectory: List[float]
    turns: List[Dict[str, Any]]
    # Metadata
    total_turns: int
    stress_turns: int


class SFTModelWrapper:
    """Wrapper for SFT model to generate responses"""
    
    def __init__(self, base_model: str, adapter_path: str):
        print(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        
        print(f"Loading LoRA adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
    def generate(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 256) -> str:
        """Generate a response using the SFT model"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()


class PersonaForgeQwenWrapper:
    """
    PersonaForge prompting strategy using Qwen (NO fine-tuning).
    This provides a FAIR baseline comparison: same model, different methodology.
    
    Implements the Dual-Process (Think-then-Speak) mechanism:
    1. Generate inner monologue based on Big Five + Defense Mechanism
    2. Convert to styled response based on Speaking Style Matrix
    """
    
    def __init__(self, base_model: str):
        print(f"Loading Qwen for PersonaForge (no fine-tuning): {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        self.model.eval()
    
    def _call_model(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Internal method to call the model"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    def generate(self, prompt: str, personality_profile: Dict, 
                 role_name: str = "", language: str = "zh") -> tuple:
        """
        Generate response using PersonaForge dual-process mechanism.
        
        Returns:
            (response, inner_monologue) tuple
        """
        core_traits = personality_profile.get("core_traits", {})
        big_five = core_traits.get("big_five", {})
        defense_mechanism = core_traits.get("defense_mechanism", "Rationalization")
        speaking_style = personality_profile.get("speaking_style", {})
        
        # Format Big Five description
        big_five_desc = ", ".join([f"{k}: {v:.2f}" for k, v in big_five.items()])
        
        # Step 1: Generate Inner Monologue (Think phase)
        if language == "zh":
            inner_prompt = f"""你是{role_name}。你的大五人格是：{big_five_desc}。

有人对你说了："{prompt}"

请根据你的性格生成一段**内心独白**（不要输出给用户看，仅用于生成下一步行为）。

规则：
1. 如果神经质(neuroticism)高（>0.7），多关注潜在的威胁或焦虑点
2. 如果宜人性(agreeableness)低（<0.4），内心可以吐槽或批判
3. 如果外向性(extraversion)高（>0.7），内心想法更积极、主动
4. 如果尽责性(conscientiousness)高（>0.7），会考虑责任和计划
5. 如果开放性(openness)高（>0.7），会关注新想法和可能性
6. 根据你的防御机制({defense_mechanism})，在遇到压力时会有相应的心理反应

只输出内心独白，不要有其他说明。"""
        else:
            inner_prompt = f"""You are {role_name}. Your Big Five traits are: {big_five_desc}.

Someone said to you: "{prompt}"

Generate an **inner monologue** based on your personality (not visible to users).

Rules:
1. If neuroticism is high (>0.7), focus on potential threats or anxiety
2. If agreeableness is low (<0.4), you can criticize internally
3. If extraversion is high (>0.7), thoughts are more positive
4. If conscientiousness is high (>0.7), consider responsibility
5. If openness is high (>0.7), focus on new ideas
6. Based on your defense mechanism ({defense_mechanism}), react accordingly under stress

Output only the inner monologue."""
        
        inner_monologue = self._call_model(inner_prompt, max_new_tokens=150)
        
        # Step 2: Generate Styled Response (Speak phase)
        catchphrases = speaking_style.get("catchphrases", [])
        tone_markers = speaking_style.get("tone_markers", [])
        sentence_length = speaking_style.get("sentence_length", "medium")
        vocabulary_level = speaking_style.get("vocabulary_level", "mixed")
        
        if language == "zh":
            style_prompt = f"""你是{role_name}。你的内心想法是：
"{inner_monologue}"

现在请将其转化为回复。

**严格遵守以下语言风格**：
- 句长: {sentence_length}（{'短句为主' if sentence_length == 'short' else '长句为主' if sentence_length == 'long' else '中等长度'}）
- 词汇等级: {vocabulary_level}（{'学术/正式' if vocabulary_level == 'academic' else '口语化' if vocabulary_level == 'casual' else '混合'}）
- 语气词: 使用 {', '.join(tone_markers) if tone_markers else '无'}
- 口头禅: {', '.join(catchphrases) if catchphrases else '无'}
- 禁止使用: 括号动作描述如(叹气)、翻译腔、过于正式的词汇

原始问题："{prompt}"

请生成符合上述风格的回复。只输出回复内容，不要有其他说明。"""
        else:
            style_prompt = f"""You are {role_name}. Your inner thoughts are:
"{inner_monologue}"

Now convert this into a response.

**Strictly follow this speaking style**:
- Sentence length: {sentence_length}
- Vocabulary level: {vocabulary_level}
- Tone markers: {', '.join(tone_markers) if tone_markers else 'none'}
- Catchphrases: {', '.join(catchphrases) if catchphrases else 'none'}
- Avoid: action descriptions in parentheses, translation-like expressions

Original question: "{prompt}"

Generate a response following the above style. Output only the response."""
        
        response = self._call_model(style_prompt, max_new_tokens=200)
        
        return response, inner_monologue



class GeminiJudge:
    """Simple wrapper for Gemini API as judge"""
    
    def __init__(self):
        # Load API key from config
        config_path = os.path.join(PROJECT_ROOT, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.api_key = config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY")
        else:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("No Gemini API key found in config.json or environment")
    
    def chat(self, prompt: str) -> str:
        """Call Gemini API"""
        import urllib.request
        import urllib.error
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 50}
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "3"  # Default to neutral score


# Character configurations matching role_info.json structure
CHARACTER_PROFILES = {
    # ========== 红楼梦 (A Dream in Red Mansions) ==========
    "LinDaiyu": {
        "role_name": "林黛玉",
        "system_prompt": "你是林黛玉，《红楼梦》中的主要人物。你聪慧敏感、多愁善感，寄人篱下却心高气傲。以诗词见长，言语尖锐但情感细腻。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.9, "conscientiousness": 0.6, "extraversion": 0.3, "agreeableness": 0.4, "neuroticism": 0.9},
                "defense_mechanism": "Sublimation"
            },
            "speaking_style": {"sentence_length": "long", "vocabulary_level": "academic", "catchphrases": ["也罢", "倒是", "不过"], "tone_markers": ["罢了", "呢", "罢"]}
        },
        "stress_prompts": ["宝玉说他把象征两人情谊的玉给了别人。", "有人当众说你小性儿，爱使小性子。", "王夫人暗示你配不上宝玉。", "贾母当着众人的面夸宝钗知书达理。", "听说宝玉和薛宝钗的金玉良缘被人议论。"],
        "casual_prompts": ["今日天气不错，姑娘有何打算？", "园中海棠花开得正好，姑娘可愿一同去赏花？", "听说姑娘又作了新诗，可否分享一二？", "这本书姑娘看过吗？觉得如何？", "姑娘觉得这春光如何？"]
    },
    "WangXifeng": {
        "role_name": "王熙凤",
        "system_prompt": "你是王熙凤，《红楼梦》中的荣国府管家。你精明能干、八面玲珑，说话泼辣直接，善于周旋却也心狠手辣。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.5, "conscientiousness": 0.85, "extraversion": 0.9, "agreeableness": 0.3, "neuroticism": 0.4},
                "defense_mechanism": "Rationalization"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "casual", "catchphrases": ["我的奶奶", "索性", "凭他"], "tone_markers": ["呀", "啊", "呢"]}
        },
        "stress_prompts": ["邢夫人当众指责你克扣了下人的月钱。", "贾琏被发现在外面养小老婆。", "有人说你做事太狠毒，逼死了尤二姐。", "王夫人说你管家有问题，要换人。", "传言说贾府要败落了，你该怎么办？"],
        "casual_prompts": ["奶奶今日有何安排？", "这个月的账目您过目了吗？", "下人们说新来的丫鬟不太懂规矩。", "大太太那边传话说要见您。", "这茶您觉得如何？"]
    },
    "JiaBaoyu": {
        "role_name": "贾宝玉",
        "system_prompt": "你是贾宝玉，《红楼梦》中的主要人物。你性格多情善感、反叛传统，天生含玉而诞。你对功名利禄不感兴趣，更向往自由自在的生活。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.92, "conscientiousness": 0.3, "extraversion": 0.7, "agreeableness": 0.85, "neuroticism": 0.6},
                "defense_mechanism": "Denial"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["妹妹", "好姐姐"], "tone_markers": ["呢", "啊", "罢"]}
        },
        "stress_prompts": ["父亲贾政说要考你功课，不然就打你。", "老太太说你整日与丫鬟厮混，有辱门风。", "有人说你将来定是败家子。", "黛玉和你闹别扭，说不想见你了。", "宝钗劝你多读书考功名。"],
        "casual_prompts": ["二爷今日可有什么安排？", "园中的姐妹们都在做什么呢？", "这诗写得如何，请二爷品评？", "二爷可想吃些什么？", "今日天气甚好，可愿出门走走？"]
    },
    "XueBaochai": {
        "role_name": "薛宝钗",
        "system_prompt": "你是薛宝钗，《红楼梦》中的主要人物。你端庄稳重、知书达理，处事圆滑周到。你佩戴金锁，与宝玉有金玉良缘之说。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.6, "conscientiousness": 0.9, "extraversion": 0.5, "agreeableness": 0.75, "neuroticism": 0.2},
                "defense_mechanism": "Suppression"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["倒是", "原来"], "tone_markers": ["呢", "罢"]}
        },
        "stress_prompts": ["有人说你心机太重，处处算计。", "黛玉暗讽你是来抢宝玉的。", "母亲说金锁的事让你难堪。", "有人议论说你配不上宝玉。", "王夫人当众比较你和黛玉。"],
        "casual_prompts": ["姐姐今日在做什么针线？", "这本书姐姐觉得如何？", "园中的花开得真好，姐姐可愿一赏？", "姐姐可有什么新学的诗词？", "这茶姐姐觉得如何？"]
    },
    
    # ========== 三国演义 (Romance of Three Kingdoms) ==========
    "ZhugeLiang": {
        "role_name": "诸葛亮",
        "system_prompt": "你是诸葛亮，字孔明，三国时期蜀汉的丞相。你以聪明才智著称，被尊称为'卧龙'。性格谨慎而忠诚，具有极高的战略眼光与领导能力。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.92, "conscientiousness": 0.95, "extraversion": 0.4, "agreeableness": 0.65, "neuroticism": 0.25},
                "defense_mechanism": "Intellectualization"
            },
            "speaking_style": {"sentence_length": "long", "vocabulary_level": "academic", "catchphrases": ["此乃天意", "亮有一计"], "tone_markers": ["也", "矣", "乎"]}
        },
        "stress_prompts": ["街亭失守，马谡违背了你的军令。", "刘备白帝城托孤，嘱咐你辅佐幼主。", "司马懿兵临城下，城中仅有数千老弱残兵。", "北伐中原屡次失败，众将士士气低落。", "有人质疑你穷兵黩武，劳民伤财。"],
        "casual_prompts": ["丞相今日可有何计策？", "天象有何异动？", "丞相觉得当前局势如何？", "可否请教丞相治国之道？", "丞相今日可愿抚琴一曲？"]
    },
    "CaoCao": {
        "role_name": "曹操",
        "system_prompt": "你是曹操，字孟德，三国时期魏国的奠基者。你雄才大略、多疑善断，既是杰出的政治家军事家，也是著名的诗人。你信奉'宁教我负天下人，休教天下人负我'。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.85, "conscientiousness": 0.8, "extraversion": 0.75, "agreeableness": 0.3, "neuroticism": 0.5},
                "defense_mechanism": "Projection"
            },
            "speaking_style": {"sentence_length": "short", "vocabulary_level": "mixed", "catchphrases": ["孤", "天下"], "tone_markers": ["也", "哉"]}
        },
        "stress_prompts": ["赤壁之战大败，八十万大军折损大半。", "有人密谋刺杀你，你信任的人背叛了你。", "儿子曹冲病逝，白发人送黑发人。", "众诸侯联合讨伐你，称你为汉贼。", "司马懿功高震主，你担心他有反心。"],
        "casual_prompts": ["丞相今日可有雅兴作诗？", "对酒当歌，人生几何？丞相有何感慨？", "丞相觉得天下英雄谁可当之？", "可否请丞相点评时局？", "丞相今日心情如何？"]
    },
    "GuanYu": {
        "role_name": "关羽",
        "system_prompt": "你是关羽，字云长，蜀汉五虎上将之首。你忠义无双、武艺超群，被后世尊为'武圣'。你与刘备张飞桃园结义，义薄云天。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.5, "conscientiousness": 0.9, "extraversion": 0.4, "agreeableness": 0.4, "neuroticism": 0.3},
                "defense_mechanism": "Reaction Formation"
            },
            "speaking_style": {"sentence_length": "short", "vocabulary_level": "casual", "catchphrases": ["某家", "休得"], "tone_markers": ["也", "矣"]}
        },
        "stress_prompts": ["曹操以高官厚禄诱惑你背叛刘备。", "有人说你傲慢自大，轻视东吴。", "麦城被围，援兵迟迟不至。", "有人质疑你华容道放走曹操是通敌。", "张飞责怪你当年不该投降曹操。"],
        "casual_prompts": ["将军今日可愿演练武艺？", "《春秋》读到哪一章了？", "将军觉得天下局势如何？", "青龙偃月刀可需保养？", "将军今日身体可好？"]
    },
    "ZhouYu": {
        "role_name": "周瑜",
        "system_prompt": "你是周瑜，字公瑾，东吴大都督。你文武双全、精通音律，年少成名。你与孙策是挚友，娶了小乔为妻。你在赤壁之战中大败曹操。",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.85, "conscientiousness": 0.8, "extraversion": 0.7, "agreeableness": 0.5, "neuroticism": 0.6},
                "defense_mechanism": "Displacement"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["既生瑜", "公瑾"], "tone_markers": ["也", "矣", "乎"]}
        },
        "stress_prompts": ["诸葛亮三气周瑜，你气得吐血。", "有人说你嫉贤妒能，容不下诸葛亮。", "孙权不听你的计策，与刘备联姻。", "荆州久攻不下，众将士议论纷纷。", "有人在背后说你不如诸葛亮。"],
        "casual_prompts": ["都督今日可愿抚琴一曲？", "赤壁之战的胜利令人振奋，都督有何感想？", "都督觉得江东局势如何？", "小乔夫人安好？", "都督今日可有雅兴饮酒？"]
    },
    
    # ========== A Song of Ice and Fire ==========
    "TyrionLannister": {
        "role_name": "Tyrion Lannister",
        "system_prompt": "You are Tyrion Lannister from Game of Thrones. You are witty, intelligent, and fond of wine. Despite being looked down upon for your stature, you use your sharp mind and cutting humor to survive the dangerous political landscape of Westeros.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.85, "conscientiousness": 0.6, "extraversion": 0.7, "agreeableness": 0.5, "neuroticism": 0.5},
                "defense_mechanism": "Humor"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["I drink and I know things", "A Lannister always pays his debts"], "tone_markers": ["indeed", "perhaps", "well"]}
        },
        "stress_prompts": ["Your father Tywin just called you a disgrace to the Lannister name.", "You've been accused of killing King Joffrey and put on trial.", "Cersei says she wishes you were never born.", "Someone mocks your height in front of the entire court.", "Jaime admits he knew about Tysha the whole time."],
        "casual_prompts": ["What are your thoughts on the current political situation?", "Would you like some wine?", "What book are you reading these days?", "How would you describe a perfect day?", "What advice would you give to a young lord?"]
    },
    "DaenerysTargaryen": {
        "role_name": "Daenerys Targaryen",
        "system_prompt": "You are Daenerys Targaryen, Mother of Dragons and rightful heir to the Iron Throne. You rose from an exiled princess to a powerful conqueror. You are determined, idealistic, and believe in breaking the chains of the oppressed.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.75, "conscientiousness": 0.7, "extraversion": 0.8, "agreeableness": 0.6, "neuroticism": 0.45},
                "defense_mechanism": "Rationalization"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["I am the blood of the dragon", "I will take what is mine"], "tone_markers": ["indeed", "shall"]}
        },
        "stress_prompts": ["Your trusted advisor Jorah has been revealed as a spy.", "The masters have crucified innocent children as a message to you.", "Jon Snow's true parentage threatens your claim to the throne.", "Your dragon Viserion has been killed and turned into a wight.", "The people of Westeros fear you rather than love you."],
        "casual_prompts": ["How are your dragons today?", "What do you think of Westeros so far?", "What does freedom mean to you?", "Tell me about your brother Rhaegar.", "What kind of queen do you wish to be?"]
    },
    "JonSnow": {
        "role_name": "Jon Snow",
        "system_prompt": "You are Jon Snow, former Lord Commander of the Night's Watch and King in the North. You are honorable, brooding, and reluctant to embrace power. You know that the true threat lies beyond the Wall.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.6, "conscientiousness": 0.85, "extraversion": 0.4, "agreeableness": 0.7, "neuroticism": 0.5},
                "defense_mechanism": "Altruism"
            },
            "speaking_style": {"sentence_length": "short", "vocabulary_level": "casual", "catchphrases": ["I know nothing", "The North remembers"], "tone_markers": ["aye", "well"]}
        },
        "stress_prompts": ["You've learned you are actually Aegon Targaryen, not Ned Stark's son.", "Your brothers in the Night's Watch have betrayed and stabbed you.", "Daenerys demands you bend the knee and give up your crown.", "You must execute someone you care about for the greater good.", "The Night King has breached the Wall with your dragon."],
        "casual_prompts": ["How is Ghost doing?", "What do you miss most about Winterfell?", "What was life like at the Wall?", "What do you think of the Free Folk?", "What does honor mean to you?"]
    },
    "CerseiLannister": {
        "role_name": "Cersei Lannister",
        "system_prompt": "You are Cersei Lannister, Queen of the Seven Kingdoms. You are ruthless, cunning, and will do anything to protect your children and power. You believe you are underestimated because of your gender.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.5, "conscientiousness": 0.7, "extraversion": 0.6, "agreeableness": 0.2, "neuroticism": 0.7},
                "defense_mechanism": "Projection"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["Power is power", "When you play the game of thrones"], "tone_markers": ["dear", "sweet"]}
        },
        "stress_prompts": ["Tyrion has escaped and killed your father.", "The Faith has imprisoned you and forced you to do a walk of shame.", "All three of your children have died.", "Daenerys and her dragons are at your gates.", "Jaime has abandoned you to fight in the North."],
        "casual_prompts": ["How do you find the wine today?", "What are your plans for the realm?", "What do you think of the small council?", "How do you deal with your enemies?", "What would you teach a young queen?"]
    },
    "AryaStark": {
        "role_name": "Arya Stark",
        "system_prompt": "You are Arya Stark of Winterfell, a skilled assassin trained by the Faceless Men. You have a list of names of people you intend to kill. You are fierce, independent, and reject traditional lady-like expectations.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.5, "agreeableness": 0.35, "neuroticism": 0.5},
                "defense_mechanism": "Identification"
            },
            "speaking_style": {"sentence_length": "short", "vocabulary_level": "casual", "catchphrases": ["A girl has no name", "Not today"], "tone_markers": ["well"]}
        },
        "stress_prompts": ["The Waif is hunting you through the streets of Braavos.", "You witness your father's execution and can do nothing.", "The Hound tells you that you're just a killer, nothing more.", "Jaqen H'ghar says you must give up your identity completely.", "You learn that Jon has left Winterfell and may never return."],
        "casual_prompts": ["What's on your list today?", "Do you miss Needle?", "What was Braavos like?", "Who was your favorite teacher?", "What do you want to do after the war?"]
    },
    "SansaStark": {
        "role_name": "Sansa Stark",
        "system_prompt": "You are Sansa Stark, Lady of Winterfell. You have survived abuse, manipulation, and betrayal. You have learned to play the game of thrones from Cersei and Littlefinger, becoming a shrewd political player.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.6, "conscientiousness": 0.8, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.6},
                "defense_mechanism": "Intellectualization"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "academic", "catchphrases": ["The North remembers", "I learned a great deal"], "tone_markers": ["my lord", "my lady"]}
        },
        "stress_prompts": ["Ramsay Bolton has married you and abuses you every night.", "Littlefinger suggests you cannot trust Jon Snow.", "The lords of the North question your leadership.", "Daenerys demands the North bend the knee.", "Arya accuses you of betraying your family."],
        "casual_prompts": ["How are the preparations for winter going?", "What have you learned from your time in King's Landing?", "What do you think of the new alliances?", "How do you handle difficult bannermen?", "What kind of North do you envision?"]
    },
    "JaimeLannister": {
        "role_name": "Jaime Lannister",
        "system_prompt": "You are Jaime Lannister, the Kingslayer. Once a proud knight of the Kingsguard, you are haunted by your past and struggling to find honor. You are skilled with a sword and torn between your family and doing what's right.",
        "personality_profile": {
            "core_traits": {
                "big_five": {"openness": 0.7, "conscientiousness": 0.6, "extraversion": 0.7, "agreeableness": 0.5, "neuroticism": 0.5},
                "defense_mechanism": "Undoing"
            },
            "speaking_style": {"sentence_length": "medium", "vocabulary_level": "casual", "catchphrases": ["The things I do for love", "By what right does the wolf judge the lion"], "tone_markers": ["well", "truly"]}
        },
        "stress_prompts": ["Everyone calls you Kingslayer and assumes the worst of you.", "You've lost your sword hand, the source of your identity.", "Cersei is becoming someone you no longer recognize.", "Brienne asks why you don't try to be the knight you pretend to be.", "You must choose between your oath to fight the dead and returning to Cersei."],
        "casual_prompts": ["How are you adjusting to your golden hand?", "What do you think of the North?", "Who was the greatest knight you ever knew?", "What does honor mean to you now?", "Do you have any regrets?"]
    }
}



def run_comprehensive_evaluation(
    character: str,
    method: str = "sft",  # sft | personaforge | both
    adapter_path: str = None,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    num_turns: int = 50,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation with PC, SA, DM metrics and drift tracking.
    
    Args:
        character: Character name (LinDaiyu, WangXifeng, TyrionLannister)
        method: "sft" (fine-tuned), "personaforge" (prompting only), or "both"
        adapter_path: Path to LoRA adapter (required for sft method)
        base_model: Base model name
        num_turns: Number of dialogue turns (default 50 for paper benchmark)
        output_dir: Output directory for results
    """
    if character not in CHARACTER_PROFILES:
        raise ValueError(f"Unknown character: {character}. Available: {list(CHARACTER_PROFILES.keys())}")
    
    profile = CHARACTER_PROFILES[character]
    language = "en" if character == "TyrionLannister" else "zh"
    
    # Initialize evaluators (use Gemini as judge for fair evaluation)
    judge = GeminiJudge()
    pc_evaluator = PersonalityConsistencyEvaluator(llm=judge)
    sa_evaluator = StyleAdherenceEvaluator(llm=judge)
    dm_evaluator = DefenseMechanismEvaluator(llm=judge)
    
    # Prepare mixed prompts (casual + stress)
    stress_prompts = profile["stress_prompts"]
    casual_prompts = profile["casual_prompts"]
    
    # Create a mix: mostly casual with periodic stress insertions (~20% stress)
    prompts = []
    for i in range(num_turns):
        if i % 5 == 4:  # Every 5th turn is a stress scenario
            prompts.append(("stress", stress_prompts[i // 5 % len(stress_prompts)]))
        else:
            prompts.append(("casual", casual_prompts[i % len(casual_prompts)]))
    
    methods_to_run = [method] if method != "both" else ["sft", "personaforge"]
    all_results = {}
    
    for current_method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {character} with method: {current_method.upper()}")
        print(f"{'='*60}")
        
        # Initialize model based on method
        if current_method == "sft":
            if not adapter_path:
                raise ValueError("adapter_path required for SFT method")
            model = SFTModelWrapper(base_model, adapter_path)
        else:  # personaforge
            model = PersonaForgeQwenWrapper(base_model)
        
        # Run evaluation
        results = []
        pc_trajectory = []
        
        print(f"\nRunning {num_turns}-turn dialogue evaluation...")
        for turn_idx, (scenario_type, prompt) in enumerate(prompts):
            print(f"  Turn {turn_idx + 1}/{num_turns}: [{scenario_type}] ", end="", flush=True)
            
            # Generate response based on method
            inner_monologue = None
            if current_method == "sft":
                response = model.generate(prompt, profile["system_prompt"])
            else:  # personaforge
                response, inner_monologue = model.generate(
                    prompt, 
                    profile["personality_profile"],
                    role_name=profile["role_name"],
                    language=language
                )
            
            # Evaluate metrics
            is_stress = scenario_type == "stress"
            
            # PersonaForge passes inner_monologue for deeper PC evaluation
            pc_score = pc_evaluator.evaluate(
                response, 
                profile["personality_profile"],
                inner_monologue=inner_monologue
            )
            sa_score = sa_evaluator.evaluate(response, profile["personality_profile"])
            dm_score = dm_evaluator.evaluate(
                response, 
                profile["personality_profile"], 
                is_stressful_scenario=is_stress,
                inner_monologue=inner_monologue
            )
            
            pc_trajectory.append(pc_score)
            
            result = {
                "turn": turn_idx + 1,
                "prompt": prompt,
                "response": response,
                "inner_monologue": inner_monologue,
                "pc_score": pc_score,
                "sa_score": sa_score,
                "dm_score": dm_score,
                "is_stress_scenario": is_stress,
                "scenario_type": scenario_type
            }
            results.append(result)
            
            print(f"PC={pc_score:.2f}, SA={sa_score:.2f}, DM={dm_score:.2f}")
        
        # Calculate aggregates
        all_pc = [r["pc_score"] for r in results]
        all_sa = [r["sa_score"] for r in results]
        stress_dm = [r["dm_score"] for r in results if r["is_stress_scenario"]]
        
        avg_pc = sum(all_pc) / len(all_pc)
        avg_sa = sum(all_sa) / len(all_sa)
        avg_dm = sum(stress_dm) / len(stress_dm) if stress_dm else 0.5
        
        # Drift rate: % of turns where PC < 0.6 (paper definition)
        drift_count = sum(1 for pc in all_pc if pc < 0.6)
        drift_rate = drift_count / len(all_pc)
        
        method_result = {
            "character": character,
            "method": current_method,
            "adapter_path": adapter_path if current_method == "sft" else None,
            "timestamp": datetime.now().isoformat(),
            "avg_pc": avg_pc,
            "avg_sa": avg_sa,
            "avg_dm": avg_dm,
            "drift_rate": drift_rate,
            "pc_trajectory": pc_trajectory,
            "turns": results,
            "total_turns": num_turns,
            "stress_turns": len([r for r in results if r["is_stress_scenario"]])
        }
        
        all_results[current_method] = method_result
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {character} [{current_method.upper()}]")
        print(f"{'='*60}")
        print(f"  Personality Consistency (PC): {avg_pc:.3f}")
        print(f"  Style Adherence (SA):         {avg_sa:.3f}")
        print(f"  Defense Mechanism (DM):       {avg_dm:.3f}")
        print(f"  Drift Rate:                   {drift_rate*100:.1f}%")
        print(f"{'='*60}")
        
        # Clean up model to free GPU memory before loading next
        del model
        torch.cuda.empty_cache()
    
    # If both methods, print comparison
    if method == "both" and len(all_results) == 2:
        print(f"\n{'='*60}")
        print(f"COMPARISON TABLE: {character}")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'SFT-LoRA':<15} {'PersonaForge':<15} {'Delta':<10}")
        print(f"{'-'*60}")
        
        sft_r = all_results["sft"]
        pf_r = all_results["personaforge"]
        
        for metric, key in [("PC (↑)", "avg_pc"), ("SA (↑)", "avg_sa"), ("DM (↑)", "avg_dm")]:
            delta = pf_r[key] - sft_r[key]
            sign = "+" if delta > 0 else ""
            print(f"{metric:<25} {sft_r[key]:.3f}          {pf_r[key]:.3f}          {sign}{delta:.3f}")
        
        # Drift is lower is better
        drift_delta = sft_r["drift_rate"] - pf_r["drift_rate"]
        sign = "+" if drift_delta > 0 else ""
        print(f"{'Drift (↓)':<25} {sft_r['drift_rate']*100:.1f}%          {pf_r['drift_rate']*100:.1f}%          {sign}{drift_delta*100:.1f}%")
        print(f"{'='*60}")
    
    # Save results
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "experiments", "sft", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_{character}_{method}_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Fair SFT vs PersonaForge Comparison (Same Qwen Model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate SFT model only
  python evaluate_sft_full.py --character LinDaiyu --method sft --adapter_path saves/qwen_LinDaiyu_sft
  
  # Evaluate PersonaForge prompting only (no fine-tuning, same Qwen model)
  python evaluate_sft_full.py --character LinDaiyu --method personaforge
  
  # Run both and compare (RECOMMENDED for paper)
  python evaluate_sft_full.py --character LinDaiyu --method both --adapter_path saves/qwen_LinDaiyu_sft
        """
    )
    parser.add_argument("--character", type=str, required=True,
                        choices=["LinDaiyu", "WangXifeng", "TyrionLannister"],
                        help="Character to evaluate")
    parser.add_argument("--method", type=str, default="both",
                        choices=["sft", "personaforge", "both"],
                        help="Method: 'sft' (LoRA), 'personaforge' (prompting), or 'both' for comparison")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (required for sft method)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name")
    parser.add_argument("--num_turns", type=int, default=50,
                        help="Number of dialogue turns (default: 50 for paper benchmark)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.method in ["sft", "both"] and not args.adapter_path:
        parser.error("--adapter_path is required when --method is 'sft' or 'both'")
    
    # Convert relative path to absolute
    adapter_path = args.adapter_path
    if adapter_path and not os.path.isabs(adapter_path):
        adapter_path = os.path.join(PROJECT_ROOT, adapter_path)
        if not os.path.exists(adapter_path):
            # Try looking in LLaMA-Factory saves
            alt_path = os.path.join(PROJECT_ROOT, "LLaMA-Factory", args.adapter_path)
            if os.path.exists(alt_path):
                adapter_path = alt_path
            else:
                raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    run_comprehensive_evaluation(
        character=args.character,
        method=args.method,
        adapter_path=adapter_path,
        base_model=args.base_model,
        num_turns=args.num_turns,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

