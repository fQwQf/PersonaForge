import os
import sys
import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm.Gemini import Gemini
from modules.utils import load_json_file, save_json_file

JUDGE_PROMPT = """
你是一位资深的文学评论家和心理学家，擅长分析人物性格和语言风格。
你的任务是评估两段由 AI 扮演的特定角色的回复（A 和 B），并决定哪一段更符合角色的人设。

【角色画像】:
{character_profile}

【情境上下文】:
{context}

【对方说】: "{trigger}"

【候选项】:
回复 A: "{response_a}"
回复 B: "{response_b}"

请从以下三个维度进行评估：
1. 人格一致性 (Personality Consistency)：哪一个回复更符合角色的大五人格特质和核心价值观？
2. 风格契合度 (Style Adherence)：哪一个回复更贴近角色特有的语言习惯（词汇选择、句式长短、语气助词）？
3. 心理真实性 (Psychological Realism)：在压力或冲突场景下，哪一个回复的反应更符合人物的防御机制和心理逻辑？

最后，请给出你的最终选择（A更优、B更优 或 平局），并简要说明理由。

请严格按以下 JSON 格式输出：
{{
    "winner": "A" / "B" / "Tie",
    "reasoning": "...",
    "scores": {{
        "consistency": {{ "A": 0-10, "B": 0-10 }},
        "style": {{ "A": 0-10, "B": 0-10 }},
        "realism": {{ "A": 0-10, "B": 0-10 }}
    }}
}}
"""

class PairwiseJudge:
    def __init__(self, model_name="gemini-2.5-flash-lite", timeout=60):
        # Load config to get API keys
        try:
            project_config = load_json_file("config.json")
            for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE"]:
                if key in project_config and project_config[key]:
                    os.environ[key] = project_config[key]
        except Exception as e:
            print(f"Warning: Could not load config.json for API keys ({e})")
            
        # We manually load API key if not in env, matching Gemini.py logic
        self.llm = Gemini(model=model_name, timeout=timeout)
        
    def compare(self, role_info, context, trigger, response_a, response_b):
        """对比两段回复，返回结果"""
        # 为了防止位置偏见，随机交换 A 和 B 的顺序
        swap = random.random() > 0.5
        if swap:
            a, b = response_b, response_a
        else:
            a, b = response_a, response_b
            
        prompt = JUDGE_PROMPT.format(
            character_profile=json.dumps(role_info.get("personality_profile", role_info), ensure_ascii=False, indent=2),
            context=context,
            trigger=trigger,
            response_a=a,
            response_b=b
        )
        
        try:
            response_text = self.llm.chat(prompt)
            # 解析 JSON
            clean_text = response_text
            if "```json" in response_text:
                clean_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                clean_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(clean_text)
            
            # 如果之前交换了顺序，这里需要换回来
            if swap:
                if result["winner"] == "A": result["winner"] = "B"
                elif result["winner"] == "B": result["winner"] = "A"
                
                # 交换分数项
                for metric in result["scores"]:
                    tmp = result["scores"][metric]["A"]
                    result["scores"][metric]["A"] = result["scores"][metric]["B"]
                    result["scores"][metric]["B"] = tmp
                    
            return result
        except Exception as e:
            print(f"评测出错: {e}")
            return None

def run_evaluation_batch(input_results_file: str):
    """
    读取实验结果 JSON，执行两两评测
    要求 JSON 格式包含历史记录，且每条记录有 baseline_response 和 ours_response
    """
    data = load_json_file(input_results_file)
    history = data.get("history", [])
    role_info = data.get("metadata", {}).get("role_info", {})
    
    judge = PairwiseJudge()
    findings = []
    
    print(f"开始评测 {len(history)} 条记录...")
    
    for i, entry in enumerate(history):
        context = entry.get("context", "")
        trigger = entry.get("trigger", entry.get("content", ""))
        resp_ours = entry.get("response_ours", entry.get("ours", ""))
        resp_baseline = entry.get("response_baseline", entry.get("baseline", ""))
        
        if not resp_ours or not resp_baseline:
            continue
            
        print(f"  [{i+1}/{len(history)}] 评测中...")
        result = judge.compare(role_info, context, trigger, resp_ours, resp_baseline)
        
        if result:
            findings.append({
                "entry_id": i,
                "context": context[:50],
                "winner": result.get("winner"),
                "scores": result.get("scores"),
                "reasoning": result.get("reasoning", "")[:100]
            })
            print(f"    Winner: {result.get('winner')}")
    
    # 统计
    wins = {"A": 0, "B": 0, "Tie": 0}
    for f in findings:
        w = f.get("winner", "Tie")
        wins[w] = wins.get(w, 0) + 1
    
    print(f"\n评测完成: Ours Wins={wins.get('A', 0)}, Baseline Wins={wins.get('B', 0)}, Ties={wins.get('Tie', 0)}")
    return findings

if __name__ == "__main__":
    # 更多是作为一个模块被调用
    print("PairwiseJudge Module Loaded.")
