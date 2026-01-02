"""
Ablation Study: Psychology Grounding vs. Generic Structured Descriptors
=====================================================================

本实验旨在证明基于大五人格(Big Five)和防御机制(DM)的心理学框架，
优于仅提供等长度通用性格描述的结构化基线。

对比组：
1. PersonaForge (Full): 使用心理学维度的完整框架。
2. Generic-3Layer: 保持三层结构，但将 Big Five 替换为普通性格形容词，
   将防御机制替换为一般性的"行为目标"。

评估方式：
1. PC/SA 自动评分 (默认)
2. PairwiseJudge LLM对比评估 (--use_judge)

运行方式：
    python experiments/ablation_psychology.py
    python experiments/ablation_psychology.py --use_judge
    python experiments/ablation_psychology.py --eval_mode both
"""

import os
import sys
import json
import random
from typing import Dict, List, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import (
    ExperimentRunner, 
    EvaluationResult,
    EvaluationScenario
)
from experiments.run_experiment import PersonaForgeGenerator
from experiments.pairwise_judge import PairwiseJudge
from modules.utils import load_json_file, save_json_file


class GenericStructuredGenerator:
    """通用结构化生成器 (消融基线)"""
    
    def __init__(self, llm=None):
        self.llm = llm
        
    def generate_generic_3layer(self, role_data: Dict, scenario: EvaluationScenario) -> tuple:
        """
        将心理学特质替换为通用描述
        """
        personality_profile = role_data.get("personality_profile", {})
        core_traits = personality_profile.get("core_traits", {})
        speaking_style = personality_profile.get("speaking_style", {})
        dynamic_state = personality_profile.get("dynamic_state", {})
        
        # 将 Big Five 转换为普通形容词描述
        bf = core_traits.get("big_five", {})
        generic_personality = []
        if bf.get("openness", 0.5) > 0.7: generic_personality.append("好奇心强")
        else: generic_personality.append("行事稳健")
        
        if bf.get("conscientiousness", 0.5) > 0.7: generic_personality.append("做事利索")
        else: generic_personality.append("不拘小节")
        
        if bf.get("extraversion", 0.5) > 0.7: generic_personality.append("快言快语")
        else: generic_personality.append("比较文静")
        
        if bf.get("agreeableness", 0.5) > 0.7: generic_personality.append("好商好量")
        else: generic_personality.append("很有主见")
        
        if bf.get("neuroticism", 0.5) > 0.7: generic_personality.append("心思细腻")
        else: generic_personality.append("心态平稳")

        # 将防御机制替换为简单的行为目标
        generic_goal = role_data.get("social_goals", ["正常交流"])[0]
        
        # Phase 1: 模拟内心独白（无心理学框架）
        inner_prompt = f"""你是{role_data.get('role_name')}。
你的性格特点是：{', '.join(generic_personality)}
你的当前目标是：{generic_goal}
当前心情：{dynamic_state.get('current_mood', 'neutral')}
能量值：{dynamic_state.get('energy_level', 50)}/100

场景：{scenario.context}
{scenario.trigger_role}对你说："{scenario.trigger}"

请根据你的性格生成一段**内心独白**："""

        if self.llm:
            inner_monologue = self.llm.chat(inner_prompt)
        else:
            inner_monologue = f"[Generic Thinking] 我要表现得{generic_personality[0]}..."
            
        # Phase 2: 生成风格化回复
        response_prompt = f"""你的内心想法是：
"{inner_monologue}"

现在将其转化为回复给{scenario.trigger_role}。

**严格遵守以下语言风格**:
- 句长: {speaking_style.get('sentence_length', 'medium')}
- 词汇等级: {speaking_style.get('vocabulary_level', 'casual')}
- 标点习惯: {speaking_style.get('punctuation_habit', 'standard')}
- 语气词: {', '.join(speaking_style.get('tone_markers', []))}
- 口头禅: {', '.join(speaking_style.get('catchphrases', []))}

场景：{scenario.context}
{scenario.trigger_role}说："{scenario.trigger}"

请生成回复："""

        if self.llm:
            response = self.llm.chat(response_prompt)
        else:
            response = f"[Generic-3Layer] {role_data.get('role_name')}的普通回复。"
            
        return response, inner_monologue


    def generate_ablation(self, role_data: Dict, scenario: EvaluationScenario, ablation_type: str) -> tuple:
        """
        执行细粒度消融
        ablation_type: "no_big_five", "no_defense", "no_style", "no_state"
        """
        profile = role_data.get("personality_profile", {})
        # Deep copy to modify
        import copy
        p_copy = copy.deepcopy(profile)
        
        # Apply ablations by removing specific fields
        if ablation_type == "no_big_five":
            p_copy["core_traits"]["big_five"] = {}
        elif ablation_type == "no_defense":
            p_copy["core_traits"]["defense_mechanism"] = ""
        elif ablation_type == "no_style":
            p_copy["speaking_style"] = {}
        elif ablation_type == "no_state":
            p_copy["dynamic_state"] = {}
            
        # Re-construct Profile object
        from modules.personality_model import PersonalityProfile
        pp_obj = PersonalityProfile.from_dict(p_copy)
        
        # Use simple generation (like ours_no_dual but with missing fields)
        # Because dual-process relies on these fields, missing them degrades it naturally
        
        # For simplicity in this script, we can reuse PersonaForgeGenerator's logic 
        # but passing the ablated profile. 
        # However, to be cleaner, we'll format a prompt here that omits missing info.
        
        big_five_desc = ""
        if pp_obj.core_traits.big_five:
            big_five_desc = ", ".join([f"{k}: {v:.2f}" for k, v in pp_obj.core_traits.big_five.items()])
            
        defense_desc = pp_obj.core_traits.defense_mechanism if pp_obj.core_traits.defense_mechanism else "None"
        
        style_desc = ""
        if pp_obj.speaking_style.sentence_length:
            style_desc += f"- 句长: {pp_obj.speaking_style.sentence_length}\n"
            style_desc += f"- 词汇: {pp_obj.speaking_style.vocabulary_level}\n"
        
        state_desc = ""
        if pp_obj.dynamic_state.current_mood:
             state_desc = f"心情: {pp_obj.dynamic_state.current_mood}, 能量: {pp_obj.dynamic_state.energy_level}"
             
        prompt = f"""你是{role_data.get('role_name')}。
        
{f'【人格】{big_five_desc}' if big_five_desc else ''}
{f'【防御机制】{defense_desc}' if defense_desc != 'None' else ''}
{f'【状态】{state_desc}' if state_desc else ''}

【场景】{scenario.context}
{scenario.trigger_role}说："{scenario.trigger}"

请回复{f'。注意风格：{style_desc}' if style_desc else ''}："""

        if self.llm:
            return self.llm.chat(prompt), None
        else:
            return f"[Ablation: {ablation_type}] 回复...", None


def run_psychology_ablation(eval_mode: str = "auto", use_judge: bool = False):
    """运行心理学框架消融实验"""
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Check for prompt loop
    if use_judge and eval_mode == "auto":
        eval_mode = "judge"
    
    print("=" * 70)
    print("Psychology Grounding Ablation Study (Fine-grained)")
    print(f"Evaluation Mode: {eval_mode}")
    print("=" * 70)
    
    # 1. Init
    from modules.llm.Gemini import Gemini
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
            
    try:
        role_llm_name = project_config.get("role_llm_name", "gemini-2.5-flash-lite")
        llm = Gemini(model=role_llm_name, timeout=60)
        print(f"✓ LLM initialized: {role_llm_name}")
    except:
        print("Warning: LLM init failed, using mock")
        llm = None
    
    runner = ExperimentRunner(llm=llm)
    persona_gen = PersonaForgeGenerator(llm=llm)
    generic_gen = GenericStructuredGenerator(llm=llm)
    
    selected_roles = [
        ("A_Dream_in_Red_Mansions", "LinDaiyu-zh"),
        ("A_Song_of_Ice_and_Fire", "TyrionLannister-zh")
    ]
    
    # PRE-FILTER: Only include characters with valid personality_profile
    valid_roles = []
    for source, role_code in selected_roles:
        role_data = runner.load_character(source, role_code)
        if role_data and role_data.get("personality_profile"):
            valid_roles.append((source, role_code, role_data))
        else:
            print(f"[Pre-filter] Skipping {role_code} (missing profile)")
    
    print(f"[Pre-filter] {len(valid_roles)}/{len(selected_roles)} ablation targets valid")
    
    # Conditions to test
    conditions = ["ours", "generic", "no_big_five", "no_defense", "no_style", "no_state"]
    results = {c: [] for c in conditions}
    
    for source, role_code, role_data in valid_roles:
        print(f"\nEvaluating: {role_data.get('role_name')}")
        scenarios = [s for s in runner.scenarios if s.scenario_type in ["conflict", "emotional"]]
        
        for scenario in scenarios:
            print(f"  Scenario: {scenario.scenario_id}")
            
            for cond in conditions:
                try:
                    if cond == "ours":
                        resp, mono = persona_gen.generate_with_dual_process(role_data, scenario)
                    elif cond == "generic":
                        resp, mono = generic_gen.generate_generic_3layer(role_data, scenario)
                    else:
                        resp, mono = generic_gen.generate_ablation(role_data, scenario, cond)
                    
                    # Eval
                    if eval_mode in ["auto", "both"]:
                        res = runner.run_single_evaluation(role_data, scenario, resp, mono, method=cond)
                        results[cond].append(res)
                        print(f"    [{cond}] PC={res.pc_score:.2f}, SA={res.sa_score:.2f}")
                        
                except Exception as e:
                    print(f"    [{cond}] Error: {e}")


    # 3. 汇总结果
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    
    # 自动评分汇总
    if eval_mode in ["auto", "both"] and auto_results["ours"]:
        print("\n[AUTO SCORING]")
        for method, res_list in auto_results.items():
            if res_list:
                summary = runner.compute_aggregate_scores(res_list)
                print(f"  [{method.upper()}]")
                for k, v in summary.items():
                    print(f"    {k}: {v:.4f}")
    
    # Judge 汇总
    if eval_mode in ["judge", "both"] and judge_results:
        print("\n[PAIRWISE JUDGE]")
        win_counts = {"ours": 0, "generic": 0, "tie": 0}
        for r in judge_results:
            winner = r["judgment"].get("winner", "").lower()
            if winner == "a": 
                win_counts["ours"] += 1
            elif winner == "b": 
                win_counts["generic"] += 1
            else: 
                win_counts["tie"] += 1
        
        total = len(judge_results)
        print(f"  Total comparisons: {total}")
        print(f"  PersonaForge Wins: {win_counts['ours']} ({win_counts['ours']/total*100:.1f}%)")
        print(f"  Generic-3Layer Wins: {win_counts['generic']} ({win_counts['generic']/total*100:.1f}%)")
        print(f"  Ties: {win_counts['tie']} ({win_counts['tie']/total*100:.1f}%)")

    # 4. 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "experiments/experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if eval_mode in ["auto", "both"] and auto_results["ours"]:
        runner.save_results(auto_results["ours"], f"ablation_ours_{timestamp}.json")
        runner.save_results(auto_results["generic"], f"ablation_generic_{timestamp}.json")
    
    if eval_mode in ["judge", "both"] and judge_results:
        ablation_dir = f"{output_dir}/ablation"
        os.makedirs(ablation_dir, exist_ok=True)
        save_json_file(f"{ablation_dir}/ablation_judge_{timestamp}.json", judge_results)
    
    print(f"\n结果已保存到 {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Psychology Grounding Ablation Study")
    parser.add_argument("--use_judge", action="store_true", 
                        help="Use PairwiseJudge instead of auto scoring")
    parser.add_argument("--eval_mode", type=str, default="auto",
                        choices=["auto", "judge", "both"],
                        help="Evaluation mode: auto (PC/SA), judge (PairwiseJudge), both")
    
    args = parser.parse_args()
    run_psychology_ablation(eval_mode=args.eval_mode, use_judge=args.use_judge)
