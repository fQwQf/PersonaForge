"""
PersonaForge Experiment Runner
==============================

完整实验运行脚本：对比 PersonaForge 与基线方法

使用方法：
    python run_experiment.py --method all --characters 10 --scenarios all
    python run_experiment.py --method ours --characters 5 --scenarios conflict
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import (
    ExperimentRunner, 
    EvaluationResult,
    EvaluationScenario
)
from modules.dual_process_agent import DualProcessAgent
from modules.personality_model import PersonalityProfile


class BaselineGenerator:
    """基线方法生成器"""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def generate_vanilla(self, role_data: Dict, scenario: EvaluationScenario) -> str:
        """
        Vanilla LLM 基线：最简单的 prompt，只给角色名
        不提供任何人格信息，让 LLM 凭自己理解生成
        """
        role_name = role_data.get('role_name', '一个角色')
        
        prompt = f"""你是{role_name}。

场景：{scenario.context}

{scenario.trigger_role}说："{scenario.trigger}"

请回复："""
        
        if self.llm:
            return self.llm.chat(prompt)
        else:
            return f"[Vanilla] 这是{role_name}的回复。"
    
    def generate_character_llm(self, role_data: Dict, scenario: EvaluationScenario) -> str:
        """
        Character-LLM 基线：使用角色描述，但不使用心理学框架
        只提供基础描述和示例
        """
        role_name = role_data.get('role_name', '一个角色')
        profile = role_data.get('profile', '')
        
        # 只使用1个示例
        style_examples = role_data.get('style_examples', [])
        examples_text = ""
        if style_examples:
            ex = style_examples[0]
            examples_text = f"\n\n参考：{ex.get('response', '')[:100]}..."
        
        prompt = f"""你是{role_name}。

{profile}
{examples_text}

场景：{scenario.context}

{scenario.trigger_role}说："{scenario.trigger}"

请以{role_name}的身份回复："""
        
        if self.llm:
            return self.llm.chat(prompt)
        else:
            return f"[CharacterLLM] 这是{role_name}的回复。"
    
    def generate_structured_cot(self, role_data: Dict, scenario: EvaluationScenario) -> str:
        """
        Structured-CoT 基线：使用 CoT 但不使用心理学框架
        只让模型"思考"，但没有大五人格等结构化人格信息
        """
        role_name = role_data.get('role_name', '一个角色')
        profile = role_data.get('profile', '')
        
        prompt = f"""你是{role_name}。

【角色描述】
{profile}

【思考步骤】
1. 考虑{role_name}会如何理解这个情况
2. 考虑{role_name}的可能反应
3. 生成符合{role_name}的回复

【场景】{scenario.context}

{scenario.trigger_role}说："{scenario.trigger}"

请先思考，然后以{role_name}的身份回复："""
        
        if self.llm:
            return self.llm.chat(prompt)
        else:
            return f"[StructuredCoT] 这是{role_name}的回复。"

    def generate_rag_persona(self, role_data: Dict, scenario: EvaluationScenario) -> str:
        """
        RAG-Persona 基线：检索增强的 Prompting
        模拟 RAG 过程：根据当前场景检索最相关的记忆/示例
        """
        role_name = role_data.get('role_name', '一个角色')
        profile = role_data.get('profile', '')
        style_examples = role_data.get('style_examples', [])
        
        # 简单的基于文本重叠的模拟检索
        # 在真实 RAG 中这里会用向量相似度
        relevant_examples = []
        if style_examples:
            # 计算简单的相似度（共有词数量）
            query_set = set(scenario.trigger)
            scored_examples = []
            for ex in style_examples:
                ex_text = ex.get('action', '') + ex.get('response', '')
                score = len(set(ex_text) & query_set)
                scored_examples.append((score, ex))
            
            # 取前3个最相关的
            scored_examples.sort(key=lambda x: x[0], reverse=True)
            relevant_examples = [x[1] for x in scored_examples[:3]]
        
        examples_text = ""
        if relevant_examples:
            examples_text = "\n【参考回忆】\n"
            for i, ex in enumerate(relevant_examples, 1):
                examples_text += f"回忆{i}: 当被问到\"{ex.get('action', '')}\"时，我回答：\"{ex.get('response', '')}\"\n"
        
        prompt = f"""你是{role_name}。

{profile}

{examples_text}
场景：{scenario.context}

{scenario.trigger_role}说："{scenario.trigger}"

请参考上述回忆（如果相关），以{role_name}的身份回复："""
        
        if self.llm:
            return self.llm.chat(prompt)
        else:
            return f"[RAG-Persona] 这是{role_name}的回复。"

    def generate_role_llm(self, role_data: Dict, scenario: EvaluationScenario) -> str:
        """
        RoleLLM 基线：Imitation Learning + Retrieval
        (Wang et al., 2023) 提出的 retrieve-then-generate 范式
        """
        role_name = role_data.get('role_name', '一个角色')
        profile = role_data.get('profile', '')
        style_examples = role_data.get('style_examples', [])
        
        # 模拟检索：基于 trigger 的语义相似度检索最相关的对话
        # 实际 RoleLLM 使用 BGE-Small 检索
        retrieved_dialogues = []
        if style_examples:
            # 简单模拟：计算关键词重叠
            query_tokens = set(scenario.trigger)
            scored_ex = []
            for ex in style_examples:
                # 假设 context 越相似，example 越相关
                ex_tokens = set(ex.get('action', ''))
                score = len(query_tokens & ex_tokens)
                scored_ex.append((score, ex))
            
            # 取前 5 个 (RoleLLM 原文 top-5)
            scored_ex.sort(key=lambda x: x[0], reverse=True)
            retrieved_dialogues = [x[1] for x in scored_ex[:5]]
            
        dialogue_text = ""
        if retrieved_dialogues:
            dialogue_text = "\\n【相关历史对话】\\n"
            for i, d in enumerate(retrieved_dialogues, 1):
                try:
                    dialogue_text += f"{i}. Q: {d.get('action', '')} A: {d.get('response', '')}\\n"
                except:
                    continue
        
        prompt = f"""Role Profile: {role_name}
        
Description: {profile}

{dialogue_text}

Instruction: Reply to the following message as {role_name}. consistent with the style above.

Input: {scenario.trigger_role}: "{scenario.trigger}"
Response: """

        if self.llm:
            return self.llm.chat(prompt)
        else:
            return f"[RoleLLM] 这是{role_name}的回复。"


class PersonaForgeGenerator:
    """PersonaForge 方法生成器 - 使用项目核心模块"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.dual_process_agent = DualProcessAgent(llm=llm, language="zh")
        print("✓ PersonaForgeGenerator initialized with DualProcessAgent from modules.dual_process_agent")
    
    def _build_personality_profile(self, role_data: Dict) -> PersonalityProfile:
        """从 role_data 构建 PersonalityProfile 对象"""
        pp_data = role_data.get("personality_profile", {})
        if not pp_data:
            # Fallback removed: strict mode requires personality_profile
            return None
        return PersonalityProfile.from_dict(pp_data)
    
    def generate_with_dual_process(self, role_data: Dict, scenario: EvaluationScenario) -> tuple:
        """
        PersonaForge 完整方法：使用项目核心模块的 DualProcessAgent
        
        Returns:
            (response, inner_monologue)
        """
        # 构建 PersonalityProfile 对象
        profile = self._build_personality_profile(role_data)
        if profile is None:
            raise ValueError(f"Character {role_data.get('role_name', 'unknown')} missing personality_profile.")
            
        style_examples = role_data.get("style_examples", profile.style_examples)
        
        # 使用核心模块生成内心独白
        inner_monologue = self.dual_process_agent.generate_inner_monologue(
            personality_profile=profile,
            action_detail=scenario.trigger,
            action_maker_name=scenario.trigger_role,
            history=""
        )
        
        # 使用核心模块生成风格化回复
        response = self.dual_process_agent.generate_styled_response(
            inner_monologue=inner_monologue,
            personality_profile=profile,
            style_examples=style_examples,
            action_detail=scenario.trigger,
            action_maker_name=scenario.trigger_role,
            history=""
        )
        
        return response, inner_monologue
    
    def generate_without_dual_process(self, role_data: Dict, scenario: EvaluationScenario) -> str:
        """
        Structured-CoT 基线：只用三层人格模型，不用双重思维链
        直接生成回复，不经过内心独白阶段
        """
        profile = self._build_personality_profile(role_data)
        if profile is None:
            raise ValueError(f"Character {role_data.get('role_name', 'unknown')} missing personality_profile.")
        
        big_five_desc = ", ".join([f"{k}: {v:.2f}" for k, v in profile.core_traits.big_five.items()])
        
        prompt = f"""你是{role_data.get('role_name')}。

【人格特质】
- MBTI: {profile.core_traits.mbti}
- 大五人格: {big_five_desc}
- 防御机制: {profile.core_traits.defense_mechanism}
- 价值观: {', '.join(profile.core_traits.values)}

【语言风格】
- 句长: {profile.speaking_style.sentence_length}
- 词汇等级: {profile.speaking_style.vocabulary_level}
- 语气词: {', '.join(profile.speaking_style.tone_markers)}
- 口头禅: {', '.join(profile.speaking_style.catchphrases)}

【场景】{scenario.context}

{scenario.trigger_role}说："{scenario.trigger}"

请以{role_data.get('role_name')}的身份，用符合人格和风格的方式回复："""
        
        if self.llm:
            return self.llm.chat(prompt)
        else:
            return f"[PersonaForge-NoDual] 这是{role_data.get('role_name')}的回复。"


def run_experiment(args):
    """运行实验"""
    # Change to project root so relative paths work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("PersonaForge ACL Experiment")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 加载配置并初始化 LLM
    from modules.llm.Gemini import Gemini
    from modules.utils import load_json_file
    
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
            
    role_llm_name = project_config.get("role_llm_name", "gemini-2.5-flash-lite")
    llm = Gemini(model=role_llm_name, timeout=60)
    print(f"已初始化 LLM: {role_llm_name}")
    
    # 初始化
    runner = ExperimentRunner(llm=llm)
    baseline_gen = BaselineGenerator(llm=llm)
    persona_gen = PersonaForgeGenerator(llm=llm)
    
    # 选择角色 - PRE-FILTER to only valid characters with personality_profile
    all_characters = runner.list_characters()
    valid_characters = []
    for source, role_code in all_characters:
        role_data = runner.load_character(source, role_code)
        if role_data and role_data.get("personality_profile"):
            valid_characters.append((source, role_code))
    
    print(f"[Pre-filter] Found {len(valid_characters)} valid characters (with personality_profile)")
    
    if args.characters == "all":
        selected_characters = valid_characters
    else:
        n = int(args.characters)
        selected_characters = random.sample(valid_characters, min(n, len(valid_characters)))
    
    print(f"选择的角色数量: {len(selected_characters)}")
    
    # 选择场景
    if args.scenarios == "all":
        selected_scenarios = runner.scenarios
    else:
        selected_scenarios = [s for s in runner.scenarios if s.scenario_type == args.scenarios]
    
    print(f"选择的场景数量: {len(selected_scenarios)}")
    
    # 存储结果
    results_by_method = {
        "vanilla": [],
        "character_llm": [],
        "structured_cot": [],
        "rag_persona": [],
        "ours": [],
        "ours_no_dual": []
    }
    
    # 遍历角色和场景
    total = len(selected_characters) * len(selected_scenarios)
    current = 0
    
    for source, role_code in selected_characters:
        role_data = runner.load_character(source, role_code)
        if not role_data:
            continue
        
        print(f"\n处理角色: {role_data.get('role_name', role_code)}")
        
        for scenario in selected_scenarios:
            current += 1
            print(f"  [{current}/{total}] 场景: {scenario.scenario_id}")
            
            # 方法 1: Vanilla LLM
            if args.method in ["all", "vanilla"]:
                response = baseline_gen.generate_vanilla(role_data, scenario)
                result = runner.run_single_evaluation(
                    role_data, scenario, response, method="vanilla"
                )
                results_by_method["vanilla"].append(result)
            
            # 方法 2: Character-LLM
            if args.method in ["all", "character_llm"]:
                response = baseline_gen.generate_character_llm(role_data, scenario)
                result = runner.run_single_evaluation(
                    role_data, scenario, response, method="character_llm"
                )
                results_by_method["character_llm"].append(result)
            
            # 方法 2.5: Structured-CoT
            if args.method in ["all", "structured_cot"]:
                response = baseline_gen.generate_structured_cot(role_data, scenario)
                result = runner.run_single_evaluation(
                    role_data, scenario, response, method="structured_cot"
                )
                results_by_method["structured_cot"].append(result)

            # 方法 2.6: RAG-Persona
            if args.method in ["all", "rag_persona"]:
                response = baseline_gen.generate_rag_persona(role_data, scenario)
                result = runner.run_single_evaluation(
                    role_data, scenario, response, method="rag_persona"
                )
                results_by_method["rag_persona"].append(result)
            
            # 方法 3: PersonaForge (完整)
            if args.method in ["all", "ours"]:
                response, inner_mono = persona_gen.generate_with_dual_process(role_data, scenario)
                result = runner.run_single_evaluation(
                    role_data, scenario, response, inner_mono, method="ours"
                )
                results_by_method["ours"].append(result)
            
            # 方法 4: PersonaForge (无双重思维链)
            if args.method in ["all", "ours_no_dual"]:
                response = persona_gen.generate_without_dual_process(role_data, scenario)
                result = runner.run_single_evaluation(
                    role_data, scenario, response, method="ours_no_dual"
                )
                results_by_method["ours_no_dual"].append(result)
    
    # 计算聚合分数
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    for method, results in results_by_method.items():
        if not results:
            continue
        
        scores = runner.compute_aggregate_scores(results)
        print(f"\n[{method.upper()}] ({len(results)} samples)")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for method, results in results_by_method.items():
        if results:
            runner.save_results(results, f"{method}_{timestamp}.json")
    
    print(f"\n实验完成！结果已保存到 {runner.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="PersonaForge Experiment Runner")
    parser.add_argument(
        "--method", 
        type=str, 
        default="all",
        choices=["all", "vanilla", "character_llm", "structured_cot", "rag_persona", "role_llm", "ours", "ours_no_dual"],
        help="要运行的方法"
    )
    parser.add_argument(
        "--characters",
        type=str,
        default="10",
        help="使用的角色数量，或 'all'"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help="场景类型: all, emotional, conflict, casual, first_encounter, decision"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_results",
        help="输出目录"
    )
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
