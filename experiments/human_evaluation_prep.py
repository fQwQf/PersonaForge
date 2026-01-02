"""
Human Evaluation Preparation
============================

人工评测数据准备脚本

对应论文:
- Section 5.5: Human Evaluation
- Table 5: Human Evaluation Results

准备内容:
1. 200 个回复对 (Structured-CoT vs. PersonaForge)
2. 10 个角色 × 4 场景类型
3. 盲审问卷生成

运行方式：
    python experiments/human_evaluation_prep.py
"""

import os
import sys
import json
import random
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import ExperimentRunner
from experiments.run_experiment import PersonaForgeGenerator
from modules.llm.Gemini import Gemini
from modules.utils import load_json_file, save_json_file


def prepare_human_evaluation(num_pairs: int = 200, num_characters: int = 10):
    """
    准备人工评测数据
    
    Args:
        num_pairs: 目标回复对数量
        num_characters: 参与角色数量
    """
    # Change to project root so relative paths work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("Human Evaluation Data Preparation")
    print(f"Target: {num_pairs} response pairs from {num_characters} characters")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. Load configuration
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
            
    role_llm_name = project_config.get("role_llm_name", "gemini-2.5-flash-lite")
    llm = Gemini(model=role_llm_name, timeout=60)
    
    # 2. Initialize
    runner = ExperimentRunner(llm=llm)
    persona_gen = PersonaForgeGenerator(llm=llm)
    
    # 3. Select characters (diverse selection)
    all_characters = runner.list_characters()
    chinese_sources = ["A_Dream_in_Red_Mansions", "Romance_of_the_Three_Kingdoms"]
    chinese_chars = [(s, r) for s, r in all_characters if s in chinese_sources]
    
    if num_characters < len(chinese_chars):
        selected_chars = random.sample(chinese_chars, num_characters)
    else:
        selected_chars = chinese_chars
    
    print(f"\nSelected {len(selected_chars)} characters")
    
    # 4. Generate response pairs
    evaluation_pairs = []
    pair_id = 0
    
    # Calculate pairs per character
    pairs_per_char = num_pairs // len(selected_chars)
    scenarios_per_char = min(pairs_per_char, len(runner.scenarios))
    
    for source, role_code in selected_chars:
        role_data = runner.load_character(source, role_code)
        if not role_data:
            continue
            
        print(f"\n[{role_data.get('role_name', role_code)}]")
        
        # Select scenarios for this character
        char_scenarios = random.sample(runner.scenarios, scenarios_per_char)
        
        for scenario in char_scenarios:
            try:
                # Generate Structured-CoT response (baseline)
                resp_cot = persona_gen.generate_without_dual_process(role_data, scenario)
                
                # Generate PersonaForge (full) response
                resp_ours, thought = persona_gen.generate_with_dual_process(role_data, scenario)
                
                # Create blind evaluation pair (randomize order)
                if random.random() > 0.5:
                    response_a, response_b = resp_cot, resp_ours
                    label_a, label_b = "structured_cot", "personaforge"
                else:
                    response_a, response_b = resp_ours, resp_cot
                    label_a, label_b = "personaforge", "structured_cot"
                
                pair = {
                    "pair_id": pair_id,
                    "character": {
                        "name": role_data.get("role_name"),
                        "code": role_code,
                        "source": source,
                        "personality_summary": _summarize_personality(role_data)
                    },
                    "scenario": {
                        "id": scenario.scenario_id,
                        "type": scenario.scenario_type,
                        "context": scenario.context,
                        "trigger": scenario.trigger,
                        "trigger_role": scenario.trigger_role
                    },
                    "responses": {
                        "A": response_a,
                        "B": response_b
                    },
                    "_labels": {  # Hidden from annotators
                        "A": label_a,
                        "B": label_b
                    }
                }
                
                evaluation_pairs.append(pair)
                pair_id += 1
                print(f"  Pair {pair_id}: {scenario.scenario_type}")
                
            except Exception as e:
                print(f"  Error generating pair: {e}")
    
    print(f"\n生成 {len(evaluation_pairs)} 个评测对")
    
    # 5. Generate annotation questionnaire
    questionnaire = _generate_questionnaire(evaluation_pairs)
    
    # 6. Save outputs
    output_dir = "experiments/experiment_results/human_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full data (with labels, for analysis)
    full_file = f"{output_dir}/human_eval_full_{timestamp}.json"
    save_json_file(full_file, {
        "pairs": evaluation_pairs,
        "config": {
            "num_characters": len(selected_chars),
            "num_pairs": len(evaluation_pairs),
            "timestamp": timestamp
        }
    })
    
    # Blind data (without labels, for annotators)
    blind_pairs = []
    for pair in evaluation_pairs:
        blind_pair = {k: v for k, v in pair.items() if not k.startswith("_")}
        blind_pairs.append(blind_pair)
    
    blind_file = f"{output_dir}/human_eval_blind_{timestamp}.json"
    save_json_file(blind_file, blind_pairs)
    
    # Questionnaire (Markdown format)
    questionnaire_file = f"{output_dir}/annotation_form_{timestamp}.md"
    with open(questionnaire_file, 'w', encoding='utf-8') as f:
        f.write(questionnaire)
    
    print(f"\n文件已保存:")
    print(f"  完整数据: {full_file}")
    print(f"  盲审数据: {blind_file}")
    print(f"  标注问卷: {questionnaire_file}")
    print("=" * 70)
    
    return evaluation_pairs


def _summarize_personality(role_data: Dict) -> str:
    """生成角色人格摘要"""
    profile = role_data.get("personality_profile", {})
    core = profile.get("core_traits", {})
    big_five = core.get("big_five", {})
    
    summary_parts = []
    
    # MBTI
    if core.get("mbti"):
        summary_parts.append(f"MBTI: {core['mbti']}")
    
    # Key Big Five traits
    high_traits = [k for k, v in big_five.items() if v > 0.7]
    low_traits = [k for k, v in big_five.items() if v < 0.3]
    
    if high_traits:
        summary_parts.append(f"高: {', '.join(high_traits)}")
    if low_traits:
        summary_parts.append(f"低: {', '.join(low_traits)}")
    
    # Defense mechanism
    if core.get("defense_mechanism"):
        summary_parts.append(f"防御机制: {core['defense_mechanism']}")
    
    return "; ".join(summary_parts) if summary_parts else "无详细信息"


def _generate_questionnaire(pairs: List[Dict]) -> str:
    """生成标注问卷"""
    md = """# 人工评测问卷

## 评测说明

您将看到一系列由 AI 扮演特定角色的回复对比。请根据以下四个维度进行评分 (1-5分)：

1. **真实性 (Authenticity)**: 回复是否像这个角色会说的话？
2. **一致性 (Consistency)**: 回复是否与角色的人格特质一致？
3. **自然度 (Naturalness)**: 回复是否自然流畅，不像机器生成？
4. **心理可信度 (Psychological Plausibility)**: 在压力场景下，角色的反应是否符合其心理特征？

---

"""
    
    for i, pair in enumerate(pairs[:10]):  # Only show first 10 as example
        md += f"""## 评测项 {i+1}

**角色**: {pair['character']['name']}
**人格摘要**: {pair['character']['personality_summary']}

**场景**: {pair['scenario']['context']}
**对方说**: "{pair['scenario']['trigger']}"

### 回复 A
{pair['responses']['A']}

### 回复 B
{pair['responses']['B']}

| 维度 | 回复 A (1-5) | 回复 B (1-5) |
|------|-------------|-------------|
| 真实性 | _____ | _____ |
| 一致性 | _____ | _____ |
| 自然度 | _____ | _____ |
| 心理可信度 | _____ | _____ |

**更优回复**: A / B / 平局

**理由** (可选): _________________________

---

"""
    
    md += f"""
## 统计信息

- 总评测对数: {len(pairs)}
- 涉及角色数: {len(set(p['character']['code'] for p in pairs))}
- 场景类型分布: {_count_scenario_types(pairs)}

---

*请在完成后将此问卷发送给研究人员。感谢您的参与！*
"""
    
    return md


def _count_scenario_types(pairs: List[Dict]) -> str:
    """统计场景类型"""
    counts = {}
    for p in pairs:
        t = p['scenario']['type']
        counts[t] = counts.get(t, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Human Evaluation Preparation")
    parser.add_argument("--num_pairs", type=int, default=200, help="Number of response pairs")
    parser.add_argument("--num_characters", type=int, default=10, help="Number of characters")
    
    args = parser.parse_args()
    prepare_human_evaluation(num_pairs=args.num_pairs, num_characters=args.num_characters)
