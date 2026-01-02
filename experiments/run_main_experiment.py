"""
Main Scenario-Based Experiment
==============================

完整场景实验脚本：对应论文 Table 1 (Main Results)

实验设置:
- 37 个角色 (红楼梦 + 三国演义)
- 8 个场景类型
- 对比方法: Vanilla LLM, Character-LLM-style, Structured-CoT, RAG-Persona, Ours

运行方式：
    python experiments/run_main_experiment.py --num_characters 10 --all_methods
"""

import os
import sys
import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import (
    ExperimentRunner, 
    EvaluationResult,
    EvaluationScenario
)
from experiments.run_experiment import PersonaForgeGenerator, BaselineGenerator
from modules.llm.Gemini import Gemini
from modules.utils import load_json_file, save_json_file


def run_main_experiment(
    num_characters: int = 10,
    all_methods: bool = True,
    output_dir: str = "experiments/experiment_results/main"
):
    """
    运行主实验
    
    Args:
        num_characters: 测试角色数量
        all_methods: 是否测试所有方法
        output_dir: 输出目录
    """
    # Change to project root so relative paths work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("PersonaForge Main Experiment (Scenario-Based)")
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
    
    # Generators
    persona_gen = PersonaForgeGenerator(llm=llm)
    baseline_gen = BaselineGenerator(llm=llm)
    
    # 3. Select characters (with personality_profile validation)
    all_characters = runner.list_characters()
    # Filter to main Chinese sources
    chinese_sources = ["A_Dream_in_Red_Mansions", "Romance_of_the_Three_Kingdoms"]
    chinese_chars = [(s, r) for s, r in all_characters if s in chinese_sources]
    
    # Validate characters have personality_profile with big_five
    valid_chars = []
    for source, role_code in chinese_chars:
        role_data = runner.load_character(source, role_code)
        if role_data:
            personality_profile = role_data.get("personality_profile", {})
            big_five = personality_profile.get("core_traits", {}).get("big_five", {})
            if big_five and len(big_five) >= 5:
                valid_chars.append((source, role_code))
            else:
                print(f"  Skipping {role_code}: missing personality_profile or big_five")
    
    print(f"\nFound {len(valid_chars)} valid characters (with personality_profile)")
    
    if num_characters > 0 and num_characters < len(valid_chars):
        selected_chars = random.sample(valid_chars, num_characters)
    else:
        selected_chars = valid_chars
        
    print(f"Selected {len(selected_chars)} characters for evaluation")
    
    # 4. Results storage
    results = {
        "vanilla": [],
        "character_llm": [],
        "structured_cot": [],
        "rag_persona": [],
        "role_llm": [],
        "ours_no_dual": [],
        "ours": []
    }
    
    methods_to_run = list(results.keys()) if all_methods else ["structured_cot", "role_llm", "ours"]
    
    # 5. Run experiments
    total_tests = len(selected_chars) * len(runner.scenarios) * len(methods_to_run)
    current = 0
    
    for source, role_code in selected_chars:
        role_data = runner.load_character(source, role_code)
        if not role_data:
            print(f"  Warning: Could not load {role_code}")
            continue
            
        print(f"\n[Character] {role_data.get('role_name', role_code)}")
        
        for scenario in runner.scenarios:
            print(f"  [Scenario] {scenario.scenario_id} ({scenario.scenario_type})")
            
            for method in methods_to_run:
                current += 1
                
                try:
                    if method == "vanilla":
                        response = baseline_gen.generate_vanilla(role_data, scenario)
                        inner_mono = None
                    elif method == "character_llm":
                        response = baseline_gen.generate_character_llm(role_data, scenario)
                        inner_mono = None
                    elif method == "structured_cot":
                        # Structured-CoT: 使用 CoT 但不使用心理学框架（与 ours_no_dual 区分）
                        response = baseline_gen.generate_structured_cot(role_data, scenario)
                        inner_mono = None
                    elif method == "rag_persona":
                        # RAG-Persona: 检索增强
                        response = baseline_gen.generate_rag_persona(role_data, scenario)
                        inner_mono = None
                    elif method == "role_llm":
                        # RoleLLM: Imitation
                        response = baseline_gen.generate_role_llm(role_data, scenario)
                        inner_mono = None
                    elif method == "ours_no_dual":
                        response = persona_gen.generate_without_dual_process(role_data, scenario)
                        inner_mono = None
                    elif method == "ours":
                        response, inner_mono = persona_gen.generate_with_dual_process(role_data, scenario)
                    else:
                        continue
                    
                    # Evaluate
                    eval_res = runner.run_single_evaluation(
                        role_data, scenario, response, inner_mono, method=method
                    )
                    results[method].append(eval_res)
                    
                    print(f"    [{method}] PC={eval_res.pc_score:.2f}, SA={eval_res.sa_score:.2f}, DM={eval_res.dm_score:.2f}")
                    
                except Exception as e:
                    print(f"    [{method}] Error: {e}")
    
    # 6. Compute aggregate scores
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    summary = {}
    for method, res_list in results.items():
        if not res_list:
            continue
        scores = runner.compute_aggregate_scores(res_list)
        summary[method] = scores
        
        print(f"\n[{method.upper()}] (n={len(res_list)})")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")
    
    # 7. Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    for method, res_list in results.items():
        if res_list:
            output_file = f"{output_dir}/main_{method}_{timestamp}.json"
            save_json_file(output_file, [asdict(r) for r in res_list])
    
    # Save summary
    summary_file = f"{output_dir}/main_summary_{timestamp}.json"
    save_json_file(summary_file, {
        "config": {
            "num_characters": len(selected_chars),
            "num_scenarios": len(runner.scenarios),
            "methods": methods_to_run,
            "llm": role_llm_name
        },
        "summary": summary,
        "timestamp": timestamp
    })
    
    print(f"\n结果已保存到 {output_dir}/")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PersonaForge Main Experiment")
    parser.add_argument("--num_characters", type=int, default=10, help="Number of characters to test")
    parser.add_argument("--all_methods", action="store_true", help="Test all methods")
    parser.add_argument("--output_dir", type=str, default="experiments/experiment_results/main")
    
    args = parser.parse_args()
    run_main_experiment(
        num_characters=args.num_characters,
        all_methods=args.all_methods,
        output_dir=args.output_dir
    )
