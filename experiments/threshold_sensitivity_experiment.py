"""
PC Threshold Sensitivity Experiment
====================================

对应论文 Table 13 (PC Threshold Sensitivity)

实验内容:
- 测试不同 PC 阈值下的 Drift 率
- 阈值: 0.5, 0.6, 0.7
- 对比: Ours vs Structured-CoT

运行方式:
    python experiments/threshold_sensitivity_experiment.py
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import ExperimentRunner, EvaluationScenario
from experiments.run_experiment import PersonaForgeGenerator, BaselineGenerator
from modules.llm.Gemini import Gemini
from modules.utils import load_json_file, save_json_file


def calculate_drift_rate(pc_scores: List[float], threshold: float) -> float:
    """
    计算给定阈值下的 Drift 率
    
    Args:
        pc_scores: PC 分数列表
        threshold: Drift 阈值
        
    Returns:
        Drift 率 (百分比)
    """
    if not pc_scores:
        return 0.0
    drift_count = sum(1 for pc in pc_scores if pc < threshold)
    return (drift_count / len(pc_scores)) * 100


def run_threshold_sensitivity_experiment(
    num_characters: int = 10,
    output_dir: str = "experiments/experiment_results/threshold_sensitivity"
):
    """
    运行 PC 阈值敏感度实验
    
    Args:
        num_characters: 测试角色数量
        output_dir: 输出目录
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("PC Threshold Sensitivity Experiment (Table 13)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load config
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
    
    role_llm_name = project_config.get("role_llm_name", "gemini-2.5-flash-lite")
    llm = Gemini(model=role_llm_name, timeout=60)
    
    runner = ExperimentRunner(llm=llm)
    persona_gen = PersonaForgeGenerator(llm=llm)
    baseline_gen = BaselineGenerator(llm=llm)
    
    # 定义阈值
    thresholds = [0.5, 0.6, 0.7]
    
    # 选择角色
    all_characters = runner.list_characters()
    chinese_sources = ["A_Dream_in_Red_Mansions", "Romance_of_the_Three_Kingdoms"]
    valid_chars = []
    
    for source, role_code in all_characters:
        if source not in chinese_sources:
            continue
        role_data = runner.load_character(source, role_code)
        if role_data:
            personality = role_data.get("personality_profile", {})
            big_five = personality.get("core_traits", {}).get("big_five", {})
            if big_five and len(big_five) >= 5:
                valid_chars.append((source, role_code))
    
    import random
    if num_characters > 0 and num_characters < len(valid_chars):
        selected_chars = random.sample(valid_chars, num_characters)
    else:
        selected_chars = valid_chars
    
    print(f"Selected {len(selected_chars)} characters")
    
    # 收集所有 PC 分数
    ours_pc_scores = []
    scot_pc_scores = []
    
    for source, role_code in selected_chars:
        role_data = runner.load_character(source, role_code)
        if not role_data:
            continue
        
        print(f"\n[Character] {role_data.get('role_name', role_code)}")
        
        for scenario in runner.scenarios:
            # Ours
            try:
                response, inner_mono = persona_gen.generate_with_dual_process(role_data, scenario)
                eval_res = runner.run_single_evaluation(
                    role_data, scenario, response, inner_mono, method="ours"
                )
                ours_pc_scores.append(eval_res.pc_score)
                print(f"  [ours] PC={eval_res.pc_score:.2f}")
            except Exception as e:
                print(f"  [ours] Error: {e}")
            
            # Structured-CoT
            try:
                response = baseline_gen.generate_structured_cot(role_data, scenario)
                eval_res = runner.run_single_evaluation(
                    role_data, scenario, response, None, method="structured_cot"
                )
                scot_pc_scores.append(eval_res.pc_score)
                print(f"  [S-CoT] PC={eval_res.pc_score:.2f}")
            except Exception as e:
                print(f"  [S-CoT] Error: {e}")
    
    # 计算不同阈值下的 Drift 率
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (Table 13)")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Ours↓':<12} {'S-CoT':<12} {'Δ':<12}")
    print("-" * 48)
    
    summary = {}
    for threshold in thresholds:
        ours_drift = calculate_drift_rate(ours_pc_scores, threshold)
        scot_drift = calculate_drift_rate(scot_pc_scores, threshold)
        delta_pct = ((ours_drift - scot_drift) / scot_drift * 100) if scot_drift > 0 else 0
        
        summary[str(threshold)] = {
            "Ours": ours_drift,
            "S-CoT": scot_drift,
            "Delta_pct": delta_pct
        }
        print(f"{threshold:<12} {ours_drift:.1f}%       {scot_drift:.1f}%       {delta_pct:.0f}%")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_json_file(f"{output_dir}/threshold_sensitivity_results_{timestamp}.json", {
        "config": {
            "num_characters": len(selected_chars),
            "thresholds": thresholds,
            "num_ours_samples": len(ours_pc_scores),
            "num_scot_samples": len(scot_pc_scores)
        },
        "raw_scores": {
            "ours": ours_pc_scores,
            "structured_cot": scot_pc_scores
        },
        "summary": summary,
        "timestamp": timestamp
    })
    
    print(f"\n结果已保存到 {output_dir}/")
    print(f"Ours samples: {len(ours_pc_scores)}, S-CoT samples: {len(scot_pc_scores)}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Threshold Sensitivity Experiment (Table 13)")
    parser.add_argument("--num_characters", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="experiments/experiment_results/threshold_sensitivity")
    
    args = parser.parse_args()
    run_threshold_sensitivity_experiment(
        num_characters=args.num_characters,
        output_dir=args.output_dir
    )
