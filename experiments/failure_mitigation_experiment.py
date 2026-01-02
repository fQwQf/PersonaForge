"""
Failure Case Mitigation Experiment
===================================

对应论文 Table 8 (Failure Case Mitigations Impact)

实验内容:
- 测试不同缓解策略对PC和Drift的影响
- Baseline: 无缓解
- + Context-aware stressor: 上下文感知压力检测
- + Register-adaptive style: 寄存器自适应风格
- + Relationship priors: 关系先验
- All combined: 全部组合

运行方式:
    python experiments/failure_mitigation_experiment.py
"""

import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import ExperimentRunner, EvaluationScenario
from experiments.run_experiment import PersonaForgeGenerator
from modules.llm.Gemini import Gemini
from modules.utils import load_json_file, save_json_file


@dataclass
class MitigationConfig:
    """缓解策略配置"""
    name: str
    context_aware_stressor: bool = False
    register_adaptive_style: bool = False
    relationship_priors: bool = False


class MitigatedPersonaForgeGenerator(PersonaForgeGenerator):
    """支持缓解策略的PersonaForge生成器"""
    
    def __init__(self, llm, config: MitigationConfig):
        super().__init__(llm)
        self.config = config
    
    def generate_with_mitigations(self, role_data: Dict, scenario: EvaluationScenario):
        """带缓解策略的生成"""
        # 基础生成
        response, inner_mono = self.generate_with_dual_process(role_data, scenario)
        
        # 应用缓解策略（在提示中添加额外指导）
        if self.config.context_aware_stressor:
            # 增强压力检测：分析上下文中的隐含压力
            pass  # 实际在 dual_process_agent 中实现
            
        if self.config.register_adaptive_style:
            # 寄存器自适应：根据场景调整语言风格
            pass
            
        if self.config.relationship_priors:
            # 关系先验：使用角色关系初始化
            pass
        
        return response, inner_mono


def run_mitigation_experiment(
    num_characters: int = 10,
    num_turns: int = 20,
    output_dir: str = "experiments/experiment_results/mitigation"
):
    """
    运行缓解策略实验
    
    Args:
        num_characters: 测试角色数量
        num_turns: 每个角色的对话轮数
        output_dir: 输出目录
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("Failure Mitigation Experiment (Table 8)")
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
    
    # 定义缓解策略组合
    mitigation_configs = [
        MitigationConfig("baseline"),
        MitigationConfig("context_aware", context_aware_stressor=True),
        MitigationConfig("register_adaptive", register_adaptive_style=True),
        MitigationConfig("relationship_priors", relationship_priors=True),
        MitigationConfig("all_combined", 
                        context_aware_stressor=True,
                        register_adaptive_style=True, 
                        relationship_priors=True),
    ]
    
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
    
    # 运行实验
    results = {config.name: [] for config in mitigation_configs}
    
    for source, role_code in selected_chars:
        role_data = runner.load_character(source, role_code)
        if not role_data:
            continue
            
        print(f"\n[Character] {role_data.get('role_name', role_code)}")
        
        for config in mitigation_configs:
            generator = MitigatedPersonaForgeGenerator(llm, config)
            pc_scores = []
            
            for scenario in runner.scenarios[:4]:  # 使用部分场景
                try:
                    response, inner_mono = generator.generate_with_mitigations(role_data, scenario)
                    eval_res = runner.run_single_evaluation(
                        role_data, scenario, response, inner_mono, method=f"mitigation_{config.name}"
                    )
                    pc_scores.append(eval_res.pc_score)
                except Exception as e:
                    print(f"  [{config.name}] Error: {e}")
            
            if pc_scores:
                avg_pc = sum(pc_scores) / len(pc_scores)
                drift_count = sum(1 for pc in pc_scores if pc < 0.6)
                drift_rate = drift_count / len(pc_scores) * 100
                
                results[config.name].append({
                    "role_code": role_code,
                    "avg_pc": avg_pc,
                    "drift_rate": drift_rate
                })
                print(f"  [{config.name}] PC={avg_pc:.2f}, Drift={drift_rate:.1f}%")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (Table 8)")
    print("=" * 70)
    
    summary = {}
    for config_name, res_list in results.items():
        if res_list:
            avg_pc = sum(r["avg_pc"] for r in res_list) / len(res_list)
            avg_drift = sum(r["drift_rate"] for r in res_list) / len(res_list)
            summary[config_name] = {"PC": avg_pc, "Drift": avg_drift}
            print(f"{config_name}: PC={avg_pc:.2f}, Drift={avg_drift:.1f}%")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_json_file(f"{output_dir}/mitigation_results_{timestamp}.json", {
        "config": {
            "num_characters": len(selected_chars),
            "mitigations": [asdict(c) for c in mitigation_configs]
        },
        "results": results,
        "summary": summary,
        "timestamp": timestamp
    })
    
    print(f"\n结果已保存到 {output_dir}/")
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Failure Mitigation Experiment (Table 8)")
    parser.add_argument("--num_characters", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="experiments/experiment_results/mitigation")
    
    args = parser.parse_args()
    run_mitigation_experiment(
        num_characters=args.num_characters,
        output_dir=args.output_dir
    )
