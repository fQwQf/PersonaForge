"""
Cross-Generator Validation Experiment
======================================

对应论文 Table 12 (Cross-Generator PC Validation)

实验内容:
- 测试不同生成器模型的PC表现
- Gemini 2.5 Flash
- Qwen-72B (通过 OpenAI 兼容 API)
- Llama 70B (通过 API)

运行方式:
    python experiments/cross_generator_experiment.py
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import ExperimentRunner, EvaluationScenario
from experiments.run_experiment import PersonaForgeGenerator, BaselineGenerator
from modules.llm.Gemini import Gemini
from modules.utils import load_json_file, save_json_file


class OpenAICompatibleLLM:
    """OpenAI 兼容 API 的 LLM 包装器"""
    
    def __init__(self, model: str, api_base: str = None, api_key: str = None):
        self.model = model
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """生成响应"""
        try:
            import openai
            client = openai.OpenAI(base_url=self.api_base, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""


def run_cross_generator_experiment(
    num_characters: int = 8,
    output_dir: str = "experiments/experiment_results/cross_generator"
):
    """
    运行跨生成器验证实验
    
    Args:
        num_characters: 测试角色数量
        output_dir: 输出目录
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("Cross-Generator Validation Experiment (Table 12)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load config
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
    
    # 定义生成器配置
    generators = {}
    
    # Gemini (主生成器)
    try:
        gemini_model = project_config.get("role_llm_name", "gemini-2.5-flash-lite")
        generators["Gemini 2.5"] = Gemini(model=gemini_model, timeout=60)
        print("✓ Gemini 2.5 initialized")
    except Exception as e:
        print(f"✗ Gemini initialization failed: {e}")
    
    # Qwen (通过 DashScope OpenAI 兼容 API)
    dashscope_api_base = project_config.get("DASHSCOPE_API_BASE")
    dashscope_api_key = project_config.get("DASHSCOPE_API_KEY")
    if dashscope_api_base and dashscope_api_key:
        try:
            generators["Qwen-Plus"] = OpenAICompatibleLLM(
                model="qwen-plus",
                api_base=dashscope_api_base,
                api_key=dashscope_api_key
            )
            print("✓ Qwen-Plus initialized")
        except Exception as e:
            print(f"✗ Qwen initialization failed: {e}")
    else:
        print("✗ Qwen skipped (no DASHSCOPE API config)")
    
    # Kimi (Moonshot)
    kimi_api_base = project_config.get("KIMI_API_BASE")
    kimi_api_key = project_config.get("KIMI_API_KEY")
    if kimi_api_base and kimi_api_key:
        try:
            generators["Kimi"] = OpenAICompatibleLLM(
                model="moonshot-v1-8k",
                api_base=kimi_api_base,
                api_key=kimi_api_key
            )
            print("✓ Kimi (Moonshot) initialized")
        except Exception as e:
            print(f"✗ Kimi initialization failed: {e}")
    else:
        print("✗ Kimi skipped (no KIMI API config)")
    
    # DeepSeek
    deepseek_api_key = project_config.get("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        try:
            generators["DeepSeek"] = OpenAICompatibleLLM(
                model="deepseek-chat",
                api_base="https://api.deepseek.com/v1",
                api_key=deepseek_api_key
            )
            print("✓ DeepSeek initialized")
        except Exception as e:
            print(f"✗ DeepSeek initialization failed: {e}")
    else:
        print("✗ DeepSeek skipped (no DEEPSEEK_API_KEY)")
    
    if not generators:
        print("ERROR: No generators available")
        return {}
    
    # 使用第一个可用的 LLM 作为评估器
    eval_llm = list(generators.values())[0]
    runner = ExperimentRunner(llm=eval_llm)
    
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
    
    print(f"\nSelected {len(selected_chars)} characters")
    
    # 运行实验
    results = {gen_name: {"ours": [], "structured_cot": []} for gen_name in generators}
    
    for gen_name, gen_llm in generators.items():
        print(f"\n{'=' * 50}")
        print(f"Testing Generator: {gen_name}")
        print("=" * 50)
        
        persona_gen = PersonaForgeGenerator(llm=gen_llm)
        baseline_gen = BaselineGenerator(llm=gen_llm)
        
        for source, role_code in selected_chars:
            role_data = runner.load_character(source, role_code)
            if not role_data:
                continue
            
            print(f"\n[Character] {role_data.get('role_name', role_code)}")
            
            for scenario in runner.scenarios[:4]:  # 使用部分场景
                # Ours
                try:
                    response, inner_mono = persona_gen.generate_with_dual_process(role_data, scenario)
                    eval_res = runner.run_single_evaluation(
                        role_data, scenario, response, inner_mono, method="ours"
                    )
                    results[gen_name]["ours"].append(eval_res.pc_score)
                except Exception as e:
                    print(f"  [ours] Error: {e}")
                
                # Structured-CoT baseline
                try:
                    response = baseline_gen.generate_structured_cot(role_data, scenario)
                    eval_res = runner.run_single_evaluation(
                        role_data, scenario, response, None, method="structured_cot"
                    )
                    results[gen_name]["structured_cot"].append(eval_res.pc_score)
                except Exception as e:
                    print(f"  [structured_cot] Error: {e}")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (Table 12)")
    print("=" * 70)
    print(f"{'Generator':<15} {'Ours':<10} {'S-CoT':<10} {'Δ':<10}")
    print("-" * 45)
    
    summary = {}
    for gen_name in generators:
        ours_scores = results[gen_name]["ours"]
        scot_scores = results[gen_name]["structured_cot"]
        
        if ours_scores and scot_scores:
            ours_avg = sum(ours_scores) / len(ours_scores)
            scot_avg = sum(scot_scores) / len(scot_scores)
            delta = ours_avg - scot_avg
            
            summary[gen_name] = {
                "Ours": ours_avg,
                "S-CoT": scot_avg,
                "Delta": delta
            }
            print(f"{gen_name:<15} {ours_avg:.2f}      {scot_avg:.2f}      +{delta:.2f}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_json_file(f"{output_dir}/cross_generator_results_{timestamp}.json", {
        "config": {
            "num_characters": len(selected_chars),
            "generators": list(generators.keys())
        },
        "results": {
            gen: {
                method: scores 
                for method, scores in methods.items()
            }
            for gen, methods in results.items()
        },
        "summary": summary,
        "timestamp": timestamp
    })
    
    print(f"\n结果已保存到 {output_dir}/")
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Generator Experiment (Table 12)")
    parser.add_argument("--num_characters", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="experiments/experiment_results/cross_generator")
    
    args = parser.parse_args()
    run_cross_generator_experiment(
        num_characters=args.num_characters,
        output_dir=args.output_dir
    )
