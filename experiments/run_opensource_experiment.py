"""
Open-Source Model Validation Experiment
========================================

验证 PersonaForge 在开源模型上的效果，以回应审稿人关于闭源模型依赖的质疑。

使用方法：
    python run_opensource_experiment.py --model qwen-max --characters 10
    python run_opensource_experiment.py --model llama3-70b --characters all
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
from experiments.run_experiment import PersonaForgeGenerator, BaselineGenerator


def get_llm_by_name(model_name: str):
    """
    根据模型名称获取 LLM 实例
    
    支持的模型：
    - qwen-max, qwen-plus, qwen-turbo (Qwen 系列)
    - llama3-70b, llama3-8b (通过 OpenRouter)
    - deepseek-v3 (通过 OpenRouter)
    - gpt-4-turbo (OpenAI)
    """
    if model_name.startswith("qwen"):
        from modules.llm.Qwen import Qwen
        return Qwen(model=model_name)
    elif model_name.startswith("llama"):
        from modules.llm.OpenRouter import OpenRouter
        model_map = {
            "llama3-70b": "meta-llama/llama-3-70b-instruct",
            "llama3-8b": "meta-llama/llama-3-8b-instruct",
        }
        return OpenRouter(model=model_map.get(model_name, model_name))
    elif model_name.startswith("deepseek"):
        from modules.llm.DeepSeek import DeepSeek
        return DeepSeek()
    elif model_name.startswith("gpt"):
        from modules.llm.LangChainGPT2 import LangChainGPT2
        return LangChainGPT2(model=model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_opensource_config():
    """加载开源模型配置"""
    # Try experiments/opensource_config.json first
    config_path = os.path.join(os.path.dirname(__file__), "opensource_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Fallback to main config.json
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_config_path = os.path.join(project_root, "config.json")
    if os.path.exists(main_config_path):
        with open(main_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return {}


def setup_api_keys(model_name: str, config: dict):
    """根据模型类型设置 API Key 环境变量"""
    if model_name.startswith("qwen"):
        qwen_config = config.get("qwen", {})
        api_key = qwen_config.get("api_key") or config.get("DASHSCOPE_API_KEY", "")
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
            print(f"✓ DASHSCOPE_API_KEY loaded")
        else:
            print("⚠ DASHSCOPE_API_KEY not found in config, please set it in experiments/opensource_config.json")
            return False
    
    elif model_name.startswith("deepseek"):
        ds_config = config.get("deepseek", {})
        api_key = ds_config.get("api_key") or config.get("DEEPSEEK_API_KEY", "")
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            print(f"✓ DEEPSEEK_API_KEY loaded")
        else:
            print("⚠ DEEPSEEK_API_KEY not found in config")
            return False
    
    elif model_name.startswith("llama"):
        or_config = config.get("openrouter", {})
        api_key = or_config.get("api_key") or config.get("OPENROUTER_API_KEY", "")
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
            print(f"✓ OPENROUTER_API_KEY loaded")
        else:
            print("⚠ OPENROUTER_API_KEY not found in config")
            return False
    
    return True


def run_opensource_experiment(args):
    """运行开源模型对比实验"""
    # Change to project root so relative paths work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("PersonaForge Open-Source Model Validation Experiment")
    print(f"Model: {args.model}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 加载配置并设置 API Key
    config = load_opensource_config()
    if not setup_api_keys(args.model, config):
        print("\n请在 experiments/opensource_config.json 中配置对应模型的 API Key")
        return
    
    # 初始化 LLM
    try:
        llm = get_llm_by_name(args.model)
        print(f"✓ LLM initialized: {args.model}")
    except Exception as e:
        print(f"✗ Failed to initialize LLM: {e}")
        return
    
    # 初始化评估框架
    runner = ExperimentRunner(llm=llm)
    persona_gen = PersonaForgeGenerator(llm=llm)
    baseline_gen = BaselineGenerator(llm=llm)
    
    # 选择角色
    all_characters = runner.list_characters()
    if args.characters == "all":
        selected_characters = all_characters
    else:
        n = int(args.characters)
        selected_characters = random.sample(all_characters, min(n, len(all_characters)))
    
    print(f"\n选择的角色数量: {len(selected_characters)}")
    
    # 存储结果
    results_personaforge = []
    results_structured_cot = []
    
    # 遍历角色和场景
    total = len(selected_characters) * len(runner.scenarios)
    current = 0
    
    for source, role_code in selected_characters:
        role_data = runner.load_character(source, role_code)
        if not role_data:
            continue
        
        print(f"\n处理角色: {role_data.get('role_name', role_code)}")
        
        for scenario in runner.scenarios:
            current += 1
            print(f"  [{current}/{total}] 场景: {scenario.scenario_id}")
            
            try:
                # PersonaForge 方法
                response_pf, inner_mono = persona_gen.generate_with_dual_process(role_data, scenario)
                result_pf = runner.run_single_evaluation(
                    role_data, scenario, response_pf, inner_mono, method="personaforge"
                )
                results_personaforge.append(result_pf)
                
                # Structured-CoT 基线
                response_cot = baseline_gen.generate_character_llm(role_data, scenario)
                result_cot = runner.run_single_evaluation(
                    role_data, scenario, response_cot, method="structured_cot"
                )
                results_structured_cot.append(result_cot)
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
    
    # 计算聚合分数
    print("\n" + "=" * 70)
    print(f"实验结果汇总 ({args.model})")
    print("=" * 70)
    
    if results_personaforge:
        scores_pf = runner.compute_aggregate_scores(results_personaforge)
        print(f"\n[PersonaForge] ({len(results_personaforge)} samples)")
        for metric, score in scores_pf.items():
            print(f"  {metric}: {score:.4f}")
    
    if results_structured_cot:
        scores_cot = runner.compute_aggregate_scores(results_structured_cot)
        print(f"\n[Structured-CoT] ({len(results_structured_cot)} samples)")
        for metric, score in scores_cot.items():
            print(f"  {metric}: {score:.4f}")
    
    # 计算提升幅度
    if results_personaforge and results_structured_cot:
        delta_pc = scores_pf.get("pc", 0) - scores_cot.get("pc", 0)
        print(f"\n[Δ PC]: {delta_pc:+.2f}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_safe = args.model.replace("/", "_")
    
    output = {
        "model": args.model,
        "timestamp": timestamp,
        "personaforge": {
            "results_count": len(results_personaforge),
            "scores": runner.compute_aggregate_scores(results_personaforge) if results_personaforge else {}
        },
        "structured_cot": {
            "results_count": len(results_structured_cot),
            "scores": runner.compute_aggregate_scores(results_structured_cot) if results_structured_cot else {}
        }
    }
    
    output_path = os.path.join(runner.output_dir, f"opensource_{model_safe}_{timestamp}.json")
    os.makedirs(runner.output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验完成！结果已保存到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PersonaForge Open-Source Model Experiment")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-v3",
        help="模型名称: qwen-max, qwen-plus, llama3-70b, deepseek-v3, gpt-4-turbo"
    )
    parser.add_argument(
        "--characters",
        type=str,
        default="10",
        help="使用的角色数量，或 'all'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_results",
        help="输出目录"
    )
    
    args = parser.parse_args()
    run_opensource_experiment(args)


if __name__ == "__main__":
    main()
