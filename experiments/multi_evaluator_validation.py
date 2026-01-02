"""
Multi-Evaluator Consistency Validation
========================================

验证多个LLM评估器的一致性，对应论文 L787-790 的声称。

实验内容:
- 使用 Gemini, Qwen, Kimi, DeepSeek 分别评估相同的响应
- 计算评估器之间的 Pearson 相关系数
- 验证所有评估器倾向一致

运行方式:
    python experiments/multi_evaluator_validation.py
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import ExperimentRunner, EvaluationScenario
from experiments.run_experiment import PersonaForgeGenerator, BaselineGenerator
from modules.llm.Gemini import Gemini
from modules.utils import load_json_file, save_json_file


class OpenAICompatibleLLM:
    """OpenAI 兼容 API 的 LLM 包装器"""
    
    def __init__(self, model: str, api_base: str, api_key: str, name: str = None):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.name = name or model
        self.messages = []
        self.system_prompt = None
        
    def initialize_message(self):
        self.messages = []
        self.system_prompt = None
        
    def system_message(self, content):
        self.messages.append({"role": "system", "content": content})
        
    def user_message(self, content):
        self.messages.append({"role": "user", "content": content})
        
    def get_response(self, temperature=0.0):
        try:
            import openai
            client = openai.OpenAI(base_url=self.api_base, api_key=self.api_key)
            
            full_messages = []
            if self.system_prompt:
                full_messages.append({"role": "system", "content": self.system_prompt})
            full_messages.extend(self.messages)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{self.name}] API error: {e}")
            return ""
    
    def chat(self, text, temperature=0.0):
        """Simple chat method for compatibility with evaluation framework"""
        self.initialize_message()
        self.user_message(text)
        return self.get_response(temperature)



def calculate_pearson_correlation(list1: List[float], list2: List[float]) -> float:
    """计算 Pearson 相关系数"""
    if len(list1) != len(list2) or len(list1) < 2:
        return 0.0
    
    n = len(list1)
    mean1 = sum(list1) / n
    mean2 = sum(list2) / n
    
    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(list1, list2))
    denom1 = sum((x - mean1) ** 2 for x in list1) ** 0.5
    denom2 = sum((y - mean2) ** 2 for y in list2) ** 0.5
    
    if denom1 == 0 or denom2 == 0:
        return 0.0
    
    return numerator / (denom1 * denom2)


def run_multi_evaluator_validation(
    num_samples: int = 30,
    output_dir: str = "experiments/experiment_results/multi_evaluator"
):
    """
    运行多评估器一致性验证实验
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("Multi-Evaluator Consistency Validation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load config
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
    
    # 初始化评估器
    evaluators = {}
    
    # Gemini
    try:
        evaluators["Gemini"] = Gemini(model="gemini-2.5-flash-lite", timeout=60)
        print("✓ Gemini evaluator initialized")
    except Exception as e:
        print(f"✗ Gemini failed: {e}")
    
    # Qwen
    if project_config.get("DASHSCOPE_API_KEY"):
        evaluators["Qwen"] = OpenAICompatibleLLM(
            model="qwen-plus",
            api_base=project_config["DASHSCOPE_API_BASE"],
            api_key=project_config["DASHSCOPE_API_KEY"],
            name="Qwen"
        )
        print("✓ Qwen evaluator initialized")
    
    # Kimi
    if project_config.get("KIMI_API_KEY"):
        evaluators["Kimi"] = OpenAICompatibleLLM(
            model="moonshot-v1-8k",
            api_base=project_config["KIMI_API_BASE"],
            api_key=project_config["KIMI_API_KEY"],
            name="Kimi"
        )
        print("✓ Kimi evaluator initialized")
    
    # DeepSeek
    if project_config.get("DEEPSEEK_API_KEY"):
        evaluators["DeepSeek"] = OpenAICompatibleLLM(
            model="deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key=project_config["DEEPSEEK_API_KEY"],
            name="DeepSeek"
        )
        print("✓ DeepSeek evaluator initialized")
    
    if len(evaluators) < 2:
        print("ERROR: Need at least 2 evaluators")
        return {}
    
    # 使用 Gemini 生成样本
    gen_llm = evaluators.get("Gemini") or list(evaluators.values())[0]
    runner = ExperimentRunner(llm=gen_llm)
    persona_gen = PersonaForgeGenerator(llm=gen_llm)
    baseline_gen = BaselineGenerator(llm=gen_llm)
    
    # 选择角色
    all_characters = runner.list_characters()
    chinese_sources = ["A_Dream_in_Red_Mansions", "Romance_of_the_Three_Kingdoms"]
    valid_chars = [(s, r) for s, r in all_characters if s in chinese_sources]
    valid_chars = valid_chars[:10]  # 限制角色数量
    
    # 生成评估样本
    print(f"\nGenerating {num_samples} evaluation samples...")
    samples = []
    
    for source, role_code in valid_chars:
        role_data = runner.load_character(source, role_code)
        if not role_data:
            continue
        
        for scenario in runner.scenarios[:3]:
            if len(samples) >= num_samples:
                break
            
            try:
                response, inner_mono = persona_gen.generate_with_dual_process(role_data, scenario)
                samples.append({
                    "role_data": role_data,
                    "scenario": scenario,
                    "response": response,
                    "inner_mono": inner_mono
                })
                print(f"  Sample {len(samples)}/{num_samples}")
            except Exception as e:
                print(f"  Error: {e}")
        
        if len(samples) >= num_samples:
            break
    
    print(f"\nGenerated {len(samples)} samples")
    
    # 用每个评估器评估所有样本
    scores = {name: [] for name in evaluators}
    
    for i, sample in enumerate(samples):
        print(f"\nEvaluating sample {i+1}/{len(samples)}...")
        
        for eval_name, eval_llm in evaluators.items():
            try:
                # 创建临时 runner 使用当前评估器
                temp_runner = ExperimentRunner(llm=eval_llm)
                eval_res = temp_runner.run_single_evaluation(
                    sample["role_data"],
                    sample["scenario"],
                    sample["response"],
                    sample["inner_mono"],
                    method="ours"
                )
                scores[eval_name].append(eval_res.pc_score)
                print(f"  [{eval_name}] PC={eval_res.pc_score:.2f}")
            except Exception as e:
                print(f"  [{eval_name}] Error: {e}")
                scores[eval_name].append(None)
    
    # 计算相关系数
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)
    
    eval_names = list(evaluators.keys())
    correlations = {}
    
    for i, name1 in enumerate(eval_names):
        for name2 in eval_names[i+1:]:
            # 过滤 None 值
            pairs = [(s1, s2) for s1, s2 in zip(scores[name1], scores[name2]) 
                     if s1 is not None and s2 is not None]
            if pairs:
                list1, list2 = zip(*pairs)
                corr = calculate_pearson_correlation(list(list1), list(list2))
                correlations[f"{name1}-{name2}"] = corr
                print(f"{name1} vs {name2}: r = {corr:.3f}")
    
    # 计算平均相关系数
    if correlations:
        avg_corr = sum(correlations.values()) / len(correlations)
        print(f"\nAverage correlation: r = {avg_corr:.3f}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_json_file(f"{output_dir}/multi_evaluator_results_{timestamp}.json", {
        "config": {
            "num_samples": len(samples),
            "evaluators": eval_names
        },
        "scores": {name: [s for s in slist if s is not None] for name, slist in scores.items()},
        "correlations": correlations,
        "average_correlation": avg_corr if correlations else 0,
        "timestamp": timestamp
    })
    
    print(f"\n结果已保存到 {output_dir}/")
    return correlations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Evaluator Validation")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="experiments/experiment_results/multi_evaluator")
    
    args = parser.parse_args()
    run_multi_evaluator_validation(
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
