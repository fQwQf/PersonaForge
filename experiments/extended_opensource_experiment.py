"""
Extended Open-Source Model Experiment
======================================

扩展的开源模型验证实验，测试 PersonaForge 在多个开源/商用 API 上的表现。

实验内容:
- Qwen-Plus (阿里云 DashScope)
- Kimi (月之暗面 Moonshot)
- DeepSeek-Chat
- 验证架构贡献是模型无关的

运行方式:
    python experiments/extended_opensource_experiment.py
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


class OpenAICompatibleLLM:
    """OpenAI 兼容 API 的 LLM 包装器，完整实现"""
    
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
        self.system_prompt = content
        
    def user_message(self, content):
        self.messages.append({"role": "user", "content": content})
        
    def ai_message(self, content):
        self.messages.append({"role": "assistant", "content": content})
        
    def get_response(self, temperature=0.7):
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
    
    def chat(self, text, temperature=0.7):
        self.initialize_message()
        self.user_message(text)
        return self.get_response(temperature)


def run_extended_opensource_experiment(
    num_characters: int = 8,
    num_scenarios: int = 4,
    output_dir: str = "experiments/experiment_results/extended_opensource"
):
    """
    运行扩展开源模型验证实验
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("Extended Open-Source Model Experiment")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load config
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
    
    # 初始化模型
    models = {}
    
    # Qwen-Plus
    if project_config.get("DASHSCOPE_API_KEY"):
        models["Qwen-Plus"] = OpenAICompatibleLLM(
            model="qwen-plus",
            api_base=project_config["DASHSCOPE_API_BASE"],
            api_key=project_config["DASHSCOPE_API_KEY"],
            name="Qwen-Plus"
        )
        print("✓ Qwen-Plus initialized")
    
    # Kimi
    if project_config.get("KIMI_API_KEY"):
        models["Kimi"] = OpenAICompatibleLLM(
            model="moonshot-v1-8k",
            api_base=project_config["KIMI_API_BASE"],
            api_key=project_config["KIMI_API_KEY"],
            name="Kimi"
        )
        print("✓ Kimi initialized")
    
    # DeepSeek
    if project_config.get("DEEPSEEK_API_KEY"):
        models["DeepSeek"] = OpenAICompatibleLLM(
            model="deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key=project_config["DEEPSEEK_API_KEY"],
            name="DeepSeek"
        )
        print("✓ DeepSeek initialized")
    
    if not models:
        print("ERROR: No open-source models available")
        return {}
    
    # 使用 Gemini 作为评估器 (保持评估一致性)
    try:
        eval_llm = Gemini(model="gemini-2.5-flash-lite", timeout=60)
        print("✓ Gemini evaluator initialized")
    except:
        eval_llm = list(models.values())[0]
        print(f"Using {eval_llm.name} as evaluator")
    
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
    results = {model_name: {"personaforge": [], "structured_cot": []} for model_name in models}
    
    for model_name, model_llm in models.items():
        print(f"\n{'=' * 50}")
        print(f"Testing Model: {model_name}")
        print("=" * 50)
        
        persona_gen = PersonaForgeGenerator(llm=model_llm)
        baseline_gen = BaselineGenerator(llm=model_llm)
        
        for source, role_code in selected_chars:
            role_data = runner.load_character(source, role_code)
            if not role_data:
                continue
            
            print(f"\n[Character] {role_data.get('role_name', role_code)}")
            
            for scenario in runner.scenarios[:num_scenarios]:
                # PersonaForge
                try:
                    response, inner_mono = persona_gen.generate_with_dual_process(role_data, scenario)
                    eval_res = runner.run_single_evaluation(
                        role_data, scenario, response, inner_mono, method="personaforge"
                    )
                    results[model_name]["personaforge"].append({
                        "pc": eval_res.pc_score,
                        "sa": eval_res.sa_score,
                        "dm": eval_res.dm_score
                    })
                    print(f"  [PersonaForge] PC={eval_res.pc_score:.2f}")
                except Exception as e:
                    print(f"  [PersonaForge] Error: {e}")
                
                # Structured-CoT baseline
                try:
                    response = baseline_gen.generate_structured_cot(role_data, scenario)
                    eval_res = runner.run_single_evaluation(
                        role_data, scenario, response, None, method="structured_cot"
                    )
                    results[model_name]["structured_cot"].append({
                        "pc": eval_res.pc_score,
                        "sa": eval_res.sa_score,
                        "dm": eval_res.dm_score
                    })
                    print(f"  [S-CoT] PC={eval_res.pc_score:.2f}")
                except Exception as e:
                    print(f"  [S-CoT] Error: {e}")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"{'Model':<15} {'Method':<15} {'PC':<8} {'SA':<8} {'DM':<8} {'n':<5}")
    print("-" * 59)
    
    summary = {}
    for model_name in models:
        summary[model_name] = {}
        
        for method in ["personaforge", "structured_cot"]:
            res_list = results[model_name][method]
            if res_list:
                avg_pc = sum(r["pc"] for r in res_list) / len(res_list)
                avg_sa = sum(r["sa"] for r in res_list) / len(res_list)
                avg_dm = sum(r["dm"] for r in res_list) / len(res_list)
                
                summary[model_name][method] = {
                    "PC": avg_pc,
                    "SA": avg_sa,
                    "DM": avg_dm,
                    "n": len(res_list)
                }
                print(f"{model_name:<15} {method:<15} {avg_pc:.2f}    {avg_sa:.2f}    {avg_dm:.2f}    {len(res_list)}")
    
    # 计算 PersonaForge 相对于 S-CoT 的提升
    print("\n" + "-" * 59)
    print("PersonaForge vs S-CoT Improvement:")
    for model_name in models:
        if "personaforge" in summary[model_name] and "structured_cot" in summary[model_name]:
            pf = summary[model_name]["personaforge"]["PC"]
            scot = summary[model_name]["structured_cot"]["PC"]
            improvement = (pf - scot) / scot * 100 if scot > 0 else 0
            print(f"  {model_name}: +{improvement:.1f}% PC")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_json_file(f"{output_dir}/extended_opensource_results_{timestamp}.json", {
        "config": {
            "num_characters": len(selected_chars),
            "num_scenarios": num_scenarios,
            "models": list(models.keys())
        },
        "results": results,
        "summary": summary,
        "timestamp": timestamp
    })
    
    print(f"\n结果已保存到 {output_dir}/")
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extended Open-Source Model Experiment")
    parser.add_argument("--num_characters", type=int, default=8)
    parser.add_argument("--num_scenarios", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="experiments/experiment_results/extended_opensource")
    
    args = parser.parse_args()
    run_extended_opensource_experiment(
        num_characters=args.num_characters,
        num_scenarios=args.num_scenarios,
        output_dir=args.output_dir
    )
