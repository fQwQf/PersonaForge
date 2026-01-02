"""
Cross-Partner Evaluation Experiment
=====================================

跨对话伙伴验证实验 - 使用真实LLM作为对话伙伴

对应论文:
- Section 5.4: Cross-Partner tests
- Appendix: Cross-Partner Evaluation Details

测试不同的交互伙伴模型对角色一致性的影响:
- Gemini 2.5 (默认)
- Qwen-Plus
- Kimi (Moonshot)
- DeepSeek

运行方式：
    python experiments/cross_partner_validation.py
    python experiments/cross_partner_validation.py --partner qwen
    python experiments/cross_partner_validation.py --partner kimi
    python experiments/cross_partner_validation.py --partner deepseek
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import ExperimentRunner, EvaluationScenario
from experiments.run_experiment import PersonaForgeGenerator
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


class RealPartnerSimulator:
    """使用真实LLM作为对话伙伴"""
    
    def __init__(self, partner_llm, role_context: str = ""):
        self.partner_llm = partner_llm
        self.role_context = role_context
        self.conversation_history = []
        
    def generate_follow_up(self, context: str, previous_response: str, turn: int) -> str:
        """使用LLM生成后续问题"""
        prompt = f"""你是一个与古典文学角色进行对话的现代人。请根据对方的回复，生成一个自然的后续问题或回应。

背景信息: {context}

对方刚才说: "{previous_response}"

请生成一个简短的回复(1-2句话)，可以是:
- 追问细节
- 表达好奇
- 分享自己的看法
- 引入新话题

要求:
- 保持对话自然流畅
- 适当挑战或质疑对方观点(第{turn}轮)
- 不要太长，控制在50字以内

直接输出回复内容，不要任何前缀说明:"""
        
        try:
            response = self.partner_llm.chat(prompt, temperature=0.8)
            # 清理可能的引号
            response = response.strip().strip('"').strip("'")
            return response[:100]  # 限制长度
        except Exception as e:
            print(f"  Partner error: {e}")
            # 备用静态回复
            fallback = ["那之后呢？", "你为什么这么想？", "能详细说说吗？", "这让你有什么感受？", "然后怎么样了？"]
            return fallback[turn % len(fallback)]


def initialize_partner_models(project_config: Dict) -> Dict:
    """初始化所有可用的伙伴模型"""
    partners = {}
    
    # Gemini (使用项目默认)
    try:
        gemini_model = project_config.get("role_llm_name", "gemini-2.5-flash-lite")
        partners["Gemini"] = Gemini(model=gemini_model, timeout=60)
        print("✓ Gemini partner initialized")
    except Exception as e:
        print(f"✗ Gemini initialization failed: {e}")
    
    # Qwen-Plus
    if project_config.get("DASHSCOPE_API_KEY"):
        try:
            partners["Qwen-Plus"] = OpenAICompatibleLLM(
                model="qwen-plus",
                api_base=project_config.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                api_key=project_config["DASHSCOPE_API_KEY"],
                name="Qwen-Plus"
            )
            print("✓ Qwen-Plus partner initialized")
        except Exception as e:
            print(f"✗ Qwen-Plus initialization failed: {e}")
    
    # Kimi (Moonshot)
    if project_config.get("KIMI_API_KEY"):
        try:
            partners["Kimi"] = OpenAICompatibleLLM(
                model="moonshot-v1-8k",
                api_base=project_config.get("KIMI_API_BASE", "https://api.moonshot.cn/v1"),
                api_key=project_config["KIMI_API_KEY"],
                name="Kimi"
            )
            print("✓ Kimi partner initialized")
        except Exception as e:
            print(f"✗ Kimi initialization failed: {e}")
    
    # DeepSeek
    if project_config.get("DEEPSEEK_API_KEY"):
        try:
            partners["DeepSeek"] = OpenAICompatibleLLM(
                model="deepseek-chat",
                api_base="https://api.deepseek.com/v1",
                api_key=project_config["DEEPSEEK_API_KEY"],
                name="DeepSeek"
            )
            print("✓ DeepSeek partner initialized")
        except Exception as e:
            print(f"✗ DeepSeek initialization failed: {e}")
    
    return partners


def run_cross_partner_validation(
    partner_model: str = "all",
    num_turns: int = 10,
    output_dir: str = "experiments/experiment_results/cross_partner"
):
    """
    运行跨对话伙伴验证实验
    
    Args:
        partner_model: 伙伴模型名称 ("gemini", "qwen", "kimi", "deepseek", "all")
        num_turns: 对话轮数
        output_dir: 输出目录
    """
    # Change to project root so relative paths work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print(f"Cross-Partner Validation Experiment")
    print(f"Partner Model: {partner_model}")
    print(f"Turns per conversation: {num_turns}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. Load configuration
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "DASHSCOPE_API_KEY", 
                "KIMI_API_KEY", "DEEPSEEK_API_KEY"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
    
    # 2. Initialize all partner models
    all_partners = initialize_partner_models(project_config)
    
    if not all_partners:
        print("ERROR: No partner models available")
        return {}
    
    # Select which partners to test
    if partner_model.lower() == "all":
        selected_partners = all_partners
    else:
        partner_map = {
            "gemini": "Gemini",
            "qwen": "Qwen-Plus",
            "kimi": "Kimi",
            "deepseek": "DeepSeek"
        }
        partner_name = partner_map.get(partner_model.lower(), partner_model)
        if partner_name in all_partners:
            selected_partners = {partner_name: all_partners[partner_name]}
        else:
            print(f"ERROR: Partner '{partner_model}' not available")
            print(f"Available: {list(all_partners.keys())}")
            return {}
    
    print(f"\nTesting with partners: {list(selected_partners.keys())}")
    
    # 3. Initialize evaluator and generator (use Gemini for consistency)
    eval_llm = all_partners.get("Gemini") or list(all_partners.values())[0]
    runner = ExperimentRunner(llm=eval_llm)
    
    # Use Gemini as the generator for role-play
    gen_llm = Gemini(model=project_config.get("role_llm_name", "gemini-2.5-flash-lite"), timeout=60)
    persona_gen = PersonaForgeGenerator(llm=gen_llm)
    
    # 4. Select test characters (use characters that exist)
    test_chars = [
        ("A_Dream_in_Red_Mansions", "LinDaiyu-zh"),
        ("A_Dream_in_Red_Mansions", "JiaBaoyu-zh"),
        ("A_Song_of_Ice_and_Fire", "TyrionLannister-zh")
    ]

    
    all_results = {}
    
    for partner_name, partner_llm in selected_partners.items():
        print(f"\n{'=' * 60}")
        print(f"Partner Model: {partner_name}")
        print("=" * 60)
        
        partner_results = []
        
        for source, role_code in test_chars:
            role_data = runner.load_character(source, role_code)
            if not role_data:
                print(f"  Warning: Could not load {role_code}")
                continue
                
            print(f"\n[Character] {role_data.get('role_name', role_code)}")
            
            # Initialize partner simulator with real LLM
            partner_sim = RealPartnerSimulator(
                partner_llm=partner_llm,
                role_context=f"与{role_data.get('role_name', role_code)}对话"
            )
            
            # Build multi-turn conversation
            conversation = []
            context = "日常闲聊场景"
            current_trigger = "你好，最近过得怎么样？"
            
            pc_scores = []
            
            for turn in range(num_turns):
                # Create scenario for this turn
                scenario = EvaluationScenario(
                    scenario_id=f"turn_{turn}",
                    scenario_type="casual" if turn < 5 else "emotional",
                    context=context,
                    trigger=current_trigger,
                    trigger_role="partner",
                    expected_traits={}
                )
                
                try:
                    # Generate response from PersonaForge
                    response, thought = persona_gen.generate_with_dual_process(role_data, scenario)
                    
                    # Evaluate
                    eval_res = runner.run_single_evaluation(
                        role_data, scenario, response, thought, method="ours"
                    )
                    pc_scores.append(eval_res.pc_score)
                    
                    conversation.append({
                        "turn": turn,
                        "partner_trigger": current_trigger,
                        "response": response,
                        "pc_score": eval_res.pc_score
                    })
                    
                    print(f"  Turn {turn}: PC={eval_res.pc_score:.2f} | Partner: {current_trigger[:30]}...")
                    
                    # Generate next trigger from real partner LLM
                    current_trigger = partner_sim.generate_follow_up(context, response, turn)
                    
                except Exception as e:
                    print(f"  Turn {turn}: Error - {e}")
                    break
            
            # Calculate drift
            if len(pc_scores) >= 5:
                early_avg = sum(pc_scores[:5]) / 5
                late_avg = sum(pc_scores[-5:]) / 5
                drift = early_avg - late_avg
            else:
                drift = 0
            
            result = {
                "role_code": role_code,
                "role_name": role_data.get("role_name"),
                "partner_model": partner_name,
                "num_turns": len(conversation),
                "avg_pc": sum(pc_scores) / len(pc_scores) if pc_scores else 0,
                "drift": drift,
                "pc_trajectory": pc_scores,
                "conversation": conversation
            }
            partner_results.append(result)
            
            print(f"  Summary: Avg PC={result['avg_pc']:.2f}, Drift={drift:.2f}")
        
        all_results[partner_name] = partner_results
    
    # 5. Aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"{'Partner':<15} {'Avg PC':<10} {'Avg Drift':<12} {'Chars':<8}")
    print("-" * 50)
    
    summary = {}
    for partner_name, results in all_results.items():
        if results:
            avg_pc = sum(r["avg_pc"] for r in results) / len(results)
            avg_drift = sum(r["drift"] for r in results) / len(results)
            summary[partner_name] = {
                "avg_pc": avg_pc,
                "avg_drift": avg_drift,
                "num_chars": len(results)
            }
            print(f"{partner_name:<15} {avg_pc:.3f}     {avg_drift:+.3f}       {len(results)}")
    
    # 6. Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if partner_model.lower() == "all":
        output_file = f"{output_dir}/cross_partner_all_{timestamp}.json"
    else:
        output_file = f"{output_dir}/cross_partner_{partner_model}_{timestamp}.json"
    
    save_json_file(output_file, {
        "config": {
            "partner_models": list(selected_partners.keys()),
            "num_turns": num_turns,
            "test_characters": [rc for _, rc in test_chars]
        },
        "results": all_results,
        "summary": summary,
        "timestamp": timestamp
    })
    
    print(f"\n结果已保存到 {output_file}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Partner Validation")
    parser.add_argument("--partner", type=str, default="all", 
                        help="Partner model: gemini, qwen, kimi, deepseek, or all")
    parser.add_argument("--turns", type=int, default=10, 
                        help="Number of conversation turns")
    parser.add_argument("--output_dir", type=str, 
                        default="experiments/experiment_results/cross_partner",
                        help="Output directory")
    
    args = parser.parse_args()
    run_cross_partner_validation(
        partner_model=args.partner,
        num_turns=args.turns,
        output_dir=args.output_dir
    )
