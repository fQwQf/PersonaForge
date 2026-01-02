"""
Cross-Domain Validation Experiment
===================================

跨域验证实验：英语文学领域角色评估

对应论文:
- Section 5.4: Cross-Domain and Cross-Partner Validation
- Table: Cross-Domain Evaluation Details (45 English characters)

测试源:
- A Song of Ice and Fire (16 characters)
- Alice's Adventures in Wonderland (29 characters)

运行方式：
    python experiments/cross_domain_validation.py
"""

import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import ExperimentRunner
from experiments.run_experiment import PersonaForgeGenerator, BaselineGenerator
from modules.llm.Gemini import Gemini
from modules.utils import load_json_file, save_json_file


def run_cross_domain_validation():
    """
    运行跨域验证实验
    """
    # Change to project root so relative paths work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("Cross-Domain Validation Experiment")
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
    baseline_gen = BaselineGenerator(llm=llm)
    
    # 3. Select English literature characters
    all_characters = runner.list_characters()
    english_sources = [
        "A_Song_of_Ice_and_Fire",
        "Alice_s_Adventures_in_Wonderland-Through_the_Looking-Glass"
    ]
    english_chars = [(s, r) for s, r in all_characters if s in english_sources]
    
    print(f"\nFound {len(english_chars)} English literature characters:")
    source_counts = {}
    for s, r in english_chars:
        source_counts[s] = source_counts.get(s, 0) + 1
    for s, c in source_counts.items():
        print(f"  - {s}: {c} characters")
    
    # 4. Results storage by domain
    results = {
        "ice_and_fire": {"ours": [], "structured_cot": []},
        "alice": {"ours": [], "structured_cot": []}
    }
    
    # 5. Run experiments
    # Use a subset of scenarios (emotional + conflict for comprehensive testing)
    test_scenarios = [s for s in runner.scenarios if s.scenario_type in ["emotional", "conflict", "casual"]][:4]
    
    for source, role_code in english_chars:
        role_data = runner.load_character(source, role_code)
        if not role_data:
            print(f"  Warning: Could not load {role_code}")
            continue
            
        domain = "ice_and_fire" if "Ice_and_Fire" in source else "alice"
        print(f"\n[{domain.upper()}] {role_data.get('role_name', role_code)}")
        
        for scenario in test_scenarios:
            try:
                # PersonaForge (Full)
                resp_ours, thought = persona_gen.generate_with_dual_process(role_data, scenario)
                eval_ours = runner.run_single_evaluation(role_data, scenario, resp_ours, thought, method="ours")
                results[domain]["ours"].append(eval_ours)
                
                # Structured-CoT baseline
                resp_cot = persona_gen.generate_without_dual_process(role_data, scenario)
                eval_cot = runner.run_single_evaluation(role_data, scenario, resp_cot, None, method="structured_cot")
                results[domain]["structured_cot"].append(eval_cot)
                
                print(f"  [{scenario.scenario_id}] Ours PC={eval_ours.pc_score:.2f}, S-CoT PC={eval_cot.pc_score:.2f}")
                
            except Exception as e:
                print(f"  [{scenario.scenario_id}] Error: {e}")
    
    # 6. Compute per-domain scores
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN RESULTS")
    print("=" * 70)
    
    summary = {}
    for domain, domain_results in results.items():
        summary[domain] = {}
        print(f"\n[{domain.upper()}]")
        
        for method, res_list in domain_results.items():
            if not res_list:
                continue
            scores = runner.compute_aggregate_scores(res_list)
            summary[domain][method] = scores
            
            print(f"  {method}: PC={scores.get('PC (Personality Consistency)', 0):.2f}, "
                  f"SA={scores.get('SA (Style Adherence)', 0):.2f}")
    
    # 7. Save results
    output_dir = "experiments/experiment_results/cross_domain"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Flatten and save
    all_results = []
    for domain, domain_results in results.items():
        for method, res_list in domain_results.items():
            for r in res_list:
                r_dict = asdict(r)
                r_dict["domain"] = domain
                all_results.append(r_dict)
    
    output_file = f"{output_dir}/cross_domain_{timestamp}.json"
    save_json_file(output_file, {
        "summary": summary,
        "config": {
            "sources": english_sources,
            "num_characters": len(english_chars),
            "num_scenarios": len(test_scenarios)
        },
        "results": all_results
    })
    
    print(f"\n结果已保存到 {output_file}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    run_cross_domain_validation()
