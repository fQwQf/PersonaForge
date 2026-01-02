#!/usr/bin/env python3
"""
Batch Evaluation Script for Scaled SFT Comparison (15 Characters)
==================================================================

This script evaluates all trained SFT models and generates:
1. Per-character results
2. Aggregate comparison table
3. Statistical analysis for paper
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Character configurations
CHARACTERS = {
    # 红楼梦 (A Dream in Red Mansions)
    "LinDaiyu": {"name": "林黛玉", "lang": "zh", "source": "红楼梦"},
    "WangXifeng": {"name": "王熙凤", "lang": "zh", "source": "红楼梦"},
    "JiaBaoyu": {"name": "贾宝玉", "lang": "zh", "source": "红楼梦"},
    "XueBaochai": {"name": "薛宝钗", "lang": "zh", "source": "红楼梦"},
    
    # 三国演义 (Romance of Three Kingdoms)
    "ZhugeLiang": {"name": "诸葛亮", "lang": "zh", "source": "三国演义"},
    "CaoCao": {"name": "曹操", "lang": "zh", "source": "三国演义"},
    "GuanYu": {"name": "关羽", "lang": "zh", "source": "三国演义"},
    "ZhouYu": {"name": "周瑜", "lang": "zh", "source": "三国演义"},
    
    # A Song of Ice and Fire
    "TyrionLannister": {"name": "Tyrion Lannister", "lang": "en", "source": "Ice and Fire"},
    "DaenerysTargaryen": {"name": "Daenerys Targaryen", "lang": "en", "source": "Ice and Fire"},
    "JonSnow": {"name": "Jon Snow", "lang": "en", "source": "Ice and Fire"},
    "CerseiLannister": {"name": "Cersei Lannister", "lang": "en", "source": "Ice and Fire"},
    "AryaStark": {"name": "Arya Stark", "lang": "en", "source": "Ice and Fire"},
    "SansaStark": {"name": "Sansa Stark", "lang": "en", "source": "Ice and Fire"},
    "JaimeLannister": {"name": "Jaime Lannister", "lang": "en", "source": "Ice and Fire"},
}


def load_existing_results(results_dir: str) -> Dict[str, Dict]:
    """Load all existing evaluation results"""
    results = {}
    for filename in os.listdir(results_dir):
        if filename.startswith("eval_") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract character name from filename
                parts = filename.replace("eval_", "").replace(".json", "").split("_")
                char_key = parts[0]
                if char_key in CHARACTERS:
                    results[char_key] = data
    return results


def run_missing_evaluations(results_dir: str, num_turns: int = 50):
    """Run evaluations for characters that don't have results yet"""
    from experiments.sft.evaluate_sft_full import run_comprehensive_evaluation, CHARACTER_PROFILES
    
    existing = load_existing_results(results_dir)
    llama_factory_saves = os.path.join(PROJECT_ROOT, "LLaMA-Factory", "saves")
    
    for char_key, char_info in CHARACTERS.items():
        if char_key in existing:
            print(f"[SKIP] {char_key}: Results exist")
            continue
        
        # Check if adapter exists
        adapter_path = os.path.join(llama_factory_saves, f"qwen_{char_key}_sft")
        if not os.path.exists(adapter_path):
            print(f"[SKIP] {char_key}: No adapter found at {adapter_path}")
            continue
        
        # Check if character profile exists in evaluate_sft_full.py
        if char_key not in CHARACTER_PROFILES:
            print(f"[SKIP] {char_key}: No profile in CHARACTER_PROFILES")
            continue
        
        print(f"[EVALUATING] {char_key}...")
        try:
            run_comprehensive_evaluation(
                character=char_key,
                method="both",
                adapter_path=adapter_path,
                num_turns=num_turns,
                output_dir=results_dir
            )
        except Exception as e:
            print(f"[ERROR] {char_key}: {e}")


def generate_aggregate_report(results_dir: str):
    """Generate aggregate statistics for paper"""
    results = load_existing_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print("\n" + "=" * 80)
    print("AGGREGATE SFT vs PersonaForge COMPARISON")
    print("=" * 80)
    
    # Collect all metrics
    sft_metrics = {"pc": [], "sa": [], "dm": [], "drift": []}
    pf_metrics = {"pc": [], "sa": [], "dm": [], "drift": []}
    
    # Per-source breakdown
    by_source = {}
    
    print(f"\n{'Character':<25} {'Source':<15} {'SFT PC':<10} {'PF PC':<10} {'SFT Drift':<12} {'PF Drift':<10}")
    print("-" * 80)
    
    for char_key, data in results.items():
        if "sft" not in data or "personaforge" not in data:
            continue
        
        sft = data["sft"]
        pf = data["personaforge"]
        char_info = CHARACTERS.get(char_key, {"name": char_key, "source": "Unknown"})
        source = char_info["source"]
        
        # Collect metrics
        sft_metrics["pc"].append(sft["avg_pc"])
        sft_metrics["sa"].append(sft["avg_sa"])
        sft_metrics["dm"].append(sft["avg_dm"])
        sft_metrics["drift"].append(sft["drift_rate"])
        
        pf_metrics["pc"].append(pf["avg_pc"])
        pf_metrics["sa"].append(pf["avg_sa"])
        pf_metrics["dm"].append(pf["avg_dm"])
        pf_metrics["drift"].append(pf["drift_rate"])
        
        # By source
        if source not in by_source:
            by_source[source] = {"sft": [], "pf": []}
        by_source[source]["sft"].append(sft)
        by_source[source]["pf"].append(pf)
        
        print(f"{char_info['name']:<25} {source:<15} {sft['avg_pc']:.3f}      {pf['avg_pc']:.3f}      {sft['drift_rate']*100:.1f}%        {pf['drift_rate']*100:.1f}%")
    
    n = len(sft_metrics["pc"])
    if n == 0:
        print("No valid results to aggregate!")
        return
    
    # Calculate averages
    avg_sft_pc = sum(sft_metrics["pc"]) / n
    avg_sft_sa = sum(sft_metrics["sa"]) / n
    avg_sft_dm = sum(sft_metrics["dm"]) / n
    avg_sft_drift = sum(sft_metrics["drift"]) / n
    
    avg_pf_pc = sum(pf_metrics["pc"]) / n
    avg_pf_sa = sum(pf_metrics["sa"]) / n
    avg_pf_dm = sum(pf_metrics["dm"]) / n
    avg_pf_drift = sum(pf_metrics["drift"]) / n
    
    print("\n" + "=" * 80)
    print(f"AGGREGATE RESULTS (N={n} characters)")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'SFT-LoRA':<15} {'PersonaForge':<15} {'Delta':<10} {'Winner':<15}")
    print("-" * 80)
    
    # PC
    delta_pc = avg_pf_pc - avg_sft_pc
    winner_pc = "PersonaForge" if delta_pc > 0 else "SFT"
    print(f"{'PC (↑)':<25} {avg_sft_pc:.3f}          {avg_pf_pc:.3f}          {'+' if delta_pc > 0 else ''}{delta_pc:.3f}      {winner_pc}")
    
    # SA
    delta_sa = avg_pf_sa - avg_sft_sa
    winner_sa = "PersonaForge" if delta_sa > 0 else "SFT"
    print(f"{'SA (↑)':<25} {avg_sft_sa:.3f}          {avg_pf_sa:.3f}          {'+' if delta_sa > 0 else ''}{delta_sa:.3f}      {winner_sa}")
    
    # DM
    delta_dm = avg_pf_dm - avg_sft_dm
    winner_dm = "PersonaForge" if delta_dm > 0 else "SFT"
    print(f"{'DM (↑)':<25} {avg_sft_dm:.3f}          {avg_pf_dm:.3f}          {'+' if delta_dm > 0 else ''}{delta_dm:.3f}      {winner_dm}")
    
    # Drift (lower is better)
    delta_drift = avg_sft_drift - avg_pf_drift
    winner_drift = "PersonaForge" if delta_drift > 0 else "SFT"
    print(f"{'Drift (↓)':<25} {avg_sft_drift*100:.1f}%          {avg_pf_drift*100:.1f}%          {'+' if delta_drift > 0 else ''}{delta_drift*100:.1f}%      {winner_drift}")
    
    # By source breakdown
    print("\n" + "=" * 80)
    print("BREAKDOWN BY SOURCE")
    print("=" * 80)
    
    for source, data in by_source.items():
        sft_list = data["sft"]
        pf_list = data["pf"]
        n_src = len(sft_list)
        
        avg_sft_pc_src = sum(s["avg_pc"] for s in sft_list) / n_src
        avg_pf_pc_src = sum(p["avg_pc"] for p in pf_list) / n_src
        avg_sft_drift_src = sum(s["drift_rate"] for s in sft_list) / n_src
        avg_pf_drift_src = sum(p["drift_rate"] for p in pf_list) / n_src
        
        print(f"\n{source} (N={n_src}):")
        print(f"  PC:    SFT={avg_sft_pc_src:.3f}, PF={avg_pf_pc_src:.3f}, Δ={avg_pf_pc_src - avg_sft_pc_src:+.3f}")
        print(f"  Drift: SFT={avg_sft_drift_src*100:.1f}%, PF={avg_pf_drift_src*100:.1f}%, Δ={avg_sft_drift_src*100 - avg_pf_drift_src*100:+.1f}%")
    
    # Generate LaTeX table for paper
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    print("""
\\begin{table}[t]
\\centering
\\small
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Method} & \\textbf{PC $\\uparrow$} & \\textbf{SA $\\uparrow$} & \\textbf{DM $\\uparrow$} & \\textbf{Drift $\\downarrow$} \\\\
\\midrule""")
    print(f"SFT-LoRA & {avg_sft_pc:.3f} & {avg_sft_sa:.3f} & {avg_sft_dm:.3f} & {avg_sft_drift*100:.1f}\\% \\\\")
    print(f"PersonaForge (Ours) & \\textbf{{{avg_pf_pc:.3f}}} & \\textbf{{{avg_pf_sa:.3f}}} & {avg_pf_dm:.3f} & \\textbf{{{avg_pf_drift*100:.1f}\\%}} \\\\")
    print(f"\\midrule")
    print(f"$\\Delta$ & {'+' if delta_pc > 0 else ''}{delta_pc*100:.1f}\\% & {'+' if delta_sa > 0 else ''}{delta_sa*100:.1f}\\% & {'+' if delta_dm > 0 else ''}{delta_dm*100:.1f}\\% & {'+' if delta_drift > 0 else ''}{delta_drift*100:.1f}\\% \\\\")
    print("""\\bottomrule
\\end{tabular}""")
    print(f"\\caption{{SFT vs. PersonaForge comparison on 50-turn dialogues ({n} characters across 3 domains). Both methods use identical Qwen2.5-7B backbone.}}")
    print("\\label{tab:sft_comparison}")
    print("\\end{table}")
    
    # Save aggregate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_characters": n,
        "aggregate": {
            "sft": {"pc": avg_sft_pc, "sa": avg_sft_sa, "dm": avg_sft_dm, "drift": avg_sft_drift},
            "personaforge": {"pc": avg_pf_pc, "sa": avg_pf_sa, "dm": avg_pf_dm, "drift": avg_pf_drift}
        },
        "by_source": {src: {"n": len(d["sft"])} for src, d in by_source.items()},
        "per_character": {k: {"sft": v.get("sft", {}), "pf": v.get("personaforge", {})} 
                         for k, v in results.items()}
    }
    
    report_file = os.path.join(results_dir, f"aggregate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch SFT Evaluation")
    parser.add_argument("--run_missing", action="store_true",
                        help="Run evaluations for characters without results")
    parser.add_argument("--num_turns", type=int, default=50,
                        help="Number of dialogue turns")
    parser.add_argument("--results_dir", type=str, 
                        default=os.path.join(PROJECT_ROOT, "experiments", "sft", "results"),
                        help="Results directory")
    
    args = parser.parse_args()
    
    if args.run_missing:
        run_missing_evaluations(args.results_dir, args.num_turns)
    
    generate_aggregate_report(args.results_dir)


if __name__ == "__main__":
    main()
