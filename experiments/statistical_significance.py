"""
Statistical Significance Testing
=================================

对实验结果进行统计显著性检验，验证论文中 p<0.01 的声称。

实验内容:
- 对现有结果进行 Wilcoxon signed-rank test
- 计算 95% 置信区间
- 生成 p-value 报告

运行方式:
    python experiments/statistical_significance.py
"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Tuple
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import load_json_file, save_json_file


def wilcoxon_signed_rank_test(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test 的简化实现
    
    Returns:
        (statistic, p_value)
    """
    if len(x) != len(y):
        raise ValueError("Sample sizes must be equal")
    
    n = len(x)
    if n < 6:
        return 0, 1.0  # 样本太小，返回不显著
    
    # 计算差值
    differences = [xi - yi for xi, yi in zip(x, y)]
    
    # 移除零差值
    nonzero = [(abs(d), i, d > 0) for i, d in enumerate(differences) if d != 0]
    
    if not nonzero:
        return 0, 1.0
    
    # 按绝对值排序并分配秩
    nonzero.sort(key=lambda x: x[0])
    
    # 计算正负秩和
    w_plus = 0
    w_minus = 0
    
    for rank, (_, _, is_positive) in enumerate(nonzero, 1):
        if is_positive:
            w_plus += rank
        else:
            w_minus += rank
    
    # 测试统计量是较小的秩和
    w = min(w_plus, w_minus)
    n_nonzero = len(nonzero)
    
    # 使用正态近似计算 p-value (n >= 6)
    mean_w = n_nonzero * (n_nonzero + 1) / 4
    std_w = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24)
    
    if std_w == 0:
        return w, 1.0
    
    z = (w - mean_w) / std_w
    
    # 使用标准正态分布的近似 CDF
    # P(Z <= z) ≈ 0.5 * (1 + erf(z / sqrt(2)))
    def norm_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    # 双尾 p-value
    p_value = 2 * norm_cdf(-abs(z))
    
    return w, p_value


def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    计算置信区间
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    if not data:
        return 0, 0, 0
    
    n = len(data)
    mean = sum(data) / n
    
    if n < 2:
        return mean, mean, mean
    
    # 计算标准误差
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_error = math.sqrt(variance / n)
    
    # 使用 z 值 (对于大样本)
    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_values.get(confidence, 1.96)
    
    margin = z * std_error
    
    return mean, mean - margin, mean + margin


def run_statistical_analysis(
    results_dir: str = "experiments/experiment_results",
    output_dir: str = "experiments/experiment_results/statistics"
):
    """
    运行统计显著性分析
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("Statistical Significance Analysis")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 查找主实验结果
    main_results_pattern = f"{results_dir}/main/main_*_*.json"
    result_files = glob.glob(main_results_pattern)
    
    if not result_files:
        print(f"No result files found in {results_dir}/main/")
        return {}
    
    print(f"Found {len(result_files)} result files")
    
    # 按方法分组收集分数
    method_scores = {
        "vanilla": {"pc": [], "sa": [], "dm": []},
        "character_llm": {"pc": [], "sa": [], "dm": []},
        "structured_cot": {"pc": [], "sa": [], "dm": []},
        "rag_persona": {"pc": [], "sa": [], "dm": []},
        "ours_no_dual": {"pc": [], "sa": [], "dm": []},
        "ours": {"pc": [], "sa": [], "dm": []}
    }
    
    for file_path in result_files:
        if "summary" in file_path:
            continue
            
        # 确定方法
        method = None
        for m in method_scores.keys():
            if f"main_{m}_" in file_path:
                method = m
                break
        
        if not method:
            continue
        
        # 加载结果
        try:
            results = load_json_file(file_path)
            for r in results:
                if "pc_score" in r:
                    method_scores[method]["pc"].append(r["pc_score"])
                if "sa_score" in r:
                    method_scores[method]["sa"].append(r["sa_score"])
                if "dm_score" in r:
                    method_scores[method]["dm"].append(r["dm_score"])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # 统计分析
    print("\n" + "=" * 70)
    print("CONFIDENCE INTERVALS (95%)")
    print("=" * 70)
    print(f"{'Method':<20} {'Metric':<8} {'Mean':<8} {'95% CI':<20}")
    print("-" * 56)
    
    ci_results = {}
    for method, scores in method_scores.items():
        ci_results[method] = {}
        for metric, values in scores.items():
            if values:
                mean, lower, upper = calculate_confidence_interval(values)
                ci_results[method][metric] = {
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n": len(values)
                }
                print(f"{method:<20} {metric.upper():<8} {mean:.3f}    [{lower:.3f}, {upper:.3f}]")
    
    # Wilcoxon 检验: Ours vs 每个 baseline
    print("\n" + "=" * 70)
    print("WILCOXON SIGNED-RANK TEST (Ours vs Baselines)")
    print("=" * 70)
    print(f"{'Comparison':<35} {'Metric':<8} {'p-value':<12} {'Significant':<12}")
    print("-" * 67)
    
    significance_results = {}
    baselines = ["vanilla", "character_llm", "structured_cot", "rag_persona", "ours_no_dual"]
    
    for baseline in baselines:
        significance_results[f"ours_vs_{baseline}"] = {}
        
        for metric in ["pc", "sa", "dm"]:
            ours_scores = method_scores["ours"][metric]
            baseline_scores = method_scores[baseline][metric]
            
            # 确保样本大小相等（取最小长度）
            min_len = min(len(ours_scores), len(baseline_scores))
            if min_len < 6:
                print(f"ours vs {baseline:<15} {metric.upper():<8} N/A          (n<6)")
                continue
            
            ours_paired = ours_scores[:min_len]
            baseline_paired = baseline_scores[:min_len]
            
            try:
                w, p_value = wilcoxon_signed_rank_test(ours_paired, baseline_paired)
                is_significant = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
                
                significance_results[f"ours_vs_{baseline}"][metric] = {
                    "w_statistic": w,
                    "p_value": p_value,
                    "significant_01": p_value < 0.01,
                    "significant_05": p_value < 0.05,
                    "n": min_len
                }
                
                print(f"ours vs {baseline:<15} {metric.upper():<8} {p_value:<12.6f} {is_significant}")
            except Exception as e:
                print(f"ours vs {baseline:<15} {metric.upper():<8} Error: {e}")
    
    # 汇总
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # 计算有多少比较在 p<0.01 水平显著
    total_tests = 0
    significant_01 = 0
    significant_05 = 0
    
    for comparison, metrics in significance_results.items():
        for metric, result in metrics.items():
            total_tests += 1
            if result.get("significant_01"):
                significant_01 += 1
            if result.get("significant_05"):
                significant_05 += 1
    
    print(f"Total comparisons: {total_tests}")
    print(f"Significant at p<0.01: {significant_01} ({significant_01/max(total_tests,1)*100:.1f}%)")
    print(f"Significant at p<0.05: {significant_05} ({significant_05/max(total_tests,1)*100:.1f}%)")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_json_file(f"{output_dir}/statistics_results_{timestamp}.json", {
        "confidence_intervals": ci_results,
        "significance_tests": significance_results,
        "summary": {
            "total_tests": total_tests,
            "significant_01": significant_01,
            "significant_05": significant_05
        },
        "timestamp": timestamp
    })
    
    print(f"\n结果已保存到 {output_dir}/")
    return significance_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical Significance Analysis")
    parser.add_argument("--results_dir", type=str, default="experiments/experiment_results")
    parser.add_argument("--output_dir", type=str, default="experiments/experiment_results/statistics")
    
    args = parser.parse_args()
    run_statistical_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
