"""
一键运行所有实验脚本
======================

运行所有 PersonaForge 实验并捕获所有异常，生成汇总报告。

使用方法：
    python experiments/run_all_experiments.py
    python experiments/run_all_experiments.py --skip authentic_long_dialogue
    python experiments/run_all_experiments.py --only run_experiment run_main_experiment
"""

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)


# Define all experiments with their entry functions
EXPERIMENTS = {
    "run_experiment": {
        "module": "experiments.run_experiment",
        "function": "main",
        "description": "主实验：对比 PersonaForge 与基线方法",
        "estimated_time": "10-30 min",
        "requires_api": True
    },
    "run_main_experiment": {
        "module": "experiments.run_main_experiment",
        "function": "run_main_experiment",
        "args": {"num_characters": 10, "all_methods": True},
        "description": "主实验：Table 1 (含 RoleLLM, RAG 等基线)",
        "estimated_time": "30-60 min",
        "requires_api": True
    },
    "run_opensource_experiment": {
        "module": "experiments.run_opensource_experiment",
        "function": "main",
        "description": "开源模型验证：测试开源 LLM 效果",
        "estimated_time": "15-30 min",
        "requires_api": True
    },
    "authentic_long_dialogue": {
        "module": "experiments.authentic_long_dialogue",
        "function": "run_long_dialogue_benchmark",
        "description": "长对话实验：10位角色 x 50轮 (Long-term Stability)",
        "estimated_time": "60-120 min",
        "requires_api": True
    },
    "ablation_psychology": {
        "module": "experiments.ablation_psychology",
        "function": "run_psychology_ablation",
        "description": "消融实验：心理学框架 (Ours vs Generic, w/o Big5/Defense/Style)",
        "estimated_time": "20-40 min",
        "requires_api": True
    },

    "cross_domain_validation": {
        "module": "experiments.cross_domain_validation",
        "function": "run_cross_domain_validation",
        "description": "跨域验证：英语文学角色评估",
        "estimated_time": "15-30 min",
        "requires_api": True
    },
    "cross_partner_validation": {
        "module": "experiments.cross_partner_validation",
        "function": "run_cross_partner_validation",
        "args": {"partner_model": "gemini", "num_turns": 10},
        "description": "跨伙伴验证：不同对话伙伴测试",
        "estimated_time": "10-20 min",
        "requires_api": True
    },
    "cost_analysis": {
        "module": "experiments.cost_analysis",
        "function": "run_cost_analysis",
        "description": "成本分析：Token 使用效率评估",
        "estimated_time": "10-20 min",
        "requires_api": True
    },
    "rolebench_expansion": {
        "module": "experiments.rolebench_expansion",
        "function": "run_rolebench_expansion",
        "description": "泛化性扩展：RoleBench 40+ 角色 (含红楼/三国/冰火/爱丽丝/天才基本法)",
        "estimated_time": "30-50 min",
        "requires_api": True
    },
    "human_evaluation_prep": {
        "module": "experiments.human_evaluation_prep",
        "function": "prepare_human_evaluation",
        "args": {"num_pairs": 50, "num_characters": 5},
        "description": "人工评测准备：生成盲审数据",
        "estimated_time": "15-30 min",
        "requires_api": True
    },
    "rag_reflect_baseline": {
        "module": "experiments.rag_reflect_baseline",
        "function": "run_rag_reflect_baseline",
        "description": "RAG-Reflect 基线对比",
        "estimated_time": "5-15 min",
        "requires_api": True
    },

    "trigger_diagnostics": {
        "module": "experiments.trigger_diagnostics",
        "function": "run_trigger_diagnostics",
        "description": "触发器诊断：测试 is_critical_interaction 的 precision/recall",
        "estimated_time": "1-5 min",
        "requires_api": False
    }
}


class ExperimentRunner:
    """实验运行器：执行所有实验并捕获异常"""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.start_time = None
        self.end_time = None
    
    def run_single_experiment(self, name: str, config: Dict) -> Dict[str, Any]:
        """运行单个实验，捕获所有异常"""
        result = {
            "name": name,
            "description": config.get("description", ""),
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "error": None,
            "traceback": None
        }
        
        print(f"\n{'='*70}")
        print(f"[{name}] {config.get('description', '')}")
        print(f"预估时间: {config.get('estimated_time', 'unknown')}")
        print(f"{'='*70}")
        
        result["start_time"] = datetime.now().isoformat()
        start = time.time()
        
        try:
            # Dynamic import
            module_name = config["module"]
            function_name = config["function"]
            
            print(f"正在导入模块: {module_name}")
            module = __import__(module_name, fromlist=[function_name])
            func = getattr(module, function_name)
            
            # Call with args if provided
            args = config.get("args", {})
            print(f"正在执行函数: {function_name}({args})")
            
            if args:
                func(**args)
            else:
                func()
            
            result["status"] = "success"
            print(f"\n✅ [{name}] 实验完成！")
            
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            result["error"] = "用户中断"
            print(f"\n⚠️ [{name}] 被用户中断")
            raise  # Re-raise to stop all experiments
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"\n❌ [{name}] 实验失败: {e}")
            print(f"详细错误:\n{traceback.format_exc()}")
        
        end = time.time()
        result["end_time"] = datetime.now().isoformat()
        result["duration_seconds"] = round(end - start, 2)
        
        print(f"耗时: {result['duration_seconds']:.2f} 秒")
        
        return result
    
    def run_all(self, skip: List[str] = None, only: List[str] = None) -> Dict[str, Any]:
        """运行所有实验"""
        self.start_time = datetime.now()
        
        # Determine which experiments to run
        experiments_to_run = list(EXPERIMENTS.keys())
        
        if only:
            experiments_to_run = [e for e in only if e in EXPERIMENTS]
        elif skip:
            experiments_to_run = [e for e in experiments_to_run if e not in skip]
        
        print("\n" + "="*70)
        print("PersonaForge 实验批量运行器")
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"计划运行 {len(experiments_to_run)} 个实验")
        print("="*70)
        
        for i, exp_name in enumerate(experiments_to_run):
            print(f"\n[{i+1}/{len(experiments_to_run)}] 准备运行: {exp_name}")
            
            config = EXPERIMENTS[exp_name]
            
            try:
                result = self.run_single_experiment(exp_name, config)
                self.results[exp_name] = result
            except KeyboardInterrupt:
                print("\n\n⚠️ 用户中断，停止后续实验")
                break
        
        self.end_time = datetime.now()
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成汇总报告"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        success_count = sum(1 for r in self.results.values() if r["status"] == "success")
        failed_count = sum(1 for r in self.results.values() if r["status"] == "failed")
        interrupted_count = sum(1 for r in self.results.values() if r["status"] == "interrupted")
        
        report = {
            "summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration_seconds": round(total_duration, 2),
                "total_experiments": len(self.results),
                "success": success_count,
                "failed": failed_count,
                "interrupted": interrupted_count
            },
            "experiments": self.results
        }
        
        # Print summary
        print("\n" + "="*70)
        print("实验汇总报告")
        print("="*70)
        print(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.1f} 分钟)")
        print(f"成功: {success_count} | 失败: {failed_count} | 中断: {interrupted_count}")
        print("-"*70)
        
        for name, result in self.results.items():
            status_icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "failed" else "⚠️"
            print(f"{status_icon} {name}: {result['status']} ({result['duration_seconds']:.2f}s)")
            if result["error"]:
                print(f"   错误: {result['error'][:100]}...")
        
        # Save report
        report_dir = "experiments/experiment_results"
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{report_dir}/batch_run_report_{timestamp}.json"
        
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存到: {report_file}")
        print("="*70)
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="一键运行所有 PersonaForge 实验")
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="跳过指定的实验 (例如: --skip authentic_long_dialogue)"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="只运行指定的实验 (例如: --only run_experiment ablation_psychology)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的实验"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n可用的实验:")
        print("-"*70)
        for name, config in EXPERIMENTS.items():
            print(f"  {name}")
            print(f"    描述: {config['description']}")
            print(f"    预估时间: {config.get('estimated_time', 'unknown')}")
            print()
        return
    
    runner = ExperimentRunner()
    
    try:
        runner.run_all(skip=args.skip, only=args.only)
    except KeyboardInterrupt:
        print("\n\n实验被用户中断。")
        runner.generate_report()


if __name__ == "__main__":
    main()
