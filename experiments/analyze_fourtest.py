import os
import json
import statistics
from collections import defaultdict

RESULTS_DIR = "/home/ubuntu/ScrollWeaver/new-experiment/new/fourtest/results/"

def analyze_results():
    stats = defaultdict(lambda: {"turns": [], "response_lengths": [], "files": 0})
    
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            group = data.get("group", "Unknown")
            logs = data.get("logs", [])
            
            stats[group]["files"] += 1
            stats[group]["turns"].append(len(logs))
            
            for turn in logs:
                response = turn.get("response", "")
                if response:
                    stats[group]["response_lengths"].append(len(response))
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"{'Group':<30} | {'Files':<5} | {'Avg Turns':<10} | {'Avg Resp Len':<12}")
    print("-" * 65)
    
    for group, data in sorted(stats.items()):
        avg_turns = statistics.mean(data["turns"]) if data["turns"] else 0
        avg_resp_len = statistics.mean(data["response_lengths"]) if data["response_lengths"] else 0
        
        print(f"{group:<30} | {data['files']:<5} | {avg_turns:<10.2f} | {avg_resp_len:<12.2f}")

if __name__ == "__main__":
    analyze_results()
