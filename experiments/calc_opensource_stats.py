import json

def calc_stats(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    summary = {}
    
    for model, protocols in results.items():
        summary[model] = {}
        for protocol, samples in protocols.items():
            pcs = [s['pc'] for s in samples if 'pc' in s]
            if pcs:
                avg_pc = sum(pcs) / len(pcs)
                summary[model][protocol] = avg_pc
    
    return summary

stats = calc_stats('experiments/experiment_results/extended_opensource/extended_opensource_results_20251229_194723.json')
print(json.dumps(stats, indent=1))
