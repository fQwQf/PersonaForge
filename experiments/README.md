# Experiments

All experiments are reproducible with **only an LLM API key** plus character
profiles you supply yourself (this repo ships no copyrighted character data —
see [`../schemas/`](../schemas/)).

## Setup

```bash
cp ../config.json.example ../config.json   # then add your API key + model names
pip install -r ../requirements.txt
```

Each script adds the project root to `sys.path`, so run them directly:

```bash
python experiments/main_scenario.py
python experiments/validations/cross_domain.py
```

## Layout

```
experiments/
├── common/            # shared library used by every experiment
│   ├── evaluation.py     # PC / SA / DM / RD evaluators + ExperimentRunner
│   ├── harness.py        # BaselineGenerator (7 baselines) + PersonaForgeGenerator
│   ├── judge.py          # pairwise LLM-as-judge (position-bias controlled)
│   ├── rag_baseline.py   # retrieval + reflection baseline
│   ├── stats.py          # Wilcoxon signed-rank + 95% CIs
│   └── api_client.py     # rate-limited API client used by validations
├── main_scenario.py   # Table 1 entrypoint
├── ablation.py        # Table 3 entrypoint
├── trigger_diagnostics.py
├── cost_analysis.py
├── sft/               # Table 4 / 5: self-contained SFT pipeline (LoRA on Qwen2.5-7B)
├── validations/       # appendix robustness / generalization studies
└── _bookworld_runtime/  # variants that need the unshipped BookWorld runtime (reference only)
```

## Paper artifact → command

| Paper result | Command |
|---|---|
| **Table 1** — main scenario (PC/SA/DM/RD across 7 baselines + PersonaForge) | `python experiments/main_scenario.py` |
| **Table 2** — 50-turn long-dialogue drift / recovery | `python experiments/sft/long_dialogue_4way.py` (Group C = PersonaForge) |
| **Table 3** — psychology-grounding ablation | `python experiments/ablation.py` |
| **Table 4 / 5** — SFT vs PersonaForge | see [`sft/README.md`](sft/README.md) |
| Selective-activation trigger F1 | `python experiments/trigger_diagnostics.py` |
| Selective-activation cost (PC / token) | `python experiments/cost_analysis.py` |
| Cross-domain generalization | `python experiments/validations/cross_domain.py` |
| Cross-partner robustness | `python experiments/validations/cross_partner.py` |
| Cross-generator (model-agnostic) | `python experiments/validations/cross_generator.py` |
| Open-source pipeline (DeepSeek/Qwen/Llama) | `python experiments/validations/open_source.py` |
| Wu Xing (ontology-agnostic) validation | `python experiments/validations/wuxing_ontology.py` |
| PC-threshold sensitivity | `python experiments/validations/threshold_sensitivity.py` |
| Multi-evaluator agreement | `python experiments/validations/multi_evaluator.py` |
| Adversarial robustness | `python experiments/validations/adversarial_robustness.py` |
| Failure-mode mitigation | `python experiments/validations/failure_mitigation.py` |
| Latency benchmark (async state) | `python experiments/validations/latency_benchmark.py` |
| Safety boundary | `python experiments/validations/safety_boundary.py` |
| Automated parameter extraction | `python experiments/validations/automated_extraction.py` |
| Data-size ablation (SFT) | `python experiments/validations/datasize_ablation.py` |
| RoleBench expansion | `python experiments/validations/rolebench_expansion.py` |

Reference aggregate numbers from our runs live in [`../reference_metrics/`](../reference_metrics/).

> The validation scripts that talk to multiple providers read API keys from
> `validations/opensource_config.json` (copy from the bundled template and fill in
> your own keys) or fall back to the top-level `config.json`.
