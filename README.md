# 🎭 PersonaForge

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-Github-b31b1b.svg)](https://github.com/fQwQf/PersonaForge/blob/main/paper/personaforge.pdf)
[![Conference](https://img.shields.io/badge/ACL_2026-Findings-success.svg)](https://2026.aclweb.org/)

**Psychology-Grounded Dual-Process Architecture for Personality-Consistent Role-Playing Agents**

</div>

> 🎉 PersonaForge has been accepted to **ACL 2026 Findings**! 

## Overview

Large Language Models excel at role-playing but struggle to maintain consistent personalities across extended multi-turn interactions. **PersonaForge** is a framework that combines a three-layer personality architecture grounded in psychological theory with a dual-process generation mechanism inspired by cognitive science to enable personality-consistent, long-term role-playing.

### Key Contributions

* **Three-Layer Personality Architecture:** A functionally decomposed structure integrating **Core Traits** (Big Five / Wu Xing), **Speaking Style**, and **Dynamic State** (mood, energy, relationships).
* **Selective Dual-Process Generation:** A "Think-then-Speak" Inner Monologue mechanism that activates only for critical interactions (~40% of turns), achieving 96% of full dual-process performance with only 13.4% token overhead.
* **Long-Dialogue Robustness:** Proven to significantly reduce drift over 50-turn conversations (6.3% drift vs. 31.7% baseline), heavily outperforming standard LoRA SFT approaches in long-context scenarios.
* **Real-Time Viable:** Features an Asynchronous State Update mechanism, reducing perceived latency to standard single-pass LLM speeds (~0.94s) without sacrificing consistency.
* **Culturally Adaptable:** Ontology-agnostic architecture compatible with diverse psychological frameworks.

---

## Repository Structure

```
PersonaForge/
├── modules/                      # Core implementation
│   ├── dual_process_agent.py     # Inner Monologue + Styled Response
│   ├── dynamic_state_manager.py  # Dynamic State Layer
│   ├── personality_model.py      # Big Five + Defense Mechanisms
│   ├── main_performer.py         # Full Performer agent
│   ├── orchestrator.py           # World-level orchestration
│   └── llm/                      # LLM interface wrappers
├── experiments/                  # Experiment scripts
│   ├── evaluation_framework.py   # Core evaluation metrics
│   ├── run_experiment.py         # Main experiment runner
│   ├── run_main_experiment.py    # Scenario-based experiments (Table 1)
│   ├── authentic_long_dialogue.py # Long dialogue benchmark (Table 2)
│   ├── ablation_psychology.py    # Ablation studies (Table 3)
│   ├── trigger_diagnostics.py    # Trigger precision/recall
│   └── configs/                  # Experiment configurations
├── data/                         # Character profiles & world settings
│   └── roles/                    # 88 character profiles (JSON)
├── rebuttal/                     # For rebuttal
└── requirements.txt              # Python dependencies
```

---

## Installation

### Requirements
- Python 3.8+
- 8GB+ RAM recommended
- API access to one of: Gemini, DeepSeek, OpenAI, or local models via Ollama

### Setup

```bash
# Clone repository
git clone https://github.com/fQwQf/PersonaForge
cd PersonaForge

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp config.json.example config.json
# Edit config.json with your API keys
```

---

## Quick Start

### Running Main Experiments

```bash
# Run scenario-based experiment (Table 1)
python experiments/run_main_experiment.py

# Run long-dialogue benchmark (Table 2)
python experiments/authentic_long_dialogue.py

# Run ablation studies (Table 3)
python experiments/ablation_psychology.py
```

### Key Configuration Options

Edit `experiments/config.json`:

```json
{
  "role_llm_name": "gemini-2.5-flash",    // Generator model
  "world_llm_name": "gemini-2.5-flash",   // Orchestrator model
  "embedding_model_name": "bge-small",     // Embedding model
  "characters_to_test": ["lin_daiyu", "tyrion_lannister", ...]
}
```

---

## Reproducing Paper Results

### Table 1: Main Results (Scenario-Based)

```bash
python experiments/run_main_experiment.py --output results/main.json
```

Reproduces: PC, SA, DM, RD metrics across all baselines.

### Table 2: Long-Dialogue Results

```bash
python experiments/authentic_long_dialogue.py \
  --turns 50 \
  --characters 10 \
  --output results/long_dialogue.json
```

Reproduces: Drift rate, Average PC, Recovery rate over 50 turns.

### Table 3: Ablation Study

```bash
python experiments/ablation_psychology.py \
  --ablations "no_dual,no_bigfive,no_dm,no_style,no_state,generic" \
  --output results/ablation.json
```

Reproduces: Per-component contribution analysis.

### Table 4: SFT Comparison

```bash
python experiments/run_experiment.py \
  --mode fourtest \
  --output results/sft_comparison.json
```

Reproduces: Comparison with LoRA-based SFT on Qwen2.5-7B.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **PC** (Personality Consistency) | Pairwise LLM-as-Judge evaluation of trait alignment |
| **SA** (Style Adherence) | Composite of sentence length, catchphrase, tone, vocabulary |
| **DM** (Defense Mechanism) | Activation precision and manifestation accuracy |
| **RD** (Response Diversity) | 1 - Self-BLEU to measure lexical diversity |
| **Drift Rate** | % of turns where PC drops below threshold |
| **Recovery Rate** | % recovering to high PC within N turns after perturbation |

---

## Character Profiles

Each character profile in `data/roles/` follows this schema:

```json
{
  "name": "Lin Daiyu",
  "big_five": {
    "openness": 0.9,
    "conscientiousness": 0.4,
    "extraversion": 0.3,
    "agreeableness": 0.5,
    "neuroticism": 0.9
  },
  "defense_mechanism": "sublimation",
  "speaking_style": {
    "sentence_length": "medium",
    "vocabulary": "academic",
    "punctuation": "excessive",
    "catchphrases": ["罢了", "你只管..."],
    "tone_markers": ["呢", "罢"]
  },
  "background": "...",
  "interests": ["poetry", "flowers"]
}
```

---

## Open-Source Pipeline

PersonaForge runs fully on open-source models with minimal degradation:

| Component | Closed-Source | Open-Source | Δ |
|-----------|---------------|-------------|---|
| Generator | Gemini 2.5 | DeepSeek-V3 | -0.02 PC |
| State Extraction | Gemini 2.5 | DeepSeek-V3 | +4.2% drift |
| Judge | Gemini 2.5 | DeepSeek-V3 | r=0.84 |

To use open-source models:

```bash
python experiments/run_opensource_experiment.py
```

---

## Baselines Implemented

- **Zero-Shot**: Role name only
- **Character-LLM-style**: Profile + few-shot exemplars
- **Structured-CoT**: Chain-of-thought with natural language descriptions
- **RAG-Persona**: Retrieval-augmented memory proxy
- **RoleLLM-style**: Role-profile-guiding with retrieval
- **Periodic Re-grounding**: Re-inject prompt every 5 turns
- **Memory+Reflection**: Simplified generative agent with 10-turn reflections

---

## Citation

If you use PersonaForge in your research, please cite:

```bibtex
@inproceedings{tong2026personaforge,
  title={PersonaForge: Psychology-Grounded Dual-Process Architecture for Personality-Consistent Role-Playing Agents},
  author={Tong, Jizhou and Zou, Sirui},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This codebase is built upon the [**BookWorld**](https://github.com/alienet1109/BookWorld) multi-agent simulation framework (Ran et al., 2025). We thank the authors for releasing their code, which served as the foundation for our agent interaction loop.
