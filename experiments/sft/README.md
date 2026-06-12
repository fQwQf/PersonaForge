# SFT comparison (Table 4 / 5)

PersonaForge is positioned as an **inference-time enhancer**, not a replacement
for SFT. These scripts reproduce the comparison against LoRA-based SFT on
Qwen2.5-7B (a realistic ~100-sample-per-character cold-start setting).

> **Data:** this directory ships no training data. `generate_sft_data.py`
> regenerates instruction-tuning data with a teacher LLM from character profiles
> you provide (see [`../../schemas/`](../../schemas/)).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Generate data | `generate_sft_data.py` | Teacher-LLM SFT data (alpaca format) per character → `data/` |
| 2. Train | `train_sft.py` / `train_lora.sh` | LoRA (QLoRA 4-bit, rank 16) on Qwen2.5-7B |
| 3. Long-dialogue eval | `long_dialogue_4way.py` | 4 groups (zero-shot / simple / **PersonaForge** / SFT) over 30–50 turns — **Table 5**, and the self-contained Table 2 long-dialogue numbers |
| 4. Metric eval | `evaluate_sft.py` | PC / SA / DM / drift for one model |
| 5. Aggregate | `evaluate_batch.py` | Aggregate across all characters → **Table 4** |

`long_dialogue_4way.py` is fully self-contained (transformers + PEFT for the SFT
model, REST calls for the judge) and is the recommended dependency-free
long-dialogue benchmark.
