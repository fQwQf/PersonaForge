# Rebuttal Documentation

This directory contains all rebuttal experiments, results, and response letters.

## Structure

```
rebuttal/
├── README.md                          # This file
├── REAL_EXPERIMENTS_README.md         # Detailed setup instructions
├── rebuttal_review1.txt               # Response to Reviewer 1
├── rebuttal_review2.txt               # Response to Reviewer 2
├── rebuttal_review3.txt               # Response to Reviewer 3
├── experiments/                       # Rebuttal experiment scripts
│   ├── api_client.py                  # Secure API client
│   ├── real_adversarial_v2.py         # Adversarial testing
│   ├── rebuttal_automated_extraction_real.py
│   ├── rebuttal_ccr_validation_real.py
│   ├── rebuttal_extended_state_real.py
│   ├── rebuttal_latency_benchmark_real.py
│   ├── rebuttal_safety_boundary_real.py
│   └── rebuttal_wuxing_ontology_real.py
└── results/                           # Experiment results
    ├── real_adversarial_results_v2.json
    ├── automated_extraction_results_real.json
    ├── ccr_validation_results_real.json
    ├── extended_state_results_real.json
    ├── latency_benchmark_results_real.json
    ├── safety_boundary_results_real.json
    ├── sft_scaling_comparison.json
    └── wuxing_ontology_results_real.json
```

### Usage

```bash
# Set API key
export GEMINI_API_KEY="your-api-key"

# Run any experiment
cd rebuttal/experiments
python rebuttal_ccr_validation_real.py

# Results saved to ../results/
```

## Repository

`https://anonymous.4open.science/r/fluffy-fishstick-2BB1`
