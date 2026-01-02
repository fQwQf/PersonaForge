#!/bin/bash
# =============================================================================
# Batch SFT Training and Evaluation for 15 Characters
# =============================================================================
# This script automates the entire SFT comparison pipeline:
# 1. Generate SFT training data for each character
# 2. Train LoRA adapters using LLaMA-Factory
# 3. Run comprehensive evaluation (PC, SA, DM, Drift)
# =============================================================================

set -e

# Configuration
PROJECT_ROOT="/home/ubuntu/ScrollWeaver"
LLAMA_FACTORY_DIR="${PROJECT_ROOT}/LLaMA-Factory"
SFT_DIR="${PROJECT_ROOT}/experiments/sft"
DATA_DIR="${SFT_DIR}/data"
RESULTS_DIR="${SFT_DIR}/results"

# 15 Characters for scaled comparison (8 Chinese, 7 English-context)
# Format: "CharacterKey|RolePath|Language|DisplayName"
CHARACTERS=(
    # 红楼梦 (A Dream in Red Mansions) - 4 characters
    "LinDaiyu|A_Dream_in_Red_Mansions/LinDaiyu-zh|zh|林黛玉"
    "WangXifeng|A_Dream_in_Red_Mansions/WangXifeng-zh|zh|王熙凤"
    "JiaBaoyu|A_Dream_in_Red_Mansions/JiaBaoyu-zh|zh|贾宝玉"
    "XueBaochai|A_Dream_in_Red_Mansions/XueBaochai-zh|zh|薛宝钗"
    
    # 三国演义 (Romance of Three Kingdoms) - 4 characters
    "ZhugeLiang|Romance_of_the_Three_Kingdoms/zhugeliang-zh|zh|诸葛亮"
    "CaoCao|Romance_of_the_Three_Kingdoms/caocao-zh|zh|曹操"
    "GuanYu|Romance_of_the_Three_Kingdoms/guanyu-zh|zh|关羽"
    "ZhouYu|Romance_of_the_Three_Kingdoms/zhouyu-zh|zh|周瑜"
    
    # 冰与火之歌 (A Song of Ice and Fire) - 7 characters
    "TyrionLannister|A_Song_of_Ice_and_Fire/TyrionLannister-zh|en|Tyrion Lannister"
    "DaenerysTargaryen|A_Song_of_Ice_and_Fire/DaenerysTargaryen-zh|en|Daenerys Targaryen"
    "JonSnow|A_Song_of_Ice_and_Fire/JonSnow-zh|en|Jon Snow"
    "CerseiLannister|A_Song_of_Ice_and_Fire/CerseiLannister-zh|en|Cersei Lannister"
    "AryaStark|A_Song_of_Ice_and_Fire/AryaStark-zh|en|Arya Stark"
    "SansaStark|A_Song_of_Ice_and_Fire/SansaStark-zh|en|Sansa Stark"
    "JaimeLannister|A_Song_of_Ice_and_Fire/JaimeLannister-zh|en|Jaime Lannister"
)

# Ensure directories exist
mkdir -p "${DATA_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LLAMA_FACTORY_DIR}/data"

echo "=============================================="
echo "Scaled SFT Comparison: 15 Characters"
echo "=============================================="
echo "Characters:"
for char_info in "${CHARACTERS[@]}"; do
    IFS='|' read -r key path lang name <<< "$char_info"
    echo "  - ${name} (${key}) [${lang}]"
done
echo "=============================================="

# Step 1: Generate SFT training data for all characters
echo ""
echo "[STEP 1/3] Generating SFT training data..."
echo "=============================================="

# Generate data for all 15 characters at once
python3 "${SFT_DIR}/generate_sft_data.py" \
    --characters "all" \
    --num_samples 100 \
    --output_dir "${DATA_DIR}" 2>&1 | head -100

# Step 2: Register datasets in LLaMA-Factory
echo ""
echo "[STEP 2/3] Registering datasets in LLaMA-Factory..."
echo "=============================================="


DATASET_INFO_FILE="${LLAMA_FACTORY_DIR}/data/dataset_info.json"

# Build dataset_info.json
echo "{" > "$DATASET_INFO_FILE"
first=true
for char_info in "${CHARACTERS[@]}"; do
    IFS='|' read -r key path lang name <<< "$char_info"
    data_file="${DATA_DIR}/${key}_sft.json"
    
    if [ -f "$data_file" ]; then
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$DATASET_INFO_FILE"
        fi
        echo "  \"${key}_sft\": {" >> "$DATASET_INFO_FILE"
        echo "    \"file_name\": \"../../experiments/sft/data/${key}_sft.json\"" >> "$DATASET_INFO_FILE"
        echo -n "  }" >> "$DATASET_INFO_FILE"
    fi
done
echo "" >> "$DATASET_INFO_FILE"
echo "}" >> "$DATASET_INFO_FILE"

echo "  Dataset info updated: ${DATASET_INFO_FILE}"

# Step 3: Train LoRA adapters for each character
echo ""
echo "[STEP 3/3] Training LoRA adapters..."
echo "=============================================="

for char_info in "${CHARACTERS[@]}"; do
    IFS='|' read -r key path lang name <<< "$char_info"
    adapter_dir="${LLAMA_FACTORY_DIR}/saves/qwen_${key}_sft"
    data_file="${DATA_DIR}/${key}_sft.json"
    
    if [ ! -f "$data_file" ]; then
        echo "  [SKIP] ${key}: No training data"
        continue
    fi
    
    if [ -d "$adapter_dir" ]; then
        echo "  [SKIP] ${key}: Adapter already exists"
        continue
    fi
    
    echo "  [TRAINING] ${key}..."
    bash "${SFT_DIR}/train_lora.sh" "$key" 2>&1 | tail -5
done

echo ""
echo "=============================================="
echo "Training complete! Run evaluation with:"
echo "  python3 experiments/sft/evaluate_sft_batch.py"
echo "=============================================="
