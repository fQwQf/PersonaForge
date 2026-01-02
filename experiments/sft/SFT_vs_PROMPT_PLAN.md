# SFT vs. Prompting (In-Context Learning) 对比实验计划

## 1. 实验背景
我们在 2025-12-30 完成了基于 15 个角色的 Batch SFT 训练。初步结果显示 SFT 在风格迁移上表现出色，但存在灾难性遗忘。为了验证 SFT 的必要性，我们需要将其与“不调参、仅使用高质量提示词”的基线进行对比。

## 2. 实验目标
探究在**同一个基座模型 (Qwen)** 上，以下哪种方式能带来更好的角色扮演体验：
1.  **SFT (微调)**: 将人设数据内化为模型参数。
2.  **Prompting (提示工程)**: 将人设数据作为上下文输入给模型。

## 3. 实验设置

### 3.1 控制变量
*   **基座模型**: 统一使用 `Qwen/Qwen1.5-7B-Chat` (或昨天训练使用的具体版本)。
*   **测试场景**: 使用 `EvaluationFramework` 中的标准测试集 (Casual, Stress/Conflict)。
*   **评测裁判**: 统一使用 `Gemini-Pro` 作为 LLM Judge。

### 3.2 对比组定义

| 组别 | 模型配置 | System Prompt (系统提示词) | 预期优势 | 潜在劣势 |
| :--- | :--- | :--- | :--- | :--- |
| **Group A: SFT** | Qwen Base + **LoRA Adapter** | **极简**: "You are {Role}." | 风格内化深，响应速度快 | 容易过拟合，灵活性差 |
| **Group B: Prompt** | Qwen Base (**无 Adapter**) | **详尽**: 包含完整的 Bio, Style, Catchphrases | 逻辑能力强，通用性好 | 容易出戏，受Context长度限制 |

## 4. 实施流程 (由脚本自动化)
1.  **加载阶段**:
    *   先加载原始 Base Model。
    *   运行 Group B (Prompt) 的生成任务。
    *   卸载或重新加载 Base Model + Adapter。
    *   运行 Group A (SFT) 的生成任务。
2.  **生成阶段**:
    *   对每个角色，抽取 5 个测试场景（包含 2 个高压冲突场景）。
3.  **评估阶段**:
    *   计算 PC, SA, DM 分数。
    *   计算 Delta (SFT Score - Prompt Score)。

## 5. 预期输出
生成一份对比报告 `sft_vs_prompt_result.json`，直观展示两种方法在不同维度上的优劣。
