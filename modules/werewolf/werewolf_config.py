# 狼人杀配置文件
import os
import json

# 读取主配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        main_config = json.load(f)
    WEREWOLF_LLM_NAME = main_config.get("LLM_NAME", "gemini-2.5-flash-lite")
except Exception as e:
    print(f"[Werewolf] 无法读取 config.json，使用默认模型: {e}")
    WEREWOLF_LLM_NAME = "gemini-2.5-flash-lite"

# 游戏配置
DEFAULT_PRESET = "simple_6"  # 默认游戏配置（降低为6人以减少API调用）

# 人类玩家默认ID
DEFAULT_HUMAN_PLAYER_ID = "player_0"
