"""
配置加载器模块

负责加载和验证狼人杀游戏配置文件。
支持从预设目录加载配置，并验证配置的有效性。
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator


class GameConfig(BaseModel):
    """游戏配置数据结构"""
    
    game_name: str = Field(..., description="游戏名称")
    total_players: int = Field(..., ge=6, le=18, description="总玩家数")
    roles: Dict[str, int] = Field(..., description="角色类型到数量的映射")
    phase_flow: List[str] = Field(..., description="游戏阶段流程")
    rules: Dict[str, Any] = Field(default_factory=dict, description="可配置规则")
    
    @validator('roles')
    def validate_roles_count(cls, v, values):
        """验证角色数量总和是否等于总玩家数"""
        if 'total_players' in values:
            total = sum(v.values())
            if total != values['total_players']:
                raise ValueError(
                    f"角色数量总和 ({total}) 必须等于总玩家数 ({values['total_players']})"
                )
        return v
    
    @validator('phase_flow')
    def validate_phase_flow(cls, v):
        """验证阶段流程非空"""
        if not v:
            raise ValueError("phase_flow 不能为空")
        return v
    
    class Config:
        extra = 'allow'  # 允许额外字段，便于扩展


class ConfigLoader:
    """配置加载器，负责加载和验证游戏配置"""
    
    def __init__(self, presets_dir: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            presets_dir: 预设配置文件目录，默认为 data/werewolf/presets/
        """
        if presets_dir is None:
            # 默认预设目录
            base_dir = Path(__file__).parent.parent.parent
            presets_dir = base_dir / "data" / "werewolf" / "presets"
        
        self.presets_dir = Path(presets_dir)
        
        # 如果目录不存在，创建它
        if not self.presets_dir.exists():
            self.presets_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ConfigLoader] 创建预设目录: {self.presets_dir}")
    
    def load_preset(self, preset_name: str) -> GameConfig:
        """
        从预设目录加载配置
        
        Args:
            preset_name: 预设名称（不含扩展名）或完整文件名
            
        Returns:
            GameConfig: 游戏配置对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置验证失败
        """
        # 处理文件名
        if not preset_name.endswith('.json'):
            preset_name += '.json'
        
        config_path = self.presets_dir / preset_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载JSON配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 验证并创建配置对象
        try:
            config = GameConfig(**config_data)
            print(f"[ConfigLoader] 成功加载配置: {config.game_name}")
            return config
        except Exception as e:
            raise ValueError(f"配置验证失败: {e}")
    
    def load_config_from_dict(self, config_data: Dict[str, Any]) -> GameConfig:
        """
        从字典加载配置
        
        Args:
            config_data: 配置字典
            
        Returns:
            GameConfig: 游戏配置对象
        """
        return GameConfig(**config_data)
    
    def save_preset(self, config: GameConfig, preset_name: str) -> None:
        """
        保存配置到预设目录
        
        Args:
            config: 游戏配置对象
            preset_name: 预设名称（不含扩展名）
        """
        if not preset_name.endswith('.json'):
            preset_name += '.json'
        
        config_path = self.presets_dir / preset_name
        
        # 保存为JSON
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.dict(), f, ensure_ascii=False, indent=2)
        
        print(f"[ConfigLoader] 配置已保存: {config_path}")
    
    def list_available_presets(self) -> List[str]:
        """
        列出所有可用的预设配置
        
        Returns:
            List[str]: 预设配置文件名列表（不含扩展名）
        """
        if not self.presets_dir.exists():
            return []
        
        presets = [
            f.stem for f in self.presets_dir.glob('*.json')
            if f.is_file()
        ]
        
        return sorted(presets)
    
    def validate_config(self, config: GameConfig) -> bool:
        """
        验证配置的有效性
        
        Args:
            config: 游戏配置对象
            
        Returns:
            bool: 配置是否有效
        """
        try:
            # Pydantic 已经在初始化时验证了，这里做额外检查
            
            # 检查至少有一个狼人和一个好人
            werewolf_count = config.roles.get('werewolf', 0)
            if werewolf_count == 0:
                print("[ConfigLoader] 警告: 没有狼人角色")
                return False
            
            villager_count = sum(
                count for role, count in config.roles.items()
                if role != 'werewolf'
            )
            if villager_count == 0:
                print("[ConfigLoader] 警告: 没有好人角色")
                return False
            
            # 检查阶段流程合理性
            if 'day_discussion' not in config.phase_flow and 'day_vote' not in config.phase_flow:
                print("[ConfigLoader] 警告: 没有白天讨论或投票阶段")
                return False
            
            return True
            
        except Exception as e:
            print(f"[ConfigLoader] 配置验证失败: {e}")
            return False


def create_default_configs():
    """创建默认的配置文件（如果不存在）"""
    loader = ConfigLoader()
    
    # 12人标准局
    standard_12 = GameConfig(
        game_name="12人标准局",
        total_players=12,
        roles={
            "werewolf": 4,
            "seer": 1,
            "witch": 1,
            "hunter": 1,
            "villager": 5
        },
        phase_flow=[
            "night_werewolf",
            "night_seer",
            "night_witch",
            "day_announce",
            "day_discussion",
            "day_vote"
        ],
        rules={
            "witch_can_self_save_first_night": True,
            "hunter_can_shoot_when_poisoned": False,
            "allow_werewolf_self_destruct": False,
            "consecutive_guard_same_player": False
        }
    )
    
    # 8人简化局
    simple_8 = GameConfig(
        game_name="8人简化局",
        total_players=8,
        roles={
            "werewolf": 2,
            "seer": 1,
            "witch": 1,
            "villager": 4
        },
        phase_flow=[
            "night_werewolf",
            "night_seer",
            "night_witch",
            "day_announce",
            "day_discussion",
            "day_vote"
        ],
        rules={
            "witch_can_self_save_first_night": True,
            "hunter_can_shoot_when_poisoned": False
        }
    )
    
    # 保存配置
    presets_dir = loader.presets_dir
    
    if not (presets_dir / "standard_12.json").exists():
        loader.save_preset(standard_12, "standard_12")
        print("[ConfigLoader] 创建默认配置: standard_12.json")
    
    if not (presets_dir / "simple_8.json").exists():
        loader.save_preset(simple_8, "simple_8")
        print("[ConfigLoader] 创建默认配置: simple_8.json")


if __name__ == "__main__":
    # 测试配置加载器
    create_default_configs()
    
    loader = ConfigLoader()
    print("\n可用预设:", loader.list_available_presets())
    
    # 加载并验证标准12人局配置
    config = loader.load_preset("standard_12")
    print(f"\n配置: {config.game_name}")
    print(f"玩家数: {config.total_players}")
    print(f"角色: {config.roles}")
    print(f"验证结果: {loader.validate_config(config)}")
