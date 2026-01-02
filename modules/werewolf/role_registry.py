"""
角色注册表模块

提供插件化的角色系统，支持动态加载角色定义。
每个角色通过独立的配置文件定义，系统自动识别和加载。
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum


class CampType(str, Enum):
    """阵营类型"""
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    NEUTRAL = "neutral"  # 预留：中立阵营


class TargetType(str, Enum):
    """目标类型"""
    SINGLE_PLAYER = "single_player"
    MULTIPLE_PLAYERS = "multiple_players"
    SELF_ONLY = "self_only"
    NO_TARGET = "no_target"


class AbilityDefinition(BaseModel):
    """技能定义"""
    
    ability_id: str = Field(..., description="技能ID")
    name: str = Field(..., description="技能名称")
    phase: str = Field(..., description="技能生效阶段")
    target_type: TargetType = Field(..., description="目标类型")
    can_target_self: bool = Field(default=False, description="是否可以选择自己")
    restrictions: List[str] = Field(default_factory=list, description="限制条件")
    description: str = Field(default="", description="技能描述")
    
    class Config:
        extra = 'allow'


class AIBehavior(BaseModel):
    """AI行为配置"""
    
    prompt_template: str = Field(..., description="提示词模板名称")
    think_chain_enabled: bool = Field(default=True, description="是否启用思维链")
    deception_level: str = Field(default="medium", description="欺骗能力等级: low/medium/high")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="生成温度")
    
    class Config:
        extra = 'allow'


class RoleDefinition(BaseModel):
    """角色定义"""
    
    role_id: str = Field(..., description="角色ID")
    role_name: str = Field(..., description="角色名称")
    camp: CampType = Field(..., description="所属阵营")
    abilities: List[AbilityDefinition] = Field(default_factory=list, description="角色技能列表")
    ai_behavior: Optional[AIBehavior] = Field(default=None, description="AI行为配置")
    description: str = Field(default="", description="角色描述")
    win_condition: str = Field(default="", description="胜利条件")
    
    class Config:
        extra = 'allow'


class RoleRegistry:
    """角色注册表，负责加载和管理所有角色定义"""
    
    def __init__(self, roles_dir: Optional[str] = None):
        """
        初始化角色注册表
        
        Args:
            roles_dir: 角色定义文件目录，默认为 modules/werewolf/roles/
        """
        if roles_dir is None:
            # 默认角色目录
            roles_dir = Path(__file__).parent / "roles"
        
        self.roles_dir = Path(roles_dir)
        self.roles: Dict[str, RoleDefinition] = {}
        
        # 如果目录不存在，创建它
        if not self.roles_dir.exists():
            self.roles_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RoleRegistry] 创建角色目录: {self.roles_dir}")
    
    def load_all_roles(self) -> Dict[str, RoleDefinition]:
        """
        加载所有角色定义
        
        Returns:
            Dict[str, RoleDefinition]: 角色ID到角色定义的映射
        """
        self.roles.clear()
        
        if not self.roles_dir.exists():
            print(f"[RoleRegistry] 警告: 角色目录不存在: {self.roles_dir}")
            return {}
        
        # 遍历所有JSON文件
        for role_file in self.roles_dir.glob('*.json'):
            try:
                with open(role_file, 'r', encoding='utf-8') as f:
                    role_data = json.load(f)
                
                role = RoleDefinition(**role_data)
                self.roles[role.role_id] = role
                print(f"[RoleRegistry] 加载角色: {role.role_name} ({role.role_id})")
                
            except Exception as e:
                print(f"[RoleRegistry] 加载角色失败 {role_file}: {e}")
        
        print(f"[RoleRegistry] 共加载 {len(self.roles)} 个角色")
        return self.roles
    
    def get_role(self, role_id: str) -> Optional[RoleDefinition]:
        """
        获取指定的角色定义
        
        Args:
            role_id: 角色ID
            
        Returns:
            Optional[RoleDefinition]: 角色定义，如果不存在则返回None
        """
        return self.roles.get(role_id)
    
    def register_custom_role(self, role_def: RoleDefinition) -> None:
        """
        注册自定义角色
        
        Args:
            role_def: 角色定义对象
        """
        self.roles[role_def.role_id] = role_def
        print(f"[RoleRegistry] 注册自定义角色: {role_def.role_name}")
    
    def get_roles_by_camp(self, camp: CampType) -> List[RoleDefinition]:
        """
        获取指定阵营的所有角色
        
        Args:
            camp: 阵营类型
            
        Returns:
            List[RoleDefinition]: 角色定义列表
        """
        return [role for role in self.roles.values() if role.camp == camp]
    
    def save_role(self, role: RoleDefinition, filename: Optional[str] = None) -> None:
        """
        保存角色定义到文件
        
        Args:
            role: 角色定义对象
            filename: 文件名（不含扩展名），默认使用role_id
        """
        if filename is None:
            filename = f"{role.role_id}.json"
        elif not filename.endswith('.json'):
            filename += '.json'
        
        filepath = self.roles_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(role.dict(), f, ensure_ascii=False, indent=2)
        
        print(f"[RoleRegistry] 角色已保存: {filepath}")
    
    def list_role_ids(self) -> List[str]:
        """
        列出所有已加载的角色ID
        
        Returns:
            List[str]: 角色ID列表
        """
        return list(self.roles.keys())


def create_default_roles():
    """创建默认的角色定义文件"""
    registry = RoleRegistry()
    roles_dir = registry.roles_dir
    
    # 狼人
    werewolf = RoleDefinition(
        role_id="werewolf",
        role_name="狼人",
        camp=CampType.WEREWOLF,
        description="狼人每晚可以击杀一名玩家",
        win_condition="所有好人被淘汰或狼人数量≥好人数量",
        abilities=[
            AbilityDefinition(
                ability_id="kill",
                name="击杀",
                phase="night_werewolf",
                target_type=TargetType.SINGLE_PLAYER,
                can_target_self=False,
                description="狼人团队选择一名玩家进行击杀"
            )
        ],
        ai_behavior=AIBehavior(
            prompt_template="werewolf_night_action",
            think_chain_enabled=True,
            deception_level="high",
            temperature=0.9
        )
    )
    
    # 预言家
    seer = RoleDefinition(
        role_id="seer",
        role_name="预言家",
        camp=CampType.VILLAGER,
        description="预言家每晚可以查验一名玩家的身份",
        win_condition="所有狼人被淘汰",
        abilities=[
            AbilityDefinition(
                ability_id="check",
                name="查验",
                phase="night_seer",
                target_type=TargetType.SINGLE_PLAYER,
                can_target_self=False,
                description="查验一名玩家，获知其是狼人还是好人"
            )
        ],
        ai_behavior=AIBehavior(
            prompt_template="seer_check",
            think_chain_enabled=True,
            deception_level="low",
            temperature=0.7
        )
    )
    
    # 女巫
    witch = RoleDefinition(
        role_id="witch",
        role_name="女巫",
        camp=CampType.VILLAGER,
        description="女巫拥有一瓶解药和一瓶毒药",
        win_condition="所有狼人被淘汰",
        abilities=[
            AbilityDefinition(
                ability_id="antidote",
                name="解药",
                phase="night_witch",
                target_type=TargetType.SINGLE_PLAYER,
                can_target_self=True,
                restrictions=["use_once_per_game", "can_only_save_killed_player"],
                description="救活被狼人击杀的玩家"
            ),
            AbilityDefinition(
                ability_id="poison",
                name="毒药",
                phase="night_witch",
                target_type=TargetType.SINGLE_PLAYER,
                can_target_self=False,
                restrictions=["use_once_per_game", "cannot_use_both_potions_same_night"],
                description="毒杀一名玩家"
            )
        ],
        ai_behavior=AIBehavior(
            prompt_template="witch_potion_use",
            think_chain_enabled=True,
            deception_level="medium",
            temperature=0.8
        )
    )
    
    # 猎人
    hunter = RoleDefinition(
        role_id="hunter",
        role_name="猎人",
        camp=CampType.VILLAGER,
        description="猎人死亡时可以开枪带走一名玩家",
        win_condition="所有狼人被淘汰",
        abilities=[
            AbilityDefinition(
                ability_id="shoot",
                name="开枪",
                phase="on_death",
                target_type=TargetType.SINGLE_PLAYER,
                can_target_self=False,
                restrictions=["trigger_on_death", "cannot_shoot_if_poisoned"],
                description="死亡时开枪带走一名玩家"
            )
        ],
        ai_behavior=AIBehavior(
            prompt_template="hunter_shoot",
            think_chain_enabled=True,
            deception_level="low",
            temperature=0.7
        )
    )
    
    # 平民
    villager = RoleDefinition(
        role_id="villager",
        role_name="平民",
        camp=CampType.VILLAGER,
        description="平民没有特殊技能",
        win_condition="所有狼人被淘汰",
        abilities=[],
        ai_behavior=AIBehavior(
            prompt_template="villager_discussion",
            think_chain_enabled=True,
            deception_level="low",
            temperature=0.8
        )
    )
    
    # 守卫（可选角色）
    guard = RoleDefinition(
        role_id="guard",
        role_name="守卫",
        camp=CampType.VILLAGER,
        description="守卫每晚可以守护一名玩家",
        win_condition="所有狼人被淘汰",
        abilities=[
            AbilityDefinition(
                ability_id="protect",
                name="守护",
                phase="night_guard",
                target_type=TargetType.SINGLE_PLAYER,
                can_target_self=True,
                restrictions=["cannot_protect_same_twice_consecutive"],
                description="守护一名玩家，使其免受狼人击杀"
            )
        ],
        ai_behavior=AIBehavior(
            prompt_template="guard_protect",
            think_chain_enabled=True,
            deception_level="low",
            temperature=0.7
        )
    )
    
    # 保存所有角色
    roles = [werewolf, seer, witch, hunter, villager, guard]
    
    for role in roles:
        filepath = roles_dir / f"{role.role_id}.json"
        if not filepath.exists():
            registry.save_role(role)
            print(f"[RoleRegistry] 创建默认角色: {role.role_name}")


if __name__ == "__main__":
    # 测试角色注册表
    create_default_roles()
    
    registry = RoleRegistry()
    registry.load_all_roles()
    
    print("\n已加载的角色:")
    for role_id in registry.list_role_ids():
        role = registry.get_role(role_id)
        print(f"  - {role.role_name} ({role.camp.value}): {len(role.abilities)} 个技能")
