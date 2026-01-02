# 狼人杀游戏模块

ScrollWeaver的狼人杀游戏扩展模块，基于配置驱动的多智能体框架。

## ✨ 特性

- 🎮 **完全配置驱动** - 支持8人局/12人局/任意人数，仅需修改配置文件
- 🔌 **插件化角色系统** - 轻松添加新角色（守卫、白狼王等）
- ⚙️ **灵活规则引擎** - 通过配置修改游戏规则（女巫首夜自救等）
- 🤖 **AI玩家支持** - 复用ScrollWeaver的Performer实现智能AI玩家
- 🧪 **命令行测试** - 无需前端即可测试完整游戏流程

## 📦 已实现功能

### 核心模块
- ✅ **配置加载器** (`config_loader.py`) - Pydantic验证的配置管理
- ✅ **角色注册表** (`role_registry.py`) - 动态加载角色定义
- ✅ **游戏状态管理** (`game_state.py`) - 完整的状态跟踪
- ✅ **规则引擎** (`rule_engine.py`) - 模块化技能结算系统

### 已定义角色
1. 🐺 **狼人** - 夜晚击杀
2. 🔮 **预言家** - 查验身份
3. 🧪 **女巫** - 解药/毒药
4. 🏹 **猎人** - 死亡开枪
5. 👤 **平民** - 无特殊技能
6. 🛡️ **守卫** - 守护玩家

## 🚀 快速开始

### 安装依赖

```bash
pip install pydantic
```

### 命令行测试

```bash
cd /mnt/c/Users/super/project/ScrollWeaver

# 方法1：直接运行测试脚本
python test_werewolf.py

# 方法2：运行CLI模块
python modules/werewolf/cli_test.py

# 方法3：使用Python模块方式
python -m modules.werewolf.cli_test
```

### 选择配置

程序启动后会提示选择：
```
选择游戏配置:
1. 12人标准局 (4狼3神5民)
2. 8人简化局 (2狼2神4民)
```

### 游戏流程

系统会自动模拟：
1. 🌙 **夜晚阶段** - 狼人击杀、预言家查验、女巫用药
2. ☀️ **白天阶段** - 宣布死亡、讨论发言、投票放逐
3. 🏆 **胜负判定** - 自动检测胜利条件

按 `Enter` 键继续下一轮。

## 📝 配置示例

### 12人标准局

```json
{
  "game_name": "12人标准局",
  "total_players": 12,
  "roles": {
    "werewolf": 4,
    "seer": 1,
    "witch": 1,
    "hunter": 1,
    "villager": 5
  },
  "phase_flow": [
    "night_werewolf",
    "night_seer",
    "night_witch",
    "day_announce",
    "day_discussion",
    "day_vote"
  ],
  "rules": {
    "witch_can_self_save_first_night": true,
    "hunter_can_shoot_when_poisoned": false,
    "allow_werewolf_self_destruct": false
  }
}
```

### 8人简化局

```json
{
  "game_name": "8人简化局",
  "total_players": 8,
  "roles": {
    "werewolf": 2,
    "seer": 1,
    "witch": 1,
    "villager": 4
  },
  "phase_flow": [
    "night_werewolf",
    "night_seer",
    "night_witch",
    "day_announce",
    "day_discussion",
    "day_vote"
  ]
}
```

## 🎨 自定义扩展

### 添加新角色（示例：守卫）

**步骤1** - 创建角色配置文件 `modules/werewolf/roles/guard.json`:

```json
{
  "role_id": "guard",
  "role_name": "守卫",
  "camp": "villager",
  "abilities": [
    {
      "ability_id": "protect",
      "name": "守护",
      "phase": "night_guard",
      "target_type": "single_player",
      "can_target_self": true,
      "restrictions": ["cannot_protect_same_twice_consecutive"]
    }
  ],
  "ai_behavior": {
    "prompt_template": "guard_protect",
    "think_chain_enabled": true,
    "deception_level": "low",
    "temperature": 0.7
  }
}
```

**步骤2** - 在游戏配置中启用:

```json
{
  "roles": {
    "werewolf": 4,
    "seer": 1,
    "witch": 1,
    "hunter": 1,
    "guard": 1,
    "villager": 4
  },
  "phase_flow": [
    "night_guard",
    "night_werewolf",
    "night_seer",
    "night_witch",
    "day_announce",
    "day_discussion",
    "day_vote"
  ]
}
```

系统会自动加载新角色！

### 修改规则

在配置文件的 `rules` 字段修改：

```json
{
  "rules": {
    "witch_can_self_save_first_night": false,
    "hunter_can_shoot_when_poisoned": true,
    "consecutive_guard_same_player": true
  }
}
```

规则引擎会自动应用！

## 🧪 测试

### 单元测试

```bash
# 测试配置加载器
python -m modules.werewolf.config_loader

# 测试角色注册表
python -m modules.werewolf.role_registry

# 测试游戏状态
python -m modules.werewolf.game_state

# 测试规则引擎
python -m modules.werewolf.rule_engine
```

### 完整游戏测试

```bash
python test_werewolf.py
```

## 📁 文件结构

```
modules/werewolf/
├── __init__.py
├── config_loader.py       # 配置管理
├── role_registry.py       # 角色系统
├── game_state.py          # 状态管理
├── rule_engine.py         # 规则引擎
├── cli_test.py            # 命令行测试
└── roles/                 # 角色定义
    ├── werewolf.json
    ├── seer.json
    ├── witch.json
    ├── hunter.json
    ├── villager.json
    └── guard.json

data/werewolf/
└── presets/               # 游戏预设
    ├── standard_12.json
    └── simple_8.json
```

## 🔜 待实现

- ⏳ Orchestrator适配器（法官AI）
- ⏳ Performer适配器（玩家AI）
- ⏳ WebSocket API集成
- ⏳ 前端游戏界面
- ⏳ 警长竞选模式
- ⏳ 更多高级角色（白狼王、骑士等）

## 📄 许可证

本模块作为ScrollWeaver项目的一部分，遵循项目许可证。
