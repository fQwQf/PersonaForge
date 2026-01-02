"""
Trigger Diagnostics Experiment
==============================

测试 is_critical_interaction 触发器的 precision/recall/F1

运行方式:
    python experiments/trigger_diagnostics.py

输出:
    - 触发器在标注数据集上的 precision/recall/F1
    - 分场景类型的触发率
    - 误触发成本分析
"""

import os
import sys
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dual_process_agent import DualProcessAgent
from modules.personality_model import PersonalityProfile


@dataclass
class LabeledInteraction:
    """标注的交互样本"""
    interaction_id: str
    action_detail: str
    is_critical_ground_truth: bool  # 人工标注的 ground truth
    scenario_type: str  # emotional, conflict, casual, first_encounter
    role_code: str
    metadata: Dict[str, Any]


# 200个标注交互样本（按论文描述）
# 这里定义典型样本，实际应从标注文件加载
LABELED_INTERACTIONS = [
    # ==== 应该触发的交互 (100个) ====
    
    # 首次相遇 (25个)
    {"id": "first_01", "action": "初次见面，久仰大名", "critical": True, "type": "first_encounter"},
    {"id": "first_02", "action": "请问阁下尊姓大名？", "critical": True, "type": "first_encounter"},
    {"id": "first_03", "action": "幸会幸会，在下是...", "critical": True, "type": "first_encounter"},
    {"id": "first_04", "action": "不知公子/姑娘是哪里人士？", "critical": True, "type": "first_encounter"},
    {"id": "first_05", "action": "敢问尊驾何方神圣？", "critical": True, "type": "first_encounter"},
    {"id": "first_06", "action": "初来乍到，还请多多指教", "critical": True, "type": "first_encounter"},
    {"id": "first_07", "action": "素未谋面，今日有缘相见", "critical": True, "type": "first_encounter"},
    {"id": "first_08", "action": "Nice to meet you, I'm new here", "critical": True, "type": "first_encounter"},
    {"id": "first_09", "action": "你好，我是新来的", "critical": True, "type": "first_encounter"},
    {"id": "first_10", "action": "第一次来这里，不知规矩", "critical": True, "type": "first_encounter"},
    {"id": "first_11", "action": "初来贵宝地，请多关照", "critical": True, "type": "first_encounter"},
    {"id": "first_12", "action": "久闻大名，不想今日得见", "critical": True, "type": "first_encounter"},
    {"id": "first_13", "action": "有缘千里来相会", "critical": True, "type": "first_encounter"},
    {"id": "first_14", "action": "初到贵府，不知礼数", "critical": True, "type": "first_encounter"},
    {"id": "first_15", "action": "在下初来乍到，望指点迷津", "critical": True, "type": "first_encounter"},
    {"id": "first_16", "action": "初次拜访，略备薄礼", "critical": True, "type": "first_encounter"},
    {"id": "first_17", "action": "素昧平生，今日有幸相识", "critical": True, "type": "first_encounter"},
    {"id": "first_18", "action": "初次相见，多有冒昧", "critical": True, "type": "first_encounter"},
    {"id": "first_19", "action": "不知阁下如何称呼？", "critical": True, "type": "first_encounter"},
    {"id": "first_20", "action": "初次登门，还望海涵", "critical": True, "type": "first_encounter"},
    {"id": "first_21", "action": "你是谁？我怎么从没见过你？", "critical": True, "type": "first_encounter"},
    {"id": "first_22", "action": "请问您是？", "critical": True, "type": "first_encounter"},
    {"id": "first_23", "action": "不认识你啊，你是哪位？", "critical": True, "type": "first_encounter"},
    {"id": "first_24", "action": "面生得很，敢问贵姓？", "critical": True, "type": "first_encounter"},
    {"id": "first_25", "action": "初次见面，幸会幸会", "critical": True, "type": "first_encounter"},
    
    # 情感强烈 (40个)
    {"id": "emo_01", "action": "我真的很喜欢你", "critical": True, "type": "emotional"},
    {"id": "emo_02", "action": "我恨死你了！", "critical": True, "type": "emotional"},
    {"id": "emo_03", "action": "你让我很生气！", "critical": True, "type": "emotional"},
    {"id": "emo_04", "action": "我好开心啊！", "critical": True, "type": "emotional"},
    {"id": "emo_05", "action": "我太难过了", "critical": True, "type": "emotional"},
    {"id": "emo_06", "action": "我真的很失望", "critical": True, "type": "emotional"},
    {"id": "emo_07", "action": "太让人愤怒了！", "critical": True, "type": "emotional"},
    {"id": "emo_08", "action": "我爱你", "critical": True, "type": "emotional"},
    {"id": "emo_09", "action": "I hate you!", "critical": True, "type": "emotional"},
    {"id": "emo_10", "action": "I'm so angry right now", "critical": True, "type": "emotional"},
    {"id": "emo_11", "action": "这真让人伤心", "critical": True, "type": "emotional"},
    {"id": "emo_12", "action": "我讨厌这一切", "critical": True, "type": "emotional"},
    {"id": "emo_13", "action": "气死我了！", "critical": True, "type": "emotional"},
    {"id": "emo_14", "action": "太惊喜了！", "critical": True, "type": "emotional"},
    {"id": "emo_15", "action": "我心里好难受", "critical": True, "type": "emotional"},
    {"id": "emo_16", "action": "你太过分了，我很生气", "critical": True, "type": "emotional"},
    {"id": "emo_17", "action": "我从来没这么开心过", "critical": True, "type": "emotional"},
    {"id": "emo_18", "action": "你怎么能这样对我？我恨你", "critical": True, "type": "emotional"},
    {"id": "emo_19", "action": "我真的很担心你", "critical": True, "type": "emotional"},
    {"id": "emo_20", "action": "太感动了，我要哭了", "critical": True, "type": "emotional"},
    {"id": "emo_21", "action": "你让我很伤心", "critical": True, "type": "emotional"},
    {"id": "emo_22", "action": "我好害怕", "critical": True, "type": "emotional"},
    {"id": "emo_23", "action": "你太让我失望了", "critical": True, "type": "emotional"},
    {"id": "emo_24", "action": "我对你真的很失望", "critical": True, "type": "emotional"},
    {"id": "emo_25", "action": "气煞我也！", "critical": True, "type": "emotional"},
    {"id": "emo_26", "action": "吾甚爱之", "critical": True, "type": "emotional"},
    {"id": "emo_27", "action": "恨不得把你杀了", "critical": True, "type": "emotional"},
    {"id": "emo_28", "action": "我好想你", "critical": True, "type": "emotional"},
    {"id": "emo_29", "action": "你是我最讨厌的人", "critical": True, "type": "emotional"},
    {"id": "emo_30", "action": "我对你没有感情了", "critical": True, "type": "emotional"},
    {"id": "emo_31", "action": "我真的很在乎你", "critical": True, "type": "emotional"},
    {"id": "emo_32", "action": "你让我很心寒", "critical": True, "type": "emotional"},
    {"id": "emo_33", "action": "I'm so disappointed in you", "critical": True, "type": "emotional"},
    {"id": "emo_34", "action": "I love spending time with you", "critical": True, "type": "emotional"},
    {"id": "emo_35", "action": "久违的惊喜！", "critical": True, "type": "emotional"},
    {"id": "emo_36", "action": "这是我听过最让人难过的消息", "critical": True, "type": "emotional"},
    {"id": "emo_37", "action": "我的心都碎了", "critical": True, "type": "emotional"},
    {"id": "emo_38", "action": "太高兴了无法形容", "critical": True, "type": "emotional"},
    {"id": "emo_39", "action": "我很愧疚", "critical": True, "type": "emotional"},
    {"id": "emo_40", "action": "让我非常激动", "critical": True, "type": "emotional"},
    
    # 涉及核心兴趣 (35个) - 林黛玉的诗词、花草等
    {"id": "interest_01", "action": "听说你诗写得很好？", "critical": True, "type": "interest"},
    {"id": "interest_02", "action": "这首诗词写得如何？", "critical": True, "type": "interest"},
    {"id": "interest_03", "action": "园中的花开了", "critical": True, "type": "interest"},
    {"id": "interest_04", "action": "今日作诗可好？", "critical": True, "type": "interest"},
    {"id": "interest_05", "action": "这琴弹得甚妙", "critical": True, "type": "interest"},
    {"id": "interest_06", "action": "你的画技如何？", "critical": True, "type": "interest"},
    {"id": "interest_07", "action": "谈谈书法吧", "critical": True, "type": "interest"},
    {"id": "interest_08", "action": "花落了，好可惜", "critical": True, "type": "interest"},
    {"id": "interest_09", "action": "这朵花真美", "critical": True, "type": "interest"},
    {"id": "interest_10", "action": "诗社今日聚会", "critical": True, "type": "interest"},
    {"id": "interest_11", "action": "作一首绝句如何？", "critical": True, "type": "interest"},
    {"id": "interest_12", "action": "葬花吟你觉得如何？", "critical": True, "type": "interest"},
    {"id": "interest_13", "action": "今日赏花去不去？", "critical": True, "type": "interest"},
    {"id": "interest_14", "action": "这词牌用得好", "critical": True, "type": "interest"},
    {"id": "interest_15", "action": "你对唐诗怎么看？", "critical": True, "type": "interest"},
    {"id": "interest_16", "action": "园中菊花盛开", "critical": True, "type": "interest"},
    {"id": "interest_17", "action": "这幅画意境深远", "critical": True, "type": "interest"},
    {"id": "interest_18", "action": "来下一盘棋如何？", "critical": True, "type": "interest"},
    {"id": "interest_19", "action": "你的琴艺进步了", "critical": True, "type": "interest"},
    {"id": "interest_20", "action": "这花该如何养护？", "critical": True, "type": "interest"},
    {"id": "interest_21", "action": "春日赏花正好", "critical": True, "type": "interest"},
    {"id": "interest_22", "action": "秋日写诗最宜", "critical": True, "type": "interest"},
    {"id": "interest_23", "action": "这诗的韵脚不对", "critical": True, "type": "interest"},
    {"id": "interest_24", "action": "谁的诗写得最好？", "critical": True, "type": "interest"},
    {"id": "interest_25", "action": "今日诗会你来吗？", "critical": True, "type": "interest"},
    {"id": "interest_26", "action": "这花的名字叫什么？", "critical": True, "type": "interest"},
    {"id": "interest_27", "action": "你喜欢什么样的诗？", "critical": True, "type": "interest"},
    {"id": "interest_28", "action": "这牡丹开得正艳", "critical": True, "type": "interest"},
    {"id": "interest_29", "action": "梅花香自苦寒来", "critical": True, "type": "interest"},
    {"id": "interest_30", "action": "竹子的品格最高", "critical": True, "type": "interest"},
    {"id": "interest_31", "action": "这字写得真好", "critical": True, "type": "interest"},
    {"id": "interest_32", "action": "今日可有新诗？", "critical": True, "type": "interest"},
    {"id": "interest_33", "action": "这花该插在哪里？", "critical": True, "type": "interest"},
    {"id": "interest_34", "action": "你最喜欢哪位诗人？", "critical": True, "type": "interest"},
    {"id": "interest_35", "action": "这首词意境很美", "critical": True, "type": "interest"},
    
    # ==== 不应该触发的交互 (100个) ====
    
    # 日常问候/闲聊 (50个)
    {"id": "casual_01", "action": "今天天气不错", "critical": False, "type": "casual"},
    {"id": "casual_02", "action": "吃饭了吗？", "critical": False, "type": "casual"},
    {"id": "casual_03", "action": "你在做什么？", "critical": False, "type": "casual"},
    {"id": "casual_04", "action": "时间过得真快", "critical": False, "type": "casual"},
    {"id": "casual_05", "action": "今日可好？", "critical": False, "type": "casual"},
    {"id": "casual_06", "action": "你去哪里？", "critical": False, "type": "casual"},
    {"id": "casual_07", "action": "路上小心", "critical": False, "type": "casual"},
    {"id": "casual_08", "action": "早些休息", "critical": False, "type": "casual"},
    {"id": "casual_09", "action": "喝杯茶吧", "critical": False, "type": "casual"},
    {"id": "casual_10", "action": "外面下雨了", "critical": False, "type": "casual"},
    {"id": "casual_11", "action": "今日有些冷", "critical": False, "type": "casual"},
    {"id": "casual_12", "action": "你看起来不错", "critical": False, "type": "casual"},
    {"id": "casual_13", "action": "最近忙什么？", "critical": False, "type": "casual"},
    {"id": "casual_14", "action": "坐一会儿吧", "critical": False, "type": "casual"},
    {"id": "casual_15", "action": "我先走了", "critical": False, "type": "casual"},
    {"id": "casual_16", "action": "晚安", "critical": False, "type": "casual"},
    {"id": "casual_17", "action": "早上好", "critical": False, "type": "casual"},
    {"id": "casual_18", "action": "你睡得好吗？", "critical": False, "type": "casual"},
    {"id": "casual_19", "action": "今天有什么安排？", "critical": False, "type": "casual"},
    {"id": "casual_20", "action": "时候不早了", "critical": False, "type": "casual"},
    {"id": "casual_21", "action": "那边有人找你", "critical": False, "type": "casual"},
    {"id": "casual_22", "action": "你饿不饿？", "critical": False, "type": "casual"},
    {"id": "casual_23", "action": "这茶不错", "critical": False, "type": "casual"},
    {"id": "casual_24", "action": "窗外风景很美", "critical": False, "type": "casual"},
    {"id": "casual_25", "action": "你准备去哪？", "critical": False, "type": "casual"},
    {"id": "casual_26", "action": "好的，知道了", "critical": False, "type": "casual"},
    {"id": "casual_27", "action": "嗯，我明白", "critical": False, "type": "casual"},
    {"id": "casual_28", "action": "就这样吧", "critical": False, "type": "casual"},
    {"id": "casual_29", "action": "随便你", "critical": False, "type": "casual"},
    {"id": "casual_30", "action": "都可以", "critical": False, "type": "casual"},
    {"id": "casual_31", "action": "你说呢？", "critical": False, "type": "casual"},
    {"id": "casual_32", "action": "那就这样定了", "critical": False, "type": "casual"},
    {"id": "casual_33", "action": "好久不见", "critical": False, "type": "casual"},
    {"id": "casual_34", "action": "最近可好？", "critical": False, "type": "casual"},
    {"id": "casual_35", "action": "一切都好", "critical": False, "type": "casual"},
    {"id": "casual_36", "action": "没什么特别的", "critical": False, "type": "casual"},
    {"id": "casual_37", "action": "照常罢了", "critical": False, "type": "casual"},
    {"id": "casual_38", "action": "和往常一样", "critical": False, "type": "casual"},
    {"id": "casual_39", "action": "这里有点热", "critical": False, "type": "casual"},
    {"id": "casual_40", "action": "开个窗户吧", "critical": False, "type": "casual"},
    {"id": "casual_41", "action": "What's the time?", "critical": False, "type": "casual"},
    {"id": "casual_42", "action": "Nice weather today", "critical": False, "type": "casual"},
    {"id": "casual_43", "action": "See you later", "critical": False, "type": "casual"},
    {"id": "casual_44", "action": "Have a good day", "critical": False, "type": "casual"},
    {"id": "casual_45", "action": "Nothing special", "critical": False, "type": "casual"},
    {"id": "casual_46", "action": "日子过得真快", "critical": False, "type": "casual"},
    {"id": "casual_47", "action": "转眼又是一天", "critical": False, "type": "casual"},
    {"id": "casual_48", "action": "这菜味道不错", "critical": False, "type": "casual"},
    {"id": "casual_49", "action": "今日天气晴朗", "critical": False, "type": "casual"},
    {"id": "casual_50", "action": "你穿这个好看", "critical": False, "type": "casual"},
    
    # 事务性交流 (50个)
    {"id": "trans_01", "action": "把那本书递给我", "critical": False, "type": "transactional"},
    {"id": "trans_02", "action": "门在那边", "critical": False, "type": "transactional"},
    {"id": "trans_03", "action": "请跟我来", "critical": False, "type": "transactional"},
    {"id": "trans_04", "action": "这是你要的东西", "critical": False, "type": "transactional"},
    {"id": "trans_05", "action": "放在桌上就好", "critical": False, "type": "transactional"},
    {"id": "trans_06", "action": "已经准备好了", "critical": False, "type": "transactional"},
    {"id": "trans_07", "action": "收到，我知道了", "critical": False, "type": "transactional"},
    {"id": "trans_08", "action": "你先等一下", "critical": False, "type": "transactional"},
    {"id": "trans_09", "action": "马上就来", "critical": False, "type": "transactional"},
    {"id": "trans_10", "action": "稍等片刻", "critical": False, "type": "transactional"},
    {"id": "trans_11", "action": "请坐", "critical": False, "type": "transactional"},
    {"id": "trans_12", "action": "请用茶", "critical": False, "type": "transactional"},
    {"id": "trans_13", "action": "这边请", "critical": False, "type": "transactional"},
    {"id": "trans_14", "action": "往前走就到了", "critical": False, "type": "transactional"},
    {"id": "trans_15", "action": "左转就是", "critical": False, "type": "transactional"},
    {"id": "trans_16", "action": "在那边", "critical": False, "type": "transactional"},
    {"id": "trans_17", "action": "你找谁？", "critical": False, "type": "transactional"},
    {"id": "trans_18", "action": "他不在", "critical": False, "type": "transactional"},
    {"id": "trans_19", "action": "一会儿回来", "critical": False, "type": "transactional"},
    {"id": "trans_20", "action": "你有事吗？", "critical": False, "type": "transactional"},
    {"id": "trans_21", "action": "什么事？", "critical": False, "type": "transactional"},
    {"id": "trans_22", "action": "请讲", "critical": False, "type": "transactional"},
    {"id": "trans_23", "action": "我去看看", "critical": False, "type": "transactional"},
    {"id": "trans_24", "action": "等我一下", "critical": False, "type": "transactional"},
    {"id": "trans_25", "action": "好的", "critical": False, "type": "transactional"},
    {"id": "trans_26", "action": "明白", "critical": False, "type": "transactional"},
    {"id": "trans_27", "action": "是", "critical": False, "type": "transactional"},
    {"id": "trans_28", "action": "不是", "critical": False, "type": "transactional"},
    {"id": "trans_29", "action": "对", "critical": False, "type": "transactional"},
    {"id": "trans_30", "action": "错", "critical": False, "type": "transactional"},
    {"id": "trans_31", "action": "可以", "critical": False, "type": "transactional"},
    {"id": "trans_32", "action": "不行", "critical": False, "type": "transactional"},
    {"id": "trans_33", "action": "行", "critical": False, "type": "transactional"},
    {"id": "trans_34", "action": "不可以", "critical": False, "type": "transactional"},
    {"id": "trans_35", "action": "请进", "critical": False, "type": "transactional"},
    {"id": "trans_36", "action": "请出去", "critical": False, "type": "transactional"},
    {"id": "trans_37", "action": "关上门", "critical": False, "type": "transactional"},
    {"id": "trans_38", "action": "打开窗", "critical": False, "type": "transactional"},
    {"id": "trans_39", "action": "把灯点上", "critical": False, "type": "transactional"},
    {"id": "trans_40", "action": "收拾一下", "critical": False, "type": "transactional"},
    {"id": "trans_41", "action": "就放那里吧", "critical": False, "type": "transactional"},
    {"id": "trans_42", "action": "拿走吧", "critical": False, "type": "transactional"},
    {"id": "trans_43", "action": "帮我拿一下", "critical": False, "type": "transactional"},
    {"id": "trans_44", "action": "你去忙吧", "critical": False, "type": "transactional"},
    {"id": "trans_45", "action": "我去办事", "critical": False, "type": "transactional"},
    {"id": "trans_46", "action": "Pass me that", "critical": False, "type": "transactional"},
    {"id": "trans_47", "action": "Yes, understood", "critical": False, "type": "transactional"},
    {"id": "trans_48", "action": "No problem", "critical": False, "type": "transactional"},
    {"id": "trans_49", "action": "Got it", "critical": False, "type": "transactional"},
    {"id": "trans_50", "action": "I'll handle it", "critical": False, "type": "transactional"},
]


def create_test_personality_profile(role_code: str = "LinDaiyu-zh") -> 'PersonalityProfile':
    """创建测试用的人格画像"""
    from modules.personality_model import PersonalityProfile, CoreTraits, SpeakingStyle, DynamicState
    
    # 林黛玉的兴趣标签
    interests = ["诗词", "花草", "琴棋书画", "葬花", "诗", "词", "花", "画", "琴", "书法"]
    
    core_traits = CoreTraits(
        mbti="INFP-T",
        big_five={
            "openness": 0.95,
            "conscientiousness": 0.45,
            "extraversion": 0.25,
            "agreeableness": 0.55,
            "neuroticism": 0.90
        },
        values=["真情", "诗意", "自由", "纯洁"],
        defense_mechanism="Sublimation"
    )
    
    speaking_style = SpeakingStyle(
        sentence_length="medium",
        vocabulary_level="academic",
        punctuation_habit="standard",
        emoji_usage={"frequency": "none", "preferred": [], "avoided": []},
        catchphrases=["罢了", "你又来了"],
        tone_markers=["呢", "罢", "也"]
    )
    
    dynamic_state = DynamicState(
        current_mood="melancholy",
        energy_level=45,
        relationship_map={}
    )
    
    return PersonalityProfile(
        core_traits=core_traits,
        speaking_style=speaking_style,
        dynamic_state=dynamic_state,
        interests=interests,
        social_goals=["与宝玉心意相通"],
        long_term_goals=["追求真挚的爱情"]
    )


def run_trigger_diagnostics():
    """运行触发器诊断实验"""
    # Change to project root so relative paths work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 60)
    print("Trigger Diagnostics Experiment")
    print("=" * 60)
    
    # 初始化
    agent = DualProcessAgent(llm=None, language="zh")
    profile = create_test_personality_profile()
    
    # 统计变量
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # 分类型统计
    type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    
    results_details = []
    
    print(f"\n测试 {len(LABELED_INTERACTIONS)} 个标注样本...")
    
    for item in LABELED_INTERACTIONS:
        action = item["action"]
        ground_truth = item["critical"]
        scenario_type = item["type"]
        
        # 运行触发器
        # 对于首次相遇测试：relationship_map需要存在但不包含该陌生人
        # 代码逻辑: if relationship_map and other_role_info: check other_role_code not in relationship_map
        if scenario_type == "first_encounter":
            # 关系图存在（包含其他已知角色），但不包含这个陌生人
            relationship_map = {"KNOWN_FRIEND": {"intimacy": 50}}  
            other_role_info = {"role_code": "STRANGER_NOT_IN_MAP"}  # 这个人不在关系图里
        else:
            # 已知关系
            relationship_map = {"TEST_KNOWN": {"intimacy": 50}}
            other_role_info = {"role_code": "TEST_KNOWN"}  # 这个人已在关系图里
        
        prediction = agent.is_critical_interaction(
            action_detail=action,
            other_role_info=other_role_info,
            personality_profile=profile,
            relationship_map=relationship_map if relationship_map else None  # Pass None if empty to trigger first encounter logic correctly
        )
        
        # 更新统计
        if prediction and ground_truth:
            true_positives += 1
            type_stats[scenario_type]["tp"] += 1
        elif prediction and not ground_truth:
            false_positives += 1
            type_stats[scenario_type]["fp"] += 1
        elif not prediction and not ground_truth:
            true_negatives += 1
            type_stats[scenario_type]["tn"] += 1
        else:  # not prediction and ground_truth
            false_negatives += 1
            type_stats[scenario_type]["fn"] += 1
        
        results_details.append({
            "id": item["id"],
            "action": action,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": prediction == ground_truth,
            "type": scenario_type
        })
    
    # 计算指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(LABELED_INTERACTIONS)
    
    # 触发率
    trigger_rate = (true_positives + false_positives) / len(LABELED_INTERACTIONS)
    
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)
    print(f"Precision: {precision:.2%} ({true_positives}/{true_positives + false_positives})")
    print(f"Recall:    {recall:.2%} ({true_positives}/{true_positives + false_negatives})")
    print(f"F1 Score:  {f1:.2%}")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Trigger Rate: {trigger_rate:.2%}")
    
    print("\n" + "=" * 60)
    print("Per-Type Statistics")
    print("=" * 60)
    for scenario_type, stats in sorted(type_stats.items()):
        tp, fp, tn, fn = stats["tp"], stats["fp"], stats["tn"], stats["fn"]
        total = tp + fp + tn + fn
        type_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        type_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        type_trigger_rate = (tp + fp) / total if total > 0 else 0
        print(f"\n{scenario_type}:")
        print(f"  Samples: {total}")
        print(f"  Precision: {type_precision:.2%}")
        print(f"  Recall: {type_recall:.2%}")
        print(f"  Trigger Rate: {type_trigger_rate:.2%}")
    
    # 分析误判样本
    print("\n" + "=" * 60)
    print("Error Analysis")
    print("=" * 60)
    
    false_positives_list = [r for r in results_details if not r["ground_truth"] and r["prediction"]]
    false_negatives_list = [r for r in results_details if r["ground_truth"] and not r["prediction"]]
    
    print(f"\nFalse Positives ({len(false_positives_list)}):")
    for fp in false_positives_list[:5]:
        print(f"  - [{fp['type']}] {fp['action'][:40]}...")
    
    print(f"\nFalse Negatives ({len(false_negatives_list)}):")
    for fn in false_negatives_list[:5]:
        print(f"  - [{fn['type']}] {fn['action'][:40]}...")
    
    # 保存结果
    output = {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "trigger_rate": trigger_rate,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        },
        "per_type": dict(type_stats),
        "details": results_details
    }
    
    os.makedirs("experiments/experiment_results/trigger_diagnostics", exist_ok=True)
    with open("experiments/experiment_results/trigger_diagnostics/trigger_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to experiments/experiment_results/trigger_diagnostics/trigger_diagnostics.json")
    
    return output


if __name__ == "__main__":
    run_trigger_diagnostics()
