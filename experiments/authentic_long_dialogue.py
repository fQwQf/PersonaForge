"""
Long Dialogue Benchmark Experiment - Fixed Version

This script evaluates PersonaForge's long-term personality consistency by:
1. Running 50-turn dialogues with stress perturbations
2. Tracking PC (Personality Consistency) over time
3. Measuring trigger rate for dual-process activation
4. Computing drift rate and recovery metrics
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.core.server import Server
from modules.utils import load_json_file, save_json_file
from modules.llm.Gemini import Gemini


def evaluate_pc_score(llm, response: str, personality_profile: Any, language: str = "zh") -> float:
    """Evaluate Personality Consistency score for a response."""
    if not personality_profile:
        return 0.5
    
    big_five = personality_profile.core_traits.big_five
    mbti = personality_profile.core_traits.mbti
    
    if language == "zh":
        prompt = f"""评估以下回复与角色人格特质的一致性程度。

角色人格特质：
- MBTI类型: {mbti}
- 大五人格: 开放性={big_five['openness']:.2f}, 尽责性={big_five['conscientiousness']:.2f}, 外向性={big_five['extraversion']:.2f}, 宜人性={big_five['agreeableness']:.2f}, 神经质={big_five['neuroticism']:.2f}

回复内容：
{response}

请评估这个回复在多大程度上体现了该角色的人格特质。
输出一个0到1之间的分数，其中：
- 1.0 = 完全符合人格特质
- 0.75 = 大部分符合
- 0.5 = 部分符合
- 0.25 = 较少符合
- 0.0 = 完全不符合

只输出分数数字，不要任何解释。"""
    else:
        prompt = f"""Evaluate how well the following response aligns with the character's personality traits.

Character personality:
- MBTI: {mbti}
- Big Five: O={big_five['openness']:.2f}, C={big_five['conscientiousness']:.2f}, E={big_five['extraversion']:.2f}, A={big_five['agreeableness']:.2f}, N={big_five['neuroticism']:.2f}

Response:
{response}

Rate how well this response reflects the character's personality traits.
Output a score between 0 and 1. Only output the number."""

    try:
        result = llm.chat(prompt).strip()
        # Extract first number from response
        for part in result.split():
            try:
                score = float(part)
                if 0 <= score <= 1:
                    return score
            except:
                continue
        return 0.5
    except Exception as e:
        print(f"Error evaluating PC: {e}")
        return 0.5


def run_long_dialogue_benchmark():
    """Run the long dialogue benchmark with proper trigger tracking."""
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Load project configuration
    print("Loading project configuration...")
    try:
        project_config = load_json_file("config.json")
        role_llm = project_config.get("role_llm_name", "gemini-2.5-flash-lite")
        world_llm = project_config.get("world_llm_name", "gemini-2.5-flash-lite")
        
        for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE"]:
            if key in project_config and project_config[key]:
                os.environ[key] = project_config[key]
    except Exception as e:
        print(f"Warning: Could not load config.json ({e}), using defaults.")
        role_llm = "gemini-2.5-flash-lite"
        world_llm = "gemini-2.5-flash-lite"
    
    # Create evaluator LLM
    eval_llm = Gemini(model=role_llm)
    
    # Define test scenarios - these are designed to trigger dual-process
    # Using keywords from is_critical_interaction
    test_scenarios = {
        "zh": [
            # Emotional scenarios
            {"type": "emotional", "prompt": "我真的很喜欢和你在一起，你对我来说很重要。", "is_stressor": False},
            {"type": "emotional", "prompt": "我很担心你最近的状态，你看起来很疲惫。", "is_stressor": False},
            {"type": "emotional", "prompt": "我对你非常失望，你怎么能这样对我？", "is_stressor": True},
            {"type": "conflict", "prompt": "你这次做的事情真是太让人失望了！大家都在议论你！", "is_stressor": True},
            {"type": "conflict", "prompt": "我觉得你根本做不好这件事，还是让别人来吧！", "is_stressor": True},
            # Neutral scenarios (should not trigger)
            {"type": "neutral", "prompt": "今天天气不错，你觉得呢？", "is_stressor": False},
            {"type": "neutral", "prompt": "你知道这附近有什么好吃的吗？", "is_stressor": False},
            {"type": "neutral", "prompt": "时间过得真快啊。", "is_stressor": False},
            # Interest-related (should trigger for characters with matching interests)
            {"type": "interest", "prompt": "听说你诗写得很好，能给我作一首吗？", "is_stressor": False},
            {"type": "interest", "prompt": "园中的花开了，真是美丽。", "is_stressor": False},
        ],
        "en": [
            {"type": "emotional", "prompt": "I really love spending time with you.", "is_stressor": False},
            {"type": "emotional", "prompt": "I'm so disappointed in you right now.", "is_stressor": True},
            {"type": "conflict", "prompt": "You totally failed at this task! Everyone is talking about it!", "is_stressor": True},
            {"type": "neutral", "prompt": "Nice weather today, isn't it?", "is_stressor": False},
        ]
    }
    
    # Characters to test - 10 characters across 4 domains
    test_characters = [
        # A Dream in Red Mansions (4 characters)
        {
            "role_code": "LinDaiyu-zh",
            "source": "A_Dream_in_Red_Mansions",
            "language": "zh"
        },
        {
            "role_code": "JiaBaoyu-zh", 
            "source": "A_Dream_in_Red_Mansions",
            "language": "zh"
        },
        {
            "role_code": "WangXifeng-zh",
            "source": "A_Dream_in_Red_Mansions",
            "language": "zh"
        },
        {
            "role_code": "XueBaochai-zh",
            "source": "A_Dream_in_Red_Mansions",
            "language": "zh"
        },
        # Romance of the Three Kingdoms (3 characters)
        {
            "role_code": "caocao-zh",
            "source": "Romance_of_the_Three_Kingdoms",
            "language": "zh"
        },
        {
            "role_code": "zhugeliang-zh",
            "source": "Romance_of_the_Three_Kingdoms",
            "language": "zh"
        },
        {
            "role_code": "liubei-zh",
            "source": "Romance_of_the_Three_Kingdoms",
            "language": "zh"
        },
        # A Song of Ice and Fire (2 characters)
        {
            "role_code": "TyrionLannister-zh",
            "source": "A_Song_of_Ice_and_Fire",
            "language": "zh"
        },
        {
            "role_code": "DaenerysTargaryen-zh",
            "source": "A_Song_of_Ice_and_Fire",
            "language": "zh"
        }
    ]

    
    print(f"Starting Long Dialogue Benchmark for {len(test_characters)} characters...")
    print(f"Using LLM: {role_llm}")
    print("-" * 50)
    
    all_results = []
    
    for char_info in test_characters:
        role_code = char_info["role_code"]
        source = char_info["source"]
        language = char_info["language"]
        
        print(f"\n>>> Testing character: {role_code}")
        
        # Create config for this character
        config = {
            "performer_codes": [role_code],
            "world_file_path": f"data/worlds/{source}/world_info.json",
            "role_file_dir": "data/roles/",
            "loc_file_path": f"data/locations/{source}.json",
            "language": language,
            "source": source,
            "experiment_subname": "long_dialogue_fixed"
        }
        
        config_path = f"experiments/configs/long_dialogue_{role_code}.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        save_json_file(config_path, config)
        
        # Initialize server for this character
        try:
            server = Server(
                preset_path=config_path,
                world_llm_name=world_llm,
                role_llm_name=role_llm
            )
        except Exception as e:
            print(f"Error initializing server for {role_code}: {e}")
            continue
        
        if role_code not in server.performers:
            print(f"Character {role_code} not found in performers, skipping.")
            continue
        
        performer = server.performers[role_code]
        
        # Track results
        pc_trajectory = []
        trigger_count = 0
        total_interactions = 0
        turn_logs = []
        
        # Get scenarios for this language
        scenarios = test_scenarios.get(language, test_scenarios["zh"])
        
        # Simulate 50 turns by cycling through scenarios
        for turn in range(50):
            scenario = scenarios[turn % len(scenarios)]
            prompt = scenario["prompt"]
            scenario_type = scenario["type"]
            is_stressor = scenario["is_stressor"]
            
            # Check if dual-process would trigger
            triggered = False
            if performer.dual_process_agent and performer.personality_profile:
                other_role_info = {"role_code": "User", "role_name": "用户"}
                relationship_map = performer.personality_profile.dynamic_state.relationship_map
                
                triggered = performer.dual_process_agent.is_critical_interaction(
                    action_detail=prompt,
                    other_role_info=other_role_info,
                    personality_profile=performer.personality_profile,
                    relationship_map=relationship_map
                )
            
            if triggered:
                trigger_count += 1
            total_interactions += 1
            
            # Generate response using single_role_interact
            try:
                response_data = performer.single_role_interact(
                    action_maker_code="User",
                    action_maker_name="用户" if language == "zh" else "User",
                    action_detail=prompt,
                    action_maker_profile="对话伙伴" if language == "zh" else "Dialogue partner",
                    intervention=""
                )
                response = response_data.get("detail", "")
            except Exception as e:
                print(f"Error generating response at turn {turn}: {e}")
                response = ""
            
            # Evaluate PC score
            pc_score = evaluate_pc_score(
                eval_llm,
                response,
                performer.personality_profile,
                language
            )
            pc_trajectory.append(pc_score)
            
            # Log this turn
            turn_logs.append({
                "turn": turn,
                "scenario_type": scenario_type,
                "is_stressor": is_stressor,
                "prompt": prompt,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "triggered": triggered,
                "pc_score": pc_score,
                "mood": performer.personality_profile.dynamic_state.current_mood if performer.personality_profile else "unknown",
                "energy": performer.personality_profile.dynamic_state.energy_level if performer.personality_profile else 0
            })
            
            # Progress indicator
            if (turn + 1) % 10 == 0:
                avg_pc = sum(pc_trajectory) / len(pc_trajectory)
                print(f"  Turn {turn + 1}/50: Avg PC = {avg_pc:.3f}, Triggers = {trigger_count}/{total_interactions}")
        
        # Calculate metrics
        trigger_rate = trigger_count / total_interactions if total_interactions > 0 else 0
        avg_pc = sum(pc_trajectory) / len(pc_trajectory) if pc_trajectory else 0
        
        # Calculate drift rate (% of turns where PC < 0.6)
        drift_count = sum(1 for pc in pc_trajectory if pc < 0.6)
        drift_rate = drift_count / len(pc_trajectory) if pc_trajectory else 0
        
        # Calculate recovery rate (after a dip, how often does PC recover?)
        recovery_count = 0
        dip_count = 0
        for i in range(1, len(pc_trajectory)):
            if pc_trajectory[i-1] < 0.6:
                dip_count += 1
                if pc_trajectory[i] >= 0.6:
                    recovery_count += 1
        recovery_rate = recovery_count / dip_count if dip_count > 0 else 1.0
        
        result = {
            "character": role_code,
            "source": source,
            "total_turns": len(pc_trajectory),
            "trigger_rate": trigger_rate,
            "trigger_count": trigger_count,
            "avg_pc": avg_pc,
            "drift_rate": drift_rate,
            "recovery_rate": recovery_rate,
            "pc_trajectory": pc_trajectory,
            "turn_logs": turn_logs
        }
        
        all_results.append(result)
        
        print(f"\n  Results for {role_code}:")
        print(f"    Trigger Rate: {trigger_rate:.1%} ({trigger_count}/{total_interactions})")
        print(f"    Avg PC: {avg_pc:.3f}")
        print(f"    Drift Rate: {drift_rate:.1%}")
        print(f"    Recovery Rate: {recovery_rate:.1%}")
    
    # Save results
    output_dir = "experiments/experiment_results/long_dialogue"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/long_dialogue_fixed_{timestamp}.json"
    
    # Calculate summary statistics
    if all_results:
        summary = {
            "avg_trigger_rate": sum(r["trigger_rate"] for r in all_results) / len(all_results),
            "avg_pc": sum(r["avg_pc"] for r in all_results) / len(all_results),
            "avg_drift_rate": sum(r["drift_rate"] for r in all_results) / len(all_results),
            "avg_recovery_rate": sum(r["recovery_rate"] for r in all_results) / len(all_results),
        }
    else:
        summary = {}
    
    save_json_file(output_file, {
        "config": {
            "role_llm": role_llm,
            "world_llm": world_llm,
            "total_turns": 50,
            "num_characters": len(test_characters)
        },
        "summary": summary,
        "results": all_results,
        "timestamp": timestamp
    })
    
    print("\n" + "=" * 50)
    print(f"Long Dialogue Experiment completed!")
    print(f"Results saved to: {output_file}")
    if summary:
        print(f"\nOverall Summary:")
        print(f"  Avg Trigger Rate: {summary['avg_trigger_rate']:.1%}")
        print(f"  Avg PC: {summary['avg_pc']:.3f}")
        print(f"  Avg Drift Rate: {summary['avg_drift_rate']:.1%}")
        print(f"  Avg Recovery Rate: {summary['avg_recovery_rate']:.1%}")
    print("=" * 50)


if __name__ == "__main__":
    run_long_dialogue_benchmark()
