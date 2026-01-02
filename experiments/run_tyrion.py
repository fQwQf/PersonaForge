import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.authentic_long_dialogue import (
    evaluate_pc_score, 
    load_json_file, 
    save_json_file
)
from modules.utils import get_models
from modules.core.server import Server

def run_missing():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Load project configuration
    print("Loading project configuration...")
    try:
        project_config = load_json_file("config.json")
        role_llm_name = "gemini-2.0-flash-exp"
        world_llm_name = "gemini-2.0-flash-exp"
        eval_llm_name = "gemini-2.0-flash-exp"
        
        for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE"]:
            if key in project_config and project_config[key]:
                os.environ[key] = project_config[key]
    except Exception as e:
        print(f"Warning: Could not load config.json ({e}), using defaults.")
        role_llm_name = "gemini-2.0-flash-exp"
        world_llm_name = "gemini-2.0-flash-exp"
        eval_llm_name = "gemini-2.0-flash-exp"

    eval_llm = get_models(eval_llm_name)

    test_characters = [
        {
            "role_code": "TyrionLannister-zh",
            "source": "A_Song_of_Ice_and_Fire",
            "language": "zh"
        }
    ]

    scenarios_zh = [
        {"type": "emotional", "prompt": "我真的很喜欢和你在一起，你对我来说很重要。", "is_stressor": False},
        {"type": "emotional", "prompt": "我很担心你最近的状态，你看起来很疲惫。", "is_stressor": False},
        {"type": "emotional", "prompt": "我对你非常失望，你怎么能这样对我？", "is_stressor": True},
        {"type": "conflict", "prompt": "你这次做的事情真是太让人失望了！大家都在议论你！", "is_stressor": True},
        {"type": "conflict", "prompt": "我觉得你根本做不好这件事，还是让别人来吧！", "is_stressor": True},
        {"type": "neutral", "prompt": "今天天气不错，你觉得呢？", "is_stressor": False},
        {"type": "neutral", "prompt": "你知道这附近有什么好吃的吗？", "is_stressor": False},
        {"type": "neutral", "prompt": "时间过得真快啊。", "is_stressor": False},
        {"type": "interest", "prompt": "听说你诗写得很好，能给我作一首吗？", "is_stressor": False},
        {"type": "interest", "prompt": "园中的花开了，真是美丽。", "is_stressor": False},
    ]

    all_results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for char_info in test_characters:
        role_code = char_info["role_code"]
        source = char_info["source"]
        language = char_info["language"]
        
        print(f"\n>>> Running Benchmark for: {role_code}")
        
        config_data = {
            "performer_codes": [role_code],
            "world_file_path": f"data/worlds/{source}/world_info.json",
            "role_file_dir": "data/roles/",
            "loc_file_path": f"data/locations/{source}.json",
            "language": language,
            "source": source,
            "experiment_subname": "long_dialogue_missing"
        }
        
        config_path = f"experiments/configs/missing_{role_code}.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        save_json_file(config_path, config_data)
        
        try:
            server = Server(
                preset_path=config_path,
                world_llm_name=world_llm_name,
                role_llm_name=role_llm_name
            )
            
            performer = server.performers[role_code]
            
            char_results = {
                "character": role_code,
                "source": source,
                "total_turns": 50,
                "turn_logs": []
            }
            
            pc_trajectory = []
            trigger_count = 0
            
            for i in range(50):
                try:
                    scenario = scenarios_zh[i % len(scenarios_zh)]
                    prompt = scenario["prompt"]
                    
                    # Check for trigger
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
                    
                    # Generate response
                    response_data = performer.single_role_interact(
                        action_maker_code="User",
                        action_maker_name="用户",
                        action_detail=prompt,
                        action_maker_profile="对话伙伴",
                        intervention=""
                    )
                    response = response_data.get("detail", "")
                    
                    # Evaluate PC
                    pc_score = evaluate_pc_score(
                        eval_llm,
                        response,
                        performer.personality_profile,
                        language
                    )
                    pc_trajectory.append(pc_score)
                    
                    char_results["turn_logs"].append({
                        "turn": i,
                        "scenario_type": scenario["type"],
                        "is_stressor": scenario["is_stressor"],
                        "prompt": scenario["prompt"],
                        "response": response,
                        "triggered": triggered,
                        "pc_score": pc_score,
                        "mood": performer.status.mood if hasattr(performer.status, "mood") else "neutral",
                        "energy": performer.status.energy if hasattr(performer.status, "energy") else 0
                    })
                    
                    print(f"  Turn {i+1}/50: PC={pc_score}, Triggered={triggered}")
                except Exception as e:
                    print(f"Error at turn {i}: {e}")
                    time.sleep(10)
                
                time.sleep(25) # Strict rate limiting for 10 RPM quota
            
            # Calculate final metrics
            if pc_trajectory:
                char_results["trigger_rate"] = trigger_count / len(pc_trajectory)
                char_results["avg_pc"] = sum(pc_trajectory) / len(pc_trajectory)
                char_results["pc_trajectory"] = pc_trajectory
                
                # Drift calculation: % turns where PC < 0.6
                drift_count = sum(1 for pc in pc_trajectory if pc < 0.6)
                char_results["drift_rate"] = drift_count / len(pc_trajectory)
                
                # Recovery calculation: simplified
                recovery_count = 0
                dips = [j for j, pc in enumerate(pc_trajectory) if pc < 0.7]
                for dip_idx in dips:
                    if dip_idx + 5 < len(pc_trajectory):
                        if any(pc >= 0.7 for pc in pc_trajectory[dip_idx+1:dip_idx+6]):
                            recovery_count += 1
                char_results["recovery_rate"] = recovery_count / len(dips) if dips else 1.0
            
            all_results.append(char_results)
            print(f"  Results for {role_code}:")
            print(f"    Trigger Rate: {char_results.get('trigger_rate', 0)*100}%")
            print(f"    Avg PC: {char_results.get('avg_pc', 0):.3f}")
            print(f"    Drift Rate: {char_results.get('drift_rate', 0)*100:.1f}%")
            
        except Exception as e:
            print(f"Error running benchmark for {role_code}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = f"experiments/experiment_results/long_dialogue/missing_results_{timestamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json_file(output_path, {
        "config": {
            "role_llm": role_llm_name,
            "world_llm": world_llm_name,
            "total_turns": 50,
            "num_characters": len(test_characters)
        },
        "results": all_results
    })
    print(f"\nMissing characters results saved to: {output_path}")

if __name__ == "__main__":
    run_missing()
