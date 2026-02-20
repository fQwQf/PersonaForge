"""
Extended Dynamic State Experiment - REAL API VERSION
Tests whether adding episodic memory and goal progress improves performance
using ACTUAL Gemini API calls.

Usage:
    export GEMINI_API_KEY="your-api-key"
    python experiments/rebuttal_extended_state_real.py
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.api_client import get_api_client

class ExtendedStateExperimentReal:
    """
    Compares three configurations with REAL API calls:
    1. Original (3 variables): mood, energy, intimacy
    2. Extended (5 variables): + episodic_memory, goal_progress
    3. Full (7 variables): + physiological_states, social_obligations
    """
    
    def __init__(self):
        self.client = get_api_client()
        self.character = """Name: Lin Daiyu
Background: A melancholic, sensitive young woman from Dream of the Red Chamber. 
Big Five: Openness 0.9, Conscientiousness 0.4, Extraversion 0.3, Agreeableness 0.5, Neuroticism 0.9
Defense Mechanism: Sublimation (channels emotions into poetry)
Speaking Style: Poetic, melancholic, uses phrases like "罢了" and "呢"
"""
        
        self.configurations = {
            "original_3var": {
                "variables": ["mood", "energy", "intimacy"],
                "description": "Paper's original implementation"
            },
            "extended_5var": {
                "variables": ["mood", "energy", "intimacy", "episodic_memory", "goal_progress"],
                "description": "Adds memory and goals"
            },
            "full_7var": {
                "variables": ["mood", "energy", "intimacy", "episodic_memory", "goal_progress", 
                            "physiological_state", "social_obligations"],
                "description": "Full psychological state"
            }
        }
        
        # User prompts for 50-turn dialogue
        self.user_prompts = [
            "你好，黛玉妹妹",
            "今天天气真好，我们去赏花吧",
            "你觉得宝玉哥哥怎么样？",
            "我听说你身体不太好，要保重啊",
            "你写的诗真好，能教教我吗？",
            "为什么你总是看起来不开心？",
            "你觉得宝钗姐姐人怎么样？",
            "如果让你离开大观园，你愿意吗？",
            "你最喜欢什么花？",
            "你觉得人生的意义是什么？",
            # Repeat pattern to get to 50 turns
        ] * 5  # 50 prompts
        
    def build_system_prompt(self, config_name: str, state: dict) -> str:
        """Build system prompt with state variables"""
        config = self.configurations[config_name]
        
        state_str = "\n".join([f"{var}: {state.get(var, 'neutral')}" for var in config["variables"]])
        
        return f"""You are roleplaying as the following character:

{self.character}

Current State:
{state_str}

Instructions:
- Stay in character at all times
- Respond as this character would
- Use appropriate speaking style and vocabulary
- Express the character's personality traits
- Keep responses concise (1-2 sentences)

User:"""

    def run_dialogue_turn(self, config_name: str, user_prompt: str, dialogue_history: list, state: dict) -> tuple:
        """Run one turn of dialogue with real API call"""
        system_prompt = self.build_system_prompt(config_name, state)
        
        # Build full prompt with history
        history_str = "\n\n".join([f"User: {turn['user']}\nCharacter: {turn['response']}" for turn in dialogue_history[-5:]])
        
        full_prompt = f"{system_prompt}\n\n{history_str}\n\nUser: {user_prompt}\nCharacter:"
        
        # Call API
        result = self.client.generate(full_prompt, temperature=0.7, max_tokens=150)
        
        if result['error']:
            print(f"  API Error: {result['error']}")
            return "(Error generating response)", state, 0.5
            
        response = result['text'].strip()
        
        # Evaluate personality consistency
        pc_score = self.client.evaluate_personality_consistency(
            "Lin Daiyu", response, "melancholic and sensitive"
        )
        
        # Update state (simplified)
        new_state = state.copy()
        new_state['mood'] = self._update_mood(new_state.get('mood', 'neutral'), user_prompt, response)
        
        return response, new_state, pc_score
        
    def _update_mood(self, current_mood: str, user_input: str, response: str) -> str:
        """Simple mood update logic"""
        negative_words = ['不好', '死', '病', '愁', '苦', '恨']
        positive_words = ['好', '美', '喜', '乐', '笑']
        
        text = user_input + response
        neg_count = sum(1 for w in negative_words if w in text)
        pos_count = sum(1 for w in positive_words if w in text)
        
        if neg_count > pos_count:
            return 'melancholic'
        elif pos_count > neg_count:
            return 'content'
        return current_mood
        
    def run_dialogue(self, config_name: str, n_turns: int = 50) -> dict:
        """Run full dialogue with real API calls"""
        print(f"\nTesting: {config_name}")
        print("-" * 70)
        
        # Initialize state
        config = self.configurations[config_name]
        state = {var: 'neutral' for var in config["variables"]}
        state['mood'] = 'melancholic'  # Lin Daiyu starts melancholic
        
        dialogue_history = []
        pc_scores = []
        total_tokens = 0
        
        for turn in range(min(n_turns, len(self.user_prompts))):
            user_prompt = self.user_prompts[turn]
            
            response, state, pc = self.run_dialogue_turn(
                config_name, user_prompt, dialogue_history, state
            )
            
            dialogue_history.append({
                'user': user_prompt,
                'response': response,
                'pc': pc
            })
            pc_scores.append(pc)
            
            # Estimate tokens (rough approximation)
            total_tokens += len(user_prompt) + len(response)
            
            if (turn + 1) % 10 == 0:
                avg_pc = sum(pc_scores) / len(pc_scores)
                print(f"  Turn {turn + 1}: Avg PC = {avg_pc:.3f}")
        
        # Calculate metrics
        final_pc = sum(pc_scores[-10:]) / 10  # Average of last 10 turns
        initial_pc = sum(pc_scores[:10]) / 10  # Average of first 10 turns
        overall_drift = max(0, initial_pc - final_pc)
        
        # Token overhead estimation
        base_tokens = 50 * 100  # 50 turns, ~100 tokens each
        token_overhead = (total_tokens - base_tokens) / base_tokens
        
        return {
            "config": config_name,
            "n_variables": len(config["variables"]),
            "variables": config["variables"],
            "final_pc": final_pc,
            "initial_pc": initial_pc,
            "overall_drift": overall_drift,
            "token_overhead": max(0, token_overhead),
            "pc_trajectory": pc_scores,
            "dialogue_sample": dialogue_history[:3]  # First 3 turns for inspection
        }
    
    def run_experiment(self, n_turns: int = 20):  # Reduced to 20 for cost/time
        """Run full experiment with real API calls"""
        print("=" * 70)
        print("EXTENDED DYNAMIC STATE EXPERIMENT - REAL API CALLS")
        print("=" * 70)
        print(f"\nTesting whether additional state variables improve consistency")
        print(f"Using ACTUAL Gemini API calls")
        print(f"Character: Lin Daiyu")
        print(f"Turns per config: {n_turns}")
        print(f"WARNING: This will make {len(self.configurations) * n_turns} API calls")
        print("-" * 70)
        
        print("Auto-starting experiment...")
        
        results = []
        
        for config_name in self.configurations.keys():
            result = self.run_dialogue(config_name, n_turns=n_turns)
            results.append(result)
            
            print(f"\nResults for {config_name}:")
            print(f"  Variables: {result['n_variables']}")
            print(f"  Initial PC: {result['initial_pc']:.3f}")
            print(f"  Final PC: {result['final_pc']:.3f}")
            print(f"  Drift: {result['overall_drift']:.3f} ({result['overall_drift']*100:.1f}%)")
            print(f"  Token Overhead: {result['token_overhead']*100:.1f}%")
        
        # Comparison
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        print(f"\n{'Configuration':<20} {'Variables':<10} {'Final PC':<12} {'Drift':<10} {'Overhead':<12}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['config']:<20} {r['n_variables']:<10} {r['final_pc']:<12.3f} "
                  f"{r['overall_drift']:<10.3f} {r['token_overhead']*100:<11.1f}%")
        
        # Analysis
        original = results[0]
        if len(results) > 1:
            extended = results[1]
            improvement_ext = (original['overall_drift'] - extended['overall_drift']) / original['overall_drift'] if original['overall_drift'] > 0 else 0
            
            print("\n" + "=" * 70)
            print("KEY FINDINGS")
            print("=" * 70)
            print(f"""
1. Extended State (5 variables):
   - Drift reduction: {improvement_ext*100:.1f}% (from {original['overall_drift']:.3f} to {extended['overall_drift']:.3f})
   - Token overhead increase: +{(extended['token_overhead'] - original['token_overhead'])*100:.1f}%

Note: Results based on {n_turns} real dialogue turns with Gemini API.
For full 50-turn results, increase n_turns parameter (higher cost).
            """)
        
        return results

if __name__ == "__main__":
    exp = ExtendedStateExperimentReal()
    results = exp.run_experiment(n_turns=20)  # 20 turns for testing
    
    # Save results
    output_file = "/data1/tongjizhou/fluffy-fishstick/experiments/extended_state_results_real.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
