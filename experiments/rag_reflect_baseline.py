import os
import sys
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation_framework import ExperimentRunner, EvaluationScenario
from modules.llm.Gemini import Gemini
from modules.embedding import get_embedding_model
from modules.utils import load_json_file, save_json_file

class RAGReflectGenerator:
    """
    RAG-Reflect Baseline:
    1. Retrieve relevant memories (RAG)
    2. Analyze character mental state/concerns (Reflect)
    3. Generate response
    """
    def __init__(self, llm, embedding_name="bge-small"):
        self.llm = llm
        try:
            self.embedding = get_embedding_model(embedding_name, "zh")
        except Exception as e:
            print(f"Warning: Could not load embedding model ({e}), using simple keyword matching")
            self.embedding = None
        
    def _simple_retrieve(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """Simple keyword-based retrieval fallback"""
        if not documents:
            return []
        
        # Score by keyword overlap
        query_words = set(query.lower())
        scored = []
        for doc in documents:
            doc_words = set(doc.lower())
            overlap = len(query_words & doc_words)
            scored.append((overlap, doc))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored[:top_k]]
        
    def generate_rag_reflect(self, role_data: Dict, scenario: EvaluationScenario) -> tuple:
        role_name = role_data.get("role_name", "一个角色")
        profile = role_data.get("profile", "")
        
        # 1. Setup Memory and Retrieve (RAG)
        # Using role-specific knowledge or history as base for RAG
        memory_data = role_data.get("knowledge", []) + [ex.get("response", "") for ex in role_data.get("style_examples", [])]
        memory_data = [m for m in memory_data if m]  # Filter empty strings
        
        # Use simple retrieval (avoiding complex DB setup)
        query = f"{scenario.context} {scenario.trigger}"
        retrieved = self._simple_retrieve(query, memory_data, top_k=5)
        retrieved_text = "\n".join([f"- {m}" for m in retrieved]) if retrieved else "无相关记忆。"
        
        # 2. Reflection Phase
        reflect_prompt = f"""作为{role_name}，根据以下相关记忆和当前场景，反思你此时的心理状态和最重要的关注点。
        
【角色背景】
{profile}

【相关记忆】
{retrieved_text}

【当前场景】
{scenario.context}
对方说："{scenario.trigger}"

请用三句话总结你的内心感悟："""
        
        reflection = self.llm.chat(reflect_prompt)
        
        # 3. Generation Phase
        gen_prompt = f"""你是{role_name}。
        
【你的档案】
{profile}

【你最近的思考】
"{reflection}"

【当前场景】
{scenario.context}
{scenario.trigger_role}说："{scenario.trigger}"

请以符合你身份和最近思考的方式进行回复："""
        
        response = self.llm.chat(gen_prompt)
        
        return response, reflection

def run_rag_reflect_baseline():
    # Change to project root so relative paths work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 70)
    print("RAG-Reflect Baseline Evaluation")
    print("=" * 70)
    
    # Initialize
    project_config = load_json_file("config.json")
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE"]:
        if key in project_config and project_config[key]:
            os.environ[key] = project_config[key]
            
    role_llm_name = project_config.get("role_llm_name", "gemini-2.5-flash-lite")
    llm = Gemini(model=role_llm_name, timeout=60)
    
    runner = ExperimentRunner(llm=llm)
    gen = RAGReflectGenerator(llm=llm)
    
    selected_roles = [
        ("A_Dream_in_Red_Mansions", "LinDaiyu-zh"),
        ("A_Song_of_Ice_and_Fire", "TyrionLannister-en")
    ]
    
    results = []
    
    for source, role_code in selected_roles:
        role_data = runner.load_character(source, role_code)
        if not role_data: continue
        
        print(f"\n[Baseline] Character: {role_data.get('role_name')}")
        
        # Test 2 high-impact scenarios per role
        for scenario in runner.scenarios[:2]:
            print(f"  Scenario: {scenario.scenario_id}")
            
            resp, reflect = gen.generate_rag_reflect(role_data, scenario)
            
            # Evaluate using standard framework
            eval_res = runner.run_single_evaluation(
                role_data, scenario, resp, reflect, method="rag_reflect"
            )
            results.append(eval_res)
            print(f"    PC Score: {eval_res.pc_score:.2f} | SA: {eval_res.sa_score:.2f}")
            
    # Save
    output_dir = "experiments/experiment_results/baselines"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/rag_reflect_results_{timestamp}.json"
    
    # Convert set and other objects for JSON
    save_json_file(output_file, [vars(r) for r in results])
    
    print(f"\nBaseline results saved to: {output_file}")

if __name__ == "__main__":
    run_rag_reflect_baseline()
