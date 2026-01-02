from experiments.run_experiment import BaselineGenerator, EvaluationScenario
from modules.llm.Gemini import Gemini
import os

# Mock objects
class MockLLM:
    def chat(self, prompt):
        return "Simulated LLM Response"

class MockScenario:
    context = "Context"
    trigger = "Trigger"
    trigger_role = "User"
    scenario_id = "test_1"
    scenario_type = "emotional"

scenario = MockScenario()
role_data = {
    "role_name": "TestRole", 
    "profile": "Test Profile",
    "style_examples": [{"action": "Act", "response": "Res"}]
}

# Test
try:
    gen = BaselineGenerator(llm=MockLLM())
    
    # Test RAG
    print("Testing RAG-Persona...")
    res_rag = gen.generate_rag_persona(role_data, scenario)
    print(f"RAG Result: {res_rag}")
    assert "Simulated" in res_rag
    
    # Test Structured-CoT
    print("Testing Structured-CoT...")
    res_cot = gen.generate_structured_cot(role_data, scenario)
    print(f"CoT Result: {res_cot}")
    assert "Simulated" in res_cot
    
    print("\n✓ All new baselines verified successfully!")
except Exception as e:
    print(f"\n❌ Verification failed: {e}")
    exit(1)
