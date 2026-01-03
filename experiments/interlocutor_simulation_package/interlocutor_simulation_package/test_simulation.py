import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import the module
sys.path.append(os.getcwd())

import run_long_dialogue_4way

class TestInterlocutorSimulation(unittest.TestCase):
    def setUp(self):
        # Mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        # Setup mock behavior for model/tokenizer if needed
        self.mock_tokenizer.apply_chat_template.return_value = "mock_prompt"
        self.mock_tokenizer.return_value = MagicMock() # for tokenizer([text])
        self.mock_model.generate.return_value = MagicMock()
        self.mock_tokenizer.batch_decode.return_value = ["Mocked response"]

    def test_simulator_init(self):
        simulator = run_long_dialogue_4way.InterlocutorSimulator(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            interlocutor_key="JiaBaoyu",
            interlocutor_role_name="贾宝玉",
            language="zh"
        )
        self.assertEqual(simulator.interlocutor_key, "JiaBaoyu")
        self.assertEqual(simulator.critical_turns, [5, 15, 20, 25])

    @patch('run_long_dialogue_4way.call_local_model')
    def test_generate_response_normal(self, mock_call_model):
        simulator = run_long_dialogue_4way.InterlocutorSimulator(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            interlocutor_key="JiaBaoyu",
            interlocutor_role_name="贾宝玉"
        )
        
        mock_call_model.return_value = "这是测试回复"
        
        history = [{"bot": "我是林黛玉", "user": "你好"}]
        response, is_shift = simulator.generate_response(
            turn_num=2, # Not critical
            char_role_name="林黛玉",
            char_response="我很好",
            history=history
        )
        
        self.assertEqual(response, "这是测试回复")
        self.assertFalse(is_shift)
        # Verify call_local_model was called with correct prompt structure
        args, _ = mock_call_model.call_args
        prompt = args[2]
        self.assertIn("林黛玉", prompt)
        self.assertIn("贾宝玉", prompt)
        self.assertIn("我很好", prompt)

    @patch('run_long_dialogue_4way.call_local_model')
    def test_generate_response_critical(self, mock_call_model):
        simulator = run_long_dialogue_4way.InterlocutorSimulator(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            interlocutor_key="JiaBaoyu",
            interlocutor_role_name="贾宝玉"
        )
        
        mock_call_model.return_value = "冲突回复"
        
        # Turn 4 (0-indexed) -> Turn 5 (1-indexed) which is critical
        response, is_shift = simulator.generate_response(
            turn_num=4, 
            char_role_name="林黛玉",
            char_response="我很好",
            history=[]
        )
        
        self.assertEqual(response, "冲突回复")
        self.assertTrue(is_shift)
        args, _ = mock_call_model.call_args
        prompt = args[2]
        # Check if instruction related to turn 5 is present
        self.assertIn("引入一个新的话题", prompt)

if __name__ == '__main__':
    unittest.main()
