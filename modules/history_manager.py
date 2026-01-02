from typing import Any, Dict, List, Optional, Literal
from collections import defaultdict
from datetime import datetime
from sw_utils import load_json_file, save_json_file, load_jsonl_file, save_jsonl_file
import os


class HistoryManager:
    def __init__(self):
        self.detailed_history = []

    def add_record(self, record):
        """添加一个事件记录
        record = {
            "cur_round":cur_round,
            "role_code":role_code,
            "detail":detail,
            "type":act_type,
            "initiator":initiator,
            "actor":actor
            "group":group,
            "other_info":other_info,
            "record_id":record_id
        }
        """
        self.detailed_history.append(record)

    def modify_record(self, record_id: str, detail: str):
        """修改特定记录"""
        for record in self.detailed_history:
            if record["record_id"] == record_id:
                record["detail"] = detail
                print(f"Record {record_id} has been modified.")
                return record['group']
    
    def search_record_detail(self, record_id: str):
        for record in self.detailed_history[::-1]:
            if record["record_id"] == record_id:
                return record["detail"]
        return None
    
    def get_recent_history(self, recent_k = 5):
        return [record["detail"] for record in self.detailed_history[-recent_k:]]
    
    def get_subsequent_history(self,start_idx):
        return [record["detail"] for record in self.detailed_history[start_idx:]]
    
    def get_complete_history(self,):
        return [record["detail"] for record in self.detailed_history[:]]
    
    def __len__(self):
        return len(self.detailed_history)
    
    def __getstate__(self):
        states = {key: value for key, value in self.__dict__.items() \
            if isinstance(value, (str, int, list, dict, bool, type(None)))}
        return states

    def __setstate__(self, states):
        self.__dict__.update(states)

    def save_to_file(self, root_dir):
        filename = os.path.join(root_dir, f"./simulation_history.json")
        save_json_file(filename, self.__getstate__() )

    def load_from_file(self, root_dir):
        filename = os.path.join(root_dir, f"./simulation_history.json")
        states = load_json_file(filename)
        self.__setstate__(states)     


