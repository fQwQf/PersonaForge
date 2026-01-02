"""
语言风格向量数据库
用于存储和检索用户的历史发言，支持风格检索和Few-Shot样本提取
"""
import os
import sys
sys.path.append("../")
from typing import List, Dict, Any, Optional
from sw_utils import build_db
from modules.embedding import get_embedding_model


class StyleVectorDB:
    """语言风格向量数据库"""
    
    def __init__(self, 
                 db_name: str,
                 embedding_name: str = "bge-small",
                 db_type: str = "chroma",
                 language: str = "zh"):
        """
        初始化风格向量数据库
        
        Args:
            db_name: 数据库名称
            embedding_name: 嵌入模型名称
            db_type: 数据库类型（chroma等）
            language: 语言设置
        """
        self.db_name = db_name
        self.db_type = db_type
        self.language = language
        
        # 获取嵌入模型
        if embedding_name:
            self.embedding = get_embedding_model(embedding_name, language=language)
        else:
            self.embedding = None
        
        # 初始化数据库（空数据库）
        self.db = build_db(
            data=[],
            db_name=db_name,
            db_type=db_type,
            embedding=self.embedding,
            save_type="persistent"
        )
        
        # 存储发言元数据（用于Few-Shot提取）
        self.utterances: List[Dict[str, Any]] = []  # 格式：{"text": "...", "context": "...", "timestamp": ...}
    
    def add_utterance(self, text: str, context: str = "", metadata: Optional[Dict[str, Any]] = None):
        """
        添加一条发言到数据库
        
        Args:
            text: 发言内容
            context: 上下文（对话背景）
            metadata: 额外的元数据
        """
        if not text or not text.strip():
            return
        
        # 添加到向量数据库
        self.db.add(text, db_name=self.db_name)
        
        # 保存元数据
        utterance_data = {
            "text": text,
            "context": context,
            "timestamp": metadata.get("timestamp") if metadata else None
        }
        if metadata:
            utterance_data.update(metadata)
        self.utterances.append(utterance_data)
    
    def add_utterances_batch(self, utterances: List[Dict[str, str]]):
        """
        批量添加发言
        
        Args:
            utterances: 发言列表，格式：[{"text": "...", "context": "..."}, ...]
        """
        texts = []
        for utt in utterances:
            if utt.get("text") and utt["text"].strip():
                texts.append(utt["text"])
                self.utterances.append({
                    "text": utt["text"],
                    "context": utt.get("context", ""),
                    "timestamp": utt.get("timestamp")
                })
        
        if texts:
            # 批量添加到向量数据库
            for text in texts:
                self.db.add(text, db_name=self.db_name)
    
    def search_similar_style(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索相似风格的发言
        
        Args:
            query: 查询文本（话题或场景）
            top_k: 返回最相似的k条
        
        Returns:
            相似发言列表，格式：[{"text": "...", "context": "...", "score": ...}, ...]
        """
        if not query or not query.strip():
            return []
        
        # 从向量数据库检索
        similar_texts = self.db.search(query, top_k, self.db_name)
        
        # 匹配元数据
        results = []
        for text in similar_texts:
            # 找到对应的元数据
            for utt in self.utterances:
                if utt["text"] == text:
                    results.append({
                        "text": utt["text"],
                        "context": utt.get("context", ""),
                        "score": 1.0  # 简化处理，实际可以计算相似度分数
                    })
                    break
        
        return results
    
    def extract_few_shot_examples(self, 
                                 topic: Optional[str] = None,
                                 num_examples: int = 5) -> List[Dict[str, str]]:
        """
        提取Few-Shot样本
        
        Args:
            topic: 话题（可选，如果提供则检索相关话题的样本）
            num_examples: 提取的样本数量
        
        Returns:
            Few-Shot样本列表，格式：[{"context": "...", "response": "..."}, ...]
        """
        if topic:
            # 如果提供了话题，检索相关样本
            similar = self.search_similar_style(topic, top_k=num_examples * 2)
            examples = []
            for item in similar[:num_examples]:
                examples.append({
                    "context": item.get("context", ""),
                    "response": item["text"]
                })
            return examples
        else:
            # 否则随机选择（或选择最近的）
            examples = []
            for utt in self.utterances[-num_examples:]:
                examples.append({
                    "context": utt.get("context", ""),
                    "response": utt["text"]
                })
            return examples
    
    def get_all_utterances(self) -> List[Dict[str, Any]]:
        """获取所有发言"""
        return self.utterances.copy()
    
    def clear(self):
        """清空数据库"""
        self.utterances = []
        # 注意：向量数据库的清空需要根据具体实现来处理
        # 这里简化处理，实际使用时可能需要重新创建数据库


def create_style_db_from_chat_history(chat_history: List[str],
                                     db_name: str,
                                     embedding_name: str = "bge-small",
                                     db_type: str = "chroma",
                                     language: str = "zh") -> StyleVectorDB:
    """
    从聊天记录创建风格向量数据库
    
    Args:
        chat_history: 聊天记录列表（每条是一个发言）
        db_name: 数据库名称
        embedding_name: 嵌入模型名称
        db_type: 数据库类型
        language: 语言设置
    
    Returns:
        StyleVectorDB实例
    """
    db = StyleVectorDB(db_name=db_name, 
                      embedding_name=embedding_name,
                      db_type=db_type,
                      language=language)
    
    # 将聊天记录转换为utterances格式
    utterances = []
    for i, text in enumerate(chat_history):
        if i > 0:
            context = chat_history[i-1]  # 上一条作为上下文
        else:
            context = ""
        
        utterances.append({
            "text": text,
            "context": context
        })
    
    db.add_utterances_batch(utterances)
    return db

