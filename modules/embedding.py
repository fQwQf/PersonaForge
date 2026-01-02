import sys
sys.path.append("../")
from chromadb.api.types import Embeddings, Documents, EmbeddingFunction, Space
from modelscope import AutoModel, AutoTokenizer
from functools import partial
from sw_utils import get_child_folders
import torch
import os
import hashlib
import random
from typing import Iterable, List

# 全局缓存：避免重复加载相同的embedding模型
_embedding_model_cache = {}


class EmbeddingModel(EmbeddingFunction[Documents]):
    def __init__(self, model_name, language='en'):
        self.model_name = model_name
        self.language = language
        self.tokenizer = None
        self.model = None
        self._fallback = False
        self._fallback_dim = 384

        try:
            # 获取缓存目录
            cache_dir = os.path.expanduser(os.environ.get("MODELSCOPE_CACHE", "~/.cache/modelscope/hub"))
            
            print(f"\n{'='*60}")
            print(f"[Embedding] Loading model: {model_name}")
            print(f"[Embedding] Cache directory: {cache_dir}")
            
            # ModelScope使用 models/PROVIDER/MODEL_NAME 格式存储
            # 例如: models/BAAI/bge-small-zh
            local_model_path = None
            if "/" in model_name:
                # ModelScope格式: models/PROVIDER/MODEL_NAME
                modelscope_path = os.path.join(cache_dir, "models", model_name)
                if os.path.exists(modelscope_path):
                    local_model_path = modelscope_path
                    print(f"[Embedding] ✓ Found cached model at: {modelscope_path}")
            
            # 如果找到本地缓存，直接加载（不联网）
            if local_model_path:
                print(f"[Embedding] Loading from local cache (offline mode)...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                print(f"[Embedding] ✓ Successfully loaded from cache")
            else:
                # 缓存不存在，从远程下载（会自动保存到cache_dir）
                print(f"[Embedding] Cache not found, downloading...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                print(f"[Embedding] ✓ Successfully downloaded and cached")
            
            print(f"{'='*60}\n")
            
        except Exception as exc:
            print(f"[Embedding] ✗ Failed to load '{model_name}': {exc}")
            print(f"[Embedding] Falling back to hash-based embeddings")
            print(f"{'='*60}\n")
            self._fallback = True

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            return self.embed_query(input)
        return self.embed_documents(list(input))

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if self._fallback or self.tokenizer is None or self.model is None:
            return [self._hash_embed(text) for text in texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].tolist()
        return embeddings

    def embed_query(self, text: str = None, **kwargs) -> List[float]:
        # Handle case where 'input' keyword argument is passed instead of positional
        if text is None and 'input' in kwargs:
            text = kwargs['input']
        if text is None:
            raise ValueError("embed_query requires 'text' or 'input' argument")
        return self.embed_documents([text])[0]

    def _hash_embed(self, text: str) -> List[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        return [rng.random() for _ in range(self._fallback_dim)]

class OpenAIEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name="text-embedding-ada-002", base_url=None, api_key_field = "OPENAI_API_KEY"):
        from openai import OpenAI
        # allow overriding base URL via OPENAI_API_BASE environment variable (e.g. apiyi mirror)
        env_base = os.getenv('OPENAI_API_BASE', '')
        if env_base:
            # If env provided, it should point to the API root (e.g. https://api.apiyi.com/v1)
            base_url = env_base
        # default: base_url may be None, OpenAI client will use default
        client_kwargs = {
            'api_key': os.environ.get(api_key_field, None)
        }
        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name

    def __call__(self, input):
        if isinstance(input, str):
            input = input.replace("\n", " ")
            return self.client.embeddings.create(input=[input], model=self.model_name).data[0].embedding
        elif isinstance(input,list):
            return [self.client.embeddings.create(input=[sentence.replace("\n", " ")], model=self.model_name).data[0].embedding for sentence in input]

def get_embedding_model(embed_name, language='en'):
    """
    获取embedding模型实例（带缓存机制，避免重复加载）
    
    Args:
        embed_name: 模型名称
        language: 语言设置
        
    Returns:
        EmbeddingModel或OpenAIEmbedding实例
    """
    # 创建缓存键
    cache_key = f"{embed_name}_{language}"
    
    # 如果缓存中存在，直接返回缓存的实例
    if cache_key in _embedding_model_cache:
        return _embedding_model_cache[cache_key]
    
    local_model_dict = {
        "bge-m3":"BAAI/bge-m3",
        "bge-large": f"BAAI/bge-large-{language}",
        "luotuo": "silk-road/luotuo-bert-medium",
        "bert": "google-bert/bert-base-multilingual-cased",
        "bge-small": f"BAAI/bge-small-{language}",
    }
    online_model_dict = {
        "openai":
            {"model_name":"text-embedding-ada-002",
             "url":"https://api.apiyi.com/v1/embeddings",
             "api_key_field":"OPENAI_API_KEY"},

    }
    
    # 创建模型实例
    if embed_name in local_model_dict:
        model_name = local_model_dict[embed_name]
        embedding = EmbeddingModel(model_name, language=language)
    elif embed_name in online_model_dict:
        model_name = online_model_dict[embed_name]["model_name"]
        api_key_field = online_model_dict[embed_name]["api_key_field"]
        base_url = online_model_dict[embed_name]["url"]
        embedding = OpenAIEmbedding(model_name=model_name, base_url=base_url, api_key_field=api_key_field)
    else:
        embedding = EmbeddingModel(embed_name, language=language)
    
    # 缓存模型实例
    _embedding_model_cache[cache_key] = embedding
    
    return embedding
