"""
Model utility functions for ScrollWeaver.
"""

import os

MODEL_NAME_DICT = {
    "gpt-3.5": "openai/gpt-3.5-turbo",
    "gpt-4": "openai/gpt-4",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v3": "deepseek/deepseek-chat",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-1.5-flash": "google/gemini-flash-1.5",
    "llama3-70b": "meta-llama/llama-3.3-70b-instruct",
    "qwen-turbo": "qwen/qwen-turbo",
    "qwen-plus": "qwen/qwen-plus",
    "qwen-max": "qwen/qwen-max",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    "phi-4": "microsoft/phi-4",
}

PROJECT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_cache")
os.environ.setdefault("MODELSCOPE_CACHE", PROJECT_CACHE_DIR)
os.environ.setdefault("HF_HOME", PROJECT_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", PROJECT_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", PROJECT_CACHE_DIR)
os.makedirs(PROJECT_CACHE_DIR, exist_ok=True)


def get_models(model_name: str):
    """Get model instance based on model name."""
    if os.getenv("OPENROUTER_API_KEY", default="") and model_name in MODEL_NAME_DICT:
        from modules.llm.OpenRouter import OpenRouter
        return OpenRouter(model=MODEL_NAME_DICT[model_name])
    elif model_name.startswith('gpt'):
        # Use the alternative LangChainGPT2 which supports custom OPENAI_API_BASE
        from modules.llm.LangChainGPT2 import LangChainGPT
        if model_name.startswith('gpt-3.5'):
            return LangChainGPT(model="gpt-3.5-turbo")
        elif model_name == 'gpt-4' or model_name == 'gpt-4-turbo':
            return LangChainGPT(model="gpt-4")
        elif model_name == 'gpt-4o':
            return LangChainGPT(model="gpt-4o")
        elif model_name == "gpt-4o-mini":
            return LangChainGPT(model="gpt-4o-mini")
    elif model_name.startswith("claude"):
        from modules.llm.Claude import Claude
        if model_name.startswith("claude-3.5-sonnet"):
            return Claude(model="claude-3-5-sonnet-latest")
        elif model_name.startswith("claude-3.7-sonnet"):
            return Claude(model="claude-3-7-sonnet-latest")
        elif model_name.startswith("claude-3.5-haiku"):
            return Claude(model="claude-3-5-haiku-latest")
        return Claude()
    elif model_name.startswith('qwen'):
        from modules.llm.Qwen import Qwen
        return Qwen(model=model_name)
    elif model_name.startswith('deepseek'):
        from modules.llm.DeepSeek import DeepSeek
        return DeepSeek(model=model_name)
    elif model_name.startswith('doubao'):
        from modules.llm.Doubao import Doubao
        return Doubao()
    elif model_name.startswith('gemini'):
        # Prefer Vertex Gemini when user indicates so via model prefix or env vars
        use_vertex_env = os.getenv("USE_VERTEX_GEMINI", "").lower() in ["1", "true", "yes"]
        has_google_creds = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "") or os.getenv("GOOGLE_CLOUD_PROJECT", ""))
        is_vertex_prefix = model_name.startswith('vertex-gemini')

        # 对于新模型（gemini-2.5-flash, gemini-2.5-flash-lite等），强制使用标准Gemini客户端
        # 因为VertexGemini不支持这些新模型的结构化输出功能
        is_new_model = any(x in model_name for x in ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.5-pro'])
        
        # 只有在明确指定vertex-gemini前缀或USE_VERTEX_GEMINI环境变量时才使用VertexGemini
        # 对于新模型，即使有Google凭证也强制使用标准Gemini客户端
        should_use_vertex = (is_vertex_prefix or use_vertex_env) and not is_new_model
        
        print(f"[model_utils] 模型选择: {model_name}, is_new_model={is_new_model}, should_use_vertex={should_use_vertex}, has_google_creds={has_google_creds}")
        
        if should_use_vertex:
            try:
                from modules.llm.VertexGemini2 import VertexGemini
                # allow model names like 'vertex-gemini:gemini-1.5-pro-002' or 'vertex-gemini/gemini-2.0'
                # parse after separator if provided
                parsed_model = model_name
                if ':' in model_name:
                    parsed_model = model_name.split(':', 1)[1]
                elif '/' in model_name:
                    parsed_model = model_name.split('/', 1)[1]

                # fall back to a sensible default if parsing produces empty
                if not parsed_model or parsed_model == 'vertex-gemini':
                    parsed_model = 'gemini-1.5-pro-002'

                return VertexGemini(model=parsed_model)
            except Exception as e:
                # if Vertex client import fails, fall back to existing Gemini wrapper
                print(f"VertexGemini import failed ({e}), falling back to generic Gemini client")

        # Use Google Gemini SDK (default)
        from modules.llm.Gemini import Gemini
        return Gemini(model=model_name)
    else:
        print(f'Warning! undefined model {model_name}, use gpt-4o-mini instead.')
        from modules.llm.LangChainGPT import LangChainGPT
        return LangChainGPT()


def build_db(data, db_name, db_type, embedding, save_type="persistent"):
    """Build database from data."""
    if not data or not db_name:
        return None
    from modules.db.ChromaDB import ChromaDB
    db = ChromaDB(embedding, save_type)
    db.init_from_data(data, db_name)
    return db


def build_orchestrator_data(world_file_path: str, max_words: int = 30):
    """Build orchestrator data from world file."""
    from .file_utils import get_child_paths, load_text_file, load_jsonl_file
    from .text_utils import split_text_by_max_words
    
    world_dir = os.path.dirname(world_file_path)
    details_dir = os.path.join(world_dir, "./world_details")
    data = []
    settings = []
    if os.path.exists(details_dir):
        for path in get_child_paths(details_dir):
            if os.path.splitext(path)[-1] == ".txt":
                text = load_text_file(path)
                data += split_text_by_max_words(text, max_words)
            if os.path.splitext(path)[-1] == ".jsonl":
                jsonl = load_jsonl_file(path)
                data += [f"{dic['term']}:{dic['detail']}" for dic in jsonl]
                settings += jsonl
    return data, settings

