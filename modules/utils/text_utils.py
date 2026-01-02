"""
Text utility functions for ScrollWeaver.
"""

import re
import json
import base64


def conceal_thoughts(detail: str) -> str:
    """Remove thoughts (text in brackets) from detail."""
    text = re.sub(r'【.*?】', '', detail)
    text = re.sub(r'\[.*?\]', '', text)
    return text


def action_detail_decomposer(detail: str) -> tuple:
    """Decompose action detail into thoughts, actions, and dialogues."""
    thoughts = re.findall(r'【(.*?)】', detail)
    actions = re.findall(r'（(.*?)）', detail)
    dialogues = re.findall(r'「(.*?)」', detail)
    return thoughts, actions, dialogues


def merge_text_with_limit(text_list: list, max_words: int, language: str = 'en') -> str:
    """
    Merge a list of text strings into one, stopping when adding another text exceeds the maximum count.

    Args:
        text_list (list): List of strings to be merged.
        max_words (int): Maximum number of characters (for Chinese) or words (for English).
        language (str): Language code ('zh' for Chinese, 'en' for English).

    Returns:
        str: The merged text, truncated as needed.
    """
    merged_text = ""
    current_count = 0

    for text in text_list:
        if language == 'zh':
            # Count Chinese characters
            text_length = len(text)
        else:
            # Count English words
            text_length = len(text.split(" "))

        if current_count + text_length > max_words:
            break

        merged_text += text + "\n"
        current_count += text_length

    return merged_text


def normalize_string(text: str) -> str:
    """Normalize string by removing whitespace and converting to lowercase."""
    return re.sub(r'[\s\,\;\t\n]+', '', text).lower()


def fuzzy_match(str1: str, str2: str, threshold: float = 0.8) -> bool:
    """Check if two strings match (normalized comparison)."""
    str1_normalized = normalize_string(str1)
    str2_normalized = normalize_string(str2)

    if str1_normalized == str2_normalized:
        return True

    return False


def split_text_by_max_words(text: str, max_words: int = 30) -> list:
    """Split text into segments with maximum word count."""
    segments = []
    current_segment = []
    current_length = 0

    lines = text.splitlines()

    for line in lines:
        words_in_line = len(line)
        current_segment.append(line + '\n')
        current_length += words_in_line

        if current_length + words_in_line > max_words:
            segments.append(''.join(current_segment))
            current_segment = []
            current_length = 0

    if current_segment:
        segments.append(''.join(current_segment))

    return segments


def lang_detect(text: str) -> str:
    """Detect language of text."""
    def count_chinese_characters(text: str) -> int:
        # 使用正则表达式匹配所有汉字字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars)

    if count_chinese_characters(text) > len(text) * 0.05:
        lang = 'zh'
    else:
        lang = 'en'
    return lang


def extract_first_number(text: str):
    """Extract the first number from text."""
    match = re.search(r'\b\d+(?:\.\d+)?\b', text)
    return int(match.group()) if match else None


def json_parser(output: str):
    """Parse JSON from text output."""
    # 首先尝试移除markdown代码块标记（新版API可能返回这些）
    original_output = output
    if "```json" in output:
        output = output.split("```json")[1].split("```")[0].strip()
    elif "```" in output:
        # 移除代码块标记
        parts = output.split("```")
        if len(parts) >= 3:
            output = parts[1].strip()
            # 移除可能的语言标识符（如 "json"）
            if output.startswith("json"):
                output = output[4:].strip()
    
    output = output.replace("\n", "")
    output = output.replace("\t", "")
    if "{" not in output:
        output = "{" + output
    if "}" not in output:
        output += "}"
    pattern = r'\{.*\}'
    matches = re.findall(pattern, output, re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON object found in output. Original: {original_output[:200]}")
    
    try:
        parsed_json = eval(matches[0])
    except:
        try:
            parsed_json = json.loads(matches[0])
        except json.JSONDecodeError:
            try:
                detail = re.search(r'"detail":\s*(.+?)\s*}', matches[0]).group(1)
                detail = f"\"{detail}\""
                new_output = re.sub(r'"detail":\s*(.+?)\s*}', f"\"detail\":{detail}}}", matches[0])
                parsed_json = json.loads(new_output)
            except Exception as e:
                raise ValueError(f"No valid JSON found in the input string. Error: {e}. Original output: {original_output[:500]}")
    return parsed_json


def clean_collection_name(name: str) -> str:
    """Clean collection name for database."""
    cleaned_name = name.replace(' ', '_')
    cleaned_name = cleaned_name.replace('.', '_')
    if not all(ord(c) < 128 for c in cleaned_name):
        encoded = base64.b64encode(cleaned_name.encode('utf-8')).decode('ascii')
        encoded = encoded[:60] if len(encoded) > 60 else encoded
        valid_name = f"mem_{encoded}"
    else:
        valid_name = cleaned_name
    valid_name = re.sub(r'[^a-zA-Z0-9_-]', '-', valid_name)
    valid_name = re.sub(r'\.\.+', '-', valid_name)
    valid_name = re.sub(r'^[^a-zA-Z0-9]+', '', valid_name)  # 移除开头非法字符
    valid_name = re.sub(r'[^a-zA-Z0-9]+$', '', valid_name)
    valid_name = valid_name[:60]
    return valid_name


def remove_markdown(text: str) -> str:
    """
    移除文本中的 Markdown 格式标记。
    
    Args:
        text: 需要清理的文本
        
    Returns:
        清理后的纯文本
    """
    if not text:
        return text
    
    # 移除代码块标记
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'```', '', text)
    
    # 移除行内代码标记
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # 移除粗体和斜体标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **粗体**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # *斜体*
    text = re.sub(r'__([^_]+)__', r'\1', text)  # __粗体__
    text = re.sub(r'_([^_]+)_', r'\1', text)  # _斜体_
    
    # 移除标题标记
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # 移除链接标记 [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 移除图片标记 ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    
    # 移除列表标记
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # 移除引用标记
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # 移除水平线（但保留包含文本的行，如 "--------- Current Event ---------"）
    # 只移除完全由 -、* 或 _ 组成的行
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # 清理多余的空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()