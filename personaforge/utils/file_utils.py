"""
File utility functions for ScrollWeaver.
"""

import os
import json


def load_text_file(path: str) -> str:
    """Load text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def save_text_file(path: str, target: str) -> None:
    """Save text to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(target)


def load_json_file(path: str):
    """Load JSON from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, target) -> None:
    """Save JSON to a file."""
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=True)


def load_jsonl_file(path: str) -> list:
    """Load JSONL from a file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl_file(path: str, target: list) -> None:
    """Save JSONL to a file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in target:
            print(json.dumps(row, ensure_ascii=False), file=f)


def get_child_paths(root_folder: str, if_full: bool = True) -> list:
    """Get all file paths in a directory."""
    paths = []
    for resource in os.listdir(root_folder):
        if if_full:
            path = os.path.join(root_folder, resource)
            if os.path.isfile(path):
                paths.append(path)
        else:
            path = resource
            if os.path.isfile(os.path.join(root_folder, path)):
                paths.append(path)
    return paths


def get_child_folders(root_folder: str, if_full: bool = True) -> list:
    """Get all folder paths in a directory."""
    folders = []
    for resource in os.listdir(root_folder):
        if if_full:
            path = os.path.join(root_folder, resource)
            if os.path.isdir(path):
                folders.append(path)
        else:
            path = resource
            if os.path.isdir(os.path.join(root_folder, path)):
                folders.append(path)
    return folders


def get_grandchild_folders(root_folder: str, if_full: bool = True) -> list:
    """Get all folder paths in subdirectories."""
    folders = []
    for resource in os.listdir(root_folder):
        subpath = os.path.join(root_folder, resource)
        if os.path.isdir(subpath):
            for folder_name in os.listdir(subpath):
                folder_path = os.path.join(subpath, folder_name)
                if os.path.isdir(folder_path):
                    if if_full:
                        folders.append(folder_path)
                    else:
                        folders.append(folder_name)
    return folders


def find_files_with_suffix(directory: str, suffix: str) -> list:
    """Find all files with a specific suffix in a directory tree."""
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                matched_files.append(os.path.join(root, file))
    return matched_files


def create_dir(dirname: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_root_dir() -> str:
    """Get the root directory of the project."""
    current_file_path = os.path.abspath(__file__)
    # Go up to the project root (from modules/utils/file_utils.py to project root)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    return root_dir


def remove_list_elements(list1: list, *args) -> list:
    """Remove elements from a list."""
    for target in args:
        if isinstance(target, list) or isinstance(target, dict):
            list1 = [i for i in list1 if i not in target]
        else:
            list1 = [i for i in list1 if i != target]
    return list1
