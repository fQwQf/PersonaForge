import os
import json
import csv
from typing import Dict, List, Any, Optional
from sw_utils import load_json_file, save_json_file

class MapDataManager:
    """管理地图数据的存储与访问，支持新旧目录结构适配。"""
    
    BASE_DATA_DIR = "data"
    
    @classmethod
    def get_world_dir(cls, source_name: str) -> str:
        """获取世界所在的目录"""
        return os.path.join(cls.BASE_DATA_DIR, "worlds", source_name)

    @classmethod
    def get_map_dir(cls, source_name: str) -> str:
        """获取地图资源目录"""
        return os.path.join(cls.get_world_dir(source_name), "map")

    @classmethod
    def get_layout_path(cls, source_name: str) -> str:
        """获取 layout.json 路径"""
        return os.path.join(cls.get_map_dir(source_name), "layout.json")

    @classmethod
    def get_background_path(cls, source_name: str) -> str:
        """获取背景图路径"""
        return os.path.join(cls.get_map_dir(source_name), "background.png")

    @classmethod
    def load_map_data(cls, source_name: str) -> Dict[str, Any]:
        """
        加载完整的地图数据（布局、建筑物、距离等）。
        优先从新结构加载，否则尝试从旧结构迁移/加载。
        """
        layout_path = cls.get_layout_path(source_name)
        
        if os.path.exists(layout_path):
            return load_json_file(layout_path)
        
        # 降级尝试：从旧文件读取并尝试返回统一格式
        return cls._load_legacy_map_data(source_name)

    @classmethod
    def _load_legacy_map_data(cls, source_name: str) -> Dict[str, Any]:
        """从旧的目录结构加载数据并转换为新格式"""
        buildings_file = os.path.join(cls.BASE_DATA_DIR, "maps", f"{source_name}_buildings.json")
        csv_file = os.path.join(cls.BASE_DATA_DIR, "maps", f"{source_name}.csv")
        locations_file = os.path.join(cls.BASE_DATA_DIR, "locations", f"{source_name}.json")
        
        data = {
            "metadata": {
                "grid": {"cols": 24, "rows": 12},
                "background": "background.png"
            },
            "locations": []
        }
        
        # 1. 加载建筑物坐标
        buildings_dict = {}
        if os.path.exists(buildings_file):
            buildings_raw = load_json_file(buildings_file)
            if isinstance(buildings_raw, dict) and "buildings" in buildings_raw:
                for b in buildings_raw["buildings"]:
                    buildings_dict[b["building_code"]] = b

        # 2. 加载地点详情
        locations_info = {}
        if os.path.exists(locations_file):
            locations_info = load_json_file(locations_file)
            if "locations" in locations_info:
                locations_info = locations_info["locations"]

        # 3. 加载距离矩阵并合并
        adjacencies = {} # code -> list of {to, distance}
        if os.path.exists(csv_file):
            try:
                with open(csv_file, mode='r', encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader)[1:]
                    for row in reader:
                        loc1 = row[0]
                        distances = row[1:]
                        adj = []
                        for i, d in enumerate(distances):
                            if i < len(header):
                                loc2 = header[i]
                                if d != '0' and d.isdigit():
                                    adj.append({"to": loc2, "distance": int(d)})
                        adjacencies[loc1] = adj
            except Exception as e:
                print(f"Error loading legacy CSV: {e}")

        # 合并所有数据
        all_codes = set(list(buildings_dict.keys()) + list(locations_info.keys()) + list(adjacencies.keys()))
        
        for code in all_codes:
            loc_data = {
                "code": code,
                "name": locations_info.get(code, {}).get("location_name", code),
                "description": locations_info.get(code, {}).get("description", ""),
                "detail": locations_info.get(code, {}).get("detail", ""),
                "view_config": buildings_dict.get(code, {}).get("coordinates", {}),
                "adjacencies": adjacencies.get(code, [])
            }
            data["locations"].append(loc_data)
            
        return data

    @classmethod
    def save_map_data(cls, source_name: str, data: Dict[str, Any]):
        """保存地图数据到新结构"""
        layout_path = cls.get_layout_path(source_name)
        save_json_file(layout_path, data)
        print(f"Map layout saved to {layout_path}")

    @classmethod
    def migrate_all(cls):
        """执行全量迁移"""
        maps_dir = os.path.join(cls.BASE_DATA_DIR, "maps")
        if not os.path.exists(maps_dir):
            return
            
        # 找到所有 csv 文件作为 source 的基准
        sources = []
        for f in os.listdir(maps_dir):
            if f.endswith(".csv") and not f.endswith("_buildings.json"):
                sources.append(f[:-4])
                
        for source in sources:
            print(f"Migrating {source}...")
            data = cls._load_legacy_map_data(source)
            
            # 确保目录存在
            os.makedirs(cls.get_map_dir(source), exist_ok=True)
            
            # 保存新 layout
            cls.save_map_data(source, data)
            
            # 尝试移动背景图 (如果存在旧的约定位置)
            # 这里简单处理：如果 data/maps/{source}/background.png 存在，则移动
            old_bg_dir = os.path.join(cls.BASE_DATA_DIR, "maps", source)
            old_bg = os.path.join(old_bg_dir, "background.png")
            if os.path.exists(old_bg):
                import shutil
                shutil.copy(old_bg, cls.get_background_path(source))
                print(f"Copied background for {source}")

