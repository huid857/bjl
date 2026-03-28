"""
数据管理模块
负责数据的输入、存储、加载和管理
"""

import json
import os
from datetime import datetime
from utils import normalize_input, validate_data


class DataManager:
    """数据管理器"""
    
    def __init__(self, data_dir='data'):
        """
        初始化数据管理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.history_file = os.path.join(data_dir, 'history.json')
        self.current_shoe = []  # 当前靴牌数据
        self.all_history = []   # 所有历史数据（跨靴牌）
        self.shoes = []         # 所有靴牌列表（用于靴内分析）
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 加载历史数据
        self.load_history()
    
    def add_result(self, result):
        """
        添加单个结果
        
        Args:
            result: 'B', 'P', 或 'T'
            
        Returns:
            是否添加成功
        """
        result = result.upper()
        if result not in ['B', 'P', 'T']:
            return False
        
        self.current_shoe.append(result)
        return True
    
    def add_batch(self, text):
        """
        批量添加数据
        
        Args:
            text: 文本数据（支持多种格式）
            
        Returns:
            (是否成功, 添加的数量)
        """
        data = normalize_input(text)
        
        if not data:
            return False, 0
        
        valid, msg = validate_data(data)
        if not valid:
            return False, 0
        
        self.current_shoe.extend(data)
        return True, len(data)
    
    def get_current_shoe(self):
        """获取当前靴牌数据"""
        return self.current_shoe.copy()
    
    def get_all_data(self):
        """获取所有数据（当前+历史）"""
        return self.all_history + self.current_shoe
    
    def get_shoes(self):
        """
        获取所有靴牌列表（用于靴内分析）
        
        Returns:
            靴牌列表，格式为:
            [
                {'name': 'shoe_xxx', 'date': '...', 'count': 54, 'data': 'PPBTP...'},
                ...
            ]
        """
        return self.shoes.copy()
    
    def save_current_shoe(self, shoe_name=None):
        """
        保存当前靴牌到历史记录
        
        Args:
            shoe_name: 靴牌名称（可选）
            
        Returns:
            是否保存成功
        """
        if not self.current_shoe:
            return False
        
        # 生成靴牌名称
        if shoe_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            shoe_name = f"shoe_{timestamp}"
        
        # 加载现有历史
        history_data = self._load_history_file()
        
        # 添加当前靴牌
        shoe_data = {
            'name': shoe_name,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'count': len(self.current_shoe),
            'data': ''.join(self.current_shoe)
        }
        
        history_data['shoes'].append(shoe_data)
        history_data['total_count'] = history_data.get('total_count', 0) + len(self.current_shoe)
        history_data['total_shoes'] = len(history_data['shoes'])
        
        # 保存到文件
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            # 更新历史数据
            self.all_history.extend(self.current_shoe)
            
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def clear_current_shoe(self):
        """清空当前靴牌数据"""
        self.current_shoe = []
    
    def new_shoe(self, save_current=True):
        """
        开始新靴牌
        
        Args:
            save_current: 是否保存当前靴牌
            
        Returns:
            是否成功
        """
        if save_current and self.current_shoe:
            self.save_current_shoe()
        
        self.current_shoe = []
        return True
    
    def load_history(self):
        """加载历史数据"""
        history_data = self._load_history_file()
        
        # 保存靴牌列表（用于靴内分析）
        self.shoes = history_data.get('shoes', [])
        
        # 合并所有历史靴牌
        self.all_history = []
        for shoe in self.shoes:
            shoe_results = list(shoe['data'])
            self.all_history.extend(shoe_results)
    
    def _load_history_file(self):
        """
        加载历史文件
        
        Returns:
            历史数据字典
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # 返回空数据结构
        return {
            'shoes': [],
            'total_count': 0,
            'total_shoes': 0
        }
    
    def get_statistics(self):
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        history_data = self._load_history_file()
        
        return {
            'current_shoe_count': len(self.current_shoe),
            'history_count': len(self.all_history),
            'total_count': len(self.all_history) + len(self.current_shoe),
            'total_shoes': history_data.get('total_shoes', 0),
            'has_enough_for_ml': (len(self.all_history) + len(self.current_shoe)) >= 300
        }
    
    def delete_last_result(self):
        """
        删除最后一个结果（撤销功能）
        
        Returns:
            是否成功
        """
        if self.current_shoe:
            self.current_shoe.pop()
            return True
        return False
    
    def import_from_file(self, filepath):
        """
        从文件导入数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            (是否成功, 导入数量)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.add_batch(content)
        except Exception as e:
            print(f"导入失败: {e}")
            return False, 0
    
    def export_current_shoe(self, filepath):
        """
        导出当前靴牌数据
        
        Args:
            filepath: 导出文件路径
            
        Returns:
            是否成功
        """
        if not self.current_shoe:
            return False
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(''.join(self.current_shoe))
            return True
        except Exception as e:
            print(f"导出失败: {e}")
            return False

