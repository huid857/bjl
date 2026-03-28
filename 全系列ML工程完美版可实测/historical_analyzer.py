"""
历史模式分析器
基于历史数据进行深度模式分析（只在靴内统计，不跨靴牌边界）
"""

from collections import defaultdict


class HistoricalPatternAnalyzer:
    """
    历史模式分析器
    
    核心原则：
    - 只在单靴内统计模式，不跨靴牌边界
    - 符合百家乐本质（每靴独立）
    - 避免虚假模式
    """
    
    def __init__(self, shoes):
        """
        初始化分析器
        
        Args:
            shoes: 靴牌列表，格式为:
                [
                    {'name': 'shoe_xxx', 'date': '...', 'count': 54, 'data': 'PPBTP...'},
                    ...
                ]
        """
        self.shoes = shoes
        self.total_shoes = len(shoes)
        self.total_rounds = sum(shoe['count'] for shoe in shoes)
        
        # 缓存分析结果
        self._streak_cache = {}
        self._sequence_cache = {}
    
    def analyze_streak_pattern(self, streak_type, streak_length):
        """
        分析连续模式：例如"庄连4次后，下一局出什么？"
        
        只在靴内统计！
        
        Args:
            streak_type: 'B' 或 'P'
            streak_length: 连续次数，如 3, 4, 5...
            
        Returns:
            {
                'B': 概率,
                'P': 概率,
                'T': 概率,
                'sample_count': 样本数量,
                'confidence': 置信度,
                'method': 'streak_within_shoe'
            }
        """
        cache_key = f"{streak_type}_{streak_length}"
        
        if cache_key in self._streak_cache:
            return self._streak_cache[cache_key]
        
        results = {'B': 0, 'P': 0, 'T': 0}
        sample_count = 0
        
        # 遍历每一靴
        for shoe in self.shoes:
            shoe_data = list(shoe['data'])
            
            # 在单靴内查找连续模式
            i = 0
            while i <= len(shoe_data) - streak_length - 1:
                # 检查是否是目标连续模式
                is_streak = True
                for j in range(streak_length):
                    if shoe_data[i + j] != streak_type:
                        is_streak = False
                        break
                
                if is_streak:
                    # 找到连续模式，记录下一局结果
                    next_result = shoe_data[i + streak_length]
                    results[next_result] += 1
                    sample_count += 1
                    
                    # 跳过这个连续段
                    i += streak_length
                else:
                    i += 1
        
        # 计算概率
        if sample_count > 0:
            result = {
                'B': results['B'] / sample_count * 100,
                'P': results['P'] / sample_count * 100,
                'T': results['T'] / sample_count * 100,
                'sample_count': sample_count,
                'confidence': min(100, sample_count * 3),  # 置信度
                'method': 'streak_within_shoe',
                'pattern': f"{streak_type}连{streak_length}次"
            }
        else:
            result = self._default_prediction()
            result['method'] = 'streak_within_shoe'
            result['pattern'] = f"{streak_type}连{streak_length}次（无样本）"
        
        # 缓存结果
        self._streak_cache[cache_key] = result
        return result
    
    def analyze_sequence_pattern(self, pattern):
        """
        分析序列模式：例如"BBP后，下一局出什么？"
        
        只在靴内统计！
        
        Args:
            pattern: 模式字符串，如 'BB', 'BBP', 'BPB' 等
            
        Returns:
            {
                'B': 概率,
                'P': 概率,
                'T': 概率,
                'sample_count': 样本数量,
                'confidence': 置信度,
                'method': 'sequence_within_shoe'
            }
        """
        if pattern in self._sequence_cache:
            return self._sequence_cache[pattern]
        
        results = {'B': 0, 'P': 0, 'T': 0}
        sample_count = 0
        pattern_len = len(pattern)
        
        # 遍历每一靴
        for shoe in self.shoes:
            shoe_data = list(shoe['data'])
            
            # 在单靴内查找模式
            for i in range(len(shoe_data) - pattern_len):
                current_pattern = ''.join(shoe_data[i:i+pattern_len])
                
                if current_pattern == pattern:
                    # 找到匹配模式，记录下一局结果
                    next_result = shoe_data[i + pattern_len]
                    results[next_result] += 1
                    sample_count += 1
        
        # 计算概率
        if sample_count > 0:
            result = {
                'B': results['B'] / sample_count * 100,
                'P': results['P'] / sample_count * 100,
                'T': results['T'] / sample_count * 100,
                'sample_count': sample_count,
                'confidence': min(100, sample_count * 3),
                'method': 'sequence_within_shoe',
                'pattern': pattern
            }
        else:
            result = self._default_prediction()
            result['method'] = 'sequence_within_shoe'
            result['pattern'] = f"{pattern}（无样本）"
        
        # 缓存结果
        self._sequence_cache[pattern] = result
        return result
    
    def get_current_pattern_info(self, current_data):
        """
        获取当前数据的模式信息
        
        Args:
            current_data: 当前数据列表 ['B', 'P', ...]
            
        Returns:
            {
                'current_streak': ('B', 4),  # 当前连续及长度
                'recent_pattern': 'BBP',      # 最近的序列模式
            }
        """
        if not current_data:
            return None
        
        # 计算当前连续
        current_streak_type = current_data[-1]
        current_streak_length = 1
        
        for i in range(len(current_data) - 2, -1, -1):
            if current_data[i] == current_streak_type:
                current_streak_length += 1
            else:
                break
        
        # 获取最近模式（2-5个字符）
        recent_patterns = {}
        for n in range(2, 6):
            if len(current_data) >= n:
                recent_patterns[n] = ''.join(current_data[-n:])
        
        return {
            'current_streak': (current_streak_type, current_streak_length),
            'recent_patterns': recent_patterns
        }
    
    def clear_cache(self):
        """清空缓存（当有新数据时调用）"""
        self._streak_cache = {}
        self._sequence_cache = {}
    
    def get_statistics(self):
        """获取分析器统计信息"""
        return {
            'total_shoes': self.total_shoes,
            'total_rounds': self.total_rounds,
            'avg_rounds_per_shoe': self.total_rounds / self.total_shoes if self.total_shoes > 0 else 0,
            'cached_patterns': len(self._streak_cache) + len(self._sequence_cache)
        }
    
    def _default_prediction(self):
        """默认预测（样本不足时）"""
        return {
            'B': 45.86,  # 百家乐理论概率
            'P': 44.62,
            'T': 9.52,
            'sample_count': 0,
            'confidence': 0
        }

