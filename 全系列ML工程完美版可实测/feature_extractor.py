"""
特征提取器
为LSTM和RandomForest模型提取特征
"""

import numpy as np
from utils import get_current_streak, get_max_streak


class FeatureExtractor:
    """
    特征提取器
    
    从百家乐数据中提取25个候选特征
    """
    
    def __init__(self):
        """初始化特征提取器"""
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self):
        """获取所有特征名称"""
        return [
            # 类别1：频率统计（6个）
            'banker_ratio_all',
            'banker_ratio_last_5',
            'banker_ratio_last_10',
            'banker_ratio_last_20',
            'tie_ratio',
            'tie_ratio_last_20',
            
            # 类别2：连续特征（5个）
            'current_streak_length',
            'current_streak_type',
            'max_streak_banker',
            'max_streak_player',
            'max_streak_overall',
            
            # 类别3：交替特征（3个）
            'alternation_rate',
            'alternation_rate_last_20',
            'last_switch_rounds',
            
            # 类别4：趋势特征（3个）
            'trend_direction',
            'trend_strength',
            'banker_momentum',
            
            # 类别5：周期特征（3个）
            'rounds_since_last_tie',
            'tie_frequency_last_10',
            'tie_cluster_score',
            
            # 类别6：数据特征（3个）
            'total_rounds',
            'data_recency',
            'shoe_progress',
            
            # 类别7：变化率特征（2个）
            'banker_change_rate',
            'volatility'
        ]
    
    def extract_features(self, shoe_data):
        """
        从靴牌数据中提取所有特征
        
        Args:
            shoe_data (list): 靴牌数据列表 ['B', 'P', 'T', ...]
            
        Returns:
            dict: 特征字典 {feature_name: value}
        """
        if len(shoe_data) < 5:
            # 数据太少，返回默认值
            return self._default_features()
        
        features = {}
        
        # 预处理
        no_tie = [x for x in shoe_data if x != 'T']
        recent_5 = shoe_data[-5:] if len(shoe_data) >= 5 else shoe_data
        recent_10 = shoe_data[-10:] if len(shoe_data) >= 10 else shoe_data
        recent_20 = shoe_data[-20:] if len(shoe_data) >= 20 else shoe_data
        
        # ===== 类别1：频率统计（6个）=====
        features['banker_ratio_all'] = self._calculate_ratio(shoe_data, 'B')
        features['banker_ratio_last_5'] = self._calculate_ratio(recent_5, 'B')
        features['banker_ratio_last_10'] = self._calculate_ratio(recent_10, 'B')
        features['banker_ratio_last_20'] = self._calculate_ratio(recent_20, 'B')
        features['tie_ratio'] = self._calculate_ratio(shoe_data, 'T')
        features['tie_ratio_last_20'] = self._calculate_ratio(recent_20, 'T')
        
        # ===== 类别2：连续特征（5个）=====
        if no_tie:
            streak_type, streak_length = get_current_streak(no_tie)
            features['current_streak_length'] = streak_length
            features['current_streak_type'] = 1.0 if streak_type == 'B' else 0.0
            features['max_streak_banker'] = get_max_streak(no_tie, 'B')
            features['max_streak_player'] = get_max_streak(no_tie, 'P')
            features['max_streak_overall'] = self._get_max_overall_streak(no_tie)
        else:
            features['current_streak_length'] = 0
            features['current_streak_type'] = 0.5
            features['max_streak_banker'] = 0
            features['max_streak_player'] = 0
            features['max_streak_overall'] = 0
        
        # ===== 类别3：交替特征（3个）=====
        features['alternation_rate'] = self._calculate_alternation_rate(no_tie)
        recent_20_no_tie = [x for x in recent_20 if x != 'T']
        features['alternation_rate_last_20'] = self._calculate_alternation_rate(recent_20_no_tie)
        features['last_switch_rounds'] = self._rounds_since_switch(no_tie)
        
        # ===== 类别4：趋势特征（3个）=====
        trend = self._analyze_trend(recent_20)
        features['trend_direction'] = trend['direction']
        features['trend_strength'] = trend['strength']
        features['banker_momentum'] = self._calculate_momentum(shoe_data)
        
        # ===== 类别5：周期特征（3个）=====
        features['rounds_since_last_tie'] = self._rounds_since_event(shoe_data, 'T')
        features['tie_frequency_last_10'] = recent_10.count('T')
        features['tie_cluster_score'] = self._calculate_tie_clustering(shoe_data)
        
        # ===== 类别6：数据特征（3个）=====
        features['total_rounds'] = len(shoe_data)
        features['data_recency'] = 1.0 / len(shoe_data) if len(shoe_data) > 0 else 0
        features['shoe_progress'] = min(len(shoe_data) / 60.0, 1.0)
        
        # ===== 类别7：变化率特征（2个）=====
        features['banker_change_rate'] = self._calculate_change_rate(shoe_data)
        features['volatility'] = self._calculate_volatility(shoe_data)
        
        return features
    
    def extract_features_array(self, shoe_data):
        """
        提取特征并返回numpy数组（用于模型输入）
        
        Args:
            shoe_data (list): 靴牌数据
            
        Returns:
            np.array: 特征向量
        """
        features = self.extract_features(shoe_data)
        return np.array([features[name] for name in self.feature_names])
    
    def _calculate_ratio(self, data, target):
        """计算特定结果的比例"""
        if not data:
            return 0.5
        return data.count(target) / len(data)
    
    def _get_max_overall_streak(self, data):
        """获取最长连续（不分类型）"""
        if not data:
            return 0
        
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return max_streak
    
    def _calculate_alternation_rate(self, data):
        """计算跳跳比例"""
        if len(data) < 2:
            return 0.5
        
        alternations = 0
        for i in range(1, len(data)):
            if data[i] != data[i-1]:
                alternations += 1
        
        return alternations / (len(data) - 1)
    
    def _rounds_since_switch(self, data):
        """距离上次转换的局数"""
        if len(data) < 2:
            return 0
        
        rounds = 0
        for i in range(len(data) - 1, 0, -1):
            if data[i] == data[i-1]:
                rounds += 1
            else:
                break
        
        return rounds
    
    def _analyze_trend(self, data):
        """
        分析趋势
        
        Returns:
            dict: {'direction': 0-1, 'strength': 0-1}
        """
        if len(data) < 10:
            return {'direction': 0.5, 'strength': 0.0}
        
        no_tie = [x for x in data if x != 'T']
        if not no_tie:
            return {'direction': 0.5, 'strength': 0.0}
        
        banker_count = no_tie.count('B')
        banker_ratio = banker_count / len(no_tie)
        
        # 方向：0=闲强, 0.5=平衡, 1=庄强
        direction = banker_ratio
        
        # 强度：离0.5越远越强
        strength = abs(banker_ratio - 0.5) * 2  # 归一化到0-1
        
        return {'direction': direction, 'strength': strength}
    
    def _calculate_momentum(self, data):
        """
        计算动量（最近趋势 - 长期趋势）
        
        正值：庄家上升趋势
        负值：庄家下降趋势
        """
        if len(data) < 20:
            return 0.0
        
        recent = data[-10:]
        earlier = data[-20:-10]
        
        recent_banker = self._calculate_ratio(recent, 'B')
        earlier_banker = self._calculate_ratio(earlier, 'B')
        
        return recent_banker - earlier_banker
    
    def _rounds_since_event(self, data, event):
        """距离上次特定事件的局数"""
        if event not in data:
            return len(data)
        
        for i in range(len(data) - 1, -1, -1):
            if data[i] == event:
                return len(data) - 1 - i
        
        return len(data)
    
    def _calculate_tie_clustering(self, data):
        """
        计算和局聚集度
        
        如果和局集中出现，得分高
        如果和局分散，得分低
        """
        ties = [i for i, x in enumerate(data) if x == 'T']
        
        if len(ties) < 2:
            return 0.0
        
        # 计算和局之间的平均间隔
        intervals = [ties[i+1] - ties[i] for i in range(len(ties) - 1)]
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # 标准差越小，越聚集
        # 归一化到0-1
        if avg_interval == 0:
            return 1.0
        
        clustering = 1.0 - min(std_interval / avg_interval, 1.0)
        return clustering
    
    def _calculate_change_rate(self, data):
        """
        计算庄比例的变化率
        
        对比前后两段的庄比例
        """
        if len(data) < 20:
            return 0.0
        
        mid = len(data) // 2
        first_half = data[:mid]
        second_half = data[mid:]
        
        ratio1 = self._calculate_ratio(first_half, 'B')
        ratio2 = self._calculate_ratio(second_half, 'B')
        
        return ratio2 - ratio1
    
    def _calculate_volatility(self, data):
        """
        计算波动性
        
        使用滑动窗口计算庄比例的标准差
        """
        if len(data) < 10:
            return 0.0
        
        window_size = 5
        ratios = []
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i+window_size]
            ratio = self._calculate_ratio(window, 'B')
            ratios.append(ratio)
        
        if not ratios:
            return 0.0
        
        return np.std(ratios)
    
    def _default_features(self):
        """数据不足时的默认特征"""
        return {name: 0.5 if 'ratio' in name or 'direction' in name else 0.0 
                for name in self.feature_names}
    
    def get_feature_descriptions(self):
        """获取特征描述（用于可视化）"""
        return {
            # 类别1：频率统计
            'banker_ratio_all': '整体庄比例',
            'banker_ratio_last_5': '最近5局庄比例',
            'banker_ratio_last_10': '最近10局庄比例',
            'banker_ratio_last_20': '最近20局庄比例',
            'tie_ratio': '和局比例',
            'tie_ratio_last_20': '最近20局和比例',
            
            # 类别2：连续特征
            'current_streak_length': '当前连续次数',
            'current_streak_type': '当前连续类型(B=1,P=0)',
            'max_streak_banker': '本靴最长庄连',
            'max_streak_player': '本靴最长闲连',
            'max_streak_overall': '本靴最长连续',
            
            # 类别3：交替特征
            'alternation_rate': '整体跳跳比例',
            'alternation_rate_last_20': '最近20局跳跳比例',
            'last_switch_rounds': '距离上次转换的局数',
            
            # 类别4：趋势特征
            'trend_direction': '趋势方向(闲0,平0.5,庄1)',
            'trend_strength': '趋势强度(0-1)',
            'banker_momentum': '庄家动量(最近-长期)',
            
            # 类别5：周期特征
            'rounds_since_last_tie': '距离上次和局',
            'tie_frequency_last_10': '最近10局和出现次数',
            'tie_cluster_score': '和局聚集度',
            
            # 类别6：数据特征
            'total_rounds': '当前总局数',
            'data_recency': '数据新鲜度(1/总局数)',
            'shoe_progress': '靴牌进度(当前局/60)',
            
            # 类别7：变化率特征
            'banker_change_rate': '庄比例变化率',
            'volatility': '波动性'
        }

