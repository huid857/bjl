"""
数据驱动预测器
基于历史统计数据进行预测
"""

from historical_analyzer import HistoricalPatternAnalyzer


class DataDrivenPredictor:
    """
    数据驱动预测器
    
    核心思想：
    - 用真实历史数据替代固定规则
    - 基于靴内统计（不跨边界）
    - 样本量检查，确保可靠性
    """
    
    def __init__(self, shoes):
        """
        初始化预测器
        
        Args:
            shoes: 靴牌列表
        """
        self.shoes = shoes
        self.analyzer = HistoricalPatternAnalyzer(shoes)
        
        # 最小样本量要求
        self.min_sample_for_high_confidence = 10  # 至少10个样本才高置信
        self.min_sample_for_medium_confidence = 5  # 至少5个样本才中等置信
    
    def predict(self, current_data):
        """
        基于当前数据预测下一局
        
        Args:
            current_data: 当前数据列表 ['B', 'P', ...]
            
        Returns:
            预测结果字典
        """
        if not current_data or len(current_data) < 2:
            return self._default_prediction()
        
        # 获取当前模式信息
        pattern_info = self.analyzer.get_current_pattern_info(current_data)
        
        # 收集多个预测
        predictions = []
        
        # 1. 基于当前连续的预测
        if pattern_info:
            streak_type, streak_length = pattern_info['current_streak']
            
            # 只对庄或闲的连续进行预测（和局不算）
            if streak_type in ['B', 'P'] and streak_length >= 2:
                streak_pred = self.analyzer.analyze_streak_pattern(streak_type, streak_length)
                
                if streak_pred['sample_count'] >= self.min_sample_for_medium_confidence:
                    predictions.append({
                        'prediction': streak_pred,
                        'weight': self._calculate_weight(streak_pred['sample_count']),
                        'reason': streak_pred['pattern']
                    })
        
        # 2. 基于序列模式的预测（优先长模式）
        if pattern_info and pattern_info['recent_patterns']:
            # 从长到短尝试
            for n in sorted(pattern_info['recent_patterns'].keys(), reverse=True):
                pattern = pattern_info['recent_patterns'][n]
                
                # 跳过包含太多和局的模式
                if pattern.count('T') / len(pattern) > 0.5:
                    continue
                
                seq_pred = self.analyzer.analyze_sequence_pattern(pattern)
                
                if seq_pred['sample_count'] >= self.min_sample_for_medium_confidence:
                    predictions.append({
                        'prediction': seq_pred,
                        'weight': self._calculate_weight(seq_pred['sample_count']) * (1 + n * 0.1),  # 长模式权重更高
                        'reason': f"模式 {pattern}"
                    })
                    break  # 找到一个有效的就够了
        
        # 3. 集成所有预测
        if predictions:
            return self._ensemble_predictions(predictions)
        else:
            # 没有足够样本，返回默认预测
            result = self._default_prediction()
            result['reason'] = "样本不足，使用理论概率"
            return result
    
    def _calculate_weight(self, sample_count):
        """
        根据样本量计算权重
        
        Args:
            sample_count: 样本数量
            
        Returns:
            权重值 (0-2.0)
        """
        if sample_count >= 20:
            return 2.0  # 高权重
        elif sample_count >= 10:
            return 1.5  # 中高权重
        elif sample_count >= 5:
            return 1.0  # 中等权重
        else:
            return 0.5  # 低权重
    
    def _ensemble_predictions(self, predictions):
        """
        集成多个预测
        
        Args:
            predictions: 预测列表
            
        Returns:
            集成后的预测结果
        """
        total_weight = sum(p['weight'] for p in predictions)
        
        # 加权平均
        weighted_probs = {'B': 0, 'P': 0, 'T': 0}
        total_samples = 0
        reasons = []
        
        for pred in predictions:
            weight = pred['weight'] / total_weight
            weighted_probs['B'] += pred['prediction']['B'] * weight
            weighted_probs['P'] += pred['prediction']['P'] * weight
            weighted_probs['T'] += pred['prediction']['T'] * weight
            total_samples += pred['prediction']['sample_count']
            reasons.append(pred['reason'])
        
        # 计算置信度
        confidence = min(100, total_samples * 2)
        
        # 选择推荐
        max_prob = max(weighted_probs.values())
        recommendation = [k for k, v in weighted_probs.items() if v == max_prob][0]
        
        return {
            'B': weighted_probs['B'],
            'P': weighted_probs['P'],
            'T': weighted_probs['T'],
            'confidence': confidence,
            'recommendation': recommendation,
            'reason': '; '.join(reasons),
            'sample_count': total_samples,
            'method': 'data_driven'
        }
    
    def _default_prediction(self):
        """默认预测（数据不足时）"""
        return {
            'B': 45.86,
            'P': 44.62,
            'T': 9.52,
            'confidence': 0,
            'recommendation': 'B',
            'reason': '数据不足',
            'sample_count': 0,
            'method': 'default'
        }
    
    def can_predict(self):
        """
        检查是否有足够数据进行预测
        
        Returns:
            是否可以预测
        """
        return len(self.shoes) > 0 and self.analyzer.total_rounds >= 50
    
    def get_statistics(self):
        """获取统计信息"""
        return self.analyzer.get_statistics()

