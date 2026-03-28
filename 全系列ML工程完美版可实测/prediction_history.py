"""
预测历史记录模块
用于记录预测结果并计算模型准确率
"""

from datetime import datetime
from collections import defaultdict


class PredictionHistory:
    """预测历史记录器"""
    
    def __init__(self, max_history=200):
        """
        初始化历史记录器

        Fix(Bug#8): max_history 从 50 提升至 200，覆盖约 3 靴数据。
        
        Args:
            max_history: 最多保留的历史记录数
        """
        self.max_history = max_history
        self.predictions = []  # 总体预测历史
        self.model_predictions = defaultdict(list)  # 各模型的预测历史
        
        # 记录会话级别的总体统计（不受 max_history 限制）
        self.session_total_valid = 0
        self.session_total_correct = 0
    
    def _evaluate_prediction(self, predicted, actual):
        """
        评估预测结果（考虑和局的特殊性）
        
        规则：
        - 预测B/P，实际T → 中性（不算对错）
        - 预测T，实际B/P → 错误
        - 预测B，实际B → 正确
        - 预测P，实际P → 正确
        - 预测T，实际T → 正确
        - 预测B，实际P → 错误
        - 预测P，实际B → 错误
        
        Args:
            predicted: 预测结果
            actual: 实际结果
            
        Returns:
            (is_correct, is_neutral)
        """
        if predicted == actual:
            return True, False
        elif predicted in ['B', 'P'] and actual == 'T':
            # 预测庄/闲，出和局 → 中性
            return False, True
        else:
            # 其他情况都是错误
            return False, False
    
    def add_prediction(self, predicted, actual, model_predictions=None):
        """
        添加一次预测记录
        
        Args:
            predicted: 预测的结果 ('B', 'P', 或 'T')
            actual: 实际的结果
            model_predictions: 各模型的预测字典 {model_name: prediction}
        """
        # 判断预测结果（考虑和局的特殊性）
        is_correct, is_neutral = self._evaluate_prediction(predicted, actual)
        
        record = {
            'predicted': predicted,
            'actual': actual,
            'correct': is_correct,
            'neutral': is_neutral,
            'timestamp': datetime.now()
        }
        
        if not is_neutral:
            self.session_total_valid += 1
            if is_correct:
                self.session_total_correct += 1
        
        self.predictions.append(record)
        
        # 记录各模型的预测
        if model_predictions:
            for model_name, model_pred in model_predictions.items():
                model_is_correct, model_is_neutral = self._evaluate_prediction(model_pred, actual)
                
                model_record = {
                    'predicted': model_pred,
                    'actual': actual,
                    'correct': model_is_correct,
                    'neutral': model_is_neutral,
                    'timestamp': datetime.now()
                }
                self.model_predictions[model_name].append(model_record)
                
                # 限制每个模型的历史记录数量
                if len(self.model_predictions[model_name]) > self.max_history:
                    self.model_predictions[model_name].pop(0)
        
        # 限制总历史记录数量
        if len(self.predictions) > self.max_history:
            self.predictions.pop(0)
    
    def get_overall_accuracy(self, window=20):
        """
        获取总体预测准确率（排除中性记录）
        
        Args:
            window: 统计窗口（最近N次）
            
        Returns:
            准确率 (0.0-1.0)
        """
        if not self.predictions:
            return 0.5  # 默认50%
        
        recent = self.predictions[-window:]
        # 排除中性记录
        valid = [p for p in recent if not p.get('neutral', False)]
        
        if not valid:
            return 0.5
        
        correct = sum(1 for p in valid if p['correct'])
        return correct / len(valid)
    
    def get_model_accuracy(self, model_name, window=20):
        """
        获取特定模型的准确率（排除中性记录）
        
        Args:
            model_name: 模型名称
            window: 统计窗口
            
        Returns:
            准确率 (0.0-1.0)
        """
        if model_name not in self.model_predictions:
            return 0.5  # 默认50%
        
        recent = self.model_predictions[model_name][-window:]
        
        # 排除中性记录
        valid = [p for p in recent if not p.get('neutral', False)]
        
        if not valid:
            return 0.5
        
        correct = sum(1 for p in valid if p['correct'])
        return correct / len(valid)
    
    def get_recent_errors(self, count=5):
        """
        获取最近的预测错误（排除中性记录）
        
        Args:
            count: 获取最近N次错误
            
        Returns:
            错误记录列表
        """
        # 排除中性记录
        errors = [p for p in self.predictions if not p['correct'] and not p.get('neutral', False)]
        return errors[-count:] if errors else []
    
    def get_consecutive_errors(self):
        """
        获取当前连续预测错误的次数（排除中性记录）
        
        Returns:
            连续错误次数
        """
        if not self.predictions:
            return 0
        
        count = 0
        for p in reversed(self.predictions):
            # 跳过中性记录
            if p.get('neutral', False):
                continue
            
            if not p['correct']:
                count += 1
            else:
                break
        
        return count
    
    def get_consecutive_corrects(self):
        """
        获取当前连续预测正确的次数（排除中性记录）
        
        Returns:
            连续正确次数
        """
        if not self.predictions:
            return 0
        
        count = 0
        for p in reversed(self.predictions):
            # 跳过中性记录
            if p.get('neutral', False):
                continue
            
            if p['correct']:
                count += 1
            else:
                break
        
        return count
    
    def get_model_consecutive_errors(self, model_name):
        """
        获取特定模型连续预测错误的次数（排除中性记录）
        
        Args:
            model_name: 模型名称
            
        Returns:
            连续错误次数
        """
        if model_name not in self.model_predictions:
            return 0
        
        records = self.model_predictions[model_name]
        if not records:
            return 0
        
        count = 0
        for p in reversed(records):
            # 跳过中性记录
            if p.get('neutral', False):
                continue
            
            if not p['correct']:
                count += 1
            else:
                break
        
        return count
    
    def get_model_recent_accuracy(self, model_name, window=20):
        """
        获取模型最近N次有效预测的准确率（排除中性）

        用于渐进式权重调整（Bug#10修复支撑方法）。

        Returns:
            (accuracy: float, valid_count: int)
        """
        if model_name not in self.model_predictions:
            return 0.5, 0
        
        records = self.model_predictions[model_name][-window:]
        valid = [p for p in records if not p.get('neutral', False)]
        if not valid:
            return 0.5, 0
        
        correct = sum(1 for p in valid if p['correct'])
        return correct / len(valid), len(valid)
    
    def get_statistics(self):
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        if not self.predictions:
            return {
                'total_predictions': 0,
                'overall_accuracy': 0.5,
                'consecutive_errors': 0,
                'model_accuracies': {}
            }
        
        # 计算各模型准确率
        model_accs = {}
        for model_name in self.model_predictions.keys():
            model_accs[model_name] = self.get_model_accuracy(model_name)
        
        return {
            'total_predictions': len(self.predictions),
            'overall_accuracy': self.get_overall_accuracy(),
            'session_valid': self.session_total_valid,
            'session_correct': self.session_total_correct,
            'session_accuracy': self.session_total_correct / self.session_total_valid if self.session_total_valid > 0 else 0.5,
            'consecutive_errors': self.get_consecutive_errors(),
            'consecutive_corrects': self.get_consecutive_corrects(),
            'model_accuracies': model_accs
        }
    
    def should_adjust_weights(self):
        """
        判断是否应该调整权重
        
        Returns:
            (是否调整, 原因)
        """
        consecutive = self.get_consecutive_errors()
        
        if consecutive >= 3:
            return True, f"连续预测错误{consecutive}次"
        
        # 检查总体准确率
        accuracy = self.get_overall_accuracy(window=20)
        if accuracy < 0.4 and len(self.predictions) >= 10:
            return True, f"最近准确率过低({accuracy*100:.1f}%)"
        
        return False, None
    
    def clear(self):
        """清空历史记录"""
        self.predictions = []
        self.model_predictions = defaultdict(list)
        self.session_total_valid = 0
        self.session_total_correct = 0
