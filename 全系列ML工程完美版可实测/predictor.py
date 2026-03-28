"""
机器学习预测模块
包含LSTM、马尔可夫链、随机森林等预测模型
"""

import numpy as np
from collections import defaultdict
from utils import result_to_number, number_to_result


class MarkovPredictor:
    """马尔可夫链预测器"""
    
    def __init__(self, data, order=2):
        """
        初始化马尔可夫链
        
        Args:
            data: 历史数据
            order: 马尔可夫链的阶数（考虑前几个状态）
        """
        self.data = [x for x in data if x != 'T']  # 排除和局
        self.order = order
        self.transition_matrix = self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        """构建状态转移矩阵"""
        if len(self.data) < self.order + 1:
            return {}
        
        transitions = defaultdict(lambda: {'B': 0, 'P': 0})
        
        for i in range(len(self.data) - self.order):
            state = ''.join(self.data[i:i+self.order])
            next_state = self.data[i+self.order]
            
            if next_state in ['B', 'P']:
                transitions[state][next_state] += 1
        
        # 转换为概率
        matrix = {}
        for state, counts in transitions.items():
            total = sum(counts.values())
            if total > 0:
                matrix[state] = {
                    'B': counts['B'] / total,
                    'P': counts['P'] / total,
                    'count': total
                }
        
        return matrix
    
    def predict(self, recent_data):
        """
        预测下一个状态
        
        Args:
            recent_data: 最近的数据
            
        Returns:
            预测字典
        """
        recent_data = [x for x in recent_data if x != 'T']
        
        if len(recent_data) < self.order:
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 20,
                'method': 'markov',
                'reason': '数据不足'
            }
        
        current_state = ''.join(recent_data[-self.order:])
        
        if current_state in self.transition_matrix:
            probs = self.transition_matrix[current_state]
            confidence = min(100, probs['count'] * 10)
            
            return {
                'B': probs['B'] * 100,
                'P': probs['P'] * 100,
                'T': 0,
                'confidence': confidence,
                'method': 'markov',
                'state': current_state,
                'sample_count': probs['count']
            }
        else:
            # 尝试降阶
            if self.order > 1:
                lower_order = MarkovPredictor(self.data, self.order - 1)
                return lower_order.predict(recent_data)
            else:
                return {
                    'B': 50,
                    'P': 50,
                    'T': 0,
                    'confidence': 30,
                    'method': 'markov',
                    'reason': '未找到匹配状态'
                }

# ──────────────────────────────────────────────────────────
# 模块级可用性标记
# 在模块加载时就做一次快速探测，避免 C 扩展崩溃（numpy 2.x 不兼容时
# import tensorflow / sklearn 会触发 AttributeError: _ARRAY_API，
# 且该异常可能绕过 Python 层的 try/except）。
# ──────────────────────────────────────────────────────────

def _probe_import(module_name):
    """用子进程探测模块是否可导入，避免 C 扩展崩溃污染主进程"""
    import subprocess, sys
    try:
        result = subprocess.run(
            [sys.executable, '-c', f'import {module_name}'],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False

_TF_UNAVAILABLE = not _probe_import('tensorflow')
_SKLEARN_UNAVAILABLE = not _probe_import('sklearn')

if _TF_UNAVAILABLE:
    print("[预检] TensorFlow 不可用（numpy 版本冲突），LSTM 模型将被跳过")
if _SKLEARN_UNAVAILABLE:
    print("[预检] sklearn 不可用（numpy 版本冲突），随机森林模型将被跳过")

class LSTMPredictor:
    """LSTM神经网络预测器（简化版）"""
    
    def __init__(self, data):
        """
        初始化LSTM预测器
        
        Args:
            data: 历史数据
        """
        self.data = [x for x in data if x != 'T']
        self.model = None
        self.is_trained = False
        self.min_data_required = 300
    
    def can_train(self):
        """检查是否有足够数据训练"""
        return len(self.data) >= self.min_data_required
    
    def train(self, sequence_length=20):
        """
        训练LSTM模型
        
        Args:
            sequence_length: 序列长度
            
        Returns:
            是否训练成功
        """
        global _TF_UNAVAILABLE
        if _TF_UNAVAILABLE:
            return False
        if not self.can_train():
            return False
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # 固定随机种子，确保训练一致性
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # 准备训练数据
            X, y = self._prepare_sequences(sequence_length)
            
            if len(X) < 50:
                return False
            
            # 构建模型
            model = keras.Sequential([
                keras.layers.LSTM(64, input_shape=(sequence_length, 1), return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(32),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(2, activation='softmax')  # B或P
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 训练模型（静默模式）
            model.fit(
                X, y,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.model = model
            self.is_trained = True
            self.sequence_length = sequence_length
            
            return True
            
        except BaseException as e:
            _TF_UNAVAILABLE = True  # 标记 TF 不可用，后续不再重试
            print(f"LSTM训练失败（将剩余计算中跳过 TF）: {e}")
            return False
    
    def _prepare_sequences(self, sequence_length):
        """准备训练序列"""
        # 转换为数字 (B=1, P=0)
        numeric_data = [1 if x == 'B' else 0 for x in self.data]
        
        X, y = [], []
        
        for i in range(len(numeric_data) - sequence_length):
            X.append(numeric_data[i:i+sequence_length])
            y.append(numeric_data[i+sequence_length])
        
        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)
        
        return X, y
    
    def predict(self, recent_data):
        """
        预测下一个结果
        
        Args:
            recent_data: 最近的数据
            
        Returns:
            预测字典
        """
        global _TF_UNAVAILABLE
        if _TF_UNAVAILABLE and not self.is_trained:
            # TF 已知不可用且模型尚未训练，直接返回空预测
            return {'B': 50, 'P': 50, 'T': 0, 'confidence': 0, 'method': 'lstm', 'reason': 'TF不可用'}
        if not self.is_trained:
            if self.can_train():
                self.train()
            else:
                return {
                    'B': 50,
                    'P': 50,
                    'T': 0,
                    'confidence': 0,
                    'method': 'lstm',
                    'reason': f'数据不足（需要{self.min_data_required}局）'
                }
        
        try:
            import tensorflow as tf
            
            # 固定随机种子，确保预测一致性
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # 准备输入数据（只使用当前靴牌数据，不混入历史）
            recent_data = [x for x in recent_data if x != 'T']
            
            # 修复：不再用历史数据填充，避免混淆当前靴牌特征
            if len(recent_data) < 10:
                # 数据太少，无法可靠预测
                return {
                    'B': 50,
                    'P': 50,
                    'T': 0,
                    'confidence': 0,
                    'method': 'lstm',
                    'reason': f'当前靴牌数据不足（{len(recent_data)}局，需要至少10局）'
                }
            
            # 数据不足sequence_length时，用当前靴牌数据的均值填充（而非历史数据）
            data_shortage_penalty = 1.0
            if len(recent_data) < self.sequence_length:
                # 计算当前靴牌的庄家比例，用作填充值
                current_b_ratio = sum(1 for x in recent_data if x == 'B') / len(recent_data)
                padding_value = current_b_ratio  # 用当前靴牌的比例填充
                
                # 填充到sequence_length长度
                padding_length = self.sequence_length - len(recent_data)
                padding = [padding_value] * padding_length
                numeric_recent = [1 if x == 'B' else 0 for x in recent_data]
                numeric_data = padding + numeric_recent
                
                # 数据不足，降低置信度
                data_shortage_penalty = len(recent_data) / self.sequence_length
            else:
                recent_data = recent_data[-self.sequence_length:]
                numeric_data = [1 if x == 'B' else 0 for x in recent_data]
            
            X = np.array(numeric_data).reshape(1, self.sequence_length, 1)
            
            # 预测
            prediction = self.model.predict(X, verbose=0)[0]
            
            banker_prob = prediction[1] * 100
            player_prob = prediction[0] * 100
            
            # 置信度基于预测概率的差异，但要考虑数据不足的惩罚
            confidence = abs(banker_prob - player_prob) * 1.5
            confidence = confidence * data_shortage_penalty  # 应用数据不足惩罚
            confidence = min(100, max(20, confidence))
            
            return {
                'B': banker_prob,
                'P': player_prob,
                'T': 0,
                'confidence': confidence,
                'method': 'lstm'
            }
            
        except Exception as e:
            print(f"LSTM预测失败: {e}")
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 0,
                'method': 'lstm',
                'reason': '预测出错'
            }


class RandomForestPredictor:
    """随机森林预测器"""
    
    def __init__(self, data):
        """
        初始化随机森林预测器
        
        Args:
            data: 历史数据
        """
        self.data = [x for x in data if x != 'T']
        self.model = None
        self.is_trained = False
        self.min_data_required = 200
    
    def can_train(self):
        """检查是否有足够数据训练"""
        return len(self.data) >= self.min_data_required
    
    def _extract_features(self, data, index):
        """
        提取特征
        
        Args:
            data: 数据列表
            index: 当前索引
            
        Returns:
            特征数组
        """
        features = []
        
        # 最近5局的结果（编码为0/1）
        for i in range(5):
            if index - i - 1 >= 0:
                features.append(1 if data[index - i - 1] == 'B' else 0)
            else:
                features.append(0.5)  # 填充值
        
        # 最近10局庄的数量
        recent_10 = data[max(0, index-10):index]
        features.append(recent_10.count('B') / 10 if len(recent_10) > 0 else 0.5)
        
        # 最近20局庄的数量
        recent_20 = data[max(0, index-20):index]
        features.append(recent_20.count('B') / 20 if len(recent_20) > 0 else 0.5)
        
        # 当前连胜数
        streak = 0
        if index > 0:
            last = data[index - 1]
            for i in range(index - 1, -1, -1):
                if data[i] == last:
                    streak += 1
                else:
                    break
        features.append(min(streak / 10, 1.0))  # 归一化
        
        return features
    
    def train(self):
        """训练随机森林模型"""
        global _SKLEARN_UNAVAILABLE
        if _SKLEARN_UNAVAILABLE or not self.can_train():
            return False
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # 准备训练数据
            X, y = [], []
            
            for i in range(20, len(self.data)):
                features = self._extract_features(self.data, i)
                label = 1 if self.data[i] == 'B' else 0
                
                X.append(features)
                y.append(label)
            
            if len(X) < 50:
                return False
            
            # 训练模型
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X, y)
            self.is_trained = True
            
            return True
            
        except BaseException as e:
            _SKLEARN_UNAVAILABLE = True  # 标记 sklearn 不可用，后续不再重试
            print(f"随机森林训练失败（将剩余计算中跳过 sklearn）: {e}")
            return False
    
    def predict(self, recent_data):
        """
        预测下一个结果
        
        Args:
            recent_data: 最近的数据
            
        Returns:
            预测字典
        """
        global _SKLEARN_UNAVAILABLE
        if _SKLEARN_UNAVAILABLE and not self.is_trained:
            return {'B': 50, 'P': 50, 'T': 0, 'confidence': 0, 'method': 'random_forest', 'reason': 'sklearn不可用'}
        if not self.is_trained:
            if self.can_train():
                self.train()
            else:
                return {
                    'B': 50,
                    'P': 50,
                    'T': 0,
                    'confidence': 0,
                    'method': 'random_forest',
                    'reason': f'数据不足（需要{self.min_data_required}局）'
                }
        
        try:
            # 准备输入数据（只使用当前靴牌数据，不混入历史）
            recent_data = [x for x in recent_data if x != 'T']
            
            # 修复：只从当前靴牌提取特征，不混入历史数据
            if len(recent_data) < 10:
                # 数据太少，无法可靠预测
                return {
                    'B': 50,
                    'P': 50,
                    'T': 0,
                    'confidence': 0,
                    'method': 'random_forest',
                    'reason': f'当前靴牌数据不足（{len(recent_data)}局，需要至少10局）'
                }
            
            # 从当前靴牌提取特征（而非combined_data）
            features = self._extract_features(recent_data, len(recent_data))
            X = np.array(features).reshape(1, -1)
            
            # 预测概率
            proba = self.model.predict_proba(X)[0]
            
            player_prob = proba[0] * 100
            banker_prob = proba[1] * 100
            
            # 置信度：考虑数据量，数据越少置信度越低
            confidence = abs(banker_prob - player_prob) * 1.2
            
            # 根据当前靴牌数据量调整置信度
            if len(recent_data) < 20:
                data_penalty = len(recent_data) / 20
                confidence = confidence * data_penalty
            
            confidence = min(100, max(20, confidence))
            
            return {
                'B': banker_prob,
                'P': player_prob,
                'T': 0,
                'confidence': confidence,
                'method': 'random_forest'
            }
            
        except Exception as e:
            print(f"随机森林预测失败: {e}")
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 0,
                'method': 'random_forest',
                'reason': '预测出错'
            }


class FrequencyPredictor:
    """频率统计预测器（简单但有效）"""
    
    def __init__(self, data):
        """
        初始化频率预测器
        
        Args:
            data: 历史数据
        """
        self.data = data
    
    def predict(self, recent_data=None, window=50):
        """
        基于频率预测
        
        Args:
            recent_data: 最近的数据（可选）
            window: 统计窗口大小
            
        Returns:
            预测字典
        """
        if recent_data:
            data_to_analyze = recent_data[-window:] if len(recent_data) > window else recent_data
        else:
            data_to_analyze = self.data[-window:] if len(self.data) > window else self.data
        
        if not data_to_analyze:
            return {
                'B': 45.86,
                'P': 44.62,
                'T': 9.52,
                'confidence': 20,
                'method': 'frequency'
            }
        
        total = len(data_to_analyze)
        b_count = data_to_analyze.count('B')
        p_count = data_to_analyze.count('P')
        t_count = data_to_analyze.count('T')
        
        b_prob = b_count / total * 100
        p_prob = p_count / total * 100
        t_prob = t_count / total * 100
        
        # 置信度基于样本量
        confidence = min(80, total * 0.8)
        
        return {
            'B': b_prob,
            'P': p_prob,
            'T': t_prob,
            'confidence': confidence,
            'method': 'frequency',
            'sample_size': total
        }

