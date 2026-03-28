"""
基于特征工程的机器学习预测器
使用精选的12个核心特征
"""

import numpy as np
from feature_extractor import FeatureExtractor
import predictor as _pred_mod  # 共享 V1 的 _TF_UNAVAILABLE / _SKLEARN_UNAVAILABLE 标记


# 精选的12个核心特征（从评估结果中得出）
SELECTED_FEATURES = [
    'current_streak_length',
    'banker_change_rate',
    'volatility',
    'alternation_rate_last_20',
    'trend_direction',
    'tie_ratio',
    'data_recency',
    'rounds_since_last_tie',
    'trend_strength',
    'tie_cluster_score',
    'max_streak_overall',
    'banker_ratio_last_10',
]


class LSTMPredictorV2:
    """
    LSTM神经网络预测器 V2 - 基于特征工程
    
    使用12个精选特征而非原始序列
    """
    
    def __init__(self, historical_shoes):
        """
        初始化LSTM预测器V2
        
        Args:
            historical_shoes (list): 历史靴牌列表
                [{'name': 'shoe1', 'data': 'BBPPTP...'}, ...]
        """
        self.historical_shoes = historical_shoes
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.is_trained = False
        self.min_data_required = 300  # 至少需要300局历史数据
        
        # 计算总数据量
        self.total_data_count = sum(len(shoe['data']) for shoe in historical_shoes)
    
    def can_train(self):
        """检查是否有足够数据训练"""
        return self.total_data_count >= self.min_data_required
    
    def train(self, sequence_length=10):
        """
        训练LSTM模型
        
        Args:
            sequence_length: 序列长度（特征向量的序列）
            
        Returns:
            是否训练成功
        """
        if _pred_mod._TF_UNAVAILABLE:
            return False
        if not self.can_train():
            return False
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # 固定随机种子
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # 准备训练数据
            X, y = self._prepare_feature_sequences(sequence_length)
            
            if len(X) < 50:
                return False
            
            # 构建模型（输入是特征序列）
            n_features = len(SELECTED_FEATURES)
            
            model = keras.Sequential([
                keras.layers.LSTM(
                    32, 
                    input_shape=(sequence_length, n_features),
                    return_sequences=True
                ),
                keras.layers.Dropout(0.3),
                keras.layers.LSTM(16),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(2, activation='softmax')  # B或P
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 训练模型
            model.fit(
                X, y,
                epochs=30,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            self.model = model
            self.is_trained = True
            self.sequence_length = sequence_length
            
            return True
            
        except BaseException as e:
            _pred_mod._TF_UNAVAILABLE = True
            print(f"LSTM V2训练失败（后续跳过 TF）: {e}")
            return False
    
    def _prepare_feature_sequences(self, sequence_length):
        """
        准备训练序列（基于特征）
        
        Returns:
            X: 特征序列 (samples, sequence_length, n_features)
            y: 标签 (samples,)
        """
        X_list = []
        y_list = []
        
        for shoe in self.historical_shoes:
            shoe_data_str = shoe['data']
            shoe_data_list = list(shoe_data_str)
            
            # 从每个靴牌中提取序列
            for i in range(15, len(shoe_data_list) - 1):
                # 构建特征序列
                feature_sequence = []
                
                for j in range(sequence_length):
                    # 每个时间步提取特征
                    step_index = i - sequence_length + j + 1
                    if step_index < 5:
                        # 数据不足，跳过
                        feature_sequence = None
                        break
                    
                    step_data = shoe_data_list[:step_index]
                    features = self.feature_extractor.extract_features(step_data)
                    
                    # 只提取精选特征
                    feature_vector = [features[name] for name in SELECTED_FEATURES]
                    feature_sequence.append(feature_vector)
                
                if feature_sequence is None:
                    continue
                
                # 目标：下一局结果
                next_result = shoe_data_list[i]
                
                # 只预测B/P，跳过和局
                if next_result in ['B', 'P']:
                    X_list.append(feature_sequence)
                    y_list.append(1 if next_result == 'B' else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def predict(self, current_shoe_data):
        """
        预测下一个结果
        
        Args:
            current_shoe_data (list): 当前靴牌数据
            
        Returns:
            预测字典
        """
        _fallback = {'B': 50, 'P': 50, 'T': 0, 'confidence': 0, 'method': 'lstm_v2', 'reason': 'TF不可用'}
        if _pred_mod._TF_UNAVAILABLE and not self.is_trained:
            return _fallback
        if not self.is_trained:
            if self.can_train():
                self.train()
            else:
                return {
                    'B': 50, 'P': 50, 'T': 0, 'confidence': 0,
                    'method': 'lstm_v2',
                    'reason': f'数据不足（需要{self.min_data_required}局）'
                }
        # train 可能已将 _TF_UNAVAILABLE 设为 True
        if _pred_mod._TF_UNAVAILABLE and not self.is_trained:
            return _fallback
        
        if len(current_shoe_data) < 15:
            return {
                'B': 50, 'P': 50, 'T': 0, 'confidence': 20,
                'method': 'lstm_v2', 'reason': '当前靴牌数据不足'
            }
        
        try:
            import tensorflow as tf
            
            # 固定随机种子
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # 构建特征序列
            feature_sequence = []
            
            for i in range(self.sequence_length):
                step_index = len(current_shoe_data) - self.sequence_length + i + 1
                if step_index < 5:
                    step_data = current_shoe_data[:5]
                else:
                    step_data = current_shoe_data[:step_index]
                
                features = self.feature_extractor.extract_features(step_data)
                feature_vector = [features[name] for name in SELECTED_FEATURES]
                feature_sequence.append(feature_vector)
            
            X = np.array(feature_sequence).reshape(1, self.sequence_length, len(SELECTED_FEATURES))
            
            # 预测
            prediction = self.model.predict(X, verbose=0)[0]
            
            banker_prob = prediction[1] * 100
            player_prob = prediction[0] * 100
            
            # 置信度基于预测概率的差异
            confidence = abs(banker_prob - player_prob) * 1.5
            confidence = min(100, max(30, confidence))
            
            return {
                'B': banker_prob, 'P': player_prob, 'T': 0,
                'confidence': confidence,
                'method': 'lstm_v2', 'reason': 'LSTM V2 (特征工程)'
            }
            
        except BaseException as e:
            _pred_mod._TF_UNAVAILABLE = True
            return {
                'B': 50, 'P': 50, 'T': 0, 'confidence': 0,
                'method': 'lstm_v2', 'reason': '预测出错'
            }


class RandomForestPredictorV2:
    """
    随机森林预测器 V2 - 基于特征工程（优化版）
    
    使用8个精选特征（经过深度优化）
    """
    
    def __init__(self, historical_shoes):
        """
        初始化随机森林预测器V2
        
        Args:
            historical_shoes (list): 历史靴牌列表
        """
        self.historical_shoes = historical_shoes
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.is_trained = False
        self.min_data_required = 150  # 至少需要150局历史数据
        
        # 使用优化后的8个特征
        self.selected_features = [
            'current_streak_length',
            'banker_change_rate',
            'volatility',
            'alternation_rate_last_20',
            'trend_direction',
            'tie_ratio',
            'data_recency',
            'rounds_since_last_tie'
        ]
        
        # 计算总数据量
        self.total_data_count = sum(len(shoe['data']) for shoe in historical_shoes)
    
    def can_train(self):
        """检查是否有足够数据训练"""
        return self.total_data_count >= self.min_data_required
    
    def train(self):
        """训练随机森林模型"""
        if _pred_mod._SKLEARN_UNAVAILABLE or not self.can_train():
            return False
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # 准备训练数据
            X, y = self._prepare_feature_data()
            
            if len(X) < 50:
                return False
            
            # 训练模型（使用优化后的参数）
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X, y)
            self.is_trained = True
            
            return True
            
        except BaseException as e:
            _pred_mod._SKLEARN_UNAVAILABLE = True
            print(f"随机森林V2训练失败（后续跳过 sklearn）: {e}")
            return False
    
    def _prepare_feature_data(self):
        """
        准备训练数据（基于特征）
        
        Returns:
            X: 特征矩阵 (samples, n_features)
            y: 标签 (samples,)
        """
        X_list = []
        y_list = []
        
        for shoe in self.historical_shoes:
            shoe_data_str = shoe['data']
            shoe_data_list = list(shoe_data_str)
            
            # 从每个靴牌中提取样本
            for i in range(10, len(shoe_data_list) - 1):
                # 提取当前数据的特征
                current_data = shoe_data_list[:i]
                features = self.feature_extractor.extract_features(current_data)
                
                # 使用优化后的8个特征
                feature_vector = [features[name] for name in self.selected_features]
                
                # 目标：下一局结果
                next_result = shoe_data_list[i]
                
                # 只预测B/P，跳过和局
                if next_result in ['B', 'P']:
                    X_list.append(feature_vector)
                    y_list.append(1 if next_result == 'B' else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def predict(self, current_shoe_data):
        """
        预测下一个结果
        
        Args:
            current_shoe_data (list): 当前靴牌数据
            
        Returns:
            预测字典
        """
        _fallback = {'B': 50, 'P': 50, 'T': 0, 'confidence': 0, 'method': 'random_forest_v2', 'reason': 'sklearn不可用'}
        if _pred_mod._SKLEARN_UNAVAILABLE and not self.is_trained:
            return _fallback
        if not self.is_trained:
            if self.can_train():
                self.train()
            else:
                return {
                    'B': 50, 'P': 50, 'T': 0, 'confidence': 0,
                    'method': 'random_forest_v2',
                    'reason': f'数据不足（需要{self.min_data_required}局）'
                }
        if _pred_mod._SKLEARN_UNAVAILABLE and not self.is_trained:
            return _fallback
        
        if len(current_shoe_data) < 10:
            return {
                'B': 50, 'P': 50, 'T': 0, 'confidence': 20,
                'method': 'random_forest_v2', 'reason': '当前靴牌数据不足'
            }
        
        try:
            # 提取特征
            features = self.feature_extractor.extract_features(current_shoe_data)
            feature_vector = [features[name] for name in self.selected_features]
            X = np.array(feature_vector).reshape(1, -1)
            
            # 预测概率
            proba = self.model.predict_proba(X)[0]
            
            player_prob = proba[0] * 100
            banker_prob = proba[1] * 100
            
            # 置信度
            confidence = abs(banker_prob - player_prob) * 1.2
            confidence = min(100, max(30, confidence))
            
            return {
                'B': banker_prob, 'P': player_prob, 'T': 0,
                'confidence': confidence,
                'method': 'random_forest_v2',
                'reason': 'RandomForest V2 (特征工程, 8特征优化)'
            }
            
        except BaseException as e:
            _pred_mod._SKLEARN_UNAVAILABLE = True
            return {
                'B': 50, 'P': 50, 'T': 0, 'confidence': 0,
                'method': 'random_forest_v2', 'reason': '预测出错'
            }
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            dict: {feature_name: importance}
        """
        if not self.is_trained or self.model is None:
            return {}
        
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.selected_features, importances)}

