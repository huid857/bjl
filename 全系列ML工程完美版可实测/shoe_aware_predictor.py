"""
靴牌感知的ML预测器
学习"靴牌演化模式"，而不是"全局概率"

核心思想：
- 不同类型的靴牌有不同的演化规律
- 在"闲强靴"中，闲可能继续强势
- 在"跳跳靴"中，可能交替出现
- 在"长龙靴"中，可能出现长连
- ML模型应该学习这些**模式**，而不是简单的概率
"""

import numpy as np
from utils import calculate_statistics


class ShoeFeatureExtractor:
    """提取靴牌级别特征"""
    
    @staticmethod
    def extract_shoe_features(shoe_data, current_position=None):
        """
        提取靴牌级别的特征
        
        Args:
            shoe_data: 靴牌数据列表
            current_position: 当前位置（如果提供，只使用到此位置的数据）
            
        Returns:
            特征字典
        """
        if current_position:
            data = shoe_data[:current_position]
        else:
            data = shoe_data
        
        if len(data) < 5:
            # 数据太少，返回默认特征
            return {
                'shoe_length': len(data),
                'shoe_stage': 0.0,  # 0-1之间
                'banker_ratio': 0.5,
                'player_ratio': 0.5,
                'tie_ratio': 0.0,
                'alternation_rate': 0.5,
                'max_banker_streak': 0,
                'max_player_streak': 0,
                'current_streak_length': 0,
                'current_streak_type': 0,  # 0=P, 1=B
                'is_banker_dominant': 0,
                'is_player_dominant': 0,
                'is_balanced': 1,
                'is_jump_shoe': 0,
                'is_long_dragon': 0,
                'volatility': 0.5
            }
        
        # 基础统计
        no_tie = [x for x in data if x != 'T']
        b_count = no_tie.count('B')
        p_count = no_tie.count('P')
        t_count = data.count('T')
        total = len(data)
        non_tie_total = len(no_tie)
        
        # 比例
        banker_ratio = b_count / non_tie_total if non_tie_total > 0 else 0.5
        player_ratio = p_count / non_tie_total if non_tie_total > 0 else 0.5
        tie_ratio = t_count / total if total > 0 else 0.0
        
        # 靴牌阶段 (0=早期, 0.5=中期, 1.0=后期)
        # 假设一靴通常50-70局
        shoe_stage = min(1.0, total / 60.0)
        
        # 交替率（跳跳程度）
        alternations = 0
        for i in range(1, len(no_tie)):
            if no_tie[i] != no_tie[i-1]:
                alternations += 1
        alternation_rate = alternations / (len(no_tie) - 1) if len(no_tie) > 1 else 0.5
        
        # 最大连胜
        max_banker_streak = ShoeFeatureExtractor._get_max_streak(no_tie, 'B')
        max_player_streak = ShoeFeatureExtractor._get_max_streak(no_tie, 'P')
        
        # 当前连胜
        current_streak_length = 0
        current_streak_type = 0
        if no_tie:
            last = no_tie[-1]
            current_streak_type = 1 if last == 'B' else 0
            for i in range(len(no_tie) - 1, -1, -1):
                if no_tie[i] == last:
                    current_streak_length += 1
                else:
                    break
        
        # 靴牌类型特征
        is_banker_dominant = 1 if banker_ratio >= 0.54 else 0
        is_player_dominant = 1 if player_ratio >= 0.54 else 0
        is_balanced = 1 if abs(banker_ratio - player_ratio) <= 0.06 else 0
        is_jump_shoe = 1 if alternation_rate >= 0.6 else 0
        is_long_dragon = 1 if max(max_banker_streak, max_player_streak) >= 6 else 0
        
        # 波动性（最近10局的变化频率）
        recent_10 = no_tie[-10:] if len(no_tie) >= 10 else no_tie
        recent_changes = sum(1 for i in range(1, len(recent_10)) if recent_10[i] != recent_10[i-1])
        volatility = recent_changes / (len(recent_10) - 1) if len(recent_10) > 1 else 0.5
        
        return {
            'shoe_length': total,
            'shoe_stage': shoe_stage,
            'banker_ratio': banker_ratio,
            'player_ratio': player_ratio,
            'tie_ratio': tie_ratio,
            'alternation_rate': alternation_rate,
            'max_banker_streak': max_banker_streak,
            'max_player_streak': max_player_streak,
            'current_streak_length': current_streak_length,
            'current_streak_type': current_streak_type,
            'is_banker_dominant': is_banker_dominant,
            'is_player_dominant': is_player_dominant,
            'is_balanced': is_balanced,
            'is_jump_shoe': is_jump_shoe,
            'is_long_dragon': is_long_dragon,
            'volatility': volatility
        }
    
    @staticmethod
    def _get_max_streak(data, target):
        """获取最大连胜数"""
        max_streak = 0
        current_streak = 0
        
        for item in data:
            if item == target:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak


class ShoeAwareLSTMPredictor:
    """
    靴牌感知的LSTM预测器
    
    学习：在不同类型的靴牌中，序列如何演化
    """
    
    def __init__(self, historical_shoes):
        """
        初始化
        
        Args:
            historical_shoes: 历史靴牌列表 [{'name': 'xxx', 'data': 'BBPP...'}, ...]
        """
        self.historical_shoes = historical_shoes
        self.model = None
        self.is_trained = False
        self.sequence_length = 15
        self.min_shoes_required = 10  # 至少需要10个靴牌
        
        self.feature_extractor = ShoeFeatureExtractor()
    
    def can_train(self):
        """检查是否有足够数据训练"""
        return len(self.historical_shoes) >= self.min_shoes_required
    
    def train(self):
        """
        训练模型
        
        核心改变：
        - 按靴牌组织数据
        - 每个样本包含：靴牌特征 + 局部序列
        - 学习："在这种靴牌特征下，序列如何演化"
        """
        if not self.can_train():
            return False
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # 准备训练数据
            X, y = self._prepare_shoe_aware_sequences()
            
            if len(X) < 100:
                return False
            
            # 输入维度：sequence_length × (1 + 靴牌特征数)
            # 1 = 序列本身(B/P编码)
            # 靴牌特征数 = 16个特征
            input_dim = 1 + 16
            
            # 构建模型
            model = keras.Sequential([
                keras.layers.LSTM(
                    64,
                    input_shape=(self.sequence_length, input_dim),
                    return_sequences=True
                ),
                keras.layers.Dropout(0.3),
                keras.layers.LSTM(32),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(2, activation='softmax')  # B或P
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 训练
            model.fit(
                X, y,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.model = model
            self.is_trained = True
            
            # 只在第一次训练时显示消息（避免过度打印）
            # print(f"ShoeAwareLSTM训练完成: {len(X)}个样本")
            
            return True
            
        except Exception as e:
            print(f"ShoeAwareLSTM训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_shoe_aware_sequences(self):
        """
        准备靴牌感知的训练序列
        
        每个样本包含：
        - 局部序列（最近15局）
        - 靴牌特征（在每个时间步重复）
        - 目标：下一局结果
        """
        X_list = []
        y_list = []
        
        for shoe in self.historical_shoes:
            shoe_data = list(shoe['data'])
            no_tie = [x for x in shoe_data if x != 'T']
            
            if len(no_tie) < self.sequence_length + 5:
                continue
            
            # 为这个靴牌的每个位置生成样本
            for i in range(self.sequence_length, len(no_tie)):
                # 提取靴牌特征（到当前位置）
                current_position_in_original = self._get_position_in_original(shoe_data, no_tie, i)
                shoe_features = self.feature_extractor.extract_shoe_features(
                    shoe_data,
                    current_position=current_position_in_original
                )
                
                # 构建特征向量（靴牌特征）
                shoe_feature_vector = [
                    shoe_features['shoe_stage'],
                    shoe_features['banker_ratio'],
                    shoe_features['player_ratio'],
                    shoe_features['tie_ratio'],
                    shoe_features['alternation_rate'],
                    shoe_features['max_banker_streak'] / 10.0,  # 归一化
                    shoe_features['max_player_streak'] / 10.0,
                    shoe_features['current_streak_length'] / 10.0,
                    shoe_features['current_streak_type'],
                    shoe_features['is_banker_dominant'],
                    shoe_features['is_player_dominant'],
                    shoe_features['is_balanced'],
                    shoe_features['is_jump_shoe'],
                    shoe_features['is_long_dragon'],
                    shoe_features['volatility'],
                    shoe_features['shoe_length'] / 70.0  # 归一化
                ]
                
                # 构建序列（最近sequence_length局）
                sequence = no_tie[i-self.sequence_length:i]
                
                # 组合：每个时间步 = [B/P编码] + [靴牌特征]
                combined_sequence = []
                for result in sequence:
                    result_encoded = 1 if result == 'B' else 0
                    timestep_features = [result_encoded] + shoe_feature_vector
                    combined_sequence.append(timestep_features)
                
                # 目标
                target = 1 if no_tie[i] == 'B' else 0
                
                X_list.append(combined_sequence)
                y_list.append(target)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def _get_position_in_original(self, original_data, no_tie_data, no_tie_index):
        """获取no_tie索引在原始数据中的位置"""
        count = 0
        for i, item in enumerate(original_data):
            if item != 'T':
                if count == no_tie_index:
                    return i
                count += 1
        return len(original_data)
    
    def predict(self, current_shoe_data):
        """
        预测下一局
        
        Args:
            current_shoe_data: 当前靴牌数据
            
        Returns:
            预测字典
        """
        if not self.is_trained:
            if self.can_train():
                self.train()
            else:
                return {
                    'B': 50,
                    'P': 50,
                    'T': 0,
                    'confidence': 0,
                    'method': 'shoe_aware_lstm',
                    'reason': f'数据不足（需要{self.min_shoes_required}个靴牌）'
                }
        
        no_tie = [x for x in current_shoe_data if x != 'T']
        
        if len(no_tie) < self.sequence_length:
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 20,
                'method': 'shoe_aware_lstm',
                'reason': f'当前靴牌数据不足（{len(no_tie)}局）'
            }
        
        try:
            import tensorflow as tf
            
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # 提取当前靴牌特征
            shoe_features = self.feature_extractor.extract_shoe_features(current_shoe_data)
            
            shoe_feature_vector = [
                shoe_features['shoe_stage'],
                shoe_features['banker_ratio'],
                shoe_features['player_ratio'],
                shoe_features['tie_ratio'],
                shoe_features['alternation_rate'],
                shoe_features['max_banker_streak'] / 10.0,
                shoe_features['max_player_streak'] / 10.0,
                shoe_features['current_streak_length'] / 10.0,
                shoe_features['current_streak_type'],
                shoe_features['is_banker_dominant'],
                shoe_features['is_player_dominant'],
                shoe_features['is_balanced'],
                shoe_features['is_jump_shoe'],
                shoe_features['is_long_dragon'],
                shoe_features['volatility'],
                shoe_features['shoe_length'] / 70.0
            ]
            
            # 构建输入序列
            sequence = no_tie[-self.sequence_length:]
            combined_sequence = []
            for result in sequence:
                result_encoded = 1 if result == 'B' else 0
                timestep_features = [result_encoded] + shoe_feature_vector
                combined_sequence.append(timestep_features)
            
            X = np.array(combined_sequence).reshape(1, self.sequence_length, -1)
            
            # 预测
            prediction = self.model.predict(X, verbose=0)[0]
            
            banker_prob = prediction[1] * 100
            player_prob = prediction[0] * 100
            
            # 置信度
            confidence = abs(banker_prob - player_prob) * 1.5
            confidence = min(100, max(30, confidence))
            
            # 生成原因说明
            shoe_type = []
            if shoe_features['is_banker_dominant']:
                shoe_type.append('庄强靴')
            elif shoe_features['is_player_dominant']:
                shoe_type.append('闲强靴')
            elif shoe_features['is_balanced']:
                shoe_type.append('平衡靴')
            
            if shoe_features['is_jump_shoe']:
                shoe_type.append('跳跳靴')
            if shoe_features['is_long_dragon']:
                shoe_type.append('长龙靴')
            
            reason = f"靴牌模式: {','.join(shoe_type)}" if shoe_type else "混合靴"
            
            return {
                'B': banker_prob,
                'P': player_prob,
                'T': 0,
                'confidence': confidence,
                'method': 'shoe_aware_lstm',
                'reason': reason
            }
            
        except Exception as e:
            print(f"ShoeAwareLSTM预测失败: {e}")
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 0,
                'method': 'shoe_aware_lstm',
                'reason': '预测出错'
            }

