"""
基于相似靴牌的预测器 (Case-Based Reasoning)

正确的思路：
1. 当前靴牌来了，分析其特征
2. 从历史中找到相似的靴牌
3. 看这些相似靴牌在相同位置后的走势
4. 基于实际走势预测

不再训练一个全局模型，而是直接利用历史案例
"""

import numpy as np
from shoe_aware_predictor import ShoeFeatureExtractor


class SimilarShoePredictor:
    """
    基于相似靴牌的预测器
    
    核心思想：
    - 不训练模型，直接使用历史案例
    - 找到与当前靴牌相似的历史靴牌
    - 基于它们的实际演化来预测
    """
    
    def __init__(self, historical_shoes):
        """
        初始化
        
        Args:
            historical_shoes: 历史靴牌列表 [{'name': 'xxx', 'data': 'BBPP...'}, ...]
        """
        self.historical_shoes = historical_shoes
        self.feature_extractor = ShoeFeatureExtractor()
        self.min_shoes_required = 5
    
    def can_predict(self):
        """检查是否有足够历史靴牌"""
        return len(self.historical_shoes) >= self.min_shoes_required
    
    def predict(self, current_shoe_data, top_n=5):
        """
        预测下一局
        
        Args:
            current_shoe_data: 当前靴牌数据
            top_n: 使用最相似的N个靴牌
            
        Returns:
            预测字典
        """
        if not self.can_predict():
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 0,
                'method': 'similar_shoe',
                'reason': f'历史靴牌不足（需要至少{self.min_shoes_required}个）'
            }
        
        if len(current_shoe_data) < 10:
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 20,
                'method': 'similar_shoe',
                'reason': '当前靴牌数据太少'
            }
        
        try:
            # 1. 提取当前靴牌的特征
            current_features = self.feature_extractor.extract_shoe_features(current_shoe_data)
            current_position = len(current_shoe_data)
            
            # 2. 找到最相似的历史靴牌
            similar_shoes = self._find_similar_shoes(current_features, current_position, top_n)
            
            if not similar_shoes:
                return {
                    'B': 50,
                    'P': 50,
                    'T': 0,
                    'confidence': 20,
                    'method': 'similar_shoe',
                    'reason': '未找到足够相似的靴牌'
                }
            
            # 3. 基于相似靴牌的后续走势预测
            prediction = self._predict_from_similar_shoes(
                similar_shoes,
                current_position,
                current_features
            )
            
            return prediction
            
        except Exception as e:
            print(f"SimilarShoe预测出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 0,
                'method': 'similar_shoe',
                'reason': '预测出错'
            }
    
    def _find_similar_shoes(self, current_features, current_position, top_n):
        """
        找到最相似的历史靴牌
        
        Args:
            current_features: 当前靴牌特征
            current_position: 当前位置（局数）
            top_n: 返回前N个
            
        Returns:
            相似靴牌列表
        """
        # 动态调整后续需要的局数（靴牌后期降低要求，避免样本不足）
        if current_position < 20:
            min_future_rounds = 10  # 前期：看后续10局
        elif current_position < 40:
            min_future_rounds = 7   # 中期：看后续7局
        else:
            min_future_rounds = 5   # 后期：看后续5局（避免样本太少）
        
        similarities = []
        
        for shoe in self.historical_shoes:
            shoe_data = list(shoe['data'])
            
            # 只考虑长度足够的靴牌（至少要有后续数据）
            if len(shoe_data) < current_position + min_future_rounds:
                continue
            
            # 提取该靴牌在相同位置的特征
            historical_features = self.feature_extractor.extract_shoe_features(
                shoe_data,
                current_position=current_position
            )
            
            # 计算相似度
            similarity = self._calculate_similarity(current_features, historical_features)
            
            similarities.append({
                'shoe': shoe,
                'similarity': similarity,
                'features': historical_features
            })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_n]
    
    def _calculate_similarity(self, features1, features2):
        """
        计算两个靴牌特征的相似度
        
        重点特征：
        - 庄闲比例
        - 靴牌类型（庄强/闲强/平衡）
        - 交替率
        - 波动性
        """
        # 权重配置（重要特征权重更高）
        weights = {
            'banker_ratio': 3.0,      # 庄闲比例最重要
            'player_ratio': 3.0,
            'alternation_rate': 2.0,  # 跳跳程度很重要
            'volatility': 2.0,        # 波动性重要
            'is_banker_dominant': 2.0,  # 类型特征重要
            'is_player_dominant': 2.0,
            'is_balanced': 1.5,
            'is_jump_shoe': 1.5,
            'is_long_dragon': 1.0,
            'tie_ratio': 0.5,         # 和局不太重要
            'shoe_stage': 0.5,        # 阶段不太重要（已经匹配位置了）
        }
        
        total_weight = sum(weights.values())
        similarity_score = 0
        
        for key, weight in weights.items():
            if key in features1 and key in features2:
                # 对于连续值特征，用1 - 差异
                if key in ['banker_ratio', 'player_ratio', 'alternation_rate', 
                          'volatility', 'tie_ratio', 'shoe_stage']:
                    diff = abs(features1[key] - features2[key])
                    # Fix(Improvement#2): 确保相似度不为负，diff > 1.0 时归零而非产生负权重
                    similarity = max(0.0, 1.0 - diff)
                    similarity_score += similarity * weight
                
                # 对于二值特征，相同得分
                elif key in ['is_banker_dominant', 'is_player_dominant', 'is_balanced',
                            'is_jump_shoe', 'is_long_dragon']:
                    if features1[key] == features2[key]:
                        similarity_score += weight
        
        # 归一化到0-100
        normalized_score = (similarity_score / total_weight) * 100
        
        return normalized_score
    
    def _predict_from_similar_shoes(self, similar_shoes, current_position, current_features):
        """
        基于相似靴牌的后续走势预测
        
        Args:
            similar_shoes: 相似靴牌列表
            current_position: 当前位置
            current_features: 当前特征
            
        Returns:
            预测字典
        """
        # 动态调整后续分析的局数（与查找逻辑保持一致）
        if current_position < 20:
            next_n_rounds = 10  # 前期：看后续10局
        elif current_position < 40:
            next_n_rounds = 7   # 中期：看后续7局
        else:
            next_n_rounds = 5   # 后期：看后续5局
        
        predictions = []
        total_weight = 0
        
        for similar in similar_shoes:
            shoe_data = list(similar['shoe']['data'])
            similarity = similar['similarity']
            
            # 提取后续N局
            next_rounds = shoe_data[current_position:current_position + next_n_rounds]
            next_rounds_no_tie = [x for x in next_rounds if x != 'T']
            
            if len(next_rounds_no_tie) < 3:
                continue
            
            # 统计后续走势
            b_count = next_rounds_no_tie.count('B')
            p_count = next_rounds_no_tie.count('P')
            total_next = len(next_rounds_no_tie)
            
            b_ratio = b_count / total_next
            p_ratio = p_count / total_next
            
            # 相似度作为权重
            weight = similarity / 100.0
            
            predictions.append({
                'B': b_ratio * 100,
                'P': p_ratio * 100,
                'weight': weight,
                'shoe_name': similar['shoe']['name'],
                'similarity': similarity
            })
            
            total_weight += weight
        
        if not predictions or total_weight == 0:
            return {
                'B': 50,
                'P': 50,
                'T': 0,
                'confidence': 20,
                'method': 'similar_shoe',
                'reason': '相似靴牌数据不足'
            }
        
        # 加权平均
        weighted_b = sum(p['B'] * p['weight'] for p in predictions) / total_weight
        weighted_p = sum(p['P'] * p['weight'] for p in predictions) / total_weight
        
        # 置信度基于相似度和一致性
        avg_similarity = sum(p['similarity'] for p in predictions) / len(predictions)
        
        # 检查预测一致性
        b_votes = sum(1 for p in predictions if p['B'] > p['P'])
        p_votes = len(predictions) - b_votes
        consistency = max(b_votes, p_votes) / len(predictions) * 100
        
        # 综合置信度
        confidence = (avg_similarity * 0.6 + consistency * 0.4)
        confidence = min(90, max(30, confidence))
        
        # 生成原因说明
        shoe_type = self._describe_shoe_type(current_features)
        similar_count = len(predictions)
        avg_sim = avg_similarity
        
        reason = f"基于{similar_count}个相似靴牌（相似度{avg_sim:.0f}%）: {shoe_type}"
        
        # 添加具体案例
        if len(predictions) > 0:
            top_similar = predictions[0]
            reason += f" | 最相似: {top_similar['shoe_name'][-8:]}({top_similar['similarity']:.0f}%)"
        
        return {
            'B': weighted_b,
            'P': weighted_p,
            'T': 0,
            'confidence': confidence,
            'method': 'similar_shoe',
            'reason': reason,
            'similar_shoes': predictions,  # 保留详细信息用于调试
            'avg_similarity': avg_similarity,
            'consistency': consistency
        }
    
    def _describe_shoe_type(self, features):
        """描述靴牌类型"""
        types = []
        
        if features['is_banker_dominant']:
            types.append('庄强靴')
        elif features['is_player_dominant']:
            types.append('闲强靴')
        elif features['is_balanced']:
            types.append('平衡靴')
        
        if features['is_jump_shoe']:
            types.append('跳跳靴')
        
        if features['is_long_dragon']:
            types.append('长龙靴')
        
        return ','.join(types) if types else '混合靴'

