"""
集成预测引擎
整合多个预测模型，给出最终预测结果和置信度评估
"""

from analyzer import BaccaratAnalyzer
from predictor import (
    LSTMPredictor, 
    RandomForestPredictor,
    FrequencyPredictor
)
from predictor_with_features import (
    LSTMPredictorV2,
    RandomForestPredictorV2
)
from historical_ensemble import HistoricalEnsemble, IntraShoeNgramPredictor
from advanced_shoe_analyzer import AdvancedShoeAnalyzer
from anomaly_detector import AnomalyDetector
from shoe_regime_detector import ShoeRegimeDetector
from derived_road_detector import DerivedRoadMetaRuleDetector
from three_bead_analyzer import ThreeBeadAnalyzer
from utils import result_to_chinese
from prediction_history import PredictionHistory
from similar_shoe_predictor import SimilarShoePredictor


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, all_data, current_shoe_data, shoes=None, prediction_history=None, enabled_models=None):
        """
        初始化集成预测器
        
        Args:
            all_data: 所有历史数据（包括跨靴牌）
            current_shoe_data: 当前靴牌数据
            shoes: 靴牌列表（用于数据驱动分析）
            prediction_history: 预测历史记录器（可选）
            enabled_models: 允许的模型名称集合，例如 {'Historical','Streak'}
                           为 None 时启用所有模型（默认行为）
        """
        self.all_data = all_data
        self.current_shoe_data = current_shoe_data
        self.combined_data = all_data + current_shoe_data
        self.shoes = shoes if shoes else []
        self.enabled_models = enabled_models  # None = 全部启用
        
        # 初始化分析器（传入当前靴牌数据，避免Streak跨靴牌）
        self.analyzer = BaccaratAnalyzer(self.combined_data, self.current_shoe_data)
        
        # 初始化历史模型集成（方案A：合并冗余模型）
        self.historical = HistoricalEnsemble(
            self.combined_data, self.analyzer, self.shoes,
            current_shoe_data=self.current_shoe_data
        )
        
        # 初始化独立预测器（原始版本）
        self.lstm = LSTMPredictor(self.combined_data)
        self.rf = RandomForestPredictor(self.combined_data)
        self.freq = FrequencyPredictor(self.combined_data)
        
        # 初始化特征工程预测器（V2版本）
        self.lstm_v2 = None
        self.rf_v2 = None
        if self.shoes and len(self.shoes) > 0:
            self.lstm_v2 = LSTMPredictorV2(self.shoes)
            self.rf_v2 = RandomForestPredictorV2(self.shoes)
        
        # 初始化相似靴牌预测器（CBR方法）
        self.similar_shoe = None
        if self.shoes and len(self.shoes) >= 5:
            self.similar_shoe = SimilarShoePredictor(self.shoes)
        
        # 初始化高级靴牌分析器
        self.shoe_analyzer = None
        if self.shoes and len(self.shoes) > 0:
            self.shoe_analyzer = AdvancedShoeAnalyzer(self.shoes)
        
        # 初始化异常检测器
        self.anomaly_detector = AnomalyDetector()

        # 初始化靴牌性格实时评估器（Regime Detector）
        self.regime_detector = ShoeRegimeDetector()

        # 初始化派生路元规律检测器
        self.derived_road = DerivedRoadMetaRuleDetector()

        # V2 新增：初始化三珠路分析引擎
        self.three_bead = ThreeBeadAnalyzer()

        # 预测历史记录
        self.history = prediction_history if prediction_history else PredictionHistory()
        
        # 最后一次的预测结果
        self.last_prediction = None
        self.last_model_predictions = {}
    
    def _model_enabled(self, model_name):
        """检查模型是否启用"""
        if self.enabled_models is None:
            return True
        return model_name in self.enabled_models
        
    def update_data(self, all_data, current_shoe_data):
        """
        更新模型底层的数据指针（关键修复：防止旧List引用导致预测冻结）

        Audit#A 修复：同步 IntraShoeNgram（原来没有同步，
        导致每局结束后的靴内N-gram一直用旧数据）
        """
        self.all_data = all_data
        self.current_shoe_data = current_shoe_data
        self.combined_data = all_data + current_shoe_data

        # 同步 Analyzer 层
        if hasattr(self, 'analyzer') and self.analyzer:
            self.analyzer.data = self.combined_data
            self.analyzer.current_shoe_data = self.current_shoe_data
            self.analyzer.data_no_tie = [x for x in self.combined_data if x != 'T']
            self.analyzer.current_shoe_no_tie = [x for x in self.current_shoe_data if x != 'T']

        # 同步 Historical 层
        if hasattr(self, 'historical') and self.historical:
            self.historical.combined_data = self.combined_data
            self.historical.current_shoe_data = self.current_shoe_data
            if hasattr(self.historical, 'markov') and self.historical.markov:
                self.historical.markov.data = [x for x in self.current_shoe_data if x != 'T']
                self.historical.markov.transition_matrix = self.historical.markov._build_transition_matrix()
            # Audit#A修复：重建 IntraShoeNgram，使其持有最新靴内数据
            if hasattr(self.historical, 'intra_ngram'):
                self.historical.intra_ngram = IntraShoeNgramPredictor(current_shoe_data)

    def predict_next(self):
        """
        预测下一局结果
        
        Returns:
            完整的预测字典
        """
        predictions = []
        
        # 记录各模型的推荐（用于更新历史）
        self.last_model_predictions = {}
        
        # ── Step 1: 靴牌性格实时评估（Regime Detector）──
        # 每次预测都从第一局到当前整靴重新分析，获取主导规律强度和置信度因子
        regime_result = self.regime_detector.analyze(self.current_shoe_data)
        regime_weight_adj = self.regime_detector.get_model_weight_adjustments(regime_result)
        
        # ── Step 2: 历史准确率权重调整（渐进式，20局窗口）──
        weight_adjustment, adjustment_reason = self._get_weight_adjustment()
        
        # ── Step 3: 高级靴牌分析 ──
        shoe_analysis = None
        shoe_weight_adj = {}
        if self.shoe_analyzer and len(self.current_shoe_data) >= 10:
            shoe_analysis = self._get_shoe_analysis()
            # Bug#13修复 + Audit#B修复：不再传入 regime_result，避免 regime 权重
            # 在 shoe_weight_adj 和 regime_weight_adj 中被重复应用。
            # Regime 权重统一由 _apply_regime_weight 负责。
            shoe_adjustment = self.shoe_analyzer.get_shoe_weight_adjustment(
                self.current_shoe_data, regime_result=None
            )
            shoe_weight_adj = shoe_adjustment.get('model_adjustments', {})

        # ── Step 4: 异常检测（Bug#14：渐进式三档）──
        anomaly_result = self.anomaly_detector.detect(self.current_shoe_data)
        
        # 1. 历史模型集成（合并N-gram + Markov + DataDriven）
        if self._model_enabled('Historical') and self.historical.can_predict():
            hist_pred = self.historical.predict()
            if hist_pred['confidence'] > 30:
                model_name = 'Historical'
                base_weight = 0.8
                adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                predictions.append({
                    'name': model_name,
                    'weight': adjusted_weight,
                    'base_weight': base_weight,
                    'prediction': hist_pred
                })
                self.last_model_predictions[model_name] = self._get_prediction_choice(hist_pred)

        # 2. 趋势预测
        if self._model_enabled('Trend'):
            trend_pred = self.analyzer.predict_by_trend()
            if trend_pred['confidence'] > 30:
                model_name = 'Trend'
                base_weight = 0.5
                adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                predictions.append({
                    'name': model_name, 'weight': adjusted_weight,
                    'base_weight': base_weight, 'prediction': trend_pred
                })
                self.last_model_predictions[model_name] = self._get_prediction_choice(trend_pred)

        # 3. 连胜预测
        if self._model_enabled('Streak'):
            streak_pred = self.analyzer.predict_by_streak()
            if streak_pred['confidence'] > 30:
                model_name = 'Streak'
                base_weight = 0.6
                adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                predictions.append({
                    'name': model_name, 'weight': adjusted_weight,
                    'base_weight': base_weight, 'prediction': streak_pred
                })
                self.last_model_predictions[model_name] = self._get_prediction_choice(streak_pred)

        # 4. 频率预测
        if self._model_enabled('Frequency'):
            freq_pred = self.freq.predict(self.current_shoe_data, window=len(self.current_shoe_data))
            if freq_pred['confidence'] > 0:
                model_name = 'Frequency'
                base_weight = 0.5
                adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                predictions.append({
                    'name': model_name, 'weight': adjusted_weight,
                    'base_weight': base_weight, 'prediction': freq_pred
                })
                self.last_model_predictions[model_name] = self._get_prediction_choice(freq_pred)

        # 5. LSTM预测
        if self._model_enabled('LSTM') and self.lstm.can_train():
            try:
                lstm_pred = self.lstm.predict(self.current_shoe_data)
                if lstm_pred['confidence'] > 0:
                    model_name = 'LSTM'
                    base_weight = 0.8
                    adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                    adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                    predictions.append({'name': model_name, 'weight': adjusted_weight, 'base_weight': base_weight, 'prediction': lstm_pred})
                    self.last_model_predictions[model_name] = self._get_prediction_choice(lstm_pred)
            except BaseException:
                pass

        # 6. 随机森林预测
        if self._model_enabled('RandomForest') and self.rf.can_train():
            try:
                rf_pred = self.rf.predict(self.current_shoe_data)
                if rf_pred['confidence'] > 0:
                    model_name = 'RandomForest'
                    base_weight = 0.5
                    adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                    adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                    predictions.append({'name': model_name, 'weight': adjusted_weight, 'base_weight': base_weight, 'prediction': rf_pred})
                    self.last_model_predictions[model_name] = self._get_prediction_choice(rf_pred)
            except BaseException:
                pass

        # 7. LSTM V2预测
        if self._model_enabled('LSTM_V2') and self.lstm_v2 and self.lstm_v2.can_train():
            try:
                lstm_v2_pred = self.lstm_v2.predict(self.current_shoe_data)
                if lstm_v2_pred['confidence'] > 0:
                    model_name = 'LSTM_V2'
                    base_weight = 0.9
                    adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                    adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                    predictions.append({'name': model_name, 'weight': adjusted_weight, 'base_weight': base_weight, 'prediction': lstm_v2_pred})
                    self.last_model_predictions[model_name] = self._get_prediction_choice(lstm_v2_pred)
            except BaseException:
                pass

        # 8. 随机森林V2预测
        if self._model_enabled('RF_V2') and self.rf_v2 and self.rf_v2.can_train():
            try:
                rf_v2_pred = self.rf_v2.predict(self.current_shoe_data)
                if rf_v2_pred['confidence'] > 0:
                    model_name = 'RF_V2'
                    base_weight = 0.7
                    adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                    adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                    predictions.append({'name': model_name, 'weight': adjusted_weight, 'base_weight': base_weight, 'prediction': rf_v2_pred})
                    self.last_model_predictions[model_name] = self._get_prediction_choice(rf_v2_pred)
            except BaseException:
                pass

        # 9. 相似靴牌预测器
        # Fix(Bug#1): SimilarShoe 权重从 4.0 降至 1.2，与 Historical/Markov 持平，避免一家独大
        if self._model_enabled('SimilarShoe') and self.similar_shoe and self.similar_shoe.can_predict():
            similar_pred = self.similar_shoe.predict(self.current_shoe_data, top_n=5)
            if similar_pred['confidence'] > 0:
                model_name = 'SimilarShoe'
                base_weight = 1.2  # Fix(Bug#1): 从 4.0 降至 1.2
                adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                
                predictions.append({
                    'name': model_name,
                    'weight': adjusted_weight,
                    'base_weight': base_weight,
                    'prediction': similar_pred
                })
                
                self.last_model_predictions[model_name] = self._get_prediction_choice(similar_pred)

        # 10. Audit#B新增：双跳（双对 BBPP）模式预测
        # 原 predict_by_double_alternation 方法已存在但未集成到 predict_next
        if self._model_enabled('DoubleAlt'):
            double_pred = self.analyzer.predict_by_double_alternation(regime_result)
            if double_pred['confidence'] > 30:
                model_name = 'DoubleAlt'
                base_weight = 0.5
                adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                predictions.append({
                    'name': model_name, 'weight': adjusted_weight,
                    'base_weight': base_weight, 'prediction': double_pred
                })
                self.last_model_predictions[model_name] = self._get_prediction_choice(double_pred)

        # 11. 派生路元规律预测（DerivedRoadMetaRule）
        # 靠靠内实时学习”逢红就续/逢蓝就跳”等元规律
        if self._model_enabled('DerivedRoad'):
            derived_result = self.derived_road.analyze(self.current_shoe_data)
            if derived_result.get('can_predict', False):
                model_name = 'DerivedRoad'
                base_weight = self.derived_road.get_prediction_weight(derived_result, regime_result)
                if base_weight > 0.1:
                    adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, {})
                    derived_pred = derived_result['prediction']
                    predictions.append({
                        'name': model_name,
                        'weight': adjusted_weight,
                        'base_weight': base_weight,
                        'prediction': derived_pred
                    })
                    self.last_model_predictions[model_name] = self._get_prediction_choice(derived_pred)

        # 12. V2 新增：三珠路分析引擎 (ThreeBead)
        if self._model_enabled('ThreeBead'):
            three_bead_result = self.three_bead.analyze(self.current_shoe_data)
            if three_bead_result.get('can_predict', False):
                tb_pred = three_bead_result['prediction']
                if tb_pred.get('confidence', 0) > 0:
                    model_name = 'ThreeBead'
                    base_weight = 0.6
                    adjusted_weight = self._adjust_model_weight(model_name, base_weight, weight_adjustment, shoe_weight_adj)
                    adjusted_weight = self._apply_regime_weight(model_name, adjusted_weight, regime_weight_adj)
                    predictions.append({
                        'name': model_name,
                        'weight': adjusted_weight,
                        'base_weight': base_weight,
                        'prediction': tb_pred
                    })
                    self.last_model_predictions[model_name] = self._get_prediction_choice(tb_pred)

        # 集成所有预测
        final_prediction = self._ensemble_predictions(predictions)
        
        # 记录最后预测（用于更新）
        self.last_prediction = final_prediction['recommendation']
        
        # 添加详细信息
        final_prediction['individual_predictions'] = predictions
        final_prediction['data_stats'] = {
            'total_data': len(self.combined_data),
            'current_shoe': len(self.current_shoe_data),
            'history_data': len(self.all_data)
        }
        
        # 添加权重调整信息
        if adjustment_reason:
            final_prediction['weight_adjustment'] = {
                'applied': True,
                'reason': adjustment_reason,
                'adjustment': weight_adjustment
            }
        
        # 添加靴牌分析信息
        if shoe_analysis:
            final_prediction['shoe_analysis'] = shoe_analysis

        # 添加 Regime 分析信息（靴牌性格评估结果）
        final_prediction['regime_analysis'] = regime_result

        # V2：添加模式断裂预警和稳定性信息
        if regime_result.get('break_detected', False):
            final_prediction['pattern_break_warning'] = True
            final_prediction['pattern_break_msg'] = regime_result.get('recommendation', '')
        final_prediction['stability_index'] = regime_result.get('stability_index', 0.5)

        # 应用 Regime 的置信度因子（会综合异常检测共同调整）
        regime_cf = regime_result.get('confidence_factor', 1.0)
        if regime_cf != 1.0:
            original_confidence = final_prediction['confidence']
            final_prediction['confidence'] = original_confidence * regime_cf
            final_prediction['regime_adjusted'] = True
        else:
            final_prediction['regime_adjusted'] = False

        # 添加异常检测信息并调整置信度
        if anomaly_result['is_anomaly']:
            final_prediction['anomaly_detection'] = anomaly_result
            
            # 根据异常程度调整置信度
            original_confidence = final_prediction['confidence']
            adjustment_factor = anomaly_result['confidence_adjustment']
            adjusted_confidence = original_confidence * adjustment_factor
            
            final_prediction['confidence'] = adjusted_confidence
            final_prediction['original_confidence'] = original_confidence
            final_prediction['anomaly_adjusted'] = True
        else:
            final_prediction['anomaly_detection'] = anomaly_result
            final_prediction['anomaly_adjusted'] = False
        
        # 添加预测历史统计
        final_prediction['history_stats'] = self.history.get_statistics()
        
        return final_prediction
    
    def _normalize_probabilities(self, b_prob, p_prob, t_prob):
        """
        概率归一化辅助方法

        Fix(Bug#2): 先将 T 概率限定在合理范围，然后将 B/P 的总和
        按原始比例缩放至 (100 - t_prob)，确保 B + P + T == 100。
        移除了原先分别夹紧 B 和 P 导致总和不为100的逻辑缺陷。

        Args:
            b_prob: 庄家概率（任意值）
            p_prob: 闲家概率（任意值）
            t_prob: 和局概率（任意值）

        Returns:
            (b, p, t) 归一化后的三元组，总和为100
        """
        # 1. 将 T 概率限制在合理范围（不允许和局概率超过30%或低于0%）
        t_prob = max(0.0, min(30.0, t_prob))
        
        # 2. B 和 P 的总和必须等于 100 - t_prob
        bp_total = 100.0 - t_prob
        
        raw_bp = b_prob + p_prob
        if raw_bp <= 0:
            # 极端情况：按理论概率分配
            b_prob = 45.86 / (45.86 + 44.62) * bp_total
            p_prob = bp_total - b_prob
        else:
            # 按原始比例缩放
            scale = bp_total / raw_bp
            b_prob = b_prob * scale
            p_prob = p_prob * scale
        
        # 3. 确保 B/P 不出现极端值（最高不超过80%，最低不低于10%）
        # 使用 softmax-like 方法而非硬夹紧
        MAX_SINGLE = 78.0
        MIN_SINGLE = 10.0
        
        if b_prob > MAX_SINGLE:
            excess = b_prob - MAX_SINGLE
            b_prob = MAX_SINGLE
            p_prob = min(p_prob + excess, bp_total - MIN_SINGLE)
        elif p_prob > MAX_SINGLE:
            excess = p_prob - MAX_SINGLE
            p_prob = MAX_SINGLE
            b_prob = min(b_prob + excess, bp_total - MIN_SINGLE)
        
        b_prob = max(MIN_SINGLE, b_prob)
        p_prob = max(MIN_SINGLE, p_prob)
        
        # 4. 最终精确归一化（消除浮点误差）
        total = b_prob + p_prob + t_prob
        if total > 0:
            factor = 100.0 / total
            b_prob *= factor
            p_prob *= factor
            t_prob *= factor
        
        return b_prob, p_prob, t_prob
    
    def _ensemble_predictions(self, predictions):
        """
        集成多个预测结果

        Fix(Bug#2): 使用 _normalize_probabilities 保证概率和为 100
        Fix(Bug#11): 简化投票/概率决策逻辑：prob_diff > 1% 选概率方，否则选投票方
        
        Args:
            predictions: 预测列表
            
        Returns:
            集成后的预测字典
        """
        if not predictions:
            return {
                'B': 45.86,
                'P': 44.62,
                'T': 9.52,
                'confidence': 20,
                'recommendation': '闲',
                'reason': '数据不足，使用理论概率',
                'strength': 'weak'
            }
        
        # 加权平均
        total_weight = 0
        weighted_b = 0
        weighted_p = 0
        weighted_t = 0
        avg_confidence = 0
        
        for pred_info in predictions:
            weight = pred_info['weight']
            pred = pred_info['prediction']
            confidence = pred['confidence']
            
            # 获取预测概率
            b_prob = pred.get('B', 0)
            p_prob = pred.get('P', 0)
            t_prob = pred.get('T', 9.52)
            
            # Fix(Bug#2): 用统一的归一化方法处理单个预测的概率
            b_prob, p_prob, t_prob = self._normalize_probabilities(b_prob, p_prob, t_prob)
            
            # 置信度因子
            confidence_factor = confidence / 100
            adjusted_weight = weight * confidence_factor
            
            weighted_b += b_prob * adjusted_weight
            weighted_p += p_prob * adjusted_weight
            weighted_t += t_prob * adjusted_weight
            
            total_weight += adjusted_weight
            avg_confidence += confidence
        
        if total_weight > 0:
            raw_b = weighted_b / total_weight
            raw_p = weighted_p / total_weight
            raw_t = weighted_t / total_weight
        else:
            raw_b, raw_p, raw_t = 45.86, 44.62, 9.52
        
        # 对最终集成结果再做一次归一化
        final_b, final_p, final_t = self._normalize_probabilities(raw_b, raw_p, raw_t)
        
        avg_confidence = avg_confidence / len(predictions) if predictions else 20
        
        # 计算投票结果（简化逻辑，只用于辅助平局决策）
        votes = {'B': 0, 'P': 0, 'T': 0}
        for pred_info in predictions:
            pred = pred_info['prediction']
            if pred['B'] >= pred['P'] and pred['B'] >= pred.get('T', 0):
                votes['B'] += 1
            elif pred['P'] >= pred['B'] and pred['P'] >= pred.get('T', 0):
                votes['P'] += 1
            else:
                votes['T'] += 1
        
        total_votes = sum(votes.values())
        consistency = max(votes.values()) / total_votes * 100 if total_votes > 0 else 50
        
        # 确定推荐 —— Fix(Bug#11): 简化为两步逻辑
        if final_b > final_p and final_b > final_t:
            prob_winner = 'B'
            prob_diff = final_b - max(final_p, final_t)
        elif final_p > final_b and final_p > final_t:
            prob_winner = 'P'
            prob_diff = final_p - max(final_b, final_t)
        else:
            prob_winner = 'T'
            prob_diff = final_t - max(final_b, final_p)
        
        # Fix(Bug#11): 简化决策：概率差 > 1% 就选概率方，否则投票方
        vote_winner = max(votes, key=votes.get)
        if prob_diff >= 1.0:
            recommendation = prob_winner
        else:
            recommendation = vote_winner
        
        # 评估预测强度
        if prob_diff > 15 and avg_confidence > 60:
            strength = 'strong'
            strength_desc = '强烈推荐'
        elif prob_diff > 8 and avg_confidence > 45:
            strength = 'medium'
            strength_desc = '推荐'
        else:
            strength = 'weak'
            strength_desc = '谨慎参考'
        
        # 生成原因说明
        recommendation_reasons = []
        other_reasons = []
        
        for pred_info in predictions:
            pred = pred_info['prediction']
            pred_choice = self._get_prediction_choice(pred)
            
            if 'reason' in pred:
                if pred_choice == recommendation:
                    recommendation_reasons.append(f"{pred_info['name']}: {pred['reason']}")
                else:
                    other_reasons.append(f"{pred_info['name']}: {pred['reason']}")
        
        all_reasons = recommendation_reasons + other_reasons
        reason = '; '.join(all_reasons[:2]) if all_reasons else '综合多模型分析'
        
        # 调整置信度（考虑一致性）
        final_confidence = (avg_confidence * 0.7 + consistency * 0.3)
        
        # 决策说明
        if recommendation != prob_winner:
            decision_note = f"概率差异极小(<1%)，采用模型投票结果"
        elif recommendation != vote_winner and vote_winner != 'T':
            decision_note = f"概率倾向{result_to_chinese(prob_winner)}，超出投票推荐，综合判断选择{result_to_chinese(recommendation)}"
        else:
            decision_note = None
        
        return {
            'B': final_b,
            'P': final_p,
            'T': final_t,
            'recommendation': recommendation,
            'recommendation_cn': result_to_chinese(recommendation),
            'confidence': final_confidence,
            'strength': strength,
            'strength_desc': strength_desc,
            'reason': reason,
            'prob_difference': prob_diff,
            'model_count': len(predictions),
            'consistency': consistency,
            'votes': votes,
            'decision_note': decision_note,
            'vote_winner': vote_winner,
            'prob_winner': prob_winner
        }
    
    def get_detailed_analysis(self):
        """
        获取详细分析报告
        
        Returns:
            完整分析字典
        """
        # 获取预测
        prediction = self.predict_next()
        
        # 获取综合分析
        analysis = self.analyzer.get_comprehensive_analysis()
        
        # 合并
        return {
            'prediction': prediction,
            'analysis': analysis
        }
    
    def _get_shoe_analysis(self):
        """
        获取当前靴牌的高级分析
        
        Returns:
            靴牌分析结果字典
        """
        if not self.shoe_analyzer or len(self.current_shoe_data) < 10:
            return None
        
        # 1. 靴牌类型识别
        shoe_type = self.shoe_analyzer.classify_shoe_type(self.current_shoe_data)
        
        # 2. 靴牌阶段分析
        shoe_phases = self.shoe_analyzer.analyze_shoe_phases(self.current_shoe_data)
        
        # 3. 查找相似靴牌
        similar_shoes = self.shoe_analyzer.find_similar_shoes(self.current_shoe_data, top_n=5)
        
        return {
            'shoe_type': shoe_type,
            'shoe_phases': shoe_phases,
            'similar_shoes': similar_shoes
        }
    
    def get_betting_suggestion(self, bankroll=1000, risk_level='medium'):
        """
        获取投注建议（包含资金管理）
        
        Args:
            bankroll: 本金
            risk_level: 风险等级 ('low', 'medium', 'high')
            
        Returns:
            投注建议字典
        """
        prediction = self.predict_next()
        
        # 风险系数
        risk_factors = {
            'low': 0.02,      # 每次投注本金的2%
            'medium': 0.05,   # 5%
            'high': 0.10      # 10%
        }
        
        base_bet = bankroll * risk_factors.get(risk_level, 0.05)
        
        # 根据置信度和强度调整投注额
        confidence = prediction['confidence']
        strength = prediction['strength']
        
        if strength == 'strong':
            bet_multiplier = 1.5
        elif strength == 'medium':
            bet_multiplier = 1.0
        else:
            bet_multiplier = 0.5
        
        # 置信度调整
        confidence_multiplier = confidence / 100
        
        suggested_bet = base_bet * bet_multiplier * confidence_multiplier
        
        # 限制最大投注（不超过本金的15%）
        suggested_bet = min(suggested_bet, bankroll * 0.15)
        
        # 最小投注
        suggested_bet = max(suggested_bet, bankroll * 0.01)
        
        return {
            'target': prediction['recommendation_cn'],
            'suggested_amount': round(suggested_bet, 2),
            'confidence': confidence,
            'strength': prediction['strength_desc'],
            'risk_level': risk_level,
            'reason': prediction['reason'],
            'warning': '请理性投注，控制风险' if strength == 'weak' else None
        }
    
    def recommend_skip(self, regime_result=None):
        """
        判断当前是否建议跳过预测（不下注）

        扩展条件（新增 Regime 驱动）：
        - 连续错误 >= 4 次
        - Regime 检测到规律切换（switch_event == True）
        - Regime 置信度因子 <= 0.65（亂路状态）

        Audit#B修复：接受外部传入的 regime_result，避免重复调用 analyze()
        导致 regime_history 被污染。

        Returns:
            (should_skip: bool, reason: str)
        """
        consecutive_errors = self.history.get_consecutive_errors()
        if consecutive_errors >= 4:
            return True, f"连续预测错误{consecutive_errors}次，建议暂时观望"

        # Regime 驱动的跳过判断（优先使用传入的结果，避免重复 analyze）
        if regime_result is None and len(self.current_shoe_data) >= 8:
            regime_result = self.regime_detector.analyze(self.current_shoe_data)

        if regime_result and regime_result.get('can_analyze', False):
            # V2：模式断裂检测
            if regime_result.get('break_detected', False):
                return True, f"⚡ 模式断裂！{regime_result['recommendation']}，建议暂停等待新模式形成"
            if regime_result.get('switch_event', False):
                return True, f"⚠️ 靴牌走势切换中：{regime_result['recommendation']}，建议观察稳定后再入场"
            if regime_result.get('confidence_factor', 1.0) <= 0.65:
                return True, f"⚠️ 当前乱路状态（主导强度仅{regime_result['regime_strength']:.0f}%），建议观望"

        return False, None
    
    def update_prediction_result(self, actual_result):
        """
        更新预测结果（在实际结果出来后调用）
        
        Args:
            actual_result: 实际开出的结果 ('B', 'P', 或 'T')
        """
        if self.last_prediction is None:
            return
        
        # 更新总体预测历史
        self.history.add_prediction(
            predicted=self.last_prediction,
            actual=actual_result,
            model_predictions=self.last_model_predictions
        )
    
    def _get_weight_adjustment(self):
        """
        获取权重调整系数

        Fix(Bug#10): 改为基于20局滑动窗口的渐进式调整，取代原来急剧的逐局响应。
        逻辑：
        - 只有连续错误 >= 3 次才触发调整
        - 调整幅度与窗口期准确率挂钩，而非急剧惩罚
        - 对表现差的模型最多降至0.4倍，对表现好的最多提至2.0倍
        
        Returns:
            (调整系数字典, 原因说明)
        """
        adjustment = {}
        reason = None
        
        # 只有连续错误 >= 3 次才触发调整
        consecutive_errors = self.history.get_consecutive_errors()
        if consecutive_errors < 3:
            return adjustment, reason
        
        reason = f"连续预测错误{consecutive_errors}次，渐进式调整权重"
        
        # 基于20局窗口的模型准确率来渐进调整
        for model_name in list(self.last_model_predictions.keys()):
            accuracy, valid_count = self.history.get_model_recent_accuracy(model_name, window=20)
            
            if valid_count < 5:
                # 历史不足，不调整
                continue
            
            # 准确率与理论基线50%的偏差决定调整幅度
            # accuracy < 0.4 → 降权（最低0.4）
            # accuracy > 0.6 → 升权（最高2.0）
            # 0.4~0.6 之间 → 轻微到无调整
            if accuracy < 0.35:
                adjustment[model_name] = 0.4   # 明显较差：降至40%权重
            elif accuracy < 0.44:
                adjustment[model_name] = 0.65  # 略差：降至65%权重
            elif accuracy > 0.65:
                adjustment[model_name] = 2.0   # 明显较好：升至200%权重
            elif accuracy > 0.56:
                adjustment[model_name] = 1.4   # 略好：升至140%权重
            # else: 在 0.44~0.56 之间，不调整
        
        return adjustment, reason
    
    def _adjust_model_weight(self, model_name, base_weight, adjustment_dict, shoe_adjustment_dict=None):
        """
        调整单个模型的权重（基于历史准确率 + 靴牌类型调整）

        Args:
            model_name: 模型名称
            base_weight: 基础权重
            adjustment_dict: 调整字典（基于预测历史准确率）
            shoe_adjustment_dict: 靴牌类型调整字典（fallback，已被 Regime 替代）

        Returns:
            调整后的权重
        """
        weight = base_weight

        if adjustment_dict and model_name in adjustment_dict:
            weight *= adjustment_dict[model_name]

        if shoe_adjustment_dict and model_name in shoe_adjustment_dict:
            weight *= shoe_adjustment_dict[model_name]

        return max(0.01, weight)

    def _apply_regime_weight(self, model_name, current_weight, regime_weight_adj):
        """
        应用 Regime Detector 的权重调整（在 shoe_weight_adj 之后叠加）

        Args:
            model_name: 模型名称
            current_weight: 当前权重（已经过历史准确率和鞋型调整）
            regime_weight_adj: ShoeRegimeDetector.get_model_weight_adjustments() 结果

        Returns:
            最终权重
        """
        if regime_weight_adj and model_name in regime_weight_adj:
            current_weight *= regime_weight_adj[model_name]
        return max(0.01, current_weight)
    
    def _get_prediction_choice(self, prediction):
        """
        从预测结果中提取推荐选择
        
        Args:
            prediction: 预测字典
            
        Returns:
            推荐结果 ('B', 'P', 或 'T')
        """
        b_prob = prediction.get('B', 0)
        p_prob = prediction.get('P', 0)
        t_prob = prediction.get('T', 0)
        
        max_prob = max(b_prob, p_prob, t_prob)
        
        if b_prob == max_prob:
            return 'B'
        elif p_prob == max_prob:
            return 'P'
        else:
            return 'T'
