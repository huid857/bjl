"""
历史模型集成
将N-gram、Markov、DataDriven合并为一个综合模型，避免冗余

重要：Markov 和 N-gram 必须只在当前靴牌内运行（current_shoe_data），
不能使用 combined_data（跨靴牌数据），否则会学到靴牌间虚假的序列关系。

Bug#15修复：新增靴内 N-gram（IntraShoeNgram），专门从当前靴牌第1局到
            当前局学习转移模式，按局数渐进加权：
              < 10 局 → 权重 0（不参与）
              10~20 局 → 权重 0.4
              >= 20 局 → 权重 1.0
"""

import warnings
from collections import defaultdict
from predictor import MarkovPredictor
from data_driven_predictor import DataDrivenPredictor


# ──────────────────────────────────────────────────────────────
# 靴内 N-gram 预测器（Bug#15 修复）
# ──────────────────────────────────────────────────────────────

class IntraShoeNgramPredictor:
    """
    从当前靴牌第1局到当前局实时学习的 N-gram 预测器。

    目的：实现用户的核心想法——"每到下一局都从第一局到当前分析规律"。
    与历史先验 N-gram 的关键区别：
      - 历史先验 N-gram 学自 15000+ 局历史（跨靴）
      - 靴内 N-gram 只从当前靴牌学习，捕捉本靴节奏特征
    """

    def __init__(self, current_shoe_data: list):
        self.data = [x for x in current_shoe_data if x != 'T']  # 去和局

    def predict(self, n=3) -> dict:
        """
        基于当前靴牌内的序列统计预测下一局。

        Returns:
            预测字典，若数据不足则 confidence=0
        """
        seq = self.data
        k = n - 1  # 需要匹配的前缀长度

        if len(seq) < n + 2:
            # 数据太少，给出 confidence=0 让上层过滤
            return {'B': 50.0, 'P': 50.0, 'T': 0.0,
                    'confidence': 0, 'method': 'intra_shoe_ngram',
                    'reason': '靴内数据不足'}

        # 构建靴内转移统计
        counts = defaultdict(lambda: {'B': 0, 'P': 0})
        for i in range(len(seq) - k):
            pattern = ''.join(seq[i: i + k])
            nxt = seq[i + k]
            if nxt in ('B', 'P'):
                counts[pattern][nxt] += 1

        # 当前匹配
        current_pattern = ''.join(seq[-k:])
        if current_pattern not in counts:
            # 降阶尝试
            if n > 2:
                lower = IntraShoeNgramPredictor(self.data)
                lower.data = self.data
                return lower.predict(n - 1)
            return {'B': 50.0, 'P': 50.0, 'T': 0.0,
                    'confidence': 0, 'method': 'intra_shoe_ngram',
                    'reason': '靴内无匹配模式'}

        c = counts[current_pattern]
        total = c['B'] + c['P']
        if total == 0:
            return {'B': 50.0, 'P': 50.0, 'T': 0.0,
                    'confidence': 0, 'method': 'intra_shoe_ngram',
                    'reason': '靴内样本为零'}

        b_prob = c['B'] / total * 100
        p_prob = c['P'] / total * 100

        # 置信度：样本数 * 8，但最高 65%
        confidence = min(65, total * 8)

        return {
            'B': b_prob,
            'P': p_prob,
            'T': 0.0,
            'confidence': confidence,
            'method': 'intra_shoe_ngram',
            'reason': f'靴内{n}-gram，模式{current_pattern}，样本{total}次',
            'sample_count': total,
        }

    @property
    def intra_weight(self) -> float:
        """
        根据靴内局数返回该预测器的参与权重（渐进加权）：

        Audit#E 修复：
          < 10 局  → 0.0（不参与）
          10 局    → 0.1（最低参与，不再是 0）
          11~19 局 → 线性从 0.1 升到 1.0
          >= 20 局 → 1.0
        """
        n = len(self.data)  # 已去和局
        if n < 10:
            return 0.0
        if n >= 20:
            return 1.0
        # 10局→0.1, 20局→1.0，线性
        return 0.1 + (n - 10) / 10.0 * 0.9


# ──────────────────────────────────────────────────────────────
# 历史模型集成主类
# ──────────────────────────────────────────────────────────────

class HistoricalEnsemble:
    """
    历史模型集成

    内部集成多个基于历史统计的模型：
    - 历史先验 N-gram（跨靴历史字典）
    - 靴内 N-gram（当前靴牌自适应，Bug#15新增）
    - Markov 链（只用当前靴）
    - DataDriven 靴内统计

    对外输出 1 个综合结果，避免模型冗余
    """

    def __init__(self, combined_data, analyzer, shoes=None, current_shoe_data=None):
        """
        初始化历史模型集成

        Args:
            combined_data: 组合数据（历史+当前），供 DataDriven 使用
            analyzer: BaccaratAnalyzer实例（用于历史先验 N-gram）
            shoes: 靴牌列表（用于DataDriven）
            current_shoe_data: 当前靴牌数据（Markov/N-gram 合法输入）
        """
        self.combined_data = combined_data
        self.analyzer = analyzer
        self.shoes = shoes if shoes else []

        # 确定 Markov/N-gram 使用的数据源：必须是当前靴牌，不能跨靴牌
        if current_shoe_data is not None:
            self.current_shoe_data = current_shoe_data
        else:
            warnings.warn(
                "[HistoricalEnsemble] current_shoe_data 未传入，Markov/N-gram 将 "
                "fallback 到 combined_data（含跨靴牌数据）。"
                "请显式传入 current_shoe_data。",
                UserWarning,
                stacklevel=2
            )
            self.current_shoe_data = combined_data

        # Markov：只用当前靴牌构建转移矩阵
        self.markov = MarkovPredictor(self.current_shoe_data, order=2)

        # 靴内 N-gram（Bug#15新增）
        self.intra_ngram = IntraShoeNgramPredictor(self.current_shoe_data)

        # DataDriven
        self.data_driven = None
        if self.shoes:
            self.data_driven = DataDrivenPredictor(self.shoes)

    def predict(self) -> dict:
        """
        集成预测

        Returns:
            综合预测结果字典
        """
        predictions = []

        # 1. 历史先验 N-gram（多个阶数）
        for n in [3, 4, 5]:
            pred = self.analyzer.predict_by_ngram(self.current_shoe_data, n=n)
            if pred['confidence'] > 30:
                predictions.append({
                    'name': f'Prior_{n}gram',
                    'weight': 1.0,
                    'prediction': pred
                })

        # 2. Markov（当前靴）
        markov_pred = self.markov.predict(self.current_shoe_data)
        if markov_pred['confidence'] > 30:
            predictions.append({
                'name': 'Markov',
                'weight': 1.2,
                'prediction': markov_pred
            })

        # 3. 靴内 N-gram（Bug#15修复：渐进加权）
        intra_w = self.intra_ngram.intra_weight
        if intra_w > 0:
            intra_pred = self.intra_ngram.predict(n=3)
            if intra_pred['confidence'] > 0:
                predictions.append({
                    'name': 'IntraShoeNgram',
                    'weight': intra_w,   # 0~1，局数越多权重越高
                    'prediction': intra_pred
                })

        # 4. DataDriven
        if self.data_driven and self.data_driven.can_predict():
            dd_pred = self.data_driven.predict(self.combined_data)
            if dd_pred['confidence'] > 0:
                predictions.append({
                    'name': 'DataDriven',
                    'weight': 1.0,
                    'prediction': dd_pred
                })

        if not predictions:
            return self._default_prediction()

        return self._ensemble_predictions(predictions)

    def _ensemble_predictions(self, predictions: list) -> dict:
        """集成多个子模型的预测"""
        total_weight = 0
        weighted_b = weighted_p = weighted_t = 0

        for pred in predictions:
            weight = pred['weight'] * (pred['prediction']['confidence'] / 100)
            weighted_b += pred['prediction']['B'] * weight
            weighted_p += pred['prediction']['P'] * weight
            weighted_t += pred['prediction'].get('T', 0) * weight
            total_weight += weight

        if total_weight == 0:
            return self._default_prediction()

        final_b = weighted_b / total_weight
        final_p = weighted_p / total_weight
        final_t = weighted_t / total_weight

        avg_confidence = sum(p['prediction']['confidence'] for p in predictions) / len(predictions)

        vote_b = sum(1 for p in predictions if p['prediction']['B'] > p['prediction']['P'])
        vote_p = len(predictions) - vote_b
        consistency = max(vote_b, vote_p) / len(predictions) * 100

        confidence = avg_confidence * (0.7 + 0.3 * (consistency / 100))
        confidence = min(100, max(30, confidence))

        winner = 'B' if final_b > final_p else 'P'
        reason = f"历史集成：{len(predictions)}个子模型，一致性{consistency:.0f}%"

        return {
            'B': final_b,
            'P': final_p,
            'T': final_t,
            'confidence': confidence,
            'method': 'historical_ensemble',
            'reason': reason,
            'sub_models': len(predictions),
            'consistency': consistency
        }

    def _default_prediction(self) -> dict:
        """默认预测（理论概率）"""
        return {
            'B': 45.86, 'P': 44.62, 'T': 9.52,
            'confidence': 30,
            'method': 'historical_ensemble',
            'reason': '历史集成：数据不足，使用理论概率'
        }

    def can_predict(self) -> bool:
        """
        检查是否可以预测。
        Bug#9修复（保持）：combined_data >= 10 AND current_shoe_data >= 3。
        """
        return len(self.combined_data) >= 10 and len(self.current_shoe_data) >= 3
