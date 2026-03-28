"""
高级靴牌分析器
识别靴牌类型、阶段特征、相似靴牌等

Bug#12 修复: _count_alternations 拆分为 _count_single_alternations
            和 _count_double_alternations，单跳/双跳不再混淆。
Bug#13 修复: get_shoe_weight_adjustment 改为接受 regime_result 参数，
            基于 ShoeRegimeDetector 的强度比例线性插值权重，不再硬编码 1.3/0.6。
"""

from collections import Counter
from utils import calculate_statistics, get_max_streak


class AdvancedShoeAnalyzer:
    """高级靴牌分析器"""

    def __init__(self, shoes):
        """
        初始化分析器

        Args:
            shoes: 所有历史靴牌列表
        """
        self.shoes = shoes

    def classify_shoe_type(self, shoe_data):
        """
        识别靴牌类型

        Args:
            shoe_data: 靴牌数据字符串或列表

        Returns:
            dict: {
                'primary_type': 主类型,
                'secondary_types': [次要类型列表],
                'characteristics': 特征描述,
                'confidence': 置信度,
                'stats': 详细统计
            }
        """
        if isinstance(shoe_data, str):
            shoe_data = list(shoe_data)

        if len(shoe_data) < 10:
            return {
                'primary_type': 'insufficient_data',
                'secondary_types': [],
                'characteristics': '数据不足',
                'confidence': 0
            }

        stats = calculate_statistics(shoe_data)
        types = []
        characteristics = []

        # 1. 基于庄闲比例分类
        banker_rate = stats['banker_rate']
        player_rate = stats['player_rate']

        if banker_rate >= 52:
            types.append('banker_dominant')
            characteristics.append(f'庄强靴(庄{banker_rate:.1f}%)')
        elif player_rate >= 52:
            types.append('player_dominant')
            characteristics.append(f'闲强靴(闲{player_rate:.1f}%)')
        elif abs(banker_rate - player_rate) <= 4:
            types.append('balanced')
            characteristics.append('平衡靴')

        # 2. 基于连续特征分类（修复Bug#12：分别统计单跳和双跳）
        no_tie = [x for x in shoe_data if x != 'T']
        long_streaks = self._count_long_streaks(no_tie)
        single_alts = self._count_single_alternations(no_tie)
        double_alts = self._count_double_alternation_pairs(no_tie)

        if long_streaks >= 3:
            types.append('long_dragon')
            characteristics.append(f'长龙靴({long_streaks}个长龙)')

        n_no_tie = len(no_tie)
        single_alt_rate = single_alts / n_no_tie if n_no_tie > 0 else 0
        double_pair_rate = double_alts * 2 / n_no_tie if n_no_tie > 0 else 0   # 每对2局

        # Bug#12修复: 分别判断单跳和双跳，不再混用
        if single_alt_rate >= 0.55 and single_alt_rate > double_pair_rate:
            types.append('alternating')
            characteristics.append(f'单跳靴(跳{single_alt_rate * 100:.0f}%)')
        elif double_pair_rate >= 0.45:
            types.append('double_alternating')
            characteristics.append(f'双跳靴(对{double_pair_rate * 100:.0f}%)')

        # 3. 基于和局比例
        tie_rate = stats['tie_rate']
        if tie_rate >= 12:
            types.append('high_tie')
            characteristics.append(f'和多靴({tie_rate:.1f}%)')

        # 确定主类型
        primary_type = types[0] if types else 'mixed'
        secondary_types = types[1:] if len(types) > 1 else []

        # 计算置信度
        confidence = min(100, len(shoe_data) * 2)

        return {
            'primary_type': primary_type,
            'secondary_types': secondary_types,
            'characteristics': ' | '.join(characteristics) if characteristics else '混合靴',
            'confidence': confidence,
            'stats': {
                'banker_rate': banker_rate,
                'player_rate': player_rate,
                'tie_rate': tie_rate,
                'long_streaks': long_streaks,
                'single_alt_rate': single_alt_rate * 100,
                'double_pair_rate': double_pair_rate * 100,
            }
        }

    def _count_long_streaks(self, data):
        """统计长龙（连续4+次）数量"""
        if not data:
            return 0

        count = 0
        current_streak = 1

        for i in range(1, len(data)):
            if data[i] == data[i - 1]:
                current_streak += 1
            else:
                if current_streak >= 4:
                    count += 1
                current_streak = 1

        if current_streak >= 4:
            count += 1

        return count

    def _count_single_alternations(self, data):
        """
        统计严格单跳局数（BPBP 交替格式）。
        Bug#12修复：只统计上一局不同、上上局也不同（严格交替）的局。

        Returns:
            处于严格单跳结构中的局数
        """
        if len(data) < 3:
            return 0

        count = 0
        for i in range(2, len(data)):
            # 严格单跳：当前 ≠ 上一，上一 ≠ 上上（BPBP）
            if data[i] != data[i - 1] and data[i - 1] != data[i - 2]:
                count += 1
        return count

    def _count_double_alternation_pairs(self, data):
        """
        统计双跳对数（BBPP 格式的完整"对"数）。
        Bug#12修复：单独统计 BBPP 结构，不与单跳混淆。

        一个"对"= 连续 2 个相同结果构成一组，相邻两组不同。

        Returns:
            完整的双跳对数
        """
        if len(data) < 4:
            return 0

        pairs_found = 0
        i = 0
        while i < len(data) - 1:
            # 找到一个"对"：data[i] == data[i+1]
            if data[i] == data[i + 1]:
                pair_val = data[i]
                i += 2
                # 下一个"对"要与当前对不同
                if i < len(data) - 1 and data[i] == data[i + 1] and data[i] != pair_val:
                    pairs_found += 1
                    # 不 advance i：让下一对从这里开始
                # else: 不是 BBPP，跳过这一"对"继续找
            else:
                i += 1

        return pairs_found

    def analyze_shoe_phases(self, shoe_data):
        """
        分析靴牌阶段特征

        Args:
            shoe_data: 靴牌数据

        Returns:
            dict: 各阶段的统计特征
        """
        if isinstance(shoe_data, str):
            shoe_data = list(shoe_data)

        total = len(shoe_data)

        # 定义阶段
        phases = {
            'early': shoe_data[:20] if total > 20 else shoe_data,
            'middle': shoe_data[20:40] if total > 40 else shoe_data[20:] if total > 20 else [],
            'late': shoe_data[40:] if total > 40 else []
        }

        result = {}

        phase_ranges = {
            'early': (1, min(20, total)),
            'middle': (21, min(40, total)) if total > 20 else None,
            'late': (41, total) if total > 40 else None
        }

        for phase_name, phase_data in phases.items():
            if not phase_data:
                result[phase_name] = None
                continue

            stats = calculate_statistics(phase_data)
            shoe_type = self.classify_shoe_type(phase_data)

            phase_range = phase_ranges.get(phase_name)
            if phase_range:
                if phase_range[0] == phase_range[1]:
                    range_str = f"第{phase_range[0]}局"
                else:
                    range_str = f"{phase_range[0]}-{phase_range[1]}局"
            else:
                range_str = "未知"

            result[phase_name] = {
                'range': range_str,
                'count': len(phase_data),
                'banker_rate': stats['banker_rate'],
                'player_rate': stats['player_rate'],
                'tie_rate': stats['tie_rate'],
                'type': shoe_type['primary_type'],
                'characteristics': shoe_type['characteristics']
            }

        if total <= 20:
            current_phase = 'early'
        elif total <= 40:
            current_phase = 'middle'
        else:
            current_phase = 'late'

        result['current_phase'] = current_phase
        result['total_count'] = total

        return result

    def find_similar_shoes(self, current_shoe_data, top_n=3, min_similarity=0.5):
        """
        找出历史中相似的靴牌

        Args:
            current_shoe_data: 当前靴牌数据
            top_n: 返回最相似的N个
            min_similarity: 最小相似度阈值

        Returns:
            list: 相似靴牌列表，按相似度排序
        """
        if isinstance(current_shoe_data, str):
            current_shoe_data = list(current_shoe_data)

        if len(current_shoe_data) < 10:
            return []

        current_type = self.classify_shoe_type(current_shoe_data)
        current_stats = current_type['stats']

        similarities = []

        for shoe in self.shoes:
            shoe_data = list(shoe['data'])

            if len(shoe_data) < 20:
                continue

            similarity = self._calculate_similarity(
                current_shoe_data,
                shoe_data,
                current_stats
            )

            if similarity >= min_similarity:
                similarities.append({
                    'shoe': shoe,
                    'similarity': similarity,
                    'type': self.classify_shoe_type(shoe_data)
                })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:top_n]

    def _calculate_similarity(self, current_data, historical_data, current_stats):
        """
        计算两个靴牌的相似度

        考虑因素：
        1. 庄闲比例相似度（40%权重）
        2. 前N局序列相似度（30%权重）
        3. 连续特征相似度（30%权重）— 修复Bug#12后用精确的单/双跳计数
        """
        hist_type = self.classify_shoe_type(historical_data)
        hist_stats = hist_type['stats']

        # 1. 庄闲比例相似度
        banker_diff = abs(current_stats['banker_rate'] - hist_stats['banker_rate'])
        player_diff = abs(current_stats['player_rate'] - hist_stats['player_rate'])
        ratio_similarity = max(0, 1 - (banker_diff + player_diff) / 100) * 0.4

        # 2. 前N局序列相似度
        compare_length = min(len(current_data), len(historical_data), 20)
        sequence_similarity = self._sequence_similarity(
            current_data[:compare_length],
            historical_data[:compare_length]
        ) * 0.3

        # 3. 连续特征相似度（单跳率 + 双跳对比率 + 长龙计数）
        current_no_tie = [x for x in current_data if x != 'T']
        hist_no_tie = [x for x in historical_data if x != 'T']

        current_long = self._count_long_streaks(current_no_tie)
        hist_long = self._count_long_streaks(hist_no_tie)

        # 单跳率
        c_single_rate = current_stats.get('single_alt_rate', 0) / 100
        h_single_rate = hist_stats.get('single_alt_rate', 0) / 100
        # 双跳对率
        c_double_rate = current_stats.get('double_pair_rate', 0) / 100
        h_double_rate = hist_stats.get('double_pair_rate', 0) / 100

        streak_diff = abs(current_long - hist_long) / max(current_long, hist_long, 1)
        alt_diff = (abs(c_single_rate - h_single_rate) + abs(c_double_rate - h_double_rate)) / 2

        feature_similarity = max(0, 1 - (streak_diff + alt_diff) / 2) * 0.3

        return ratio_similarity + sequence_similarity + feature_similarity

    def _sequence_similarity(self, seq1, seq2):
        """计算序列相似度（简单匹配）"""
        if not seq1 or not seq2:
            return 0

        matches = sum(1 for i in range(min(len(seq1), len(seq2))) if seq1[i] == seq2[i])
        return matches / min(len(seq1), len(seq2))

    def get_shoe_weight_adjustment(self, current_shoe_data, regime_result=None):
        """
        根据靴牌类型和 Regime 分析建议权重调整。

        Bug#13修复：若传入 regime_result，使用强度比例线性插值权重；
                    否则 fallback 到分类类型（保持向后兼容）。

        Args:
            current_shoe_data: 当前靴牌数据
            regime_result: ShoeRegimeDetector.analyze() 的结果（可选）

        Returns:
            dict: {'reason': str, 'model_adjustments': dict}
        """
        # 优先使用 Regime 结果（Bug#13修复路径）
        if regime_result and regime_result.get('can_analyze', False):
            try:
                from shoe_regime_detector import ShoeRegimeDetector
                detector = ShoeRegimeDetector()
                model_adj = detector.get_model_weight_adjustments(regime_result)
                return {
                    'reason': f"Regime 驱动 ({regime_result['recommendation']})",
                    'model_adjustments': model_adj
                }
            except ImportError:
                pass  # fallback 到旧逻辑

        # Fallback: 旧的二元分类调整（兼容性保留）
        shoe_type = self.classify_shoe_type(current_shoe_data)
        primary = shoe_type['primary_type']

        adjustments = {
            'reason': f"靴牌类型: {shoe_type['characteristics']}",
            'model_adjustments': {}
        }

        if primary == 'long_dragon':
            adjustments['model_adjustments'] = {
                'Streak': 1.3, 'Trend': 0.8, 'DataDriven': 1.2
            }
        elif primary == 'alternating':
            adjustments['model_adjustments'] = {
                'Streak': 0.6, 'Trend': 1.3, 'Frequency': 1.2
            }
        elif primary == 'double_alternating':
            adjustments['model_adjustments'] = {
                'Streak': 0.5, 'IntraShoeNgram': 1.4, 'Frequency': 1.1
            }
        elif primary == 'banker_dominant':
            adjustments['model_adjustments'] = {'Frequency': 1.1}
        elif primary == 'player_dominant':
            adjustments['model_adjustments'] = {'Frequency': 1.1}

        return adjustments

    def get_statistics(self):
        """获取所有靴牌的统计信息"""
        if not self.shoes:
            return {}

        all_types = []
        for shoe in self.shoes:
            shoe_type = self.classify_shoe_type(shoe['data'])
            all_types.append(shoe_type['primary_type'])

        type_counts = Counter(all_types)

        return {
            'total_shoes': len(self.shoes),
            'type_distribution': dict(type_counts),
            'most_common_type': type_counts.most_common(1)[0] if type_counts else None
        }
