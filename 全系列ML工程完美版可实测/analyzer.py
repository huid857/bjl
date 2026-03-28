"""
统计分析引擎 - V2 增强版

V2 优化内容：
  1. predict_by_streak：自适应跟龙阈值，基于靴内平均连长动态判断
  2. predict_by_trend：自适应窗口 + 趋势强度动态概率
  3. predict_by_double_alternation：自适应强度门控，不再固定50%
  4. _identify_pattern：使用靴内全量列数据，不再限制"最近10列"
  5. _determine_trend：自适应平衡阈值，基于靴内数据量动态调整
  6. 新增 _compute_avg_run_length 用于自适应连龙判断
"""

from collections import defaultdict, Counter
from utils import calculate_statistics, get_current_streak, get_max_streak


class BaccaratAnalyzer:
    """百家乐统计分析器"""

    def __init__(self, data, current_shoe_data=None):
        """
        初始化分析器

        Args:
            data: 历史数据列表 ['B', 'P', 'T', ...] (可以是跨靴牌的)
            current_shoe_data: 当前靴牌数据 (用于Streak/Trend分析，避免跨靴牌)
        """
        self.data = data
        self.data_no_tie = [x for x in data if x != 'T']

        self.current_shoe_data = current_shoe_data if current_shoe_data is not None else data
        self.current_shoe_no_tie = [x for x in self.current_shoe_data if x != 'T']

    def analyze_ngram(self, n=3, exclude_tie=True):
        """
        N-gram模式分析
        """
        data = self.data_no_tie if exclude_tie else self.data

        if len(data) < n:
            return {}

        patterns = defaultdict(lambda: {'B': 0, 'P': 0, 'T': 0})

        for i in range(len(data) - n):
            pattern = ''.join(data[i:i+n-1])
            next_result = data[i+n-1]
            patterns[pattern][next_result] += 1

        result = {}
        for pattern, counts in patterns.items():
            total = sum(counts.values())
            if total > 0:
                result[pattern] = {
                    'B': counts['B'] / total * 100,
                    'P': counts['P'] / total * 100,
                    'T': counts['T'] / total * 100,
                    'count': total
                }

        return result

    def predict_by_ngram(self, recent_data, n=3, exclude_tie=True):
        """
        基于N-gram预测下一局
        """
        if exclude_tie:
            recent_data = [x for x in recent_data if x != 'T']

        if len(recent_data) < n - 1:
            return self._default_prediction()

        current_pattern = ''.join(recent_data[-(n-1):])
        ngram_stats = self.analyze_ngram(n, exclude_tie)

        if current_pattern in ngram_stats:
            stats = ngram_stats[current_pattern]
            confidence = min(100, stats['count'] * 5)

            return {
                'B': stats['B'],
                'P': stats['P'],
                'T': stats['T'],
                'confidence': confidence,
                'method': f'{n}-gram',
                'pattern': current_pattern,
                'sample_count': stats['count']
            }
        else:
            if n > 2:
                return self.predict_by_ngram(recent_data, n-1, exclude_tie)
            else:
                return self._default_prediction()

    def analyze_road_map(self):
        """
        路单分析（大路）
        """
        if not self.data_no_tie:
            return [], {}

        road = []
        current_column = []
        last_result = None

        for result in self.data_no_tie:
            if result == last_result or last_result is None:
                current_column.append(result)
            else:
                if current_column:
                    road.append(current_column)
                current_column = [result]
            last_result = result

        if current_column:
            road.append(current_column)

        stats = {
            'columns': len(road),
            'max_column_height': max([len(col) for col in road]) if road else 0,
            'avg_column_height': sum([len(col) for col in road]) / len(road) if road else 0,
            'long_dragons': self._count_long_dragons(road),
            'pattern_type': self._identify_pattern(road)
        }

        return road, stats

    def _count_long_dragons(self, road):
        """统计长龙（连续5次以上）"""
        count = 0
        for column in road:
            if len(column) >= 5:
                count += 1
        return count

    def _identify_pattern(self, road):
        """
        识别路单模式类型。

        V2 优化：使用靴内全量列数据分析，不再限制"最近10列"。
        当列数多时，用加权平均（近期列权重更高）。
        """
        if len(road) < 3:
            return 'insufficient_data'

        # V2：使用全量列数据，但近期列权重更高
        n_cols = len(road)
        weighted_height_sum = 0
        weight_sum = 0
        decay = 0.95

        for i, col in enumerate(road):
            w = decay ** (n_cols - 1 - i)
            weighted_height_sum += len(col) * w
            weight_sum += w

        avg_height = weighted_height_sum / weight_sum if weight_sum > 0 else 0

        # Audit#C修复：改用「长龙列占比」而非median×1.5的自适应阈值。
        # 原逻辑问题：当所有列都是长龙时，median本身就大，1.5倍后阈值过高，
        # 导致 [5,3,4] 这种明显长龙靴被误判为mixed。
        heights = [len(col) for col in road]
        n_cols = len(heights)

        # 长龙列：高度>=3 的列
        long_cols = sum(1 for h in heights if h >= 3)
        long_ratio = long_cols / n_cols

        # 短列：高度==1 的列（单跳特征）
        short_cols = sum(1 for h in heights if h == 1)
        short_ratio = short_cols / n_cols

        if long_ratio >= 0.5:
            # 超过一半的列是长龙 → 长龙靴
            banker_long = sum(1 for col in road if len(col) >= 3 and col[0] == 'B')
            player_long = sum(1 for col in road if len(col) >= 3 and col[0] == 'P')
            if banker_long >= player_long:
                return 'long_banker'
            else:
                return 'long_player'
        elif short_ratio >= 0.7:
            # 70%以上的列高度为1 → 单跳靴
            return 'alternating'
        else:
            return 'mixed'


    def analyze_trend(self, window_sizes=None):
        """
        趋势分析

        V2 优化：自适应窗口大小，基于当前靴牌长度动态选择。
        """
        shoe_len = len(self.current_shoe_data)
        if window_sizes is None:
            if shoe_len < 15:
                window_sizes = [shoe_len]
            elif shoe_len < 25:
                window_sizes = [10, shoe_len]
            elif shoe_len < 40:
                window_sizes = [10, 20, shoe_len]
            else:
                window_sizes = [10, 20, 30, shoe_len]

        result = {}
        base_data = self.current_shoe_data if self.current_shoe_data else self.data

        for window in window_sizes:
            if len(base_data) >= window:
                recent = base_data[-window:]
                stats = calculate_statistics(recent)
                result[f'last_{window}'] = stats

        streak_type, streak_count = get_current_streak(self.current_shoe_no_tie)
        result['current_streak'] = {
            'type': streak_type,
            'count': streak_count
        }

        result['max_streaks'] = {
            'banker': get_max_streak(self.current_shoe_data, 'B'),
            'player': get_max_streak(self.current_shoe_data, 'P'),
            'tie': get_max_streak(self.current_shoe_data, 'T')
        }

        result['trend_direction'] = self._determine_trend()

        return result

    def _determine_trend(self):
        """
        判断当前趋势方向

        V2 优化：
          - 自适应最小数据量要求（从固定10局改为6局即可开始分析）
          - 自适应窗口大小
          - 自适应平衡阈值（数据量少时容忍更大偏差）
        """
        shoe_no_tie = self.current_shoe_no_tie
        n = len(shoe_no_tie)

        if n < 6:
            return 'insufficient_data'

        window = min(n, 30)
        recent = self.current_shoe_data[-window:]

        streak_type, streak_length = get_current_streak(shoe_no_tie)
        avg_run = self._compute_avg_run_length(shoe_no_tie)
        streak_thresh = max(2, int(avg_run * 1.5))
        if streak_length >= streak_thresh:
            return 'banker_strong' if streak_type == 'B' else 'player_strong'

        stats = calculate_statistics(recent)
        banker_rate = stats['banker_rate']
        player_rate = stats['player_rate']
        diff = abs(banker_rate - player_rate)

        # V2：自适应平衡阈值
        balance_thresh = max(10, 40 - n * 1.0)

        if diff < balance_thresh:
            return 'balanced'
        elif banker_rate > player_rate:
            return 'banker_strong'
        else:
            return 'player_strong'

    def _compute_avg_run_length(self, seq):
        """V2 新增：计算序列的平均连续段长度"""
        if not seq:
            return 1.0
        runs = []
        current_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        return sum(runs) / len(runs) if runs else 1.0

    def predict_by_trend(self):
        """
        基于趋势预测

        V2 优化：趋势概率不再固定55/40，而是基于趋势强度动态调整。
        """
        trend = self.analyze_trend()
        direction = trend['trend_direction']

        if direction == 'insufficient_data' or direction == 'balanced':
            return self._default_prediction()

        shoe_no_tie = self.current_shoe_no_tie
        if len(shoe_no_tie) < 6:
            return self._default_prediction()

        b_count = shoe_no_tie.count('B')
        p_count = shoe_no_tie.count('P')
        total = b_count + p_count
        if total == 0:
            return self._default_prediction()

        b_rate = b_count / total
        p_rate = p_count / total
        imbalance = abs(b_rate - p_rate)

        offset = min(8.0, imbalance * 30)
        confidence = min(70, 45 + imbalance * 100)

        if direction == 'banker_strong':
            b_prob = 50.0 + offset
            p_prob = 100.0 - 5.0 - b_prob
            reason = f'庄家趋势强势（庄{b_rate*100:.0f}%）'
        else:
            p_prob = 49.0 + offset
            b_prob = 100.0 - 5.0 - p_prob
            reason = f'闲家趋势强势（闲{p_rate*100:.0f}%）'

        return {
            'B': b_prob,
            'P': max(10.0, p_prob),
            'T': 5,
            'confidence': confidence,
            'method': 'trend',
            'reason': reason
        }

    def predict_by_streak(self):
        """
        基于连胜情况预测

        V2 优化：自适应跟龙阈值、基于靴内平均连长动态调整置信度
        """
        streak_type, streak_count = get_current_streak(self.current_shoe_no_tie)

        if streak_count < 2:
            return self._default_prediction()

        avg_run = self._compute_avg_run_length(self.current_shoe_no_tie)
        relative_streak = streak_count / max(1.0, avg_run)

        if streak_type == 'B':
            b_prob = min(58.0, 50.0 + streak_count * 1.5)
            p_prob = 100.0 - 5.0 - b_prob
            base_conf = 35 + streak_count * 4
            if relative_streak >= 2.0:
                base_conf += 10
            elif relative_streak < 1.0:
                base_conf -= 5
            confidence = min(65, max(30, base_conf))
            return {
                'B': b_prob,
                'P': max(10.0, p_prob),
                'T': 5,
                'confidence': confidence,
                'method': 'follow_streak',
                'reason': f'庄连{streak_count}次（均连{avg_run:.1f}），跟龙'
            }
        else:
            p_prob = min(56.0, 49.0 + streak_count * 1.5)
            b_prob = 100.0 - 5.0 - p_prob
            base_conf = 35 + streak_count * 4
            if relative_streak >= 2.0:
                base_conf += 10
            elif relative_streak < 1.0:
                base_conf -= 5
            confidence = min(65, max(30, base_conf))
            return {
                'B': max(10.0, b_prob),
                'P': p_prob,
                'T': 5,
                'confidence': confidence,
                'method': 'follow_streak',
                'reason': f'闲连{streak_count}次（均连{avg_run:.1f}），跟龙'
            }

    def predict_by_double_alternation(self, regime_result=None):
        """
        基于双跳（双对 BBPP）模式预测。

        V2 优化：自适应强度门控 + 利用 gap_to_second 判断
        """
        if regime_result is not None:
            dominant = regime_result.get('dominant_regime', '')
            strength = regime_result.get('regime_strength', 0)
            gap = regime_result.get('gap_to_second', 0)

            if dominant != 'double_alt':
                scores = regime_result.get('regime_scores', {})
                double_score = scores.get('double_alt', 0)
                if double_score < 20:
                    return self._default_prediction()
            elif strength < 25 and gap < 5:
                return self._default_prediction()

        seq = self.current_shoe_no_tie
        n = len(seq)

        if n < 6:
            return self._default_prediction()

        window = min(n, max(6, n // 2))
        recent = seq[-window:]

        new_pair_val = None
        for j in range(len(recent) - 1, 0, -1):
            if recent[j] != recent[j - 1]:
                new_pair_val = recent[j]
                break

        if new_pair_val is None:
            return self._default_prediction()

        run_start = len(recent) - 1
        while run_start > 0 and recent[run_start] == recent[run_start - 1]:
            run_start -= 1
        run_len = len(recent) - run_start

        if run_len == 1:
            predict_side = new_pair_val
            reason = f'双跳模式，当前{new_pair_val}对第1局，预测继续'
        else:
            predict_side = 'B' if new_pair_val == 'P' else 'P'
            reason = f'双跳模式，{new_pair_val}对已完成，预测切换到{predict_side}'

        if regime_result is not None:
            strength_pct = regime_result.get('regime_strength', 50)
            gap_pct = regime_result.get('gap_to_second', 10)
            confidence = min(60, 30 + strength_pct * 0.15 + gap_pct * 0.3)
        else:
            confidence = 40

        b_prob = 55.0 if predict_side == 'B' else 40.0
        p_prob = 100.0 - b_prob - 5.0

        return {
            'B': b_prob,
            'P': p_prob,
            'T': 5.0,
            'confidence': confidence,
            'method': 'double_alt',
            'reason': reason,
        }

    def _default_prediction(self):
        """默认预测（基于理论概率）"""
        return {
            'B': 45.86,
            'P': 44.62,
            'T': 9.52,
            'confidence': 30,
            'method': 'theoretical',
            'reason': '数据不足，使用理论概率'
        }

    def get_comprehensive_analysis(self):
        """获取综合分析报告"""
        if len(self.data) < 10:
            return {
                'error': '数据不足（至少需要10局）',
                'data_count': len(self.data)
            }

        basic_stats = calculate_statistics(self.data)
        ngram_2 = self.analyze_ngram(2)
        ngram_3 = self.analyze_ngram(3)
        ngram_4 = self.analyze_ngram(4)
        road, road_stats = self.analyze_road_map()
        trend = self.analyze_trend()

        return {
            'basic_stats': basic_stats,
            'ngram': {
                '2-gram': ngram_2,
                '3-gram': ngram_3,
                '4-gram': ngram_4
            },
            'road_map': road_stats,
            'trend': trend,
            'data_count': len(self.data)
        }
