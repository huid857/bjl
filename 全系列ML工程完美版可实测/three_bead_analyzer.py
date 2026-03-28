"""
三珠路分析引擎 (Three-Bead Road Analyzer)

实现百家乐"三珠路"打法的系统化分析：
  - 将去和局序列每3局一组分块
  - 统计8种三珠组合 (BBB/BBP/BPB/BPP/PBB/PBP/PPB/PPP) 出现频率
  - 判断当前靴牌适合"正打"还是"反打"
  - 基于靴内三珠转移模式预测下一局

三珠路打法源自澳门赌王叶汉：
  - 正打：BBB/PPP（跟趋势），适合长龙靴
  - 反打：BPP/PBB（逆趋势），适合短路/跳路靴
  - ED式：正反动态切换
"""


class ThreeBeadAnalyzer:
    """三珠路分析引擎"""

    # 8种三珠组合
    ALL_PATTERNS = ['BBB', 'BBP', 'BPB', 'BPP', 'PBB', 'PBP', 'PPB', 'PPP']

    # 正式组合（趋势型）：连续同方
    ZHENG_PATTERNS = {'BBB', 'PPP'}
    # 反式组合（短路型）：一方后接两个另一方
    FAN_PATTERNS = {'BPP', 'PBB'}
    # 交替型组合
    ALT_PATTERNS = {'BPB', 'PBP'}
    # 混合型
    MIX_PATTERNS = {'BBP', 'PPB'}

    def __init__(self):
        pass

    def analyze(self, shoe_data: list) -> dict:
        """
        主分析接口：分析当前靴牌的三珠路特征。

        Args:
            shoe_data: 当前靴牌数据（含和局）

        Returns:
            {
                'can_predict': bool,
                'style': str,         # 'zheng'(正打)|'fan'(反打)|'alt'(交替)|'mixed'
                'style_strength': float,  # 0~100
                'prediction': dict,    # 下一局预测
                'pattern_stats': dict, # 8种三珠组合的统计
                'narrative': str,      # 中文描述
                'beads': list,         # 三珠分块列表
            }
        """
        seq = [x for x in shoe_data if x != 'T']
        n = len(seq)

        if n < 6:
            return self._no_data_result(n)

        # Step 1: 分块（每3个一组）
        beads = self._split_beads(seq)
        if len(beads) < 2:
            return self._no_data_result(n)

        # Step 2: 统计各三珠组合出现频率
        pattern_stats = self._count_patterns(beads)

        # Step 3: 判断打法风格
        style, style_strength = self._determine_style(pattern_stats, len(beads))

        # Step 4: 基于三珠转移模式预测
        prediction = self._predict_next(seq, beads, pattern_stats, style)

        # Step 5: 描述
        narrative = self._build_narrative(style, style_strength, beads)

        return {
            'can_predict': prediction.get('confidence', 0) > 0,
            'style': style,
            'style_strength': round(style_strength, 1),
            'prediction': prediction,
            'pattern_stats': pattern_stats,
            'narrative': narrative,
            'beads': beads,
            'bead_count': len(beads),
        }

    def _split_beads(self, seq: list) -> list:
        """将去和局序列每3个一组分块"""
        beads = []
        for i in range(0, len(seq) - 2, 3):
            bead = ''.join(seq[i:i+3])
            beads.append(bead)
        return beads

    def _count_patterns(self, beads: list) -> dict:
        """统计各三珠组合出现频率"""
        counts = {p: 0 for p in self.ALL_PATTERNS}
        for bead in beads:
            if bead in counts:
                counts[bead] += 1

        total = len(beads)
        stats = {}
        for p in self.ALL_PATTERNS:
            stats[p] = {
                'count': counts[p],
                'rate': counts[p] / total * 100 if total > 0 else 0
            }
        return stats

    def _determine_style(self, pattern_stats: dict, total_beads: int):
        """
        判断当前靴牌适合正打、反打还是交替。

        Returns:
            (style: str, strength: float)
        """
        zheng_count = sum(pattern_stats[p]['count'] for p in self.ZHENG_PATTERNS)
        fan_count = sum(pattern_stats[p]['count'] for p in self.FAN_PATTERNS)
        alt_count = sum(pattern_stats[p]['count'] for p in self.ALT_PATTERNS)
        mix_count = sum(pattern_stats[p]['count'] for p in self.MIX_PATTERNS)

        total = total_beads
        if total == 0:
            return 'mixed', 0.0

        scores = {
            'zheng': zheng_count / total * 100,
            'fan': fan_count / total * 100,
            'alt': alt_count / total * 100,
            'mixed': mix_count / total * 100,
        }

        best_style = max(scores, key=scores.get)
        best_score = scores[best_style]

        # 要求主导风格至少占 30%（8种组合理论均匀分布时每种12.5%）
        if best_score < 30:
            return 'mixed', best_score

        return best_style, best_score

    def _predict_next(self, seq: list, beads: list, pattern_stats: dict, style: str) -> dict:
        """
        基于三珠转移模式预测下一局。

        策略：
          1. 看当前在三珠周期的哪个位置（第1局/第2局/第3局）
          2. 基于靴内已出现的三珠模式预测接下来的走向
        """
        n = len(seq)

        # 当前在三珠周期内的位置 (0, 1, 2)
        position_in_bead = n % 3

        if position_in_bead == 0:
            # 新三珠组的第一局：看前一组的模式来推断
            if len(beads) < 2:
                return self._default_pred()

            last_bead = beads[-1]
            # 统计：在此靴中，last_bead后面的三珠组第一局通常是什么？
            transitions = {'B': 0, 'P': 0}
            for i in range(len(beads) - 1):
                if beads[i] == last_bead and i + 1 < len(beads):
                    next_first = beads[i + 1][0]
                    if next_first in transitions:
                        transitions[next_first] += 1

            return self._make_pred_from_counts(transitions, f'三珠转移：{last_bead}后')

        elif position_in_bead == 1:
            # 当前三珠组的第2局：知道第1局是什么
            first = seq[-1]
            # 看靴内以该字母开头的三珠组，第2局通常是什么
            transitions = {'B': 0, 'P': 0}
            for bead in beads:
                if bead[0] == first:
                    if bead[1] in transitions:
                        transitions[bead[1]] += 1

            return self._make_pred_from_counts(transitions, f'三珠位置2：首局{first}后')

        else:
            # 当前三珠组的第3局：知道前2局是什么
            prefix = ''.join(seq[-2:])
            transitions = {'B': 0, 'P': 0}
            for bead in beads:
                if bead[:2] == prefix:
                    if bead[2] in transitions:
                        transitions[bead[2]] += 1

            return self._make_pred_from_counts(transitions, f'三珠位置3：前缀{prefix}后')

    def _make_pred_from_counts(self, transitions: dict, reason_prefix: str) -> dict:
        """根据转移计数生成预测"""
        total = transitions['B'] + transitions['P']
        if total == 0:
            return self._default_pred()

        b_rate = transitions['B'] / total
        p_rate = transitions['P'] / total

        # 概率范围限制在 15~85，避免极端值
        b_prob = max(15.0, min(85.0, b_rate * 100))
        p_prob = max(15.0, min(85.0, p_rate * 100))
        # 归一化
        bp_sum = b_prob + p_prob
        b_prob = b_prob / bp_sum * 100
        p_prob = p_prob / bp_sum * 100

        # Audit#C修复：置信度以样本量为主要因子，1样本不应超30%
        # 原公式 30+total*5+imbalance*40 在 total=1,imbalance=1 时给出65，过高
        imbalance = abs(b_rate - p_rate)
        if total <= 1:
            confidence = min(30, 15 + imbalance * 15)
        elif total <= 3:
            confidence = min(45, 20 + total * 6 + imbalance * 15)
        else:
            confidence = min(65, 25 + total * 5 + imbalance * 20)

        predict_side = 'B' if b_rate > p_rate else 'P'
        reason = f'{reason_prefix}，样本{total}次，{predict_side}占{max(b_rate,p_rate)*100:.0f}%'

        return {
            'B': max(10.0, b_prob),
            'P': max(10.0, p_prob),
            'T': 0.0,
            'confidence': confidence,
            'method': 'three_bead',
            'reason': reason,
            'sample_count': total,
        }

    def _default_pred(self) -> dict:
        return {
            'B': 50.0, 'P': 50.0, 'T': 0.0,
            'confidence': 0, 'method': 'three_bead',
            'reason': '三珠路数据不足'
        }

    def _build_narrative(self, style: str, strength: float, beads: list) -> str:
        style_names = {
            'zheng': '正打（趋势型BBB/PPP为主）',
            'fan': '反打（短路型BPP/PBB为主）',
            'alt': '交替型（BPB/PBP为主）',
            'mixed': '混合型（无明显偏向）',
        }
        name = style_names.get(style, style)

        if strength >= 50:
            desc = '高度匹配'
        elif strength >= 35:
            desc = '中等匹配'
        else:
            desc = '弱匹配'

        return f'三珠路分析：{name}（{desc}，强度{strength:.0f}%，已分析{len(beads)}组）'

    def _no_data_result(self, n: int) -> dict:
        return {
            'can_predict': False,
            'style': 'unknown',
            'style_strength': 0.0,
            'prediction': self._default_pred(),
            'pattern_stats': {},
            'narrative': f'ℹ️ 数据不足（当前{n}局，去和局后需≥6局）',
            'beads': [],
            'bead_count': 0,
        }
