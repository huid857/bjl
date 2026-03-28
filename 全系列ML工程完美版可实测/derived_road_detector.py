"""
派生路元规律检测器 (Derived Road Meta-Rule Detector)

设计思想：
  百家乐三条"派生路"（大眼仔/小路/曱甴路）本质上是大路的"规律的规律"。
  在实战中，玩家观察的核心问题是：
    "在这靴牌里，红信号预示下一局续跟还是跳换？"
    "在这靴牌里，蓝信号预示下一局续跟还是跳换？"
  即"逢红就续/逢红就跳/逢蓝就续/逢蓝就跳"这四种元规律中，
  哪种在本靴中更占主导。

算法核心：
  1. 从大路（实际结果序列）构建列结构
  2. 对三条派生路分别计算 红/蓝 信号序列（使用列高度对比法）
  3. 统计靴内"信号→下一实际结果（续/跳）"的条件频率
  4. 三条路的结论综合，给出主导元规律和置信度
  5. 基于最新信号+主导元规律，给出下一局的预测

核心约束：
  - 靴内样本 < MIN_SAMPLES 个信号对时，不输出预测（避免过拟合）
  - 置信度基于样本数量和三路一致性
"""


class DerivedRoadMetaRuleDetector:
    """派生路元规律检测器"""

    MIN_SAMPLES = 6    # 每条路最少需要6个信号-结果对
    MAX_CONFIDENCE = 72  # 最高置信度 72%（元规律本就不稳定）

    def __init__(self):
        pass

    # ──────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────

    def analyze(self, shoe_data: list) -> dict:
        """
        主分析接口：实时分析整靴的派生路元规律。

        Args:
            shoe_data: 当前靴牌数据（含和局），如 ['B','P','T','B',...]

        Returns:
            {
                'can_predict': bool,
                'dominant_rule': str,       # 'red_continue'|'red_jump'|
                                            # 'blue_continue'|'blue_jump'|'unstable'
                'rule_confidence': float,   # 发现元规律的置信度（不是预测置信度）
                'prediction': dict,         # 下一局预测（B%, P%, confidence）
                'road_rules': dict,         # 三条路各自的统计
                'narrative': str,           # 中文描述
                'sample_count': int,        # 最少那条路的有效样本数
            }
        """
        # 去和局
        seq = [x for x in shoe_data if x != 'T']
        n = len(seq)

        if n < 8:
            return self._no_data_result(n)

        # Step 1: 构建大路列结构
        columns = self._build_columns(seq)
        if len(columns) < 5:
            return self._no_data_result(n)

        # Step 2: 对三条路计算信号序列 + 统计条件频率
        rules = {}
        for offset, road_name in [(1, 'big_eye'), (2, 'small_road'), (3, 'cockroach')]:
            signals, outcomes = self._compute_derived_road_signals(columns, offset)
            stats = self._compute_conditional_stats(signals, outcomes)
            rules[road_name] = stats

        # Step 3: 综合三路，确定主导元规律
        dominant, consistency, narrative = self._determine_dominant_rule(rules)

        # Step 3.5 V2：三路共振信号检测
        resonance = self._compute_resonance(rules)

        # Step 4: 最小样本检查
        min_samples = min(r.get('total', 0) for r in rules.values())
        if min_samples < self.MIN_SAMPLES:
            return self._insufficient_result(min_samples)

        # Step 5: 计算规则置信度（样本量 × 一致性）
        sample_factor = min(1.0, min_samples / 20)
        rule_confidence = consistency * sample_factor * 100

        # V2：共振加成/衰减
        rule_confidence *= resonance['confidence_boost']

        # Step 6: 基于主导规律和最新信号预测
        prediction = self._make_prediction(columns, dominant, rules, rule_confidence)

        # V2：共振信息追加到 narrative
        if resonance['resonance'] == 'strong_red':
            narrative += ' | 三路共振全红（强趋势）'
        elif resonance['resonance'] == 'strong_blue':
            narrative += ' | 三路共振全蓝（乱路信号）'

        return {
            'can_predict': prediction.get('confidence', 0) > 0,
            'dominant_rule': dominant,
            'rule_confidence': round(rule_confidence, 1),
            'prediction': prediction,
            'road_rules': rules,
            'narrative': narrative,
            'sample_count': min_samples,
            'resonance': resonance,  # V2 新增
        }

    def get_prediction_weight(self, analyze_result: dict, regime_result: dict = None) -> float:
        """
        基于分析结果和 Regime 状态，计算该模型在集成中的权重。

        Args:
            analyze_result: analyze() 的返回值
            regime_result: ShoeRegimeDetector.analyze() 的结果（可选）

        Returns:
            建议权重（0.0~2.0）
        """
        if not analyze_result.get('can_predict', False):
            return 0.0

        rule_conf = analyze_result.get('rule_confidence', 0) / 100
        base_weight = 1.3 * rule_conf  # 规则置信度越高，权重越大

        if regime_result and regime_result.get('can_analyze', False):
            dominant = regime_result.get('dominant_regime', '')
            switch = regime_result.get('switch_event', False)

            if switch:
                base_weight *= 0.4   # 切换中：派生路信号本身也不稳定
            elif dominant == 'single_alt':
                base_weight *= 1.2   # 单跳时派生路信号意义更强
            elif dominant == 'long_streak':
                base_weight *= 0.7   # 长龙时派生路信号意义下降
            elif dominant == 'chaos':
                base_weight *= 0.5   # 亂路时所有信号都减弱

        return min(2.0, max(0.0, base_weight))

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _build_columns(self, seq: list) -> list:
        """
        从去和局序列构建大路列结构。

        Returns:
            list of list: columns[i] 是第 i 列的所有元素（相同结果的连续段）
                         columns[i][j] = 'B' or 'P'
        """
        if not seq:
            return []

        columns = []
        current_col = [seq[0]]

        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                current_col.append(seq[i])
            else:
                columns.append(current_col)
                current_col = [seq[i]]

        columns.append(current_col)
        return columns

    def _compute_derived_road_signals(self, columns: list, offset: int):
        """
        计算派生路的红/蓝信号序列及其后续结果（续/跳）。

        派生路规则（列高度对比法）：
          对于第 i 列（i >= offset + 1），看：
            当前列第 j 行 vs 参照列（第 i - offset - 1 列）第 j 行
            - 参照位置存在：红信号（规律性相同）
            - 参照位置不存在（列更短）：蓝信号（规律性不同）

        为实用起见，我们简化为：
          对于每个新列 i（i >= offset + 1）的第1个元素：
            如果 len(columns[i]) == len(columns[i - offset - 1])：→ 中性（跳过）
            如果 len(columns[i]) >= len(columns[i - offset - 1])：→ 红（续规律）
            如果 len(columns[i]) <  len(columns[i - offset - 1])：→ 蓝（破规律）
          该信号对应的"结果"是：
            columns[i][0] 与 columns[i-1][-1] 相同（续跟）还是不同（跳换）

        Args:
            columns: 大路列结构
            offset: 1=大眼仔, 2=小路, 3=曱甴路

        Returns:
            (signals: list, outcomes: list)
            signals[k] = 'R'（红）or 'B'（蓝）
            outcomes[k] = 'C'（续跟 continue）or 'J'（跳换 jump）
        """
        signals = []
        outcomes = []

        # 从第 offset+2 列开始（需要 offset+1 列的参照）
        start_col = offset + 1
        for i in range(start_col, len(columns)):
            ref_col_idx = i - offset - 1
            if ref_col_idx < 0:
                continue

            ref_len = len(columns[ref_col_idx])
            cur_len = len(columns[i])

            # 信号判断
            if cur_len > ref_len:
                sig = 'R'   # 红：当前列比参照列更长（规律延续）
            elif cur_len < ref_len:
                sig = 'B'   # 蓝：当前列比参照列更短（规律中断）
            else:
                # 相等：也视为红（大多数实现中相等=红）
                sig = 'R'

            # 结果判断：这个新列的起点是续还是跳
            # 新列的第一个结果 = columns[i][0]
            # 前一列的最后结果 = columns[i-1][-1]
            if columns[i][0] != columns[i - 1][-1]:
                outcome = 'J'  # 跳换（开了新列本来就跳了）
            else:
                # 这其实不可能，因为开新列必然换方向
                outcome = 'C'

            # 更有意义的结果定义：
            # 信号对应"下一局的续跳"而非"是否开新列"
            # 重新定义：
            # outcome = 新列的第一个结果（B/P）
            # 然后在统计阶段计算 B的出现率 →  预测B的概率

            signals.append(sig)
            outcomes.append(columns[i][0])   # 存实际结果（B/P）而非C/J

        return signals, outcomes

    def _compute_conditional_stats(self, signals: list, outcomes: list) -> dict:
        """
        统计给定信号序列下的条件概率。

        Returns:
            {
                'red_b': 红信号后出现B的次数,
                'red_p': 红信号后出现P的次数,
                'blue_b': 蓝信号后出现B的次数,
                'blue_p': 蓝信号后出现P的次数,
                'red_total': 红信号总次数,
                'blue_total': 蓝信号总次数,
                'total': 总信号数,
                'red_dominant': 'B' | 'P' | 'tie',  ← 红信号下更可能出现B还是P
                'blue_dominant': 'B' | 'P' | 'tie',
                'red_confidence': %,    ← 红信号下主导方的占比
                'blue_confidence': %,
            }
        """
        red_b = red_p = blue_b = blue_p = 0

        for sig, out in zip(signals, outcomes):
            if sig == 'R':
                if out == 'B':
                    red_b += 1
                else:
                    red_p += 1
            else:  # sig == 'B'
                if out == 'B':
                    blue_b += 1
                else:
                    blue_p += 1

        red_total = red_b + red_p
        blue_total = blue_b + blue_p
        total = red_total + blue_total

        red_dominant = 'tie'
        red_conf = 50.0
        if red_total > 0:
            if red_b > red_p:
                red_dominant = 'B'
                red_conf = red_b / red_total * 100
            elif red_p > red_b:
                red_dominant = 'P'
                red_conf = red_p / red_total * 100

        blue_dominant = 'tie'
        blue_conf = 50.0
        if blue_total > 0:
            if blue_b > blue_p:
                blue_dominant = 'B'
                blue_conf = blue_b / blue_total * 100
            elif blue_p > blue_b:
                blue_dominant = 'P'
                blue_conf = blue_p / blue_total * 100

        return {
            'red_b': red_b, 'red_p': red_p,
            'blue_b': blue_b, 'blue_p': blue_p,
            'red_total': red_total, 'blue_total': blue_total,
            'total': total,
            'red_dominant': red_dominant,
            'red_confidence': round(red_conf, 1),
            'blue_dominant': blue_dominant,
            'blue_confidence': round(blue_conf, 1),
        }

    def _determine_dominant_rule(self, rules: dict):
        """
        综合三条路的统计，确定主导元规律。

        V2 优化：增加三路共振信号检测。

        元规律类型：
          'red_B'  = 红信号后大概率出B（逢红开庄）
          'red_P'  = 红信号后大概率出P（逢红开闲）
          'blue_B' = 蓝信号后大概率出B（逢蓝开庄）
          'blue_P' = 蓝信号后大概率出P（逢蓝开闲）
          'unstable' = 三路结论不一致

        Returns:
            (dominant: str, consistency: float, narrative: str)
        """
        road_names = ['big_eye', 'small_road', 'cockroach']

        votes_red_b = votes_red_p = 0
        votes_blue_b = votes_blue_p = 0
        total_conf_red = total_conf_blue = 0

        for road in road_names:
            r = rules[road]
            if r['red_total'] >= 3:
                total_conf_red += r['red_confidence']
                if r['red_dominant'] == 'B':
                    votes_red_b += 1
                elif r['red_dominant'] == 'P':
                    votes_red_p += 1

            if r['blue_total'] >= 3:
                total_conf_blue += r['blue_confidence']
                if r['blue_dominant'] == 'B':
                    votes_blue_b += 1
                elif r['blue_dominant'] == 'P':
                    votes_blue_p += 1

        results = [
            ('red_B', votes_red_b, total_conf_red / 3),
            ('red_P', votes_red_p, total_conf_red / 3),
            ('blue_B', votes_blue_b, total_conf_blue / 3),
            ('blue_P', votes_blue_p, total_conf_blue / 3),
        ]
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)

        best_rule, best_votes, best_conf = results[0]

        if best_votes < 2:
            return 'unstable', 0.3, '⚠️ 派生路信号尚无一致规律，建议继续观察'

        consistency = best_votes / 3

        rule_desc = {
            'red_B': '逢红出庄（红信号强势跟庄）',
            'red_P': '逢红出闲（红信号强势跟闲）',
            'blue_B': '逢蓝出庄（蓝信号强势跟庄）',
            'blue_P': '逢蓝出闲（蓝信号强势跟闲）',
        }.get(best_rule, best_rule)

        roads_agree = f'{best_votes}/3条派生路'
        narrative = f'元规律：{rule_desc}（{roads_agree}一致，强度{best_conf:.0f}%）'

        return best_rule, consistency, narrative

    def _compute_resonance(self, rules: dict) -> dict:
        """
        V2 新增：三路共振信号检测。

        检查三条派生路的最新信号（红/蓝）是否一致：
          - 三路全红 → 强趋势信号（resonance='strong_red'）
          - 三路全蓝 → 混沌信号（resonance='strong_blue'）
          - 混合 → 中性（resonance='mixed'）

        Returns:
            {
                'resonance': str,           # 'strong_red'|'strong_blue'|'mixed'
                'red_count': int,           # 红信号数（0~3）
                'blue_count': int,          # 蓝信号数（0~3）
                'confidence_boost': float,  # 共振加成（0.8~1.2）
            }
        """
        red_count = 0
        blue_count = 0

        for road_name in ['big_eye', 'small_road', 'cockroach']:
            r = rules.get(road_name, {})
            # 看最近的信号——用红总数和蓝总数的比来推断最近趋势
            red_total = r.get('red_total', 0)
            blue_total = r.get('blue_total', 0)
            if red_total > blue_total:
                red_count += 1
            elif blue_total > red_total:
                blue_count += 1

        if red_count == 3:
            return {
                'resonance': 'strong_red',
                'red_count': red_count,
                'blue_count': blue_count,
                'confidence_boost': 1.15,
            }
        elif blue_count == 3:
            return {
                'resonance': 'strong_blue',
                'red_count': red_count,
                'blue_count': blue_count,
                'confidence_boost': 0.80,
            }
        else:
            return {
                'resonance': 'mixed',
                'red_count': red_count,
                'blue_count': blue_count,
                'confidence_boost': 1.0,
            }

    def _make_prediction(self, columns: list, dominant_rule: str,
                         rules: dict, rule_confidence: float) -> dict:
        """
        基于最新派生路信号和主导元规律，预测下一局结果。

        策略：
          1. 计算当前最新一列结束后，各条派生路的下一个信号（R/B）
          2. 用主导规律 + 该信号 → 预测 B 或 P
          3. 置信度 = rule_confidence × 衰减（避免虚高）
        """
        if dominant_rule == 'unstable' or not columns:
            return {'B': 50.0, 'P': 50.0, 'T': 0.0,
                    'confidence': 0, 'method': 'derived_road',
                    'reason': '元规律不稳定，不输出预测'}

        # 最新信号：看最后一列与倒数第2列的高度关系（简化：用大眼仔逻辑）
        last_len = len(columns[-1]) if len(columns) >= 1 else 0
        ref_len = len(columns[-3]) if len(columns) >= 3 else 0

        if last_len >= ref_len:
            latest_signal = 'R'
        else:
            latest_signal = 'B'

        # 基于主导规律查表
        # dominant_rule = 'red_B' | 'red_P' | 'blue_B' | 'blue_P'
        rule_signal, rule_side = dominant_rule.split('_')  # e.g., 'red', 'B'

        if rule_signal.upper() == latest_signal:
            # 信号匹配主导规律 → 预测相应方向
            predict_side = rule_side
            confidence_multiplier = 1.0
        else:
            # 信号不匹配 → 反向（但置信度打折）
            predict_side = 'P' if rule_side == 'B' else 'B'
            confidence_multiplier = 0.6

        # 计算最终置信度（封顶 MAX_CONFIDENCE）
        confidence = min(self.MAX_CONFIDENCE,
                         rule_confidence * confidence_multiplier)

        b_prob = 55.0 if predict_side == 'B' else 40.0
        p_prob = 100.0 - b_prob - 5.0

        reason = (f'元规律:{dominant_rule}，'
                  f'最新信号:{"红" if latest_signal == "R" else "蓝"}，'
                  f'预测{"庄" if predict_side == "B" else "闲"}')

        return {
            'B': b_prob,
            'P': p_prob,
            'T': 5.0,
            'confidence': round(confidence, 1),
            'method': 'derived_road',
            'reason': reason,
            'latest_signal': latest_signal,
            'dominant_rule': dominant_rule,
        }

    def _no_data_result(self, n: int) -> dict:
        return {
            'can_predict': False,
            'dominant_rule': 'unknown',
            'rule_confidence': 0.0,
            'prediction': {'B': 50.0, 'P': 50.0, 'T': 0.0,
                           'confidence': 0, 'method': 'derived_road',
                           'reason': f'数据不足（{n}局，去和局后需≥8局）'},
            'road_rules': {},
            'narrative': f'ℹ️ 数据不足（当前{n}局）',
            'sample_count': 0,
        }

    def _insufficient_result(self, samples: int) -> dict:
        return {
            'can_predict': False,
            'dominant_rule': 'learning',
            'rule_confidence': 0.0,
            'prediction': {'B': 50.0, 'P': 50.0, 'T': 0.0,
                           'confidence': 0, 'method': 'derived_road',
                           'reason': f'靴内样本不足（{samples}/{self.MIN_SAMPLES}）'},
            'road_rules': {},
            'narrative': f'ℹ️ 派生路正在学习中（已积累{samples}个信号对，需≥{self.MIN_SAMPLES}）',
            'sample_count': samples,
        }
