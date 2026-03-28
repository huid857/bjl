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

        Audit#C重写：
          原实现用 columns[i][0]（该列方向B/P）作为结果，但大路中
          每列方向必然交替（B列→P列→B列），导致红信号后出B/P的比例
          永远约50/50，统计完全无意义。

          修正：结果改为"续/跳"——该信号对应的列是否延续了长度≥2。
            续(C) = 列长度 > 1（模式有持续性，该列至少连了2局）
            跳(J) = 列长度 == 1（立刻就断了，换方向）

          这样红→续/红→跳 的统计才有实际预测意义：
            "在这靴牌中，看到红信号后，当前趋势是否倾向延续？"

        Args:
            columns: 大路列结构
            offset: 1=大眼仔, 2=小路, 3=曱甴路

        Returns:
            (signals: list, outcomes: list)
            signals[k] = 'R'（红）or 'B'（蓝）
            outcomes[k] = 'C'（续：列长>1）or 'J'（跳：列长==1）
        """
        signals = []
        outcomes = []

        start_col = offset + 1
        for i in range(start_col, len(columns)):
            ref_col_idx = i - offset - 1
            if ref_col_idx < 0:
                continue

            ref_len = len(columns[ref_col_idx])
            cur_len = len(columns[i])

            # 信号判断：当前列与参照列高度比较
            if cur_len >= ref_len:
                sig = 'R'   # 红：当前列不比参照列短（规律延续）
            else:
                sig = 'B'   # 蓝：当前列比参照列短（规律中断）

            # Audit#C修正：结果改为续/跳
            # 看该信号对应列的下一列（i+1）是续还是跳
            if i + 1 < len(columns):
                outcome = 'C' if len(columns[i + 1]) > 1 else 'J'
            else:
                # 最后一列（还在进行中），跳过不记
                continue

            signals.append(sig)
            outcomes.append(outcome)

        return signals, outcomes

    def _compute_conditional_stats(self, signals: list, outcomes: list) -> dict:
        """
        Audit#C重写：统计给定信号序列下"续/跳"的条件概率。

        Returns:
            {
                'red_c': 红信号后续(Continue)的次数,
                'red_j': 红信号后跳(Jump)的次数,
                'blue_c': 蓝信号后续的次数,
                'blue_j': 蓝信号后跳的次数,
                'red_total': 红信号总次数,
                'blue_total': 蓝信号总次数,
                'total': 总信号数,
                'red_dominant': 'C'|'J'|'tie', ← 红信号后更可能续还是跳
                'blue_dominant': 'C'|'J'|'tie',
                'red_confidence': %,
                'blue_confidence': %,
            }
        """
        red_c = red_j = blue_c = blue_j = 0

        for sig, out in zip(signals, outcomes):
            if sig == 'R':
                if out == 'C':
                    red_c += 1
                else:
                    red_j += 1
            else:  # sig == 'B'
                if out == 'C':
                    blue_c += 1
                else:
                    blue_j += 1

        red_total = red_c + red_j
        blue_total = blue_c + blue_j
        total = red_total + blue_total

        red_dominant = 'tie'
        red_conf = 50.0
        if red_total > 0:
            if red_c > red_j:
                red_dominant = 'C'
                red_conf = red_c / red_total * 100
            elif red_j > red_c:
                red_dominant = 'J'
                red_conf = red_j / red_total * 100

        blue_dominant = 'tie'
        blue_conf = 50.0
        if blue_total > 0:
            if blue_c > blue_j:
                blue_dominant = 'C'
                blue_conf = blue_c / blue_total * 100
            elif blue_j > blue_c:
                blue_dominant = 'J'
                blue_conf = blue_j / blue_total * 100

        return {
            'red_c': red_c, 'red_j': red_j,
            'blue_c': blue_c, 'blue_j': blue_j,
            'red_total': red_total, 'blue_total': blue_total,
            'total': total,
            'red_dominant': red_dominant,
            'red_confidence': round(red_conf, 1),
            'blue_dominant': blue_dominant,
            'blue_confidence': round(blue_conf, 1),
        }

    def _determine_dominant_rule(self, rules: dict):
        """
        Audit#C重写：综合三条路的续/跳统计，确定主导元规律。

        元规律类型：
          'red_C' = 红信号后大概率续（逢红续跟，有路）
          'red_J' = 红信号后大概率跳（逢红换方向）
          'blue_C' = 蓝信号后大概率续
          'blue_J' = 蓝信号后大概率跳（逢蓝跳换）
          'unstable' = 三路结论不一致

        Returns:
            (dominant: str, consistency: float, narrative: str)
        """
        road_names = ['big_eye', 'small_road', 'cockroach']

        votes_red_c = votes_red_j = 0
        votes_blue_c = votes_blue_j = 0
        total_conf_red = total_conf_blue = 0

        for road in road_names:
            r = rules[road]
            if r['red_total'] >= 3:
                total_conf_red += r['red_confidence']
                if r['red_dominant'] == 'C':
                    votes_red_c += 1
                elif r['red_dominant'] == 'J':
                    votes_red_j += 1

            if r['blue_total'] >= 3:
                total_conf_blue += r['blue_confidence']
                if r['blue_dominant'] == 'C':
                    votes_blue_c += 1
                elif r['blue_dominant'] == 'J':
                    votes_blue_j += 1

        results = [
            ('red_C', votes_red_c, total_conf_red / 3),
            ('red_J', votes_red_j, total_conf_red / 3),
            ('blue_C', votes_blue_c, total_conf_blue / 3),
            ('blue_J', votes_blue_j, total_conf_blue / 3),
        ]
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)

        best_rule, best_votes, best_conf = results[0]

        if best_votes < 2:
            return 'unstable', 0.3, '⚠️ 派生路信号尚无一致规律，建议继续观察'

        consistency = best_votes / 3

        rule_desc = {
            'red_C': '逢红续跟（红信号后趋势延续）',
            'red_J': '逢红跳换（红信号后趋势断裂）',
            'blue_C': '逢蓝续跟（蓝信号后趋势延续）',
            'blue_J': '逢蓝跳换（蓝信号后趋势断裂）',
        }.get(best_rule, best_rule)

        roads_agree = f'{best_votes}/3条派生路'
        narrative = f'元规律：{rule_desc}（{roads_agree}一致，强度{best_conf:.0f}%）'

        return best_rule, consistency, narrative

    def _compute_resonance(self, rules: dict) -> dict:
        """
        V2 新增：三路共振信号检测。

        Audit#B修复：使用各路最近信号的趋势（最近3个信号中红/蓝的多数）
        而非全靴红蓝总数，更准确反映当前状态。

        检查三条派生路的最新信号趋势是否一致：
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
            # Audit#B修复：用最近信号的红蓝趋势代替全靴总数
            # 通过最近3个信号-结果对的红蓝占比来判断
            red_total = r.get('red_total', 0)
            blue_total = r.get('blue_total', 0)
            total = red_total + blue_total
            if total == 0:
                continue
            # 最近信号趋势：红占比 > 60% 视为红，蓝占比 > 60% 视为蓝
            red_ratio = red_total / total
            if red_ratio > 0.6:
                red_count += 1
            elif red_ratio < 0.4:
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
        Audit#C重写：基于最新派生路信号和续/跳元规律，预测下一局结果。

        策略：
          1. 计算最新信号（R/B）
          2. 用主导规律(red_C/red_J/blue_C/blue_J) + 信号 → 预测续跟还是跳换
          3. 续跟 = 押当前列方向（最后一列的side），跳换 = 押反方向
        """
        if dominant_rule == 'unstable' or not columns:
            return {'B': 50.0, 'P': 50.0, 'T': 0.0,
                    'confidence': 0, 'method': 'derived_road',
                    'reason': '元规律不稳定，不输出预测'}

        # 最新信号
        last_len = len(columns[-1]) if len(columns) >= 1 else 0
        ref_len = len(columns[-3]) if len(columns) >= 3 else 0

        if last_len >= ref_len:
            latest_signal = 'R'
        else:
            latest_signal = 'B'

        # 当前列方向（最后一列的side）
        current_side = columns[-1][0]  # 'B' or 'P'

        # 基于主导规律查表
        # dominant_rule = 'red_C' | 'red_J' | 'blue_C' | 'blue_J'
        rule_signal, rule_action = dominant_rule.split('_')  # e.g., 'red', 'C'

        if rule_signal.upper() == latest_signal:
            # 信号匹配主导规律方向
            if rule_action == 'C':
                # 续跟 → 押当前列方向
                predict_side = current_side
            else:
                # 跳换 → 押反方向
                predict_side = 'P' if current_side == 'B' else 'B'
            confidence_multiplier = 1.0
        else:
            # 信号不匹配 → 反向推断（置信度打折）
            if rule_action == 'C':
                predict_side = 'P' if current_side == 'B' else 'B'
            else:
                predict_side = current_side
            confidence_multiplier = 0.6

        confidence = min(self.MAX_CONFIDENCE,
                         rule_confidence * confidence_multiplier)

        b_prob = 55.0 if predict_side == 'B' else 40.0
        p_prob = 100.0 - b_prob - 5.0

        action_desc = '续跟' if rule_action == 'C' else '跳换'
        reason = (f'元规律:{action_desc}({dominant_rule})，'
                  f'信号:{"红" if latest_signal == "R" else "蓝"}，'
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
