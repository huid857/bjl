"""
靴牌性格实时评估器 (Shoe Regime Detector) - V2 增强版

V2 优化内容：
  1. 渐进式模式轨迹追踪：每局记录累计分析快照，可回溯模式演变历程
  2. 自适应阈值：用「相对差距」取代固定 55%/40% 门槛
  3. 模式断裂检测：当稳定模式突然被打破时自动预警
  4. 模式稳定性指数：连续 N 局主导模式不变 → 可信度高
  5. 靴内全量滚动分析（从第1局到当前局累计，非固定窗口）

设计思想：
  每新增一局，从第一局到当前整靴累计分析，识别主导模式（长龙/单跳/双跳/混沌）
  及其强度，同时检测规律切换事件，用以动态调整集成预测的置信度和权重。

关键特性：
  - 累计全靴分析（非固定窗口）
  - 近期指数加权（支持切换检测）
  - 输出连续强度分数（0~100%），而非是/否判断
  - 渐进式模式轨迹（regime_history）
  - 模式稳定性指数和断裂检测
  - 自适应阈值（基于模式间相对差距）
"""


class ShoeRegimeDetector:
    """靴牌性格实时评估器 V2"""

    # 近期衰减因子：越近期的局，权重越高
    # 0.92 意味着距现在 10 局前的数据权重约为最近局的 (0.92^10 ≈ 0.43)
    RECENCY_DECAY = 0.92

    # 自适应阈值参数（取代原固定 DOMINANT_THRESH=0.55 和 WEAK_THRESH=0.40）
    # 「主导」：最高模式分数与第二高的差距 ≥ GAP_DOMINANT_MIN 即可视为主导
    GAP_DOMINANT_MIN = 0.10   # 最低差距 10% 就可认定主导
    # 当最高模式分数本身 < FLOOR_STRENGTH 时，无论差距多大都视为混沌
    FLOOR_STRENGTH = 0.25
    # 当差距 < GAP_CHAOS 时，视为混沌
    GAP_CHAOS = 0.05

    # — 置信度因子 —
    CF_STRONG   = 1.05   # 主导强度 ≥ 70%，稳定
    CF_NORMAL   = 1.00   # 正常主导
    CF_UNCLEAR  = 0.85   # 不明确
    CF_SWITCH   = 0.70   # 正在切换中
    CF_CHAOS    = 0.65   # 混沌
    CF_BREAK    = 0.60   # 模式断裂

    def __init__(self):
        """初始化"""
        # V2 新增：模式轨迹历史，每次 analyze 自动追加
        self.regime_history = []
        # V2 新增：断裂检测状态
        self._last_stable_regime = None
        self._stable_count = 0
        self._peak_stable_strength = 0  # 稳定期间的峰值强度

    # ──────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────

    def analyze(self, shoe_data: list) -> dict:
        """
        主分析接口：从第一局到当前整靴分析。

        Args:
            shoe_data: 当前靴牌数据列表（含和局），如 ['B','P','T','B',...]

        Returns:
            字典（详见下方）
        """
        # 去除和局
        seq = [x for x in shoe_data if x != 'T']
        n = len(seq)

        # 数据不足：返回中性结果
        if n < 6:
            result = self._neutral_result(n)
            self._update_history(n, result)
            return result

        # Step 1: 给序列中每个位置标注模式类型
        labels = self._label_sequence(seq)

        # Step 2: 用指数衰减权重计算各模式的加权占比
        scores = self._compute_weighted_scores(labels)

        # Step 3: 自适应阈值确定主导模式和强度
        dominant, strength, gap = self._get_dominant_adaptive(scores)

        # Step 4: 检测规律趋势（与前半段比较）
        trend, switch_event = self._detect_regime_trend(seq, dominant)

        # Step 5: 模式断裂检测（V2 新增）
        break_detected = self._detect_pattern_break(dominant, strength)

        # Step 6: 计算模式稳定性指数（V2 新增）
        stability_index = self._compute_stability_index()

        # Step 7: 计算置信度因子
        cf = self._compute_confidence_factor(strength, trend, break_detected, stability_index)

        # Step 8: 生成描述
        recommendation = self._build_recommendation(
            dominant, strength, trend, switch_event,
            break_detected, stability_index
        )

        result = {
            'dominant_regime': dominant,
            'regime_strength': round(strength * 100, 1),
            'regime_scores': {k: round(v * 100, 1) for k, v in scores.items()},
            'regime_trend': trend,
            'switch_event': switch_event,
            'confidence_factor': cf,
            'recommendation': recommendation,
            'can_analyze': True,
            'seq_length': n,
            # V2 新增字段
            'gap_to_second': round(gap * 100, 1),    # 与第二名的差距
            'break_detected': break_detected,          # 模式断裂
            'stability_index': stability_index,         # 稳定性指数 (0~1)
            'regime_history_len': len(self.regime_history),
        }

        # 更新历史轨迹
        self._update_history(n, result)

        return result

    def get_model_weight_adjustments(self, regime_result: dict) -> dict:
        """
        根据 Regime 结果返回各模型的权重调整系数。
        V2：增加了断裂检测和稳定性指数的影响。
        """
        if not regime_result.get('can_analyze', False):
            return {}

        dominant = regime_result['dominant_regime']
        strength = regime_result['regime_strength'] / 100.0
        trend = regime_result['regime_trend']
        switch = regime_result['switch_event']
        break_detected = regime_result.get('break_detected', False)
        stability = regime_result.get('stability_index', 0.5)

        def lerp(base_min, base_max, s):
            """在 strength 0.30~0.85 之间线性插值（自适应阈值下起点更低）"""
            t = min(1.0, max(0.0, (s - 0.30) / 0.55))
            return base_min + t * (base_max - base_min)

        adj = {}

        if dominant == 'long_streak':
            s = max(strength, 0.30)
            adj = {
                'Streak':         lerp(1.0, 1.6, s),
                'Historical':     lerp(1.0, 1.1, s),
                'SimilarShoe':    lerp(1.0, 1.2, s),
                'IntraShoeNgram': lerp(1.0, 1.4, s),
                'Trend':          lerp(0.8, 0.6, s),
                'Frequency':      lerp(0.9, 0.8, s),
            }

        elif dominant == 'single_alt':
            s = max(strength, 0.30)
            adj = {
                'Streak':         lerp(0.7, 0.3, s),
                'Historical':     lerp(0.9, 0.8, s),
                'SimilarShoe':    lerp(1.0, 1.0, s),
                'IntraShoeNgram': lerp(1.2, 1.6, s),
                'Trend':          lerp(1.0, 0.9, s),
                'Frequency':      lerp(1.0, 1.1, s),
            }

        elif dominant == 'double_alt':
            s = max(strength, 0.30)
            adj = {
                'Streak':         lerp(0.8, 0.5, s),
                'Historical':     lerp(0.9, 0.8, s),
                'SimilarShoe':    lerp(1.0, 0.9, s),
                'IntraShoeNgram': lerp(1.2, 1.6, s),
                'DoubleAlt':      lerp(1.2, 1.8, s),
                'Trend':          lerp(0.9, 0.8, s),
                'Frequency':      lerp(1.0, 1.1, s),
            }

        else:
            # chaos / switching：所有模型均降权
            chaos_factor = 0.7 if not switch else 0.6
            adj = {
                'Streak': chaos_factor, 'Historical': chaos_factor,
                'SimilarShoe': chaos_factor, 'IntraShoeNgram': chaos_factor,
                'Trend': chaos_factor, 'Frequency': chaos_factor,
                'LSTM': chaos_factor, 'RandomForest': chaos_factor,
                'LSTM_V2': chaos_factor, 'RF_V2': chaos_factor,
            }

        # V2：切换中额外惩罚
        if switch and dominant != 'chaos':
            adj = {k: v * 0.85 for k, v in adj.items()}

        # V2：模式断裂额外惩罚（所有模型降 30%）
        if break_detected:
            adj = {k: v * 0.70 for k, v in adj.items()}

        # V2：稳定性加成（稳定模式加权，不稳定模式降权）
        if stability > 0.7 and dominant != 'chaos':
            stability_bonus = 1.0 + (stability - 0.7) * 0.3  # 最高 1.09
            adj = {k: v * stability_bonus for k, v in adj.items()}
        elif stability < 0.3:
            stability_penalty = 0.8 + stability * 0.67  # 最低 0.8
            adj = {k: v * stability_penalty for k, v in adj.items()}

        return adj

    def get_regime_trajectory(self) -> list:
        """
        V2 新增：获取完整的模式演变轨迹。

        Returns:
            list of dict: 每局的模式快照
        """
        return list(self.regime_history)

    def reset(self):
        """重置状态（新靴牌开始时调用）"""
        self.regime_history = []
        self._last_stable_regime = None
        self._stable_count = 0
        self._peak_stable_strength = 0

    # ──────────────────────────────────────────────
    # V2 新增：模式轨迹追踪
    # ──────────────────────────────────────────────

    def _update_history(self, round_num: int, result: dict):
        """记录当前局的模式快照到轨迹历史"""
        snapshot = {
            'round': round_num,
            'dominant': result.get('dominant_regime', 'unknown'),
            'strength': result.get('regime_strength', 0),
            'trend': result.get('regime_trend', 'stable'),
            'stability': result.get('stability_index', 0),
        }
        self.regime_history.append(snapshot)

    # ──────────────────────────────────────────────
    # V2 新增：模式断裂检测
    # ──────────────────────────────────────────────

    def _detect_pattern_break(self, current_dominant: str, current_strength: float) -> bool:
        """
        检测模式断裂：之前稳定的模式突然消失或强度骤降。

        定义：
          - 过去连续 ≥ 4 局的主导模式相同（稳定期），且当前局主导模式改变 → 断裂
          - 主导模式不变但强度从峰值下降超过 25% → 断裂（模式衰减）

        Returns:
            bool: 是否检测到断裂
        """
        if not self.regime_history:
            self._last_stable_regime = current_dominant
            self._stable_count = 1
            self._peak_stable_strength = current_strength * 100
            return False

        last = self.regime_history[-1]
        last_dominant = last.get('dominant', 'unknown')

        if current_dominant == last_dominant and current_dominant != 'chaos':
            self._stable_count += 1
            # 更新峰值强度
            current_str_pct = current_strength * 100
            if current_str_pct > self._peak_stable_strength:
                self._peak_stable_strength = current_str_pct
            if self._stable_count >= 4:
                self._last_stable_regime = current_dominant

            # 检查强度从峰值骤降（即使模式没变）
            if (self._stable_count >= 4
                    and self._peak_stable_strength - current_str_pct > 25):
                # Audit#C修复：断裂后重置峰值和计数，避免后续每局都报断裂
                self._peak_stable_strength = current_str_pct
                self._stable_count = 1
                return True
        else:
            # 模式变了
            if (self._last_stable_regime is not None
                    and self._stable_count >= 4
                    and current_dominant != self._last_stable_regime):
                # 断裂！之前稳定的模式被打破
                self._stable_count = 1
                self._peak_stable_strength = current_strength * 100
                return True

            self._stable_count = 1
            self._peak_stable_strength = current_strength * 100

        return False

    # ──────────────────────────────────────────────
    # V2 新增：模式稳定性指数
    # ──────────────────────────────────────────────

    def _compute_stability_index(self) -> float:
        """
        计算模式稳定性指数（0~1）。

        定义：最近 N 局中，主导模式未变化的比例。
        N = min(10, len(history))

        Returns:
            float: 0.0（极不稳定） ~ 1.0（完全稳定）
        """
        if len(self.regime_history) < 3:
            return 0.5  # 数据不足，中性

        window = min(10, len(self.regime_history))
        recent = self.regime_history[-window:]

        # 统计最频繁的主导模式
        mode_counts = {}
        for snap in recent:
            d = snap.get('dominant', 'unknown')
            mode_counts[d] = mode_counts.get(d, 0) + 1

        if not mode_counts:
            return 0.5

        most_common_count = max(mode_counts.values())
        stability = most_common_count / window

        return round(stability, 3)

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _label_sequence(self, seq: list) -> list:
        """
        逐位给序列元素打标签，返回与 seq 等长的标签列表。

        标签类型：
          'long_streak' — 处于真实长龙段（连续段长度 ≥3）
          'single_alt'  — 处于严格单跳（BPBP）
          'double_alt'  — 处于双对交替（BBPP）
          'chaos'       — 不属于任何清晰模式
        """
        n = len(seq)
        labels = ['chaos'] * n

        if n < 3:
            return labels

        # 预计算每个位置的"连续段长度"（向左延伸）
        run_len = [1] * n
        for i in range(1, n):
            if seq[i] == seq[i - 1]:
                run_len[i] = run_len[i - 1] + 1
            else:
                run_len[i] = 1

        for i in range(2, n):
            same_prev = (seq[i] == seq[i - 1])

            if same_prev:
                rl = run_len[i]
                if rl >= 3:
                    # 真实长龙：连续段长度 ≥3
                    labels[i] = 'long_streak'

                elif rl == 2:
                    # 段长==2时，判断是真正对还是双跳对内
                    pair_start = i - 1
                    if pair_start >= 1:
                        prev_pair_len = run_len[pair_start - 1]
                        if prev_pair_len == 2:
                            labels[i] = 'double_alt'
                        elif prev_pair_len == 1:
                            labels[i] = 'chaos'
                        else:
                            labels[i] = 'long_streak'
                    else:
                        labels[i] = 'long_streak'
                continue

            else:  # diff_prev
                # Audit#C修复：双跳判断需要更严格的条件
                # 不仅前段长度==2，还要再往前也是长度2的不同值段（BBPP结构）
                if i >= 3:
                    if run_len[i - 1] == 2 and run_len[i] == 1:
                        # 找到前段(长度2)的起始位置，再看更前一段
                        prev_seg_start = i - 2  # 前段起始（长度2的段头）
                        if prev_seg_start >= 2:
                            # 更前一段的末尾在 prev_seg_start - 1
                            before_prev_len = run_len[prev_seg_start - 1]
                            if before_prev_len == 2:
                                # 确认是 XX YY Z 结构（真正双跳）
                                labels[i] = 'double_alt'
                                continue
                        # 否则不是双跳：可能是长龙末尾的PP→B

                if seq[i - 2] != seq[i - 1]:
                    labels[i] = 'single_alt'
                    continue

        return labels

    def _compute_weighted_scores(self, labels: list) -> dict:
        """
        对标签列表进行近期指数加权，计算四种模式的归一化占比。
        """
        n = len(labels)
        if n == 0:
            return {'long_streak': 0.0, 'single_alt': 0.0,
                    'double_alt': 0.0, 'chaos': 0.0}

        totals = {'long_streak': 0.0, 'single_alt': 0.0,
                  'double_alt': 0.0, 'chaos': 0.0}
        weight_sum = 0.0

        for i, label in enumerate(labels):
            w = self.RECENCY_DECAY ** (n - 1 - i)
            totals[label] += w
            weight_sum += w

        if weight_sum > 0:
            return {k: v / weight_sum for k, v in totals.items()}
        return totals

    def _get_dominant_adaptive(self, scores: dict):
        """
        V2：自适应阈值确定主导模式和强度。

        不再使用固定的 DOMINANT_THRESH=0.55 和 WEAK_THRESH=0.40，
        而是基于最高分与次高分的「相对差距」来判断。

        规则：
          1. 最高分 < FLOOR_STRENGTH → 强制 chaos
          2. 最高分与次高分差距 < GAP_CHAOS → chaos（势均力敌）
          3. 差距 ≥ GAP_DOMINANT_MIN → 主导模式确认
          4. 中间地带 → 使用最高分模式但标记为不确定

        Returns:
            (dominant: str, strength: float, gap: float)
        """
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_mode, best_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0

        gap = best_score - second_score

        # 规则 1：绝对强度过低 → chaos
        if best_score < self.FLOOR_STRENGTH:
            return 'chaos', best_score, gap

        # 规则 2：差距太小 → chaos
        if gap < self.GAP_CHAOS:
            return 'chaos', best_score, gap

        # 规则 3 & 4：差距足够 → 主导；差距中等 → 也是主导但强度偏低
        return best_mode, best_score, gap

    def _detect_regime_trend(self, seq: list, current_dominant: str):
        """
        比较"前段"与"近期窗口"的主导模式，判断趋势。
        """
        n = len(seq)
        switch_event = False
        trend = 'stable'

        if n < 10:
            return trend, switch_event

        recent_window = min(15, max(8, n // 5))
        recent_labels = self._label_sequence(seq[-recent_window:])
        recent_scores = self._compute_weighted_scores(recent_labels)
        recent_dominant, recent_strength, _ = self._get_dominant_adaptive(recent_scores)

        if n >= 15:
            early_labels = self._label_sequence(seq[:-recent_window])
            early_scores = self._compute_weighted_scores(early_labels)
            early_dominant, early_strength, _ = self._get_dominant_adaptive(early_scores)
        else:
            early_dominant = current_dominant
            early_strength = 0.5

        if recent_dominant != early_dominant and recent_dominant != 'chaos':
            switch_event = True
            trend = 'switching'
        elif recent_dominant == 'chaos' and early_dominant != 'chaos':
            switch_event = True
            trend = 'switching'
        else:
            delta = recent_strength - early_strength
            if delta > 0.10:
                trend = 'strengthening'
            elif delta < -0.10:
                trend = 'weakening'
            else:
                trend = 'stable'

        return trend, switch_event

    def _compute_confidence_factor(self, strength: float, trend: str,
                                    break_detected: bool,
                                    stability_index: float) -> float:
        """V2：根据强度、趋势、断裂和稳定性综合计算置信度因子"""
        # 模式断裂优先
        if break_detected:
            return self.CF_BREAK

        if trend == 'switching':
            return self.CF_SWITCH

        if strength >= 0.70:
            base_cf = self.CF_STRONG
        elif strength >= 0.45:
            if trend in ('stable', 'strengthening'):
                base_cf = self.CF_NORMAL
            else:
                base_cf = self.CF_UNCLEAR
        elif strength >= 0.30:
            base_cf = self.CF_UNCLEAR
        else:
            base_cf = self.CF_CHAOS

        # V2：稳定性修正
        if stability_index >= 0.8:
            base_cf = min(1.10, base_cf * 1.05)
        elif stability_index < 0.3:
            base_cf = base_cf * 0.90

        return round(base_cf, 3)

    def _build_recommendation(self, dominant: str, strength: float,
                               trend: str, switch_event: bool,
                               break_detected: bool, stability_index: float) -> str:
        """生成给用户看的中文描述"""
        names = {
            'long_streak': '长龙',
            'single_alt': '单跳',
            'double_alt': '双跳（双对）',
            'chaos': '混沌（乱路）',
        }
        name = names.get(dominant, dominant)
        s_pct = f"{strength * 100:.0f}%"

        # V2：断裂预警
        if break_detected:
            return (f"⚡ 模式断裂！原{self._last_stable_regime or ''}模式被打破，"
                    f"当前转向{name}，建议暂停观察")

        if switch_event:
            return f"⚠️ 走势切换中（原{name}→新规律形成中），置信度已降低，建议观察"

        if dominant == 'chaos' or strength < self.FLOOR_STRENGTH:
            return f"⚠️ 当前走势混乱（乱路，强度仅 {s_pct}），建议以观察为主"

        trend_desc = {
            'stable': '稳定',
            'strengthening': '走势增强',
            'weakening': '走势减弱',
            'switching': '切换中',
        }.get(trend, trend)

        # V2：加入稳定性信息
        if stability_index >= 0.8:
            stability_text = '，模式高度稳定'
        elif stability_index >= 0.5:
            stability_text = '，模式相对稳定'
        elif stability_index >= 0.3:
            stability_text = '，模式不太稳定'
        else:
            stability_text = '，模式很不稳定'

        return f"当前主导模式：{name}（强度 {s_pct}，{trend_desc}{stability_text}）"

    def _neutral_result(self, n: int) -> dict:
        """数据不足时返回默认中性结果"""
        return {
            'dominant_regime': 'unknown',
            'regime_strength': 0.0,
            'regime_scores': {'long_streak': 0.0, 'single_alt': 0.0,
                              'double_alt': 0.0, 'chaos': 100.0},
            'regime_trend': 'stable',
            'switch_event': False,
            'confidence_factor': 1.0,
            'recommendation': f'ℹ️ 数据不足（当前{n}局，去和局后需≥6局才能分析）',
            'can_analyze': False,
            'seq_length': n,
            'gap_to_second': 0.0,
            'break_detected': False,
            'stability_index': 0.5,
            'regime_history_len': len(self.regime_history),
        }
