"""
靴牌性格实时评估器 (Shoe Regime Detector)

设计思想：
  每新增一局，从第一局到当前整靴累计分析，识别主导模式（长龙/单跳/双跳/混沌）
  及其强度，同时检测规律切换事件，用以动态调整集成预测的置信度和权重。

关键特性：
  - 累计全靴分析（非固定窗口）
  - 近期指数加权（支持切换检测）
  - 输出连续强度分数（0~100%），而非是/否判断
  - 置信度因子用于乘以所有预测模型的置信度
"""


class ShoeRegimeDetector:
    """靴牌性格实时评估器"""

    # 近期衰减因子：越近期的局，权重越高
    # 0.92 意味着距现在 10 局前的数据权重约为最近局的 (0.92^10 ≈ 0.43)
    RECENCY_DECAY = 0.92

    # 主导模式的强度阈值
    DOMINANT_THRESH = 0.55   # 超过55%算"主导"
    WEAK_THRESH = 0.40       # 低于40%算"混沌/亂路"

    # — 置信度因子 —
    CF_STRONG   = 1.05   # 主导强度 ≥ 70%，稳定
    CF_NORMAL   = 1.00   # 主导强度 55~70%，稳定或增强
    CF_UNCLEAR  = 0.85   # 强度 40~55%，不明确
    CF_SWITCH   = 0.70   # 正在切换中
    CF_CHAOS    = 0.65   # 主导强度 < 40%（亂路）

    def __init__(self):
        """初始化"""
        pass

    # ──────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────

    def analyze(self, shoe_data: list) -> dict:
        """
        主分析接口：从第一局到当前整靴分析。

        Args:
            shoe_data: 当前靴牌数据列表（含和局），如 ['B','P','T','B',...]

        Returns:
            {
                'dominant_regime': str,     # 主导模式名
                'regime_strength': float,   # 0~100，主导模式加权强度占比
                'regime_scores': dict,      # 四种模式各自分数
                'regime_trend': str,        # 'stable'|'strengthening'|'weakening'|'switching'
                'switch_event': bool,       # 最近5局内是否发生主导模式切换
                'confidence_factor': float, # 乘以预测置信度的系数（0.5~1.1）
                'recommendation': str,      # 给用户看的中文描述
                'can_analyze': bool,        # 数据是否足够
            }
        """
        # 去除和局
        seq = [x for x in shoe_data if x != 'T']
        n = len(seq)

        # 数据不足：返回中性结果
        if n < 8:
            return self._neutral_result(n)

        # Step 1: 给序列中每个位置标注模式类型
        labels = self._label_sequence(seq)

        # Step 2: 用指数衰减权重计算各模式的加权占比
        scores = self._compute_weighted_scores(labels)

        # Step 3: 确定主导模式和强度
        dominant, strength = self._get_dominant(scores)

        # Step 4: 检测规律趋势（与前半段比较）
        trend, switch_event = self._detect_regime_trend(seq, dominant)

        # Step 5: 计算置信度因子
        cf = self._compute_confidence_factor(strength, trend)

        # Step 6: 生成描述
        recommendation = self._build_recommendation(dominant, strength, trend, switch_event)

        return {
            'dominant_regime': dominant,
            'regime_strength': round(strength * 100, 1),
            'regime_scores': {k: round(v * 100, 1) for k, v in scores.items()},
            'regime_trend': trend,
            'switch_event': switch_event,
            'confidence_factor': cf,
            'recommendation': recommendation,
            'can_analyze': True,
            'seq_length': n,
        }

    def get_model_weight_adjustments(self, regime_result: dict) -> dict:
        """
        根据 Regime 结果返回各模型的权重调整系数。

        Args:
            regime_result: analyze() 的返回值。

        Returns:
            {model_name: multiplier}
        """
        if not regime_result.get('can_analyze', False):
            return {}

        dominant = regime_result['dominant_regime']
        strength = regime_result['regime_strength'] / 100.0   # 转回 0~1
        trend = regime_result['regime_trend']
        switch = regime_result['switch_event']

        # 基础调整表（主导模式 → 各模型乘数）
        # 线性插值：strength 在 [0.55, 0.85] 之间时，系数从 1.0 渐变到最大/最小值
        def lerp(base_min, base_max, s):
            """在 strength 0.55~0.85 之间线性插值"""
            t = min(1.0, max(0.0, (s - 0.55) / 0.30))
            return base_min + t * (base_max - base_min)

        adj = {}

        if dominant == 'long_streak':
            s = strength if strength >= 0.55 else 0.55
            adj = {
                'Streak':      lerp(1.0, 1.6, s),
                'Historical':  lerp(1.0, 1.1, s),
                'SimilarShoe': lerp(1.0, 1.2, s),
                'IntraShoeNgram': lerp(1.0, 1.4, s),
                'Trend':       lerp(0.8, 0.6, s),
                'Frequency':   lerp(0.9, 0.8, s),
            }

        elif dominant == 'single_alt':
            s = strength if strength >= 0.55 else 0.55
            adj = {
                'Streak':      lerp(0.7, 0.3, s),   # 单跳时连龙模型应大幅降低
                'Historical':  lerp(0.9, 0.8, s),
                'SimilarShoe': lerp(1.0, 1.0, s),
                'IntraShoeNgram': lerp(1.2, 1.6, s),
                'Trend':       lerp(1.0, 0.9, s),
                'Frequency':   lerp(1.0, 1.1, s),
            }

        elif dominant == 'double_alt':
            s = strength if strength >= 0.55 else 0.55
            adj = {
                'Streak':      lerp(0.8, 0.5, s),
                'Historical':  lerp(0.9, 0.8, s),
                'SimilarShoe': lerp(1.0, 0.9, s),
                'IntraShoeNgram': lerp(1.2, 1.6, s),
                'DoubleAlt':   lerp(1.2, 1.8, s),   # 双跳专用模型权重大幅提升
                'Trend':       lerp(0.9, 0.8, s),
                'Frequency':   lerp(1.0, 1.1, s),
            }

        else:
            # chaos / switching：所有模型均降权
            chaos_factor = 0.7 if not switch else 0.6
            adj = {
                'Streak':      chaos_factor,
                'Historical':  chaos_factor,
                'SimilarShoe': chaos_factor,
                'IntraShoeNgram': chaos_factor,
                'Trend':       chaos_factor,
                'Frequency':   chaos_factor,
                'LSTM':        chaos_factor,
                'RandomForest': chaos_factor,
                'LSTM_V2':     chaos_factor,
                'RF_V2':       chaos_factor,
            }

        # 切换中：额外惩罚
        if switch and dominant != 'chaos':
            adj = {k: v * 0.85 for k, v in adj.items()}

        return adj

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _label_sequence(self, seq: list) -> list:
        """
        逐位给序列元素打标签，返回与 seq 等长的标签列表。

        Audit#B 修复：引入"run_len==2的匹配居中续"=double_alt区分条件。
        真实长龙：连续段 ≥3 局。
        双跳对内：连续段恰好 = 2（上一对也是2）。

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
                    # Audit#B 修复：段长==2时，判断是真正对还是双跳对内
                    # 寻找当前段的起始位x（run_len[x]==1）
                    pair_start = i - 1  # 当前对的第1局
                    if pair_start >= 1:
                        # 前一局是切换局（前一对的第2局）
                        # 上一对的长度
                        prev_pair_len = run_len[pair_start - 1]  # pair_start-1 是前一对的最后一局
                        if prev_pair_len == 2:
                            # 上一对也恰好是2：典型 BBPP 结构 → double_alt
                            labels[i] = 'double_alt'
                        elif prev_pair_len == 1:
                            # 上一局是单局（上一对也是旭始）: 可能是 BBBP后起的雌形 → chaos
                            labels[i] = 'chaos'
                        else:
                            # 上一对长度 ≥3：上一对是长龙后接双跳 → 保守标 long_streak
                            labels[i] = 'long_streak'
                    else:
                        labels[i] = 'long_streak'

                # else: rl==1 在下面的 diff_prev 分支处理
                continue

            else:  # diff_prev
                if i >= 3:
                    # 双跳檢测：当前是新对的第1局，前一对完整（run_len==2）
                    if run_len[i - 1] == 2 and run_len[i] == 1:
                        labels[i] = 'double_alt'
                        continue

                # 单跳：当前与上一局不同，上一局与上上局也不同（BPBP）
                if seq[i - 2] != seq[i - 1]:
                    labels[i] = 'single_alt'
                    continue

            # else → 默认 chaos

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
            # 权重：越靠近末尾（近期）权重越大
            w = self.RECENCY_DECAY ** (n - 1 - i)
            totals[label] += w
            weight_sum += w

        # 归一化
        if weight_sum > 0:
            return {k: v / weight_sum for k, v in totals.items()}
        return totals

    def _get_dominant(self, scores: dict):
        """确定主导模式和强度"""
        dominant = max(scores, key=scores.get)
        strength = scores[dominant]

        # 如果最高分低于 WEAK_THRESH，强制标为 chaos
        if strength < self.WEAK_THRESH:
            dominant = 'chaos'
            strength = scores.get('chaos', strength)

        return dominant, strength

    def _detect_regime_trend(self, seq: list, current_dominant: str):
        """
        比较"前段"与"近期窗口"的主导模式，判断趋势。

        Audit#C 修复：近期窗口改为 max(8, n//5)，封顶 15 局，
        避免固定 8 局窗口在长靴中触发误报 switch_event=True。

        Returns:
            (trend: str, switch_event: bool)
        """
        n = len(seq)
        switch_event = False
        trend = 'stable'

        if n < 10:
            return trend, switch_event

        # Audit#C修复：动态窗口 = max(8, n//5)，最多 15 局
        recent_window = min(15, max(8, n // 5))
        recent_labels = self._label_sequence(seq[-recent_window:])
        recent_scores = self._compute_weighted_scores(recent_labels)
        recent_dominant, recent_strength = self._get_dominant(recent_scores)

        # 前段：去掉近期窗口后的剩余序列
        if n >= 15:
            early_labels = self._label_sequence(seq[:-recent_window])
            early_scores = self._compute_weighted_scores(early_labels)
            early_dominant, early_strength = self._get_dominant(early_scores)
        else:
            early_dominant = current_dominant
            early_strength = 0.5

        # 切换检测
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

    def _compute_confidence_factor(self, strength: float, trend: str) -> float:
        """根据强度和趋势计算置信度因子"""
        if trend == 'switching':
            return self.CF_SWITCH
        if strength >= 0.70:
            return self.CF_STRONG
        if strength >= self.DOMINANT_THRESH:
            if trend in ('stable', 'strengthening'):
                return self.CF_NORMAL
            else:
                return self.CF_UNCLEAR
        if strength >= self.WEAK_THRESH:
            return self.CF_UNCLEAR
        return self.CF_CHAOS

    def _build_recommendation(self, dominant: str, strength: float,
                               trend: str, switch_event: bool) -> str:
        """生成给用户看的中文描述"""
        names = {
            'long_streak': '长龙',
            'single_alt': '单跳',
            'double_alt': '双跳（双对）',
            'chaos': '混沌（亂路）',
        }
        name = names.get(dominant, dominant)
        s_pct = f"{strength * 100:.0f}%"

        if switch_event:
            return f"⚠️ 走势切换中（原{name}→新规律形成中），置信度已降低，建议观察"

        if dominant == 'chaos' or strength < self.WEAK_THRESH:
            return f"⚠️ 当前走势混乱（亂路，强度仅 {s_pct}），建议以观察为主"

        trend_desc = {
            'stable': '稳定',
            'strengthening': '走势增强',
            'weakening': '走势减弱',
            'switching': '切换中',
        }.get(trend, trend)

        return f"当前主导模式：{name}（强度 {s_pct}，{trend_desc}）"

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
            'recommendation': f'ℹ️ 数据不足（当前{n}局，去和局后需≥8局才能分析）',
            'can_analyze': False,
            'seq_length': n,
        }
