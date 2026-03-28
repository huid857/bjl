"""
统计分析引擎
包含N-gram分析、路单分析、趋势分析等
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
        self.data_no_tie = [x for x in data if x != 'T']  # 去除和局的数据
        
        # 当前靴牌数据（用于Streak/Trend分析）
        self.current_shoe_data = current_shoe_data if current_shoe_data is not None else data
        self.current_shoe_no_tie = [x for x in self.current_shoe_data if x != 'T']
    
    def analyze_ngram(self, n=3, exclude_tie=True):
        """
        N-gram模式分析
        
        Args:
            n: N-gram的长度（默认3，即看前2局预测下一局）
            exclude_tie: 是否排除和局
            
        Returns:
            字典 {模式: {B: 概率, P: 概率, T: 概率}}
        """
        data = self.data_no_tie if exclude_tie else self.data
        
        if len(data) < n:
            return {}
        
        patterns = defaultdict(lambda: {'B': 0, 'P': 0, 'T': 0})
        
        # 统计每个模式后面出现的结果
        for i in range(len(data) - n):
            pattern = ''.join(data[i:i+n-1])  # 前n-1个作为模式
            next_result = data[i+n-1]  # 第n个作为结果
            patterns[pattern][next_result] += 1
        
        # 转换为概率
        result = {}
        for pattern, counts in patterns.items():
            total = sum(counts.values())
            if total > 0:
                result[pattern] = {
                    'B': counts['B'] / total * 100,
                    'P': counts['P'] / total * 100,
                    'T': counts['T'] / total * 100,
                    'count': total  # 该模式出现的总次数
                }
        
        return result
    
    def predict_by_ngram(self, recent_data, n=3, exclude_tie=True):
        """
        基于N-gram预测下一局
        
        Args:
            recent_data: 最近的数据（用于匹配模式）
            n: N-gram长度
            exclude_tie: 是否排除和局
            
        Returns:
            预测结果字典 {'B': 概率, 'P': 概率, 'T': 概率, 'confidence': 置信度}
        """
        if exclude_tie:
            recent_data = [x for x in recent_data if x != 'T']
        
        if len(recent_data) < n - 1:
            return self._default_prediction()
        
        # 获取当前模式
        current_pattern = ''.join(recent_data[-(n-1):])
        
        # 分析N-gram
        ngram_stats = self.analyze_ngram(n, exclude_tie)
        
        # 查找匹配的模式
        if current_pattern in ngram_stats:
            stats = ngram_stats[current_pattern]
            confidence = min(100, stats['count'] * 5)  # 置信度基于出现次数
            
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
            # 如果没有找到精确匹配，尝试更短的模式
            if n > 2:
                return self.predict_by_ngram(recent_data, n-1, exclude_tie)
            else:
                return self._default_prediction()
    
    def analyze_road_map(self):
        """
        路单分析（大路）
        
        Returns:
            路单矩阵和统计信息
        """
        if not self.data_no_tie:
            return [], {}
        
        # 构建大路矩阵
        road = []
        current_column = []
        last_result = None
        
        for result in self.data_no_tie:
            if result == last_result or last_result is None:
                # 继续当前列
                current_column.append(result)
            else:
                # 开始新列
                if current_column:
                    road.append(current_column)
                current_column = [result]
            
            last_result = result
        
        # 添加最后一列
        if current_column:
            road.append(current_column)
        
        # 分析路单特征
        stats = {
            'columns': len(road),
            'max_column_height': max([len(col) for col in road]) if road else 0,
            'avg_column_height': sum([len(col) for col in road]) / len(road) if road else 0,
            'long_dragons': self._count_long_dragons(road),  # 长龙数量（5+）
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

        Audit#F 修复：不再用"最后一列的方向"判断庄/闲龙，
        改为用全部历史列的高度分布判断，并用庄/闲总计数决定龙的方向。

        Returns:
            'long_banker', 'long_player', 'alternating', 'mixed'
        """
        if len(road) < 3:
            return 'insufficient_data'

        # 计算最近10列的平均高度
        recent_columns = road[-10:] if len(road) >= 10 else road
        avg_height = sum([len(col) for col in recent_columns]) / len(recent_columns)

        if avg_height >= 3:
            # 用所有列统计哪一方的龙更多
            banker_long = sum(1 for col in road if len(col) >= 3 and col[0] == 'B')
            player_long = sum(1 for col in road if len(col) >= 3 and col[0] == 'P')
            if banker_long >= player_long:
                return 'long_banker'
            else:
                return 'long_player'
        elif avg_height <= 1.5:
            return 'alternating'  # 单跳或双跳
        else:
            return 'mixed'

    
    def analyze_trend(self, window_sizes=[10, 20, 30]):
        """
        趋势分析

        Fix(Bug#7): 窗口统计优先使用 current_shoe_data，避免跨靴牌污染。
        
        Args:
            window_sizes: 滑动窗口大小列表
            
        Returns:
            趋势统计字典
        """
        result = {}
        
        # Fix(Bug#7): 优先用当前靴牌数据做窗口统计
        base_data = self.current_shoe_data if self.current_shoe_data else self.data
        
        for window in window_sizes:
            if len(base_data) >= window:
                recent = base_data[-window:]
                stats = calculate_statistics(recent)
                result[f'last_{window}'] = stats
        
        # 当前连胜情况（使用当前靴牌）
        streak_type, streak_count = get_current_streak(self.current_shoe_no_tie)
        result['current_streak'] = {
            'type': streak_type,
            'count': streak_count
        }
        
        # 最长连胜记录（使用当前靴牌）
        result['max_streaks'] = {
            'banker': get_max_streak(self.current_shoe_data, 'B'),
            'player': get_max_streak(self.current_shoe_data, 'P'),
            'tie': get_max_streak(self.current_shoe_data, 'T')
        }
        
        # 判断趋势方向
        result['trend_direction'] = self._determine_trend()
        
        return result
    
    def _determine_trend(self):
        """
        判断当前趋势方向
        
        Returns:
            'banker_strong', 'player_strong', 'balanced'
        """
        # 趋势只能在当前靴牌看，绝不能跨靴牌
        if len(self.current_shoe_data) < 10:
            return 'insufficient_data'
        
        # 获取最近20局(或更少)
        recent = self.current_shoe_data[-20:]
        
        # 如果存在明显的连龙，连龙代表当前绝对的短期趋势！
        streak_type, streak_length = get_current_streak([x for x in recent if x != 'T'])
        if streak_length >= 3:
            return 'banker_strong' if streak_type == 'B' else 'player_strong'
        
        # 没有连龙时，才看大区间偏离
        stats = calculate_statistics(recent)
        
        banker_rate = stats['banker_rate']
        player_rate = stats['player_rate']
        
        diff = abs(banker_rate - player_rate)
        
        if diff < 15:
            return 'balanced'
        elif banker_rate > player_rate:
            return 'banker_strong'
        else:
            return 'player_strong'
    
    def predict_by_trend(self):
        """
        基于趋势预测
        
        Returns:
            预测字典
        """
        trend = self.analyze_trend()
        direction = trend['trend_direction']
        
        if direction == 'banker_strong':
            return {
                'B': 55,
                'P': 40,
                'T': 5,
                'confidence': 60,
                'method': 'trend',
                'reason': '庄家趋势强势'
            }
        elif direction == 'player_strong':
            return {
                'B': 40,
                'P': 55,
                'T': 5,
                'confidence': 60,
                'method': 'trend',
                'reason': '闲家趋势强势'
            }
        else:
            return self._default_prediction()
    
    def predict_by_streak(self):
        """
        基于连胜情况预测

        Fix(Bug#3, Bug#6):
        - 移除硬编码魔数概率（35/60），改为固定但合理的偏移
        - 统一跟龙逻辑：连2+就跟龙（低置信度），无魔数反龙
        - 置信度封顶65%，防止连龙越长越虚高
        
        注意：只分析当前靴牌的连续，不跨靴牌边界
        
        Returns:
            预测字典
        """
        # 使用当前靴牌数据，避免跨靴牌统计
        streak_type, streak_count = get_current_streak(self.current_shoe_no_tie)
        
        if streak_count < 2:
            return self._default_prediction()
        
        # Fix(Bug#6): 统一跟龙逻辑，不做硬编码反龙
        # 连龙时预测跟龙：短连低置信度，长连略提升置信度但封顶65%
        if streak_type == 'B':
            # 庄连，跟庄
            b_prob = min(58.0, 50.0 + streak_count * 1.5)
            p_prob = 100.0 - 5.0 - b_prob
            confidence = min(65, 38 + streak_count * 5)
            return {
                'B': b_prob,
                'P': max(10.0, p_prob),
                'T': 5,
                'confidence': confidence,
                'method': 'follow_streak',
                'reason': f'庄连{streak_count}次，跟龙'
            }
        else:
            # 闲连，跟闲
            p_prob = min(56.0, 49.0 + streak_count * 1.5)
            b_prob = 100.0 - 5.0 - p_prob
            confidence = min(65, 38 + streak_count * 5)
            return {
                'B': max(10.0, b_prob),
                'P': p_prob,
                'T': 5,
                'confidence': confidence,
                'method': 'follow_streak',
                'reason': f'闲连{streak_count}次，跟龙'
            }

    def predict_by_double_alternation(self, regime_result=None):
        """
        基于双跳（双对 BBPP）模式预测。

        只在 ShoeRegimeDetector 报告 dominant_regime == 'double_alt'
        且强度 ≥ 50% 时才应参与集成；若无 regime_result，则内部
        自行判断双跳强度。

        置信度封顶 60%（低于单跳），因为双跳更容易被误识别。

        Returns:
            预测字典，若当前无双跳模式则返回 _default_prediction()
        """
        # 若有 regime_result，直接用强度做门控
        if regime_result is not None:
            if (regime_result.get('dominant_regime') != 'double_alt'
                    or regime_result.get('regime_strength', 0) < 50):
                return self._default_prediction()

        seq = self.current_shoe_no_tie
        n = len(seq)

        if n < 6:
            return self._default_prediction()

        # 识别最近一个完整"双对"的方向
        # 向后扫描找到最后一个完整对（连续2个相同）
        # 格式最近: ... X X Y Y → 预测下一为 Y Y（继续当前对的第二局）
        #           或 ... X X Y （当前是新对开头） → 预测 Y
        recent = seq[-8:]  # 取最近8局分析

        # 扫描最近的完整 BBPP 对
        i = len(recent) - 1
        last_pair_val = None
        in_new_pair = False
        new_pair_val = None

        # 从后往前找最后一个"跳点"（前后不同的位置）
        for j in range(len(recent) - 1, 0, -1):
            if recent[j] != recent[j - 1]:
                # recent[j] 是当前对的开头
                new_pair_val = recent[j]
                in_new_pair = True
                # recent[j-1] 是前一对末尾
                # 找前一对的方向
                if j >= 2 and recent[j - 1] == recent[j - 2]:
                    last_pair_val = recent[j - 1]
                break

        if new_pair_val is None:
            return self._default_prediction()

        # 判断我们处于新对的第几局
        # 找从 new_pair_val 开始的连续段
        run_start = len(recent) - 1
        while run_start > 0 and recent[run_start] == recent[run_start - 1]:
            run_start -= 1

        run_len = len(recent) - run_start

        if run_len == 1:
            # 新对第一局，预测继续这个对（第二局同方）
            predict_side = new_pair_val
            reason = f'双跳模式，当前{new_pair_val}对第1局，预测继续'
        else:
            # 新对第二局已出，预测切换到另一方对
            predict_side = 'B' if new_pair_val == 'P' else 'P'
            reason = f'双跳模式，{new_pair_val}对已完成，预测切换到{predict_side}'

        # 计算强度（若无 regime_result，用简单双对计数评估置信度）
        if regime_result is not None:
            strength_pct = regime_result.get('regime_strength', 50)
        else:
            strength_pct = 50

        confidence = min(60, 35 + strength_pct * 0.25)

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
        """
        默认预测（基于理论概率）

        Returns:
            预测字典
        """
        return {
            'B': 45.86,
            'P': 44.62,
            'T': 9.52,
            'confidence': 30,
            'method': 'theoretical',
            'reason': '数据不足，使用理论概率'
        }

    def get_comprehensive_analysis(self):
        """
        获取综合分析报告

        Returns:
            完整的分析字典
        """
        if len(self.data) < 10:
            return {
                'error': '数据不足（至少需要10局）',
                'data_count': len(self.data)
            }

        # 基础统计
        basic_stats = calculate_statistics(self.data)

        # N-gram分析
        ngram_2 = self.analyze_ngram(2)
        ngram_3 = self.analyze_ngram(3)
        ngram_4 = self.analyze_ngram(4)

        # 路单分析
        road, road_stats = self.analyze_road_map()

        # 趋势分析
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
