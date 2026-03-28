"""
异常检测器
基于统计学理论和百家乐理论概率检测异常靴牌
"""

import numpy as np
from utils import get_max_streak, get_current_streak


def _get_overall_max_streak(data):
    """
    获取所有类型中的最大连胜
    
    Args:
        data: 结果列表
        
    Returns:
        最长连胜次数
    """
    if not data:
        return 0
    
    max_streak = 0
    current_val = None
    current_streak = 0
    
    for result in data:
        if result == current_val:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_val = result
            current_streak = 1
    
    return max_streak


class AnomalyDetector:
    """
    异常检测器
    
    检测原理：
    1. 统计显著性检验（Z-score）
    2. 极端长连检测
    3. 和局比例异常
    4. 卡方检验
    """
    
    def __init__(self):
        """初始化异常检测器"""
        # 百家乐理论概率（8副牌）
        self.THEORETICAL_PROBS = {
            'banker': 0.4586,  # 庄家赢
            'player': 0.4462,  # 闲家赢
            'tie': 0.0952      # 和局
        }
        
        # 非和局中的比例
        self.BANKER_IN_NON_TIE = 0.507  # 庄家在非和局中占50.7%
        self.PLAYER_IN_NON_TIE = 0.493  # 闲家在非和局中占49.3%
        
        # 检测阈值
        self.thresholds = {
            'z_score_warning': 2.58,      # 99%置信度 (p<0.01)
            'z_score_critical': 3.29,     # 99.9%置信度 (p<0.001)
            'streak_warning': 12,         # 12连概率<0.01%
            'streak_critical': 15,        # 15连概率<0.001%
            'tie_rate_warning': 15,       # 和局>15%
            'tie_rate_critical': 20,      # 和局>20%
            'min_sample_size': 30         # 至少30局才检测
        }
    
    def detect(self, shoe_data):
        """
        综合异常检测

        Bug#14修复：改为渐进式三档检测：
          < 10 局  → 跳过
          10~29 局 → 仅做连龙+和局检测（宽松阈值）
          30+ 局   → 完整四项检测

        Args:
            shoe_data (list): 靴牌数据

        Returns:
            dict: 异常检测结果
        """
        n = len(shoe_data)

        # 档位 0：完全不足
        if n < 10:
            return {
                'is_anomaly': False,
                'severity_score': 0,
                'severity_level': 'insufficient_data',
                'anomalies': [],
                'recommendation': f'ℹ️ 数据不足（当前{n}局，需至少10局）',
                'confidence_adjustment': 1.0
            }

        anomalies = []
        severity_score = 0
        details = {}

        # 档位 1（10~29局）：仅做连龙检测 + 和局检测（宽松阈值）
        if n < self.thresholds['min_sample_size']:
            # 连龙检测（宽松：10连报 warning, 13连报 critical）
            streak_result = self._extreme_streak_test(
                shoe_data,
                warn_override=10, critical_override=13
            )
            details['streak_test'] = streak_result
            if streak_result['is_anomaly']:
                anomalies.append(streak_result['message'])
                severity_score += streak_result['severity']

            # 和局检测（宽松：20%报 warning, 30%报 critical）
            tie_result = self._tie_anomaly_test(
                shoe_data,
                warn_override=20, critical_override=30
            )
            details['tie_test'] = tie_result
            if tie_result['is_anomaly']:
                anomalies.append(tie_result['message'])
                severity_score += tie_result['severity']

            severity_level = self._get_severity_level(severity_score)
            recommendation = self._get_recommendation(severity_score)
            confidence_adjustment = self._get_confidence_adjustment(severity_score)

            return {
                'is_anomaly': severity_score > 0,
                'severity_score': severity_score,
                'severity_level': severity_level,
                'anomalies': anomalies,
                'recommendation': recommendation,
                'confidence_adjustment': confidence_adjustment,
                'partial_detection': True,
                'details': details
            }

        # 档位 2（30+ 局）：完整四项检测
        stat_result = self._statistical_test(shoe_data)
        details['statistical_test'] = stat_result
        if stat_result['is_anomaly']:
            anomalies.append(stat_result['message'])
            severity_score += stat_result['severity']

        streak_result = self._extreme_streak_test(shoe_data)
        details['streak_test'] = streak_result
        if streak_result['is_anomaly']:
            anomalies.append(streak_result['message'])
            severity_score += streak_result['severity']

        tie_result = self._tie_anomaly_test(shoe_data)
        details['tie_test'] = tie_result
        if tie_result['is_anomaly']:
            anomalies.append(tie_result['message'])
            severity_score += tie_result['severity']

        chi_result = self._chi_square_test(shoe_data)
        details['chi_square_test'] = chi_result
        if chi_result['is_anomaly']:
            anomalies.append(chi_result['message'])
            severity_score += chi_result['severity']

        severity_level = self._get_severity_level(severity_score)
        recommendation = self._get_recommendation(severity_score)
        confidence_adjustment = self._get_confidence_adjustment(severity_score)

        return {
            'is_anomaly': severity_score > 0,
            'severity_score': severity_score,
            'severity_level': severity_level,
            'anomalies': anomalies,
            'recommendation': recommendation,
            'confidence_adjustment': confidence_adjustment,
            'details': details
        }

    
    def _statistical_test(self, shoe_data):
        """
        统计显著性检验（Z-score）
        
        检查庄闲比例是否显著偏离理论值
        """
        no_tie = [x for x in shoe_data if x != 'T']
        n = len(no_tie)
        
        if n < 20:
            return {'is_anomaly': False, 'severity': 0}
        
        # 实际观察
        actual_banker = no_tie.count('B')
        
        # 理论期望
        expected_banker = n * self.BANKER_IN_NON_TIE
        
        # 标准差（二项分布）
        std_dev = np.sqrt(n * self.BANKER_IN_NON_TIE * self.PLAYER_IN_NON_TIE)
        
        # Z-score
        z_score = abs(actual_banker - expected_banker) / std_dev
        
        # 判断
        if z_score > self.thresholds['z_score_critical']:
            return {
                'is_anomaly': True,
                'severity': 4,
                'z_score': z_score,
                'message': f'庄闲比例严重异常 (Z={z_score:.2f}, p<0.001)',
                'level': 'critical'
            }
        elif z_score > self.thresholds['z_score_warning']:
            return {
                'is_anomaly': True,
                'severity': 2,
                'z_score': z_score,
                'message': f'庄闲比例异常 (Z={z_score:.2f}, p<0.01)',
                'level': 'warning'
            }
        else:
            return {
                'is_anomaly': False,
                'severity': 0,
                'z_score': z_score,
                'level': 'normal'
            }
    
    def _extreme_streak_test(self, shoe_data, warn_override=None, critical_override=None):
        """
        极端长连检测

        Bug#14修复（新增参数）：
          warn_override / critical_override 允许调用方传入宽松阈值，
          用于早期靴牌（10~29局）的渐进检测。

        Fix(Improvement#3): 使用 0.5 基概率 + 多起始位置期望公式
        """
        no_tie = [x for x in shoe_data if x != 'T']

        if len(no_tie) < 10:
            return {'is_anomaly': False, 'severity': 0}

        max_streak = _get_overall_max_streak(no_tie)
        n = len(no_tie)
        k = max_streak

        # 正确计算多起始位置概率
        base_prob = 0.5
        single_streak_prob = base_prob ** k
        positions = max(1, n - k + 1)
        probability_pct = (1.0 - (1.0 - single_streak_prob) ** positions) * 100
        probability_pct = min(100.0, probability_pct)

        # 使用 override 阈值（若有），否则用默认配置
        thresh_warn = warn_override if warn_override is not None else self.thresholds['streak_warning']
        thresh_critical = critical_override if critical_override is not None else self.thresholds['streak_critical']

        if max_streak >= thresh_critical:
            return {
                'is_anomaly': True,
                'severity': 5,
                'max_streak': max_streak,
                'probability': probability_pct,
                'message': f'极端长连 ({max_streak}连, 出现概率≈{probability_pct:.4f}%)',
                'level': 'critical'
            }
        elif max_streak >= thresh_warn:
            return {
                'is_anomaly': True,
                'severity': 3,
                'max_streak': max_streak,
                'probability': probability_pct,
                'message': f'异常长连 ({max_streak}连, 出现概率≈{probability_pct:.3f}%)',
                'level': 'warning'
            }
        else:
            return {
                'is_anomaly': False,
                'severity': 0,
                'max_streak': max_streak,
                'probability': probability_pct,
                'level': 'normal'
            }
    
    def _tie_anomaly_test(self, shoe_data, warn_override=None, critical_override=None):
        """
        和局比例异常检测（理论和局率：9.52%）

        Bug#14修复（新增参数）：
          warn_override / critical_override 允许调用方传入宽松阈值。
        """
        n = len(shoe_data)
        ties = shoe_data.count('T')
        tie_rate = ties / n * 100

        expected_ties = n * self.THEORETICAL_PROBS['tie']
        std_dev = np.sqrt(n * self.THEORETICAL_PROBS['tie'] * (1 - self.THEORETICAL_PROBS['tie']))
        z_score = abs(ties - expected_ties) / std_dev if std_dev > 0 else 0

        # 使用 override 阈值（若有）
        thresh_warn = warn_override if warn_override is not None else self.thresholds['tie_rate_warning']
        thresh_critical = critical_override if critical_override is not None else self.thresholds['tie_rate_critical']

        if tie_rate > thresh_critical:
            return {
                'is_anomaly': True,
                'severity': 3,
                'tie_rate': tie_rate,
                'z_score': z_score,
                'message': f'和局严重过多 ({tie_rate:.1f}%, 理论9.5%)',
                'level': 'critical'
            }
        elif tie_rate > thresh_warn or z_score > self.thresholds['z_score_warning']:
            return {
                'is_anomaly': True,
                'severity': 2,
                'tie_rate': tie_rate,
                'z_score': z_score,
                'message': f'和局偏多 ({tie_rate:.1f}%, 理论9.5%)',
                'level': 'warning'
            }
        elif tie_rate < 3:
            return {
                'is_anomaly': True,
                'severity': 1,
                'tie_rate': tie_rate,
                'z_score': z_score,
                'message': f'和局偏少 ({tie_rate:.1f}%, 理论9.5%)',
                'level': 'warning'
            }
        else:
            return {
                'is_anomaly': False,
                'severity': 0,
                'tie_rate': tie_rate,
                'level': 'normal'
            }
    
    def _chi_square_test(self, shoe_data):
        """
        卡方检验
        
        检验观察分布是否符合理论分布
        """
        n = len(shoe_data)
        
        # 观察频数
        observed = [
            shoe_data.count('B'),
            shoe_data.count('P'),
            shoe_data.count('T')
        ]
        
        # 理论期望频数
        expected = [
            n * self.THEORETICAL_PROBS['banker'],
            n * self.THEORETICAL_PROBS['player'],
            n * self.THEORETICAL_PROBS['tie']
        ]
        
        # 卡方值 χ² = Σ [(观察-期望)² / 期望]
        chi_square = sum(
            (obs - exp) ** 2 / exp 
            for obs, exp in zip(observed, expected)
            if exp > 0
        )
        
        # 自由度 = 类别数 - 1 = 2
        # 临界值（α=0.01, df=2）= 9.21
        # 临界值（α=0.001, df=2）= 13.82
        
        if chi_square > 13.82:
            return {
                'is_anomaly': True,
                'severity': 3,
                'chi_square': chi_square,
                'message': f'整体分布严重异常 (χ²={chi_square:.2f}, p<0.001)',
                'level': 'critical'
            }
        elif chi_square > 9.21:
            return {
                'is_anomaly': True,
                'severity': 2,
                'chi_square': chi_square,
                'message': f'整体分布异常 (χ²={chi_square:.2f}, p<0.01)',
                'level': 'warning'
            }
        else:
            return {
                'is_anomaly': False,
                'severity': 0,
                'chi_square': chi_square,
                'level': 'normal'
            }
    
    def _get_severity_level(self, score):
        """根据综合评分确定严重程度"""
        if score >= 10:
            return 'critical'      # 严重异常
        elif score >= 6:
            return 'high'          # 高度异常
        elif score >= 3:
            return 'moderate'      # 中度异常
        elif score > 0:
            return 'low'           # 轻微异常
        else:
            return 'normal'        # 正常
    
    def _get_recommendation(self, score):
        """根据异常程度给出建议"""
        if score >= 10:
            return "⛔ 强烈建议停止预测！靴牌极度异常，可能存在问题"
        elif score >= 6:
            return "🚫 建议谨慎！靴牌高度异常，预测可靠性极低"
        elif score >= 3:
            return "⚠️ 注意风险！靴牌存在明显异常，建议降低投注"
        elif score > 0:
            return "💡 轻微异常，适当谨慎"
        else:
            return "✅ 靴牌正常"
    
    def _get_confidence_adjustment(self, score):
        """
        根据异常程度调整置信度
        
        返回调整系数（0-1）
        """
        if score >= 10:
            return 0.2   # 严重异常：置信度降至20%
        elif score >= 6:
            return 0.4   # 高度异常：置信度降至40%
        elif score >= 3:
            return 0.6   # 中度异常：置信度降至60%
        elif score > 0:
            return 0.8   # 轻微异常：置信度降至80%
        else:
            return 1.0   # 正常：不调整
    
    def get_summary_text(self, detection_result):
        """
        获取异常检测的摘要文本（用于显示）
        
        Args:
            detection_result: detect()返回的结果
            
        Returns:
            str: 格式化的摘要文本
        """
        if not detection_result['is_anomaly']:
            return "✅ 靴牌数据正常"
        
        level_emoji = {
            'low': '💡',
            'moderate': '⚠️',
            'high': '🚫',
            'critical': '⛔'
        }
        
        emoji = level_emoji.get(detection_result['severity_level'], '⚠️')
        
        summary = f"{emoji} 检测到异常 (严重度: {detection_result['severity_score']})\n"
        
        for anomaly in detection_result['anomalies']:
            summary += f"  • {anomaly}\n"
        
        summary += f"\n{detection_result['recommendation']}"
        
        return summary

