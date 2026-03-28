"""
Walk-Forward 回测框架

核心原则：
- 严格数据隔离：预测第 N 局时，只能看到该靴牌第 N-1 局及之前的数据
  + 该靴牌之前所有已完成的历史靴牌
- 绝不使用"未来靴牌"数据（防止前视偏差 look-ahead bias）
- 直接调用现有 EnsemblePredictor，不修改任何预测逻辑
- 和局处理：预测B/P但实际出T → 中性（不计入胜率，与 PredictionHistory 一致）
- 输出包含朴素基线对比（永远猜庄 / 永远猜闲）

用法：
    python backtester.py              # 直接运行
    或通过 main.py 菜单选项调用
"""

import json
import os
import time
from datetime import datetime
from collections import defaultdict


class WalkForwardBacktester:
    """
    Walk-Forward 逐局回测器

    每次预测时，EnsemblePredictor 接收到的数据严格限制为：
        all_prior_shoes   = 当前靴牌之前已完成的所有历史靴牌
        current_shoe_data = 当前靴牌到第 N-1 局为止的数据
    从不让模型看到任何"未来"信息。
    """

    def __init__(self, history_path=None):
        """
        Args:
            history_path: history.json 路径，默认为 data/history.json
        """
        if history_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            history_path = os.path.join(base_dir, 'data', 'history.json')
        self.history_path = history_path
        self.shoes = []
        self._load_data()

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _load_data(self):
        """从 history.json 加载所有靴牌"""
        if not os.path.exists(self.history_path):
            print(f"[回测] 未找到历史数据文件: {self.history_path}")
            return
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.shoes = data.get('shoes', [])
            # 保证每靴牌的 data 字段是列表格式
            for shoe in self.shoes:
                if isinstance(shoe.get('data'), str):
                    shoe['data'] = list(shoe['data'])
            print(f"[回测] 加载 {len(self.shoes)} 个历史靴牌")
        except Exception as e:
            print(f"[回测] 加载数据失败: {e}")

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, warmup_shoes=5, min_rounds_before_predict=10, verbose=True, enabled_models=None):
        """
        运行完整回测

        Args:
            warmup_shoes: 前 N 个靴牌仅用于热身
            min_rounds_before_predict: 每靴牌至少看到 N 局后才开始预测
            verbose: 是否打印进度
            enabled_models: 允许的模型名称集合（None=全部启用）

        Returns:
            results dict
        """
        if len(self.shoes) < warmup_shoes + 1:
            print(f"[回测] 数据不足，至少需要 {warmup_shoes + 1} 个靴牌")
            return {}

        config_desc = "全部模型" if enabled_models is None else ", ".join(sorted(enabled_models))
        print(f"\n{'='*60}")
        print(f"  Walk-Forward 回测")
        print(f"  数据：{len(self.shoes)} 靴牌，热身：前 {warmup_shoes} 靴")
        print(f"  模型配置：{config_desc}")
        print(f"{'='*60}\n")

        all_results = []
        total_shoes_evaluated = 0
        start_time = time.time()
        self._err_shown = False  # 重置错误显示标记

        for shoe_idx in range(warmup_shoes, len(self.shoes)):
            prior_shoes = self.shoes[:shoe_idx]
            current_shoe_full = self.shoes[shoe_idx]

            shoe_result = self._simulate_shoe(
                shoe_idx,
                current_shoe_full,
                prior_shoes,
                min_rounds_before_predict,
                verbose,
                enabled_models
            )

            if shoe_result['predictions_made'] > 0:
                all_results.append(shoe_result)
                total_shoes_evaluated += 1

        elapsed = time.time() - start_time

        summary = self._aggregate_results(all_results, elapsed)
        summary['config'] = config_desc
        if not verbose:
            # 消融批量模式只打印单行摘要
            acc = summary.get('overall_accuracy', 0) * 100
            preds = summary.get('valid_predictions', 0)
            print(f"  [{config_desc:<35}] 胜率 {acc:5.2f}%  ({preds} 局)")
        else:
            self.print_report(summary)
        self._save_report(summary)

        return summary

    # ------------------------------------------------------------------
    # 单靴牌模拟
    # ------------------------------------------------------------------

    def _simulate_shoe(self, shoe_idx, shoe, prior_shoes, min_rounds, verbose, enabled_models=None):
        """
        模拟一个完整靴牌的逐局预测

        严格规则：
            - 预测第 i 局时，只能看到 shoe['data'][:i]（前 i 局）
            - 历史只有 prior_shoes（当前靴牌之前的靴牌）
        """
        shoe_data = shoe['data']  # 完整靴牌（仅用于取出"实际结果"）
        shoe_name = shoe.get('name', f'shoe_{shoe_idx}')

        result = {
            'shoe_name': shoe_name,
            'shoe_idx': shoe_idx,
            'total_rounds': len(shoe_data),
            'predictions_made': 0,
            'correct': 0,
            'wrong': 0,
            'wrong': 0,
            'neutral': 0,          # 预测B/P, 实际T
            'model_stats': defaultdict(lambda: {'correct': 0, 'wrong': 0, 'neutral': 0}),
            'accuracy': 0.0,
            'predictions_log': []  # 记录每次预测的具体信息
        }

        # 构建"至今可见的历史"作为 all_history 列表
        # 这是传给 EnsemblePredictor 的扁平化历史序列
        prior_all_history = []
        prior_shoes_list = []
        for s in prior_shoes:
            prior_all_history.extend(s['data'])
            prior_shoes_list.append({
                'name': s.get('name', ''),
                'date': s.get('date', ''),
                'count': s.get('count', len(s['data'])),
                'data': ''.join(s['data'])
            })

        for i in range(min_rounds, len(shoe_data) - 1):
            # 当前可见的本靴数据（第 0 ~ i-1 局，共 i 局）
            visible_shoe = shoe_data[:i]
            actual_result = shoe_data[i]   # 这是要预测的结果

            try:
                prediction_result, model_preds = self._call_predictor(
                    prior_all_history,
                    visible_shoe,
                    prior_shoes_list,
                    enabled_models
                )
            except BaseException as e:
                # 不中断回测，展示错误然后跳过（BaseException 覆盖 C 扩展错误）
                import traceback
                if not getattr(self, '_err_shown', False):
                    print(f"\n  [DEBUG] 预测异常: {e}")
                    traceback.print_exc()
                    self._err_shown = True
                continue

            if prediction_result is None:
                continue

            predicted = prediction_result.get('recommendation', '')
            # 兼容中文推荐字段
            if predicted in ['庄', '庄家']:
                predicted = 'B'
            elif predicted in ['闲', '闲家']:
                predicted = 'P'
            elif predicted in ['和', '和局']:
                predicted = 'T'

            if predicted not in ('B', 'P', 'T'):
                continue

            result['predictions_made'] += 1

            # 评估结果
            is_correct, is_neutral = self._evaluate(predicted, actual_result)

            if is_neutral:
                result['neutral'] += 1
            elif is_correct:
                result['correct'] += 1
            else:
                result['wrong'] += 1
                
            result['predictions_log'].append({
                'confidence': prediction_result.get('confidence', 0),
                'correct': is_correct,
                'neutral': is_neutral
            })

            # 各子模型统计（来自 predictor.last_model_predictions）
            for model_name, model_pred_val in model_preds.items():
                mc, mn = self._evaluate(model_pred_val, actual_result)
                if mn:
                    result['model_stats'][model_name]['neutral'] += 1
                elif mc:
                    result['model_stats'][model_name]['correct'] += 1
                else:
                    result['model_stats'][model_name]['wrong'] += 1

        # 计算本靴胜率（排除中性）
        valid = result['correct'] + result['wrong']
        result['accuracy'] = result['correct'] / valid if valid > 0 else 0.0

        if verbose:
            print(f"  [{shoe_idx+1:3d}] {shoe_name}  "
                  f"预测{result['predictions_made']:3d}局  "
                  f"胜率 {result['accuracy']*100:5.1f}%  "
                  f"(✓{result['correct']} ✗{result['wrong']} ≈{result['neutral']})")

        return result

    # ------------------------------------------------------------------
    # 调用预测器（严格数据隔离入口）
    # ------------------------------------------------------------------

    def _call_predictor(self, all_history, current_shoe_data, shoes_list, enabled_models=None):
        """
        实例化 EnsemblePredictor，传入严格裁剪的数据，返回预测结果。

        Args:
            all_history:       仅包含当前预测时点之前已完成靴牌的历史数据（列表）
            current_shoe_data: 当前靴牌到第 N-1 局为止的数据（列表）
            shoes_list:        prior shoes 的结构化列表（供 SimilarShoe/DataDriven）

        Returns:
            (prediction_dict, model_predictions_dict)
        """
        from ensemble import EnsemblePredictor

        # 注意：第一个参数名是 all_data，不是 all_history
        predictor = EnsemblePredictor(
            all_history,
            current_shoe_data,
            shoes=shoes_list,
            enabled_models=enabled_models
        )
        result = predictor.predict_next()
        # 各子模型推荐存在 predictor.last_model_predictions，格式: {name: 'B'/'P'/'T'}
        model_preds = dict(predictor.last_model_predictions)
        return result, model_preds

    # ------------------------------------------------------------------
    # 评估函数（与 PredictionHistory 规则一致）
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate(predicted, actual):
        """
        Returns:
            (is_correct: bool, is_neutral: bool)
        规则：
            预测B/P，实际T → 中性（不算对错）
            其他不匹配 → 错误
        """
        if predicted == actual:
            return True, False
        elif predicted in ('B', 'P') and actual == 'T':
            return False, True
        else:
            return False, False

    # ------------------------------------------------------------------
    # 朴素基线计算
    # ------------------------------------------------------------------

    def _compute_baselines(self, shoes, warmup_shoes):
        """
        计算朴素基线：永远猜庄 / 永远猜闲 / 永远猜和
        从 warmup_shoes 之后开始统计，与回测范围一致。

        Returns:
            dict of baseline name → accuracy
        """
        results_b = {'correct': 0, 'wrong': 0}
        results_p = {'correct': 0, 'wrong': 0}

        for shoe in shoes[warmup_shoes:]:
            for result in shoe['data']:
                if result == 'T':
                    # 基线猜庄/闲时，出和局算"中性"——同样排除
                    continue
                if result == 'B':
                    results_b['correct'] += 1
                    results_p['wrong'] += 1
                else:  # P
                    results_b['wrong'] += 1
                    results_p['correct'] += 1

        total = results_b['correct'] + results_b['wrong']
        return {
            'always_banker': results_b['correct'] / total if total > 0 else 0,
            'always_player': results_p['correct'] / total if total > 0 else 0,
            'total_non_tie': total
        }

    # ------------------------------------------------------------------
    # 结果汇总
    # ------------------------------------------------------------------

    def _aggregate_results(self, all_results, elapsed):
        """汇总所有靴牌的回测结果"""
        if not all_results:
            return {}

        total_predictions = sum(r['predictions_made'] for r in all_results)
        total_correct = sum(r['correct'] for r in all_results)
        total_wrong = sum(r['wrong'] for r in all_results)
        total_neutral = sum(r['neutral'] for r in all_results)
        valid = total_correct + total_wrong
        overall_accuracy = total_correct / valid if valid > 0 else 0

        # 各模型汇总
        model_totals = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'neutral': 0})
        for r in all_results:
            for model_name, stats in r['model_stats'].items():
                model_totals[model_name]['correct'] += stats['correct']
                model_totals[model_name]['wrong'] += stats['wrong']
                model_totals[model_name]['neutral'] += stats['neutral']

        model_accuracies = {}
        for model_name, stats in model_totals.items():
            v = stats['correct'] + stats['wrong']
            model_accuracies[model_name] = stats['correct'] / v if v > 0 else 0

        # 靴牌胜率分布
        shoe_accuracies = [r['accuracy'] for r in all_results]
        shoes_above_50 = sum(1 for a in shoe_accuracies if a > 0.5)
        shoes_below_50 = sum(1 for a in shoe_accuracies if a < 0.5)

        # 连败分析
        # 连败分析
        all_outcomes = []
        for r in all_results:
            all_outcomes.extend(
                [1] * r['correct'] + [0] * r['wrong']
            )
        max_loss_streak = self._max_streak(all_outcomes, 0)
        avg_loss_streak = self._avg_streak(all_outcomes, 0)

        # 置信度分层统计
        conf_buckets = {
            '<50%': {'correct': 0, 'wrong': 0, 'valid': 0},
            '50-60%': {'correct': 0, 'wrong': 0, 'valid': 0},
            '60-70%': {'correct': 0, 'wrong': 0, 'valid': 0},
            '70-80%': {'correct': 0, 'wrong': 0, 'valid': 0},
            '80-90%': {'correct': 0, 'wrong': 0, 'valid': 0},
            '>90%': {'correct': 0, 'wrong': 0, 'valid': 0}
        }
        
        for r in all_results:
            for p in r.get('predictions_log', []):
                if p['neutral']:
                    continue
                c = p['confidence']
                if c < 50: b = '<50%'
                elif c < 60: b = '50-60%'
                elif c < 70: b = '60-70%'
                elif c < 80: b = '70-80%'
                elif c < 90: b = '80-90%'
                else: b = '>90%'
                
                conf_buckets[b]['valid'] += 1
                if p['correct']:
                    conf_buckets[b]['correct'] += 1
                else:
                    conf_buckets[b]['wrong'] += 1
                    
        confidence_stats = {}
        for b, s in conf_buckets.items():
            if s['valid'] > 0:
                acc = s['correct'] / s['valid']
                confidence_stats[b] = {'accuracy': acc, 'valid': s['valid'], 'correct': s['correct']}

        # 朴素基线（用原始靴牌数据，而非预测结果）
        warmup = len(self.shoes) - len(all_results)
        baselines = self._compute_baselines(self.shoes, warmup)

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': round(elapsed, 1),
            'total_shoes_evaluated': len(all_results),
            'total_predictions': total_predictions,
            'total_correct': total_correct,
            'total_wrong': total_wrong,
            'total_neutral': total_neutral,
            'valid_predictions': valid,
            'overall_accuracy': overall_accuracy,
            'model_accuracies': dict(sorted(model_accuracies.items(),
                                            key=lambda x: -x[1])),
            'shoe_accuracies': shoe_accuracies,
            'shoes_above_50pct': shoes_above_50,
            'shoes_below_50pct': shoes_below_50,
            'shoes_below_50pct': shoes_below_50,
            'max_loss_streak': max_loss_streak,
            'avg_loss_streak': round(avg_loss_streak, 1),
            'confidence_stats': confidence_stats,
            'baselines': baselines,
        }

    @staticmethod
    def _max_streak(outcomes, target_value):
        """计算最大连续 target_value 的长度"""
        max_s = current_s = 0
        for v in outcomes:
            if v == target_value:
                current_s += 1
                max_s = max(max_s, current_s)
            else:
                current_s = 0
        return max_s

    @staticmethod
    def _avg_streak(outcomes, target_value):
        """计算 target_value 的平均连续长度"""
        streaks = []
        current_s = 0
        for v in outcomes:
            if v == target_value:
                current_s += 1
            else:
                if current_s > 0:
                    streaks.append(current_s)
                current_s = 0
        if current_s > 0:
            streaks.append(current_s)
        return sum(streaks) / len(streaks) if streaks else 0

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------

    def print_report(self, summary):
        """打印格式化回测报告"""
        if not summary:
            print("[回测] 无结果可报告")
            return

        sep = '=' * 60
        print(f"\n{sep}")
        print("  回测结果报告")
        print(f"  生成时间: {summary['timestamp']}")
        print(f"  耗时: {summary['elapsed_seconds']}s")
        print(sep)

        acc = summary['overall_accuracy'] * 100
        b_base = summary['baselines']['always_banker'] * 100
        p_base = summary['baselines']['always_player'] * 100

        print(f"\n📊 总体胜率")
        print(f"  系统预测胜率:   {acc:.2f}%")
        print(f"  ─────────────────────────────")
        print(f"  朴素基线(永远猜庄): {b_base:.2f}%   {'✅ 超过' if acc > b_base else '❌ 未超过'}")
        print(f"  朴素基线(永远猜闲): {p_base:.2f}%   {'✅ 超过' if acc > p_base else '❌ 未超过'}")
        print(f"  有效预测局数:   {summary['valid_predictions']}")
        print(f"  中性(出和局):   {summary['total_neutral']}")

        print(f"\n📈 各子模型胜率（从高到低）")
        for model_name, acc_m in summary['model_accuracies'].items():
            bar_len = int(acc_m * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            print(f"  {model_name:<18} {bar}  {acc_m*100:.1f}%")

        print(f"\n🎯 置信度分层表现（寻找高胜率区间）")
        conf_stats = summary.get('confidence_stats', {})
        for b in ['<50%', '50-60%', '60-70%', '70-80%', '80-90%', '>90%']:
            if b in conf_stats:
                s = conf_stats[b]
                c_acc = s['accuracy'] * 100
                c_valid = s['valid']
                bar_len = int(c_acc * 0.3)
                bar = '█' * bar_len + '░' * (30 - bar_len)
                beat = "✅ 超过庄基线" if c_acc > b_base else "❌"
                print(f"  {b:>6} {bar} {c_acc:5.2f}%  {beat}  (样本: {c_valid}局)")

        print(f"\n🎯 靴牌胜率分布")
        total_shoes = summary['total_shoes_evaluated']
        print(f"  胜率>50% 的靴牌: {summary['shoes_above_50pct']}/{total_shoes}")
        print(f"  胜率<50% 的靴牌: {summary['shoes_below_50pct']}/{total_shoes}")
        shoe_accs = summary['shoe_accuracies']
        if shoe_accs:
            print(f"  最高单靴胜率:   {max(shoe_accs)*100:.1f}%")
            print(f"  最低单靴胜率:   {min(shoe_accs)*100:.1f}%")
            print(f"  中位单靴胜率:   {sorted(shoe_accs)[len(shoe_accs)//2]*100:.1f}%")

        print(f"\n⚠️  风险指标")
        print(f"  最大连败局数:   {summary['max_loss_streak']}")
        print(f"  平均连败长度:   {summary['avg_loss_streak']}")

        print(f"\n{sep}\n")

    def _save_report(self, summary):
        """将回测结果保存为 JSON 文件"""
        if not summary:
            print("[回测] 无有效结果，跳过保存")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        report_dir = os.path.join(base_dir, 'backtest_results')
        os.makedirs(report_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(report_dir, f'backtest_{ts}.json')

        # shoe_accuracies 列表转成摘要和平均值保存
        save_data = {k: v for k, v in summary.items() if k != 'shoe_accuracies'}
        shoe_accs = summary.get('shoe_accuracies', [])
        save_data['shoe_accuracy_avg'] = (
            sum(shoe_accs) / len(shoe_accs) if shoe_accs else 0
        )

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"[回测] 结果已保存: {path}")


# ------------------------------------------------------------------
# 直接运行入口
# ------------------------------------------------------------------

if __name__ == '__main__':
    bt = WalkForwardBacktester()
    bt.run(warmup_shoes=5, min_rounds_before_predict=10, verbose=True)
