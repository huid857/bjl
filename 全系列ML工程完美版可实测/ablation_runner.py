"""
消融测试运行器

逐一禁用/启用各子模型，跑相同的 Walk-Forward 回测，
最终输出对比表，找出谁在拖后腿。

用法：
    python ablation_runner.py
"""

import json
import os
import time
from datetime import datetime
from backtester import WalkForwardBacktester


# ──────────────────────────────────────────────────────────
# 消融配置：每组给一个标签 + 启用的模型集合
# ──────────────────────────────────────────────────────────

ALL_NON_ML = {'Historical', 'Trend', 'Streak', 'Frequency', 'SimilarShoe'}

ABLATION_CONFIGS = [
    # ---- 单模型 ----
    ("仅 Historical (Markov+Ngram)", {'Historical'}),
    ("仅 Streak",                    {'Streak'}),
    ("仅 SimilarShoe",               {'SimilarShoe'}),

    # ---- 去掉一个 ----
    ("全量 - Frequency - Trend",     ALL_NON_ML - {'Frequency', 'Trend'}),
    ("全量 - SimilarShoe",           ALL_NON_ML - {'SimilarShoe'}),
    ("全量 - Streak",                ALL_NON_ML - {'Streak'}),
    ("全量 - Historical",            ALL_NON_ML - {'Historical'}),

    # ---- 对照组 ----
    ("全量非ML",                     ALL_NON_ML),
]


def run_ablation():
    bt = WalkForwardBacktester()
    if not bt.shoes:
        print("[消融] 无数据")
        return

    results = []
    total_start = time.time()

    print("\n" + "=" * 65)
    print("  消融测试（共 {} 组配置）".format(len(ABLATION_CONFIGS)))
    print("=" * 65)

    for label, models in ABLATION_CONFIGS:
        print(f"\n▶ {label}")
        summary = bt.run(
            warmup_shoes=5,
            min_rounds_before_predict=10,
            verbose=False,
            enabled_models=models
        )
        if summary:
            results.append({
                'label': label,
                'models': sorted(models),
                'accuracy': summary.get('overall_accuracy', 0),
                'valid': summary.get('valid_predictions', 0),
                'correct': summary.get('total_correct', 0),
                'wrong': summary.get('total_wrong', 0),
                'neutral': summary.get('total_neutral', 0),
                'model_accs': summary.get('model_accuracies', {}),
                'shoes_above_50': summary.get('shoes_above_50pct', 0),
                'shoes_total': summary.get('total_shoes_evaluated', 0),
                'max_loss_streak': summary.get('max_loss_streak', 0),
            })
        else:
            results.append({
                'label': label, 'models': sorted(models),
                'accuracy': 0, 'valid': 0
            })

    total_elapsed = time.time() - total_start

    # ──────────────────────────────────────────────────────
    # 打印对比表
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  消融测试结果对比")
    print("=" * 65)

    # 按胜率降序排列
    results.sort(key=lambda r: -r['accuracy'])

    banker_baseline = 0.5141  # 来自 Phase 1 回测

    for r in results:
        acc = r['accuracy'] * 100
        valid = r.get('valid', 0)
        delta = (r['accuracy'] - banker_baseline) * 100
        sign = "+" if delta >= 0 else ""
        bar_len = int(acc * 0.4)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        beat = "✅" if r['accuracy'] > banker_baseline else "❌"
        print(f"  {r['label']:<28} {bar} {acc:5.2f}%  {sign}{delta:+.2f}pp  {beat}  ({valid}局)")

    print(f"\n  庄基线:                          {'░' * 40} {banker_baseline*100:5.2f}%  (参考)")
    print(f"\n  总耗时: {total_elapsed:.0f}s")

    # ──────────────────────────────────────────────────────
    # 保存结果
    # ──────────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(base_dir, 'backtest_results')
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(report_dir, f'ablation_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {path}")
    print("=" * 65)


if __name__ == '__main__':
    run_ablation()
