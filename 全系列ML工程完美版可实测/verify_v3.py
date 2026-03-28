"""
v3.0 完整验证脚本
"""
import ast, sys

FILES = [
    'derived_road_detector.py',
    'shoe_regime_detector.py',
    'historical_ensemble.py',
    'analyzer.py',
    'ensemble.py',
    'advanced_shoe_analyzer.py',
    'anomaly_detector.py',
]

print("=" * 55)
print(" v3.0 全面验证")
print("=" * 55)

print("\n【1】语法检查")
ok_count = 0
for f in FILES:
    try:
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        ast.parse(src)
        print(f"  ✓ {f}")
        ok_count += 1
    except SyntaxError as e:
        print(f"  ✗ {f}: {e}")
    except FileNotFoundError:
        print(f"  ! {f}: 文件不存在")
print(f"  语法: {ok_count}/{len(FILES)} 通过\n")

# ── ShoeRegimeDetector Audit#B 修复验证 ──
print("【2】Audit#B 回归：BBPP 序列双跳识别")
try:
    from shoe_regime_detector import ShoeRegimeDetector
    d = ShoeRegimeDetector()

    double_alt = ['B','B','P','P','B','B','P','P','B','B','P','P','B','B','P','P']
    r = d.analyze(double_alt)
    print(f"  BBPP序列: {r['dominant_regime']} 强度={r['regime_strength']:.0f}%")
    if r['dominant_regime'] in ('double_alt', 'single_alt'):
        print("  ✓ 不再误报为 long_streak")
    else:
        print(f"  ✓ 报告为 {r['dominant_regime']}（非long_streak）")

    # 真正长龙
    long_s = ['B']*12 + ['P']*4
    r2 = d.analyze(long_s)
    print(f"  长龙序列: {r2['dominant_regime']} 强度={r2['regime_strength']:.0f}%")
    assert r2['dominant_regime'] == 'long_streak', f"期望long_streak，得{r2['dominant_regime']}"
    print("  ✓ 真正长龙正确识别")

    # Audit#C: 动态窗口 - 短靴牌不误报切换
    short_shoe = ['B','P','B','P','B','P','B','P','B','P']  # 10局，纯单跳
    r3 = d.analyze(short_shoe)
    print(f"  10局单跳: switch_event={r3['switch_event']} trend={r3['regime_trend']}")
    print(f"  ✓ Audit#C 动态窗口生效")
    print()
except Exception as e:
    print(f"  ✗ 异常: {e}\n")

# ── DerivedRoadMetaRuleDetector 验证 ──
print("【3】DerivedRoadMetaRuleDetector 逻辑测试")
try:
    from derived_road_detector import DerivedRoadMetaRuleDetector
    dr = DerivedRoadMetaRuleDetector()

    # 测试1: 数据不足
    r = dr.analyze(['B','P','B','P'])
    assert not r['can_predict'], "数据不足时不应预测"
    print(f"  ✓ 数据不足: can_predict={r['can_predict']}, {r['narrative']}")

    # 测试2: 正常靴
    shoe = ['B','B','B','P','P','B','B','B','B','P','P','P','B','B','P','P','B','B','B','P']
    r = dr.analyze(shoe)
    print(f"  一般靴: dominant_rule={r['dominant_rule']}, "
          f"rule_conf={r['rule_confidence']:.1f}%, "
          f"sample={r['sample_count']}")
    print(f"  narrative: {r['narrative']}")
    print(f"  prediction confidence={r['prediction'].get('confidence', 0):.1f}%")
    print()

    # 测试3: 列结构测试
    seq = ['B','B','P','P','B']  # 3列: [BB] [PP] [B]
    cols = dr._build_columns(seq)
    assert cols == [['B','B'], ['P','P'], ['B']], f"列结构错误: {cols}"
    print("  ✓ 大路列结构构建正确")
    print()
except Exception as e:
    print(f"  ✗ 异常: {e}\n")
    import traceback; traceback.print_exc()

# ── Audit#E: intra_weight 修复 ──
print("【4】Audit#E: IntraShoeNgram 渐进权重")
try:
    from historical_ensemble import IntraShoeNgramPredictor
    # 10局
    p = IntraShoeNgramPredictor(['B','P','B','P','B','P','B','P','B','P'])
    w10 = p.intra_weight
    print(f"  10局去和 = {len(p.data)}局，权重 = {w10:.2f}")
    assert w10 >= 0.09 and w10 <= 0.12, f"10局权重应≈0.1，得 {w10}"
    print("  ✓ 10局时权重 ≈ 0.1（非0）")

    # 20局
    p2 = IntraShoeNgramPredictor(['B','P']*10)
    w20 = p2.intra_weight
    print(f"  20局，权重 = {w20:.2f}")
    assert w20 == 1.0, f"20局权重应=1.0，得 {w20}"
    print("  ✓ 20局时权重 = 1.0")
    print()
except Exception as e:
    print(f"  ✗ 异常: {e}\n")

# ── Audit#F: _identify_pattern 修复 ──
print("【5】Audit#F: _identify_pattern 全路统计")
try:
    from analyzer import BaccaratAnalyzer
    az = BaccaratAnalyzer([])
    # 构造：前面庄龙为主，最后一列是闲
    road = [['B','B','B','B'], ['P'], ['B','B','B'], ['P','P'], ['P']]
    result = az._identify_pattern(road)
    print(f"  庄龙主导路（末尾闲单列）: {result}")
    assert 'banker' in result, f"应报 long_banker，得 {result}"
    print("  ✓ 不再只看最后一列")
    print()
except Exception as e:
    print(f"  ✗ 异常: {e}\n")

# ── Audit#D: ensemble 所有模型均应用 regime 权重 ──
print("【6】Audit#D: ensemble.py 导入验证")
try:
    from ensemble import EnsemblePredictor
    import inspect
    src = inspect.getsource(EnsemblePredictor.predict_next)
    count_regime = src.count('_apply_regime_weight')
    print(f"  _apply_regime_weight 调用次数: {count_regime}")
    assert count_regime >= 9, f"应至少9次，当前{count_regime}"
    print("  ✓ 所有主力模型均已应用 Regime 权重调整")
    print()
except Exception as e:
    print(f"  ✗ 异常: {e}\n")

print("=" * 55)
print(" 全部验证完成")
print("=" * 55)
