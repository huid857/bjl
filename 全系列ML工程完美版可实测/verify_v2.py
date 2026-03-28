"""
快速语法 + 逻辑验证脚本
验证所有修改文件的语法正确性和 ShoeRegimeDetector 的逻辑
"""
import ast, sys, os

FILES = [
    'shoe_regime_detector.py',
    'advanced_shoe_analyzer.py',
    'analyzer.py',
    'historical_ensemble.py',
    'anomaly_detector.py',
    'ensemble.py',
]

print("=== 语法检查 ===")
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

print(f"\n语法: {ok_count}/{len(FILES)} 通过\n")

# ──────────────────────────────────────────────
# 逻辑单元测试：ShoeRegimeDetector
# ──────────────────────────────────────────────
print("=== ShoeRegimeDetector 逻辑测试 ===")
try:
    from shoe_regime_detector import ShoeRegimeDetector
    d = ShoeRegimeDetector()

    # 测试 1: 纯单跳 BPBPBP...
    single_alt = ['B','P','B','P','B','P','B','P','B','P','B','P','B','P','B','P','B','P']
    r = d.analyze(single_alt)
    assert r['dominant_regime'] == 'single_alt', f"期望 single_alt, 得到 {r['dominant_regime']}"
    print(f"  ✓ 单跳: {r['dominant_regime']} 强度={r['regime_strength']:.0f}%")

    # 测试 2: 纯长龙 BBBBBBBBB...
    long_streak = ['B']*15 + ['P']*3
    r = d.analyze(long_streak)
    assert r['dominant_regime'] == 'long_streak', f"期望 long_streak, 得到 {r['dominant_regime']}"
    print(f"  ✓ 长龙: {r['dominant_regime']} 强度={r['regime_strength']:.0f}%")

    # 测试 3: 双跳 BBPPBBPPBBPP
    double_alt = ['B','B','P','P','B','B','P','P','B','B','P','P','B','B','P','P']
    r = d.analyze(double_alt)
    print(f"  ℹ 双跳测试: {r['dominant_regime']} 强度={r['regime_strength']:.0f}% (次优=双跳专用)")

    # 测试 4: 混沌
    chaos = ['B','B','P','B','P','P','B','P','B','B','P','B','P','B','P','P','B']
    r = d.analyze(chaos)
    print(f"  ℹ 混沌: {r['dominant_regime']} 强度={r['regime_strength']:.0f}%")

    # 测试 5: 中性（数据不足）
    r = d.analyze(['B','P','T'])
    assert not r['can_analyze'], "数据 < 8 应该 can_analyze=False"
    print(f"  ✓ 数据不足: can_analyze={r['can_analyze']}")

    # 测试 6: 规律切换检测
    switch_seq = ['B','B','B','B','B','B','B','B'] + ['P','B','P','B','P','B']  # 前段长龙→后段单跳
    r = d.analyze(switch_seq)
    print(f"  ℹ 切换测试: {r['dominant_regime']} trend={r['regime_trend']} switch={r['switch_event']}")

    print("\nShoeRegimeDetector 逻辑测试完成!")

except ImportError as e:
    print(f"  ! 无法导入: {e}")
except AssertionError as e:
    print(f"  ✗ 逻辑错误: {e}")
except Exception as e:
    print(f"  ✗ 异常: {e}")

# ──────────────────────────────────────────────
# 逻辑单元测试：advanced_shoe_analyzer Bug#12
# ──────────────────────────────────────────────
print("\n=== AdvancedShoeAnalyzer Bug#12 测试 ===")
try:
    from advanced_shoe_analyzer import AdvancedShoeAnalyzer
    aa = AdvancedShoeAnalyzer([])

    # 单跳序列：单跳数应 > 0，双跳对数应 = 0
    single_seq = ['B','P','B','P','B','P','B','P','B','P','B','P']
    s = aa._count_single_alternations(single_seq)
    dp = aa._count_double_alternation_pairs(single_seq)
    print(f"  单跳率测试 - 单跳局={s}, 双跳对={dp}")
    assert s > 0, "单跳序列中单跳局数应 > 0"
    assert dp == 0, f"纯单跳序列中双跳对应=0，得到 {dp}"
    print("  ✓ 单跳 vs 双跳正确区分")

    # 双跳序列：双跳对数应 > 0
    double_seq = ['B','B','P','P','B','B','P','P','B','B','P','P']
    s2 = aa._count_single_alternations(double_seq)
    dp2 = aa._count_double_alternation_pairs(double_seq)
    print(f"  双跳率测试 - 单跳局={s2}, 双跳对={dp2}")
    assert dp2 > 0, f"双跳序列中双跳对应 > 0，得到 {dp2}"
    print("  ✓ 双跳对正确统计")

    # 鞋型分类
    t1 = aa.classify_shoe_type(double_seq + double_seq)
    print(f"  分类结果: {t1['primary_type']} - {t1['characteristics']}")

    print("\nAdvancedShoeAnalyzer Bug#12 测试完成!")

except ImportError as e:
    print(f"  ! 无法导入: {e}")
except AssertionError as e:
    print(f"  ✗ Bug#12 未修复: {e}")
except Exception as e:
    print(f"  ✗ 异常: {e}")

print("\n=== 所有验证完成 ===")
