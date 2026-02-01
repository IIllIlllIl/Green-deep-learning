#!/usr/bin/env python3
"""
测试sign函数与CTF论文对齐

验证场景：
1. 正常ATE（>0或<0）- 应该完全对齐
2. ATE=0边缘情况 - 分析差异
"""

import sys
sys.path.insert(0, 'utils')

from tradeoff_detection import create_sign_func_from_rule

def test_ctf_alignment():
    """测试与CTF论文的对齐情况"""

    print("=" * 60)
    print("测试sign函数与CTF论文对齐")
    print("=" * 60)

    # CTF论文逻辑
    def ctf_logic(ate, rule):
        """CTF论文的improve判断逻辑"""
        direction = '+' if ate > 0 else '-'
        improve = (direction == rule)
        return direction, improve

    # 我们的sign函数逻辑
    def our_logic(ate, rule):
        """我们的sign函数"""
        sign_func = create_sign_func_from_rule(rule)
        sign = sign_func(0, ate)  # current_value=0
        improve = (sign == '+')
        return sign, improve

    # 测试场景
    test_cases = [
        # (ate, rule, description)
        (0.1, '+', '性能指标，正ATE'),
        (-0.1, '+', '性能指标，负ATE'),
        (0.0, '+', '性能指标，零ATE'),
        (0.1, '-', '能耗指标，正ATE'),
        (-0.1, '-', '能耗指标，负ATE'),
        (0.0, '-', '能耗指标，零ATE'),
    ]

    print("\n测试结果：")
    print("-" * 60)
    print(f"{'场景':<20} {'ATE':>6} {'规则':>4} {'CTF':<15} {'我们':<15} {'对齐':>4}")
    print("-" * 60)

    aligned_count = 0
    total_count = 0

    for ate, rule, desc in test_cases:
        total_count += 1

        # CTF逻辑
        ctf_direction, ctf_improve = ctf_logic(ate, rule)

        # 我们的逻辑
        our_sign, our_improve = our_logic(ate, rule)

        # 判断improve是否对齐
        is_aligned = (ctf_improve == our_improve)
        if is_aligned:
            aligned_count += 1

        aligned_mark = "✅" if is_aligned else "❌"

        print(f"{desc:<20} {ate:>6.2f} {rule:>4} "
              f"d={ctf_direction},i={ctf_improve:<5} "
              f"s={our_sign},i={our_improve:<5} "
              f"{aligned_mark:>4}")

    print("-" * 60)
    print(f"对齐率: {aligned_count}/{total_count} ({aligned_count/total_count*100:.1f}%)")
    print()

    # 详细分析ATE=0的情况
    print("=" * 60)
    print("ATE=0边缘情况详细分析")
    print("=" * 60)

    for rule in ['+', '-']:
        ate = 0.0
        ctf_direction, ctf_improve = ctf_logic(ate, rule)
        our_sign, our_improve = our_logic(ate, rule)

        print(f"\n规则'{rule}':")
        print(f"  CTF: direction={ctf_direction}, improve={ctf_improve}")
        print(f"  我们: sign={our_sign}, improve={our_improve}")

        if ctf_improve != our_improve:
            print(f"  ⚠️  差异分析:")
            print(f"    CTF认为: ATE=0时direction='{ctf_direction}'")
            print(f"    由于rule='{rule}', improve={ctf_improve}")
            print(f"    我们认为: ATE=0表示'无效应', sign='{our_sign}'")
            print(f"    实际影响: {'可能影响权衡检测' if ate == 0 else '无影响'}")

    # 统计分析
    print("\n" + "=" * 60)
    print("影响评估")
    print("=" * 60)
    print("1. 正常ATE数据（≠0）: ✅ 完全对齐CTF论文")
    print("2. ATE=0边缘情况: ⚠️  语义不同但影响可控")
    print("   - CTF: ATE=0时direction='-'，可能判定为改善")
    print("   - 我们: ATE=0时sign='-'（恶化），认为无效应")
    print("   - 缓解: require_significance=True会过滤ATE=0")
    print("   - 实际影响: 极低（统计显著性的ATE不会恰好为0）")
    print()

    if aligned_count == total_count:
        print("✅ 所有测试场景完全对齐CTF论文")
    else:
        print(f"⚠️  {total_count - aligned_count}个场景存在差异（ATE=0）")

    return aligned_count == total_count

if __name__ == "__main__":
    result = test_ctf_alignment()
    sys.exit(0 if result else 1)
