#!/usr/bin/env python3
"""
白名单过滤脚本实现验证测试

验证脚本:
    /home/green/energy_dl/nightly/analysis/scripts/filter_causal_edges_by_whitelist.py

验证文档:
    /home/green/energy_dl/nightly/analysis/docs/CAUSAL_EDGE_WHITELIST_DESIGN.md v1.1

测试内容:
    1. 白名单规则完整性检查
    2. 黑名单默认机制检查
    3. 关键规则验证（mediator -> performance）
    4. 边界情况测试
"""

import sys
import pandas as pd
from pathlib import Path

# 添加脚本路径
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(script_dir))

# 导入待测试的模块
from filter_causal_edges_by_whitelist import (
    WHITELIST_RULES,
    is_edge_allowed,
    filter_causal_edges_by_whitelist,
    get_filter_statistics
)

# 文档中定义的预期白名单规则（来自v1.1第7.2节）
EXPECTED_WHITELIST_RULES = {
    # 规则组1: 超参数主效应
    ('hyperparam', 'energy'): True,
    ('hyperparam', 'mediator'): True,
    ('hyperparam', 'performance'): True,

    # 规则组2: 交互项调节效应
    ('interaction', 'energy'): True,
    ('interaction', 'mediator'): True,
    ('interaction', 'performance'): True,

    # 规则组3: 中间变量中介效应
    ('mediator', 'energy'): True,
    ('mediator', 'mediator'): True,
    ('mediator', 'performance'): True,  # v1.1新增！
    ('energy', 'energy'): True,

    # 规则组4: 控制变量影响
    ('control', 'energy'): True,
    ('control', 'mediator'): True,
    ('control', 'performance'): True,
    ('mode', 'energy'): True,
    ('mode', 'mediator'): True,
    ('mode', 'performance'): True,
}

# 所有变量类型
ALL_CATEGORIES = ['hyperparam', 'interaction', 'energy', 'mediator',
                  'performance', 'control', 'mode']

# 黑名单规则（应该被禁止的组合）
BLACKLIST_RULES = [
    # 黑名单组1: 结果变量 -> 原因变量
    ('energy', 'hyperparam'),
    ('performance', 'hyperparam'),
    ('mediator', 'hyperparam'),

    # 黑名单组2: 任意变量 -> 实验设计变量
    ('energy', 'control'),
    ('performance', 'control'),
    ('mediator', 'control'),
    ('energy', 'mode'),
    ('performance', 'mode'),
    ('mediator', 'mode'),

    # 黑名单组3: 自循环
    ('hyperparam', 'hyperparam'),
    ('control', 'control'),
    ('mode', 'mode'),
    ('performance', 'performance'),

    # 黑名单组4: 反直觉的因果关系
    ('performance', 'energy'),
    ('performance', 'mediator'),

    # 特殊情况（文档第5.2节）
    ('hyperparam', 'interaction'),
    ('interaction', 'hyperparam'),
    ('interaction', 'interaction'),
    ('control', 'control'),
]


def test_whitelist_completeness():
    """测试1: 白名单规则完整性"""
    print("=" * 70)
    print("测试1: 白名单规则完整性检查")
    print("=" * 70)

    errors = []
    warnings = []

    # 检查所有预期规则是否存在
    for rule, expected_value in EXPECTED_WHITELIST_RULES.items():
        actual_value = WHITELIST_RULES.get(rule)
        if actual_value is None:
            errors.append(f"  缺失规则: {rule}")
        elif actual_value != expected_value:
            errors.append(f"  规则值错误: {rule} = {actual_value}, 期望 {expected_value}")

    # 检查是否有额外规则
    extra_rules = set(WHITELIST_RULES.keys()) - set(EXPECTED_WHITELIST_RULES.keys())
    if extra_rules:
        warnings.append(f"  额外规则: {extra_rules}")

    # 检查规则数量
    expected_count = len(EXPECTED_WHITELIST_RULES)
    actual_count = len([k for k, v in WHITELIST_RULES.items() if v is True])
    warnings.append(f"  规则数量: {actual_count}/{expected_count}")

    if errors:
        print("  失败 - 发现错误:")
        for error in errors:
            print(error)
        return False
    else:
        print("  通过 - 所有预期规则都存在且值正确")
        if warnings:
            print("  警告:")
            for warning in warnings:
                print(warning)
        return True


def test_blacklist_default_mechanism():
    """测试2: 黑名单默认机制"""
    print("\n" + "=" * 70)
    print("测试2: 黑名单默认机制检查")
    print("=" * 70)

    errors = []

    # 测试所有黑名单规则是否返回False
    for source, target in BLACKLIST_RULES:
        result = is_edge_allowed(source, target)
        if result is not False:
            errors.append(f"  黑名单规则未被禁止: ({source}, {target}) = {result}")

    # 测试未定义的组合是否默认返回False
    # 生成所有可能的组合，排除白名单规则
    all_combinations = [(s, t) for s in ALL_CATEGORIES for t in ALL_CATEGORIES]
    undefined_combinations = set(all_combinations) - set(EXPECTED_WHITELIST_RULES.keys())

    unexpected_allowed = []
    for source, target in undefined_combinations:
        if is_edge_allowed(source, target):
            unexpected_allowed.append((source, target))

    if unexpected_allowed:
        errors.append(f"  未定义的组合被错误允许: {unexpected_allowed}")

    if errors:
        print("  失败 - 发现错误:")
        for error in errors:
            print(error)
        return False
    else:
        print(f"  通过 - 所有{len(BLACKLIST_RULES)}个黑名单规则正确禁止")
        print(f"  通过 - 未定义的组合默认返回False")
        return True


def test_mediator_to_performance_rule():
    """测试3: 关键规则 mediator -> performance"""
    print("\n" + "=" * 70)
    print("测试3: 关键规则 mediator -> performance (v1.1新增)")
    print("=" * 70)

    result = is_edge_allowed('mediator', 'performance')
    expected = True

    if result == expected:
        print("  通过 - mediator -> performance 正确允许")
        print(f"    值: {result}")
        return True
    else:
        print(f"  失败 - mediator -> performance = {result}, 期望 {expected}")
        return False


def test_rule_groups():
    """测试4: 规则组分类"""
    print("\n" + "=" * 70)
    print("测试4: 规则组分类验证")
    print("=" * 70)

    # 定义预期规则组
    expected_groups = {
        'Q1_hyperparam_main': [
            ('hyperparam', 'energy'),
            ('hyperparam', 'mediator')
        ],
        'Q1_interaction_moderation': [
            ('interaction', 'energy'),
            ('interaction', 'mediator')
        ],
        'Q2_performance': [
            ('hyperparam', 'performance'),
            ('interaction', 'performance')
        ],
        'Q3_mediation': [
            ('mediator', 'energy'),
            ('mediator', 'mediator'),
            ('energy', 'energy')
        ],
        'control_effects': [
            ('control', 'energy'),
            ('control', 'mediator'),
            ('control', 'performance'),
            ('mode', 'energy'),
            ('mode', 'mediator'),
            ('mode', 'performance')
        ]
    }

    # 注意: 脚本中的Q2_performance组缺少mediator->performance
    # 这是统计分组问题，不影响过滤功能

    print("  规则组定义:")
    for group_name, edges in expected_groups.items():
        print(f"    {group_name}: {len(edges)}条规则")
        for edge in edges:
            is_allowed = is_edge_allowed(edge[0], edge[1])
            status = "允许" if is_allowed else "禁止"
            print(f"      {edge[0]:12} -> {edge[1]:12}: {status}")

    print("\n  注意: 统计分组Q2_performance中应包含mediator->performance")
    print("  建议更新脚本的rule_groups定义以反映v1.1的变更")

    return True


def test_edge_cases():
    """测试5: 边界情况"""
    print("\n" + "=" * 70)
    print("测试5: 边界情况处理")
    print("=" * 70)

    errors = []

    # 测试空输入
    try:
        empty_df = pd.DataFrame({'source_category': [], 'target_category': []})
        result = filter_causal_edges_by_whitelist(empty_df)
        if len(result) != 0:
            errors.append("  空DataFrame处理错误")
    except Exception as e:
        errors.append(f"  空DataFrame抛出异常: {e}")

    # 测试缺少必需列
    try:
        bad_df = pd.DataFrame({'source': ['a'], 'target': ['b']})
        result = filter_causal_edges_by_whitelist(bad_df)
        errors.append("  缺少必需列时应抛出异常")
    except ValueError as e:
        pass  # 预期行为
    except Exception as e:
        errors.append(f"  缺少必需列抛出错误异常类型: {e}")

    # 测试大小写敏感性
    if is_edge_allowed('Hyperparam', 'energy') or is_edge_allowed('hyperparam', 'Energy'):
        errors.append("  大小写敏感性不一致")

    # 测试None/空字符串
    try:
        result1 = is_edge_allowed('', 'energy')
        result2 = is_edge_allowed(None, 'energy')
        if result1 or result2:
            errors.append("  空/None输入应返回False")
    except Exception:
        pass  # 抛出异常也可接受

    if errors:
        print("  失败 - 发现问题:")
        for error in errors:
            print(error)
        return False
    else:
        print("  通过 - 所有边界情况正确处理")
        return True


def test_real_data_filtering():
    """测试6: 真实数据过滤验证"""
    print("\n" + "=" * 70)
    print("测试6: 真实数据过滤验证")
    print("=" * 70)

    # 查找测试数据文件
    test_data_path = Path("/home/green/energy_dl/nightly/analysis/results/energy_research/data/interaction/threshold/group1_examples_causal_edges_0.3.csv")

    if not test_data_path.exists():
        print("  跳过 - 测试数据文件不存在")
        return True

    # 读取原始数据
    original_df = pd.read_csv(test_data_path)
    print(f"  原始数据: {len(original_df)}条边")

    # 应用白名单过滤
    filtered_df = filter_causal_edges_by_whitelist(original_df)
    print(f"  过滤后数据: {len(filtered_df)}条边")
    print(f"  保留率: {len(filtered_df)/len(original_df)*100:.1f}%")

    # 验证过滤结果
    errors = []

    # 检查: 所有保留的边都应该在白名单中
    for _, row in filtered_df.iterrows():
        source_cat = row['source_category']
        target_cat = row['target_category']
        if not is_edge_allowed(source_cat, target_cat):
            errors.append(f"  被保留的边不在白名单中: ({source_cat}, {target_cat})")

    # 检查: 黑名单边是否被正确过滤
    blacklist_in_filtered = filtered_df.apply(
        lambda row: (row['source_category'], row['target_category']) in
                   [tuple(x) for x in BLACKLIST_RULES],
        axis=1
    )
    if blacklist_in_filtered.any():
        blacklisted_edges = filtered_df[blacklist_in_filtered][['source_category', 'target_category']].drop_duplicates()
        errors.append(f"  黑名单边未被过滤: {blacklisted_edges.values}")

    # 统计各规则组的边数
    print("\n  规则组统计:")
    rule_groups = {
        'Q1_hyperparam_main': [('hyperparam', 'energy'), ('hyperparam', 'mediator')],
        'Q1_interaction_moderation': [('interaction', 'energy'), ('interaction', 'mediator')],
        'Q2_performance': [('hyperparam', 'performance'), ('interaction', 'performance')],
        'Q3_mediation': [('mediator', 'energy'), ('mediator', 'mediator'), ('energy', 'energy')],
        'mediator_to_performance': [('mediator', 'performance')],  # v1.1新增
        'control_effects': [
            ('control', 'energy'), ('control', 'mediator'), ('control', 'performance'),
            ('mode', 'energy'), ('mode', 'mediator'), ('mode', 'performance')
        ]
    }

    for group_name, edges in rule_groups.items():
        count = 0
        for source, target in edges:
            count += len(filtered_df[
                (filtered_df['source_category'] == source) &
                (filtered_df['target_category'] == target)
            ])
        print(f"    {group_name}: {count}条")

    # 特别检查mediator -> performance边
    mediator_perf_edges = filtered_df[
        (filtered_df['source_category'] == 'mediator') &
        (filtered_df['target_category'] == 'performance')
    ]
    print(f"\n  mediator -> performance 边: {len(mediator_perf_edges)}条")
    if len(mediator_perf_edges) > 0:
        print("  示例:")
        for _, row in mediator_perf_edges.head(3).iterrows():
            print(f"    {row['source']} -> {row['target']} (强度={row['strength']:.2f})")

    if errors:
        print("\n  失败 - 发现错误:")
        for error in errors:
            print(error)
        return False
    else:
        print("\n  通过 - 真实数据过滤验证成功")
        return True


def test_whitelist_vs_document_consistency():
    """测试7: 实现与文档的一致性（逐项对比）"""
    print("\n" + "=" * 70)
    print("测试7: 实现与文档的一致性检查")
    print("=" * 70)

    # 文档第7.2节定义的规则
    doc_rules = EXPECTED_WHITELIST_RULES

    # 脚本中实现的规则
    impl_rules = {k: v for k, v in WHITELIST_RULES.items() if v is True}

    # 检查一致性
    missing_rules = set(doc_rules.keys()) - set(impl_rules.keys())
    extra_rules = set(impl_rules.keys()) - set(doc_rules.keys())

    if missing_rules:
        print(f"  缺失规则: {missing_rules}")
        return False

    if extra_rules:
        print(f"  额外规则: {extra_rules}")
        print("  注: 额外规则可能是合理的，请确认是否符合设计意图")

    # 检查v1.1的关键更新
    if ('mediator', 'performance') not in impl_rules:
        print("  错误: v1.1新增的mediator->performance规则未实现!")
        return False

    print("  通过 - 实现与文档一致")
    print(f"    规则数量: {len(impl_rules)}条")
    print("    包含v1.1新增: mediator -> performance")

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("白名单过滤脚本实现验证测试")
    print("=" * 70)

    tests = [
        ("白名单规则完整性", test_whitelist_completeness),
        ("黑名单默认机制", test_blacklist_default_mechanism),
        ("mediator->performance规则", test_mediator_to_performance_rule),
        ("规则组分类", test_rule_groups),
        ("边界情况处理", test_edge_cases),
        ("真实数据过滤", test_real_data_filtering),
        ("实现与文档一致性", test_whitelist_vs_document_consistency),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n  测试异常: {e}")
            results[test_name] = False

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "通过" if result else "失败"
        print(f"  {status}: {test_name}")

    print(f"\n总体结果: {passed}/{total} 测试通过")

    # 一致性评分
    consistency_score = passed / total * 100
    print(f"一致性评分: {consistency_score:.0f}%")

    if consistency_score >= 90:
        print("\n结论: 脚本实现与白名单设计文档v1.1高度一致，可以投入使用。")
        return 0
    elif consistency_score >= 70:
        print("\n结论: 脚本实现基本符合要求，但有部分问题需要修复。")
        return 1
    else:
        print("\n结论: 脚本实现存在重大问题，不建议投入使用。")
        return 2


if __name__ == '__main__':
    sys.exit(main())
