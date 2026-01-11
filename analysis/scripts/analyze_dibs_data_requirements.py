#!/usr/bin/env python3
"""
DiBS数据需求分析脚本

功能：
1. 分析当前实验数据的完整性（能耗、性能）
2. 按照12分组方案（6任务组 × 2场景）统计样本量
3. 评估DiBS所需的最小样本量
4. 计算还需要多少实验

作者：Green + Claude
日期：2026-01-11
"""

import csv
from collections import defaultdict
from pathlib import Path

# 数据文件路径
DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'raw_data.csv'

# 任务组定义（与QUESTION1方案一致）
TASK_GROUPS = {
    'group1a_examples': [
        'examples/mnist',
        'examples/mnist_ff',
        'examples/mnist_rnn',
        'examples/siamese'
    ],
    'group1b_resnet': [
        'pytorch_resnet_cifar10/resnet20'
    ],
    'group2_person_reid': [
        'Person_reID_baseline_pytorch/densenet121',
        'Person_reID_baseline_pytorch/hrnet18',
        'Person_reID_baseline_pytorch/pcb'
    ],
    'group3_vulberta': [
        'VulBERTa/mlp'
    ],
    'group4_bug_localization': [
        'bug-localization-by-dnn-and-rvsm/default'
    ],
    'group5_mrt_oast': [
        'MRT-OAST/default'
    ]
}

# 能耗列
ENERGY_COLS = [
    'energy_cpu_pkg_joules',
    'energy_cpu_ram_joules',
    'energy_gpu_total_joules',
    'energy_cpu_total_joules'
]

# 性能列（至少需要一个）
PERF_COLS_PREFIXES = ['perf_']


def parse_model_name(row):
    """从行数据中解析模型名称"""
    repo = row.get('repository', '')
    model = row.get('model', '')

    # 处理并行模式的情况
    if not repo or not model:
        # 尝试从前台数据获取
        fg_repo = row.get('fg_repository', '')
        fg_model = row.get('fg_model', '')
        if fg_repo and fg_model:
            return f"{fg_repo}/{fg_model}"
        # 尝试从后台数据获取
        bg_repo = row.get('bg_repository', '')
        bg_model = row.get('bg_model', '')
        if bg_repo and bg_model:
            return f"{bg_repo}/{bg_model}"

    return f"{repo}/{model}" if repo and model else None


def is_parallel_mode(row):
    """判断是否为并行模式"""
    mode = row.get('mode', '')

    # 如果有明确的mode字段
    if mode:
        return 'parallel' in mode.lower() or 'background' in mode.lower()

    # 检查是否有fg_或bg_前缀的列有数据
    has_fg = any(row.get(f'fg_{col}') for col in ['repository', 'model'])
    has_bg = any(row.get(f'bg_{col}') for col in ['repository', 'model'])

    return has_fg or has_bg


def check_energy_complete(row):
    """检查能耗数据是否完整"""
    for col in ENERGY_COLS:
        val = row.get(col, '')
        if not val or val == '':
            return False
    return True


def check_perf_complete(row):
    """检查性能数据是否完整（至少有一个性能指标）"""
    for key in row.keys():
        if key.startswith('perf_'):
            val = row.get(key, '')
            if val and val != '':
                return True
    return False


def analyze_data():
    """分析当前数据"""

    print("=" * 80)
    print("DiBS数据需求分析")
    print("=" * 80)

    # 读取数据
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"\n总实验数: {total_rows}")

    # 统计完整性
    energy_complete = 0
    perf_complete = 0
    both_complete = 0

    # 按任务组和场景分组统计
    group_stats = defaultdict(lambda: {
        'total': 0,
        'energy': 0,
        'perf': 0,
        'both': 0,
        'non_parallel_total': 0,
        'non_parallel_energy': 0,
        'non_parallel_perf': 0,
        'non_parallel_both': 0,
        'parallel_total': 0,
        'parallel_energy': 0,
        'parallel_perf': 0,
        'parallel_both': 0
    })

    for row in rows:
        model_name = parse_model_name(row)
        is_parallel = is_parallel_mode(row)
        has_energy = check_energy_complete(row)
        has_perf = check_perf_complete(row)
        has_both = has_energy and has_perf

        # 全局统计
        if has_energy:
            energy_complete += 1
        if has_perf:
            perf_complete += 1
        if has_both:
            both_complete += 1

        # 找到所属任务组
        for group_name, models in TASK_GROUPS.items():
            if model_name in models:
                stats = group_stats[group_name]
                stats['total'] += 1

                if is_parallel:
                    stats['parallel_total'] += 1
                    if has_energy:
                        stats['parallel_energy'] += 1
                    if has_perf:
                        stats['parallel_perf'] += 1
                    if has_both:
                        stats['parallel_both'] += 1
                else:
                    stats['non_parallel_total'] += 1
                    if has_energy:
                        stats['non_parallel_energy'] += 1
                    if has_perf:
                        stats['non_parallel_perf'] += 1
                    if has_both:
                        stats['non_parallel_both'] += 1

                if has_energy:
                    stats['energy'] += 1
                if has_perf:
                    stats['perf'] += 1
                if has_both:
                    stats['both'] += 1
                break

    # 打印全局统计
    print("\n" + "=" * 80)
    print("全局数据完整性")
    print("=" * 80)
    print(f"能耗数据完整:           {energy_complete}/{total_rows} ({energy_complete/total_rows*100:.1f}%)")
    print(f"性能数据完整:           {perf_complete}/{total_rows} ({perf_complete/total_rows*100:.1f}%)")
    print(f"能耗+性能都完整:        {both_complete}/{total_rows} ({both_complete/total_rows*100:.1f}%)")

    # 打印12分组统计
    print("\n" + "=" * 80)
    print("12分组方案样本量统计（6任务组 × 2场景）")
    print("=" * 80)

    print(f"\n{'任务组':<30} {'场景':<15} {'总计':<8} {'能耗完整':<10} {'性能完整':<10} {'都完整':<10}")
    print("-" * 90)

    total_6group_energy = 0
    total_6group_both = 0
    total_12group_both = 0

    for group_name in sorted(TASK_GROUPS.keys()):
        stats = group_stats[group_name]

        # 打印非并行
        print(f"{group_name:<30} {'非并行':<15} {stats['non_parallel_total']:<8} "
              f"{stats['non_parallel_energy']:<10} {stats['non_parallel_perf']:<10} "
              f"{stats['non_parallel_both']:<10}")

        # 打印并行
        print(f"{'':<30} {'并行':<15} {stats['parallel_total']:<8} "
              f"{stats['parallel_energy']:<10} {stats['parallel_perf']:<10} "
              f"{stats['parallel_both']:<10}")

        # 小计
        print(f"{'':<30} {'小计':<15} {stats['total']:<8} "
              f"{stats['energy']:<10} {stats['perf']:<10} "
              f"{stats['both']:<10}")
        print("-" * 90)

        total_6group_energy += stats['energy']
        total_6group_both += stats['both']
        total_12group_both += stats['non_parallel_both'] + stats['parallel_both']

    print(f"{'总计（6分组方案）':<30} {'':<15} {'':<8} "
          f"{total_6group_energy:<10} {'':<10} {total_6group_both:<10}")
    print(f"{'总计（12分组方案）':<30} {'':<15} {'':<8} "
          f"{'':<10} {'':<10} {total_12group_both:<10}")

    # DiBS需求分析
    print("\n" + "=" * 80)
    print("DiBS数据需求分析")
    print("=" * 80)

    print("\nDiBS因果图学习的样本量要求：")
    print("  - 最小样本量（每组）: 30-50个（可运行，但结果不稳定）")
    print("  - 推荐样本量（每组）: 100-150个（较稳定）")
    print("  - 理想样本量（每组）: 200+个（高质量结果）")

    print("\n当前6分组方案（不区分场景）：")
    print(f"  - 可用数据量: {total_6group_both} 个实验（能耗+性能都完整）")
    print(f"  - 平均每组:   {total_6group_both/6:.1f} 个实验")
    print(f"  - 状态评估:   {'✅ 达到推荐水平' if total_6group_both/6 >= 100 else '⚠️ 接近最小要求' if total_6group_both/6 >= 50 else '❌ 低于最小要求'}")

    print("\n当前12分组方案（区分场景）：")
    print(f"  - 可用数据量: {total_12group_both} 个实验（能耗+性能都完整）")
    print(f"  - 平均每组:   {total_12group_both/12:.1f} 个实验")
    print(f"  - 状态评估:   {'✅ 达到推荐水平' if total_12group_both/12 >= 100 else '⚠️ 接近最小要求' if total_12group_both/12 >= 50 else '❌ 低于最小要求'}")

    # 计算每组详细需求
    print("\n" + "=" * 80)
    print("各组详细评估与建议")
    print("=" * 80)

    recommendations = []

    for group_name in sorted(TASK_GROUPS.keys()):
        stats = group_stats[group_name]

        print(f"\n{group_name}:")
        print(f"  当前数据量（都完整）:")
        print(f"    - 非并行: {stats['non_parallel_both']} 个")
        print(f"    - 并行:   {stats['parallel_both']} 个")
        print(f"    - 合计:   {stats['both']} 个")

        # 评估状态
        min_per_scenario = 50  # 每个场景的最小样本量
        recommended_per_scenario = 100  # 每个场景的推荐样本量

        non_parallel_need = max(0, recommended_per_scenario - stats['non_parallel_both'])
        parallel_need = max(0, recommended_per_scenario - stats['parallel_both'])
        total_need = non_parallel_need + parallel_need

        if stats['non_parallel_both'] >= recommended_per_scenario and stats['parallel_both'] >= recommended_per_scenario:
            status = "✅ 数据充足"
        elif stats['non_parallel_both'] >= min_per_scenario and stats['parallel_both'] >= min_per_scenario:
            status = "⚠️ 数据基本满足，建议补充"
        else:
            status = "❌ 数据不足，需要补充"

        print(f"  状态: {status}")
        print(f"  建议补充:")
        print(f"    - 非并行: {non_parallel_need} 个")
        print(f"    - 并行:   {parallel_need} 个")
        print(f"    - 合计:   {total_need} 个")

        recommendations.append({
            'group': group_name,
            'current': stats['both'],
            'non_parallel_current': stats['non_parallel_both'],
            'parallel_current': stats['parallel_both'],
            'non_parallel_need': non_parallel_need,
            'parallel_need': parallel_need,
            'total_need': total_need
        })

    # 总体建议
    print("\n" + "=" * 80)
    print("总体实验需求建议")
    print("=" * 80)

    total_current = sum(r['current'] for r in recommendations)
    total_needed = sum(r['total_need'] for r in recommendations)

    print(f"\n当前可用数据: {total_current} 个实验（能耗+性能都完整）")
    print(f"建议补充实验: {total_needed} 个")
    print(f"补充后总数:   {total_current + total_needed} 个")
    print(f"平均每组:     {(total_current + total_needed)/12:.1f} 个")

    print("\n" + "=" * 80)
    print("实验设计建议")
    print("=" * 80)

    print("\n如果采用12分组DiBS分析方案，建议：")
    print(f"  1. 当前数据量: {total_12group_both} 个（能耗+性能都完整）")
    print(f"  2. 推荐目标:   1200 个（100/组 × 12组）")
    print(f"  3. 需要补充:   约 {max(0, 1200 - total_12group_both)} 个实验")
    print(f"  4. 补充方式:   在当前模型基础上，增加多参数变异实验")
    print(f"  5. 实验分配:   每个模型平均增加 {max(0, 1200 - total_12group_both)/11:.0f} 个实验")

    print("\n注意事项：")
    print("  - DiBS需要能耗+性能都完整的数据")
    print("  - 如果只做回归分析（不需要性能数据），当前数据已经充足")
    print("  - 12分组会稀释样本，每组平均 {:.1f} 个实验".format(total_12group_both/12))
    print("  - 建议优先补充样本量不足的组")

    # 保存结果
    output_file = Path(__file__).parent.parent / 'docs' / 'reports' / 'DIBS_DATA_REQUIREMENTS_ANALYSIS.md'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# DiBS数据需求分析报告\n\n")
        f.write(f"**生成时间**: 2026-01-11\n\n")
        f.write(f"## 数据概况\n\n")
        f.write(f"- 总实验数: {total_rows}\n")
        f.write(f"- 能耗数据完整: {energy_complete} ({energy_complete/total_rows*100:.1f}%)\n")
        f.write(f"- 性能数据完整: {perf_complete} ({perf_complete/total_rows*100:.1f}%)\n")
        f.write(f"- 能耗+性能都完整: {both_complete} ({both_complete/total_rows*100:.1f}%)\n")
        f.write(f"\n## 12分组方案样本量\n\n")
        f.write("| 任务组 | 场景 | 能耗+性能都完整 | 建议补充 |\n")
        f.write("|--------|------|----------------|----------|\n")
        for r in recommendations:
            f.write(f"| {r['group']} | 非并行 | {r['non_parallel_current']} | {r['non_parallel_need']} |\n")
            f.write(f"| {r['group']} | 并行 | {r['parallel_current']} | {r['parallel_need']} |\n")
        f.write(f"\n## 总体建议\n\n")
        f.write(f"- 当前可用: {total_12group_both} 个实验\n")
        f.write(f"- 推荐目标: 1200 个实验（100/组）\n")
        f.write(f"- 需要补充: 约 {max(0, 1200 - total_12group_both)} 个实验\n")

    print(f"\n分析报告已保存到: {output_file}")


if __name__ == '__main__':
    analyze_data()
