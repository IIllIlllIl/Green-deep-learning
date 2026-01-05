#!/usr/bin/env python3
"""
验证所有Stage配置文件的runs_per_config定义是否正确

检查并行模式配置中的runs_per_config定义位置，确保与v4.7.2修复后的代码兼容。
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent

def check_config_file(config_path):
    """检查单个配置文件的runs_per_config定义"""

    print(f"\n{'=' * 80}")
    print(f"检查配置: {config_path.name}")
    print('=' * 80)

    with open(config_path, 'r') as f:
        config = json.load(f)

    issues = []
    warnings = []
    good_practices = []

    experiments = config.get('experiments', [])
    print(f"实验配置项数量: {len(experiments)}")

    for i, exp in enumerate(experiments, 1):
        exp_mode = exp.get('mode', 'mutation')

        # 跳过注释行
        if 'comment' in exp and len(exp) == 1:
            continue

        if exp_mode == 'parallel':
            foreground = exp.get('foreground', {})

            # 检查runs_per_config的定义位置
            outer_runs = exp.get('runs_per_config')
            fg_runs = foreground.get('runs_per_config')

            print(f"\n  配置项 {i} (并行模式):")
            print(f"    前景模型: {foreground.get('repo', 'N/A')}/{foreground.get('model', 'N/A')}")
            print(f"    变异参数: {foreground.get('mutate', 'N/A')}")
            print(f"    外层 runs_per_config: {outer_runs}")
            print(f"    foreground runs_per_config: {fg_runs}")

            if outer_runs and fg_runs:
                warnings.append(f"配置项{i}: 同时定义了外层和foreground的runs_per_config（外层优先级更高）")
                print(f"    ⚠️  两处都有定义（外层={outer_runs}优先）")
            elif outer_runs:
                good_practices.append(f"配置项{i}: 使用外层runs_per_config={outer_runs} ✅（推荐）")
                print(f"    ✅ 使用外层定义（推荐）")
            elif fg_runs:
                good_practices.append(f"配置项{i}: 使用foreground runs_per_config={fg_runs} ✅（支持）")
                print(f"    ✅ 使用foreground定义（支持）")
            else:
                issues.append(f"配置项{i}: 未定义runs_per_config（将使用全局默认值1）")
                print(f"    ❌ 未定义（将fallback到全局，可能为1）")

        elif exp_mode in ['mutation', 'nonparallel', 'default']:
            outer_runs = exp.get('runs_per_config')
            print(f"\n  配置项 {i} ({exp_mode}模式):")
            print(f"    模型: {exp.get('repo', 'N/A')}/{exp.get('model', 'N/A')}")
            print(f"    runs_per_config: {outer_runs}")

            if outer_runs:
                good_practices.append(f"配置项{i}: 定义了runs_per_config={outer_runs} ✅")
                print(f"    ✅ 已定义")
            else:
                warnings.append(f"配置项{i}: 未定义runs_per_config（将使用全局默认值）")
                print(f"    ⚠️  未定义（将fallback到全局）")

    # 汇总报告
    print(f"\n{'-' * 80}")
    print("汇总:")
    print(f"  良好实践: {len(good_practices)}")
    print(f"  警告: {len(warnings)}")
    print(f"  问题: {len(issues)}")

    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"    - {issue}")

    if warnings:
        print(f"\n⚠️  {len(warnings)} 个警告:")
        for warning in warnings:
            print(f"    - {warning}")

    return len(issues) == 0


def main():
    """检查所有待执行的Stage配置"""

    print("\n" + "🔍" * 40)
    print("Stage配置文件验证工具 (v4.7.2)")
    print("🔍" * 40)

    # 待执行的配置文件
    configs_to_check = [
        'settings/stage11_parallel_hrnet18.json',
        'settings/stage12_parallel_pcb.json',
        'settings/stage13_merged_final_supplement.json'
    ]

    all_passed = True
    results = {}

    for config_file in configs_to_check:
        config_path = project_root / config_file

        if not config_path.exists():
            print(f"\n❌ 文件不存在: {config_file}")
            all_passed = False
            continue

        passed = check_config_file(config_path)
        results[config_file] = passed

        if not passed:
            all_passed = False

    # 最终总结
    print("\n" + "=" * 80)
    print("最终总结")
    print("=" * 80)

    for config_file, passed in results.items():
        status = "✅ 通过" if passed else "❌ 有问题"
        print(f"  {config_file.split('/')[-1]}: {status}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有配置文件验证通过！")
        print("=" * 80)
        return 0
    else:
        print("⚠️  部分配置文件存在问题，请检查上述详情")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
