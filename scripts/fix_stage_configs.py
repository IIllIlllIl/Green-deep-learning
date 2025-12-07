#!/usr/bin/env python3
"""
修复Stage7-13配置文件的参数拆分问题

问题：配置文件将多个参数放在一个配置项中，导致混合变异
解决：拆分为每个参数独立的配置项

使用方法：
    python3 scripts/fix_stage_configs.py [--dry-run]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def expand_experiment_config(exp: Dict[str, Any], base_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将包含多个mutate_params的配置项拆分为单参数配置项

    Args:
        exp: 原始实验配置
        base_info: 基本信息（如comment, version等）

    Returns:
        拆分后的配置项列表
    """
    # 获取mutate_params
    mutate_params = None
    is_parallel = 'foreground' in exp

    if is_parallel:
        mutate_params = exp['foreground'].get('mutate', exp['foreground'].get('mutate_params'))
    else:
        mutate_params = exp.get('mutate', exp.get('mutate_params'))

    # 如果只有1个参数或'all'，不需要拆分
    if not mutate_params or mutate_params == ['all'] or len(mutate_params) == 1:
        return [exp]

    # 拆分为多个单参数配置
    expanded = []
    for param in mutate_params:
        new_exp = exp.copy()

        if is_parallel:
            # 并行模式：复制整个foreground/background结构
            new_exp['foreground'] = exp['foreground'].copy()
            new_exp['background'] = exp['background'].copy()
            new_exp['foreground']['mutate'] = [param]
            new_exp['foreground']['comment'] = f"仅变异{param}参数"
        else:
            # 非并行模式：直接修改mutate_params
            if 'mutate' in new_exp:
                new_exp['mutate'] = [param]
            else:
                new_exp['mutate_params'] = [param]
            new_exp['comment'] = f"仅变异{param}参数"

        expanded.append(new_exp)

    return expanded

def fix_config_file(config_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """
    修复单个配置文件

    Args:
        config_path: 配置文件路径
        dry_run: 如果为True，只打印修改不实际保存

    Returns:
        修复统计信息
    """
    print(f"\n{'='*80}")
    print(f"处理: {config_path.name}")
    print(f"{'='*80}")

    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    original_experiments = config.get('experiments', [])
    original_count = len(original_experiments)
    original_estimated = config.get('estimated_experiments', 0)

    print(f"原配置: {original_count}个配置项, 预期{original_estimated}个实验")

    # 拆分实验配置
    expanded_experiments = []
    for exp in original_experiments:
        expanded = expand_experiment_config(exp, config)
        expanded_experiments.extend(expanded)

    new_count = len(expanded_experiments)

    # 计算新的estimated_experiments
    # 简单估算：每个配置项的runs_per_config之和
    new_estimated = 0
    for exp in expanded_experiments:
        if 'foreground' in exp:
            runs = exp['foreground'].get('runs_per_config', exp.get('runs_per_config', 1))
        else:
            runs = exp.get('runs_per_config', 1)
        new_estimated += runs

    print(f"新配置: {new_count}个配置项, 预计{new_estimated}个实验")
    print(f"变化: +{new_count - original_count}个配置项, +{new_estimated - original_estimated}个实验")

    # 打印详细变化
    print(f"\n详细变化:")
    exp_idx = 0
    for orig_exp in original_experiments:
        is_parallel = 'foreground' in orig_exp

        if is_parallel:
            repo = orig_exp['foreground']['repo']
            model = orig_exp['foreground']['model']
            mutate_params = orig_exp['foreground'].get('mutate', orig_exp['foreground'].get('mutate_params', []))
            runs = orig_exp['foreground'].get('runs_per_config', orig_exp.get('runs_per_config', 1))
        else:
            repo = orig_exp['repo']
            model = orig_exp['model']
            mutate_params = orig_exp.get('mutate', orig_exp.get('mutate_params', []))
            runs = orig_exp.get('runs_per_config', 1)

        num_params = len(mutate_params) if mutate_params != ['all'] else 1
        mode_str = "并行" if is_parallel else "非并行"

        if num_params > 1:
            print(f"  ❌ [{mode_str}] {repo}/{model}: {runs}次 × {num_params}参数")
            for param in mutate_params:
                print(f"     → 拆分为: {runs}次 × 1参数 ({param})")
            exp_idx += num_params
        else:
            print(f"  ✓ [{mode_str}] {repo}/{model}: {runs}次 × {num_params}参数 (无需修改)")
            exp_idx += 1

    # 更新配置
    config['experiments'] = expanded_experiments
    config['estimated_experiments'] = new_estimated

    # 更新version和comment
    if 'version' in config:
        config['version'] = "4.7.1-fixed"
    if 'comment' in config:
        config['comment'] = f"{config['comment']} [已修复: 拆分多参数配置项]"

    # 保存修复后的配置
    if not dry_run:
        backup_path = config_path.with_suffix('.json.bak')
        config_path.rename(backup_path)
        print(f"\n✓ 备份原文件: {backup_path.name}")

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存新配置: {config_path.name}")
    else:
        print(f"\n[DRY RUN] 不保存更改")

    return {
        'file': config_path.name,
        'original_items': original_count,
        'new_items': new_count,
        'original_estimated': original_estimated,
        'new_estimated': new_estimated,
        'added_items': new_count - original_count,
        'added_experiments': new_estimated - original_estimated
    }

def main():
    """主函数"""
    dry_run = '--dry-run' in sys.argv

    project_root = Path(__file__).parent.parent
    settings_dir = project_root / 'settings'

    # Stage7-13配置文件
    config_files = [
        'stage7_nonparallel_fast_models.json',
        'stage8_nonparallel_medium_slow_models.json',
        'stage9_nonparallel_hrnet18.json',
        'stage10_nonparallel_pcb.json',
        'stage11_parallel_hrnet18.json',
        'stage12_parallel_pcb.json',
        'stage13_parallel_fast_models_supplement.json'
    ]

    print("="*80)
    print("Stage7-13 配置文件修复工具")
    print("="*80)
    print(f"模式: {'DRY RUN (不保存)' if dry_run else 'LIVE (将保存更改)'}")
    print(f"待处理: {len(config_files)}个配置文件")

    stats = []
    for config_file in config_files:
        config_path = settings_dir / config_file
        if not config_path.exists():
            print(f"\n⚠️  文件不存在: {config_file}")
            continue

        try:
            stat = fix_config_file(config_path, dry_run)
            stats.append(stat)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    # 打印总结
    print(f"\n\n{'='*80}")
    print("修复总结")
    print(f"{'='*80}")

    total_original_items = sum(s['original_items'] for s in stats)
    total_new_items = sum(s['new_items'] for s in stats)
    total_original_exp = sum(s['original_estimated'] for s in stats)
    total_new_exp = sum(s['new_estimated'] for s in stats)

    print(f"\n处理文件: {len(stats)}/{len(config_files)}")
    print(f"\n配置项变化:")
    print(f"  原始: {total_original_items}个配置项")
    print(f"  修复后: {total_new_items}个配置项")
    print(f"  新增: +{total_new_items - total_original_items}个配置项")
    print(f"\n实验数变化:")
    print(f"  原估计: {total_original_exp}个实验")
    print(f"  新估计: {total_new_exp}个实验")
    print(f"  应等于: 370个实验 (设计目标)")

    if total_new_exp == 370:
        print(f"  ✅ 修复正确！")
    else:
        print(f"  ⚠️  与预期不符，需要检查")

    print(f"\n详细统计:")
    for stat in stats:
        print(f"  {stat['file']}:")
        print(f"    配置项: {stat['original_items']} → {stat['new_items']} (+{stat['added_items']})")
        print(f"    实验数: {stat['original_estimated']} → {stat['new_estimated']} (+{stat['added_experiments']})")

    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN 模式 - 未保存任何更改")
        print("移除 --dry-run 参数以实际保存修复")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("✅ 所有更改已保存")
        print("原文件已备份为 .json.bak")
        print(f"{'='*80}")

if __name__ == '__main__':
    main()
