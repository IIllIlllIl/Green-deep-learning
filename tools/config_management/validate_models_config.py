#!/usr/bin/env python3
"""验证models_config.json的完整性和有效性

用途: 检查所有超参数是否有默认值定义
作者: Claude
日期: 2025-12-24
"""

import json
from pathlib import Path
from collections import defaultdict


def validate_models_config():
    """验证models_config.json的完整性"""

    config_file = Path('mutation/models_config.json')

    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return None

    with open(config_file) as f:
        config = json.load(f)

    print("=" * 80)
    print("models_config.json 完整性验证")
    print("=" * 80)

    # 统计
    stats = {
        'total_repos': 0,
        'total_models': 0,
        'total_hyperparams': 0,
        'missing_defaults': [],
        'missing_type': [],
        'missing_range': []
    }

    hyperparam_coverage = defaultdict(int)
    hyperparam_defaults = defaultdict(list)  # 记录每个超参数在不同repo的默认值

    for repo, repo_config in config['models'].items():
        stats['total_repos'] += 1
        models = repo_config.get('models', [])
        stats['total_models'] += len(models)

        print(f"\n{'='*80}")
        print(f"仓库: {repo}")
        print(f"{'='*80}")
        print(f"  模型: {', '.join(models)}")

        supported = repo_config.get('supported_hyperparams', {})
        print(f"  超参数数量: {len(supported)}")

        if not supported:
            print("  ⚠️ 没有定义支持的超参数")
            continue

        for param_name, param_config in supported.items():
            stats['total_hyperparams'] += 1
            hyperparam_coverage[param_name] += 1

            # 提取字段
            default_val = param_config.get('default')
            param_type = param_config.get('type')
            param_range = param_config.get('range')
            flag = param_config.get('flag', 'N/A')

            # 检查必需字段
            issues = []

            if default_val is None:
                stats['missing_defaults'].append(f"{repo}.{param_name}")
                issues.append("缺少default")
            else:
                hyperparam_defaults[param_name].append({
                    'repo': repo,
                    'value': default_val,
                    'type': param_type
                })

            if param_type is None:
                stats['missing_type'].append(f"{repo}.{param_name}")
                issues.append("缺少type")

            if param_range is None:
                stats['missing_range'].append(f"{repo}.{param_name}")
                issues.append("缺少range")

            # 打印超参数信息
            status = "✅" if not issues else "⚠️"
            print(f"    {status} {param_name}:")
            print(f"       - flag: {flag}")
            print(f"       - type: {param_type or '未定义'}")
            print(f"       - default: {default_val if default_val is not None else '未定义'}")
            print(f"       - range: {param_range or '未定义'}")

            if issues:
                print(f"       - 问题: {', '.join(issues)}")

    # 打印超参数覆盖统计
    print("\n" + "=" * 80)
    print("超参数覆盖统计（按使用频率排序）")
    print("=" * 80)
    for param, count in sorted(hyperparam_coverage.items(), key=lambda x: -x[1]):
        print(f"{param:20s}: {count} 个仓库使用")

    # 打印超参数默认值对比
    print("\n" + "=" * 80)
    print("超参数默认值对比（相同超参数在不同仓库的默认值）")
    print("=" * 80)
    for param in sorted(hyperparam_defaults.keys()):
        defaults_list = hyperparam_defaults[param]
        if len(defaults_list) > 1:  # 只显示被多个仓库使用的
            print(f"\n{param}:")
            for item in defaults_list:
                print(f"  - {item['repo']:35s}: {item['value']} ({item['type']})")

    # 打印总体统计
    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80)
    print(f"仓库数: {stats['total_repos']}")
    print(f"模型总数: {stats['total_models']}")
    print(f"超参数总数: {stats['total_hyperparams']}")
    print(f"唯一超参数名: {len(hyperparam_coverage)}")

    # 检查问题
    print("\n" + "=" * 80)
    print("完整性检查")
    print("=" * 80)

    if stats['missing_defaults']:
        print(f"\n❌ 缺少默认值的参数 ({len(stats['missing_defaults'])} 个):")
        for param in stats['missing_defaults']:
            print(f"  - {param}")
    else:
        print("\n✅ 所有超参数都有默认值定义")

    if stats['missing_type']:
        print(f"\n⚠️ 缺少类型定义的参数 ({len(stats['missing_type'])} 个):")
        for param in stats['missing_type']:
            print(f"  - {param}")
    else:
        print("✅ 所有超参数都有类型定义")

    if stats['missing_range']:
        print(f"\n⚠️ 缺少范围定义的参数 ({len(stats['missing_range'])} 个):")
        for param in stats['missing_range']:
            print(f"  - {param}")
    else:
        print("✅ 所有超参数都有范围定义")

    # 阶段1验证结论
    print("\n" + "=" * 80)
    print("阶段1验证结论")
    print("=" * 80)

    if not stats['missing_defaults']:
        print("✅ models_config.json 完全符合v2.0方案要求")
        print("✅ 可以安全地进行默认值回溯")
        print("✅ 建议进入阶段2: 实现数据提取脚本")
    else:
        print("⚠️ 存在缺失默认值的超参数")
        print("⚠️ 需要先补充默认值或使用降级策略（range中位数）")

    return stats


def generate_field_mapping_table(config):
    """
    生成超参数字段映射表（用于阶段2）

    Args:
        config: models_config.json的配置字典

    Returns:
        dict: 字段映射表
    """
    print("\n" + "=" * 80)
    print("字段映射表生成（用于阶段2数据提取）")
    print("=" * 80)

    # 全局映射规则
    GLOBAL_MAPPING = {
        # 训练迭代次数统一
        'epochs': 'training_duration',
        'max_iter': 'training_duration',
        'num_iters': 'training_duration',

        # L2正则化统一
        'weight_decay': 'l2_regularization',

        # 学习率统一
        'learning_rate': 'hyperparam_learning_rate',
        'lr': 'hyperparam_learning_rate',

        # 批量大小统一
        'batch_size': 'hyperparam_batch_size',
        'train_batch_size': 'hyperparam_batch_size',

        # Dropout统一
        'dropout': 'hyperparam_dropout',
        'droprate': 'hyperparam_dropout',
        'dropout_rate': 'hyperparam_dropout',

        # 其他参数保持原名
        'seed': 'seed',
        'kfold': 'hyperparam_kfold',
    }

    # 特殊仓库映射
    REPO_SPECIFIC_MAPPING = {
        'bug-localization-by-dnn-and-rvsm': {
            'alpha': 'l2_regularization',  # Bug定位的alpha是L2正则化
        },
        'MRT-OAST': {
            'alpha': 'l2_regularization',  # MRT-OAST的alpha也是L2正则化
        }
    }

    print("\n全局映射规则:")
    for original, unified in sorted(GLOBAL_MAPPING.items()):
        print(f"  {original:20s} → {unified}")

    print("\n特殊仓库映射:")
    for repo, mappings in REPO_SPECIFIC_MAPPING.items():
        print(f"  {repo}:")
        for original, unified in mappings.items():
            print(f"    {original:20s} → {unified}")

    # 分析models_config.json中实际使用的超参数
    print("\n实际使用的超参数及其映射:")
    for repo, repo_config in config['models'].items():
        supported = repo_config.get('supported_hyperparams', {})
        if not supported:
            continue

        print(f"\n  {repo}:")
        for param_name in supported.keys():
            # 确定映射后的字段名
            if repo in REPO_SPECIFIC_MAPPING and param_name in REPO_SPECIFIC_MAPPING[repo]:
                unified_name = REPO_SPECIFIC_MAPPING[repo][param_name]
            elif param_name in GLOBAL_MAPPING:
                unified_name = GLOBAL_MAPPING[param_name]
            else:
                unified_name = f'hyperparam_{param_name}'

            print(f"    {param_name:20s} → {unified_name}")

    return {
        'global': GLOBAL_MAPPING,
        'repo_specific': REPO_SPECIFIC_MAPPING
    }


if __name__ == '__main__':
    # 验证完整性
    stats = validate_models_config()

    if stats:
        # 加载配置生成映射表
        config_file = Path('mutation/models_config.json')
        with open(config_file) as f:
            config = json.load(f)

        field_mapping = generate_field_mapping_table(config)

        # 保存映射表到JSON（用于阶段2）
        mapping_output = Path('analysis/docs/reports/hyperparam_field_mapping.json')
        mapping_output.parent.mkdir(parents=True, exist_ok=True)

        with open(mapping_output, 'w', encoding='utf-8') as f:
            json.dump(field_mapping, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 字段映射表已保存到: {mapping_output}")
