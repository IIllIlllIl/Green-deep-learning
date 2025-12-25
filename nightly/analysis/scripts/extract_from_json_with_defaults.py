#!/usr/bin/env python3
"""从experiment.json + models_config.json联合提取完整数据

用途: 重新提取能耗因果分析数据，解决缺失值问题
方法: 结合实验JSON记录值 + model config默认值
作者: Claude
日期: 2025-12-24
版本: v2.0
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys


# 全局映射规则
GLOBAL_FIELD_MAPPING = {
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


def load_models_config():
    """加载models_config.json"""
    # 从analysis/scripts/运行，需要../../mutation/models_config.json
    config_path = Path('../../mutation/models_config.json')

    if not config_path.exists():
        # 尝试从主项目根目录
        config_path = Path('../mutation/models_config.json')

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print(f"   当前工作目录: {Path.cwd()}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def apply_field_mapping(repo, param_name):
    """
    应用字段映射规则，统一超参数命名

    Args:
        repo: 仓库名
        param_name: 原始参数名

    Returns:
        str: 统一后的字段名
    """
    # 先检查仓库特定映射
    if repo in REPO_SPECIFIC_MAPPING and param_name in REPO_SPECIFIC_MAPPING[repo]:
        return REPO_SPECIFIC_MAPPING[repo][param_name]

    # 再检查全局映射
    if param_name in GLOBAL_FIELD_MAPPING:
        return GLOBAL_FIELD_MAPPING[param_name]

    # 未映射的保持原名（加hyperparam_前缀）
    return f'hyperparam_{param_name}'


def extract_complete_hyperparams(exp_data, models_config):
    """
    从experiment.json + models_config联合提取完整超参数

    核心逻辑:
    1. 先从experiment.json提取记录值（被变异的超参数）
    2. 对于未记录的超参数，从models_config提取默认值
    3. 应用字段映射统一命名

    Args:
        exp_data: 实验数据字典
        models_config: models_config.json配置

    Returns:
        dict: 完整的超参数字典（所有参数都有值）
    """
    is_parallel = exp_data.get('mode') == 'parallel'

    # 提取repository
    if is_parallel:
        repo = exp_data.get('foreground', {}).get('repository')
        recorded_params = exp_data.get('foreground', {}).get('hyperparameters', {})
    else:
        repo = exp_data.get('repository')
        recorded_params = exp_data.get('hyperparameters', {})

    # 获取该仓库支持的所有超参数定义
    if repo not in models_config['models']:
        print(f"⚠️ 仓库 {repo} 不在 models_config.json 中")
        return {}

    supported_params = models_config['models'][repo].get('supported_hyperparams', {})

    # 合并：记录值优先，未记录的用默认值
    complete_params = {}

    for param_name, param_config in supported_params.items():
        if param_name in recorded_params:
            # 使用实验中记录的值（被变异的）
            value = recorded_params[param_name]
            source = 'recorded'
        else:
            # 使用model config中的默认值（未变异的）
            value = param_config.get('default')
            source = 'default'

            if value is None:
                print(f"⚠️ {repo}.{param_name} 缺少默认值")
                continue

        # 应用字段映射（统一命名）
        unified_name = apply_field_mapping(repo, param_name)
        complete_params[unified_name] = value

    return complete_params


def extract_energy_metrics(exp_data):
    """
    提取能耗指标

    Args:
        exp_data: 实验数据字典

    Returns:
        dict: 能耗指标字典
    """
    is_parallel = exp_data.get('mode') == 'parallel'

    if is_parallel:
        energy_data = exp_data.get('foreground', {}).get('energy_metrics', {})
    else:
        energy_data = exp_data.get('energy_metrics', {})

    # 提取关键能耗指标
    return {
        'energy_cpu_total_joules': energy_data.get('cpu_energy_total_joules'),
        'energy_gpu_total_joules': energy_data.get('gpu_energy_total_joules'),
        'gpu_power_avg_watts': energy_data.get('gpu_power_avg_watts'),
        'gpu_power_max_watts': energy_data.get('gpu_power_max_watts'),
        'gpu_power_min_watts': energy_data.get('gpu_power_min_watts'),
        'gpu_util_avg': energy_data.get('gpu_util_avg_percent'),
        'gpu_temp_max': energy_data.get('gpu_temp_max_celsius'),
        'gpu_temp_avg': energy_data.get('gpu_temp_avg_celsius'),
    }


def calculate_derived_energy_metrics(row):
    """
    计算派生的能耗指标（中介变量）

    Args:
        row: 数据行字典

    Returns:
        dict: 派生指标
    """
    derived = {}

    # CPU能耗占比
    cpu_energy = row.get('energy_cpu_total_joules')
    gpu_energy = row.get('energy_gpu_total_joules')
    if cpu_energy is not None and gpu_energy is not None and (cpu_energy + gpu_energy) > 0:
        derived['cpu_pkg_ratio'] = cpu_energy / (cpu_energy + gpu_energy)
    else:
        derived['cpu_pkg_ratio'] = None

    # GPU功率波动性
    max_watts = row.get('gpu_power_max_watts')
    min_watts = row.get('gpu_power_min_watts')
    if max_watts is not None and min_watts is not None:
        derived['gpu_power_fluctuation'] = max_watts - min_watts
    else:
        derived['gpu_power_fluctuation'] = None

    # GPU温度波动性
    max_temp = row.get('gpu_temp_max')
    avg_temp = row.get('gpu_temp_avg')
    if max_temp is not None and avg_temp is not None:
        derived['gpu_temp_fluctuation'] = max_temp - avg_temp
    else:
        derived['gpu_temp_fluctuation'] = None

    return derived


def extract_performance_metrics(exp_data, repo):
    """
    提取性能指标（不同任务有不同指标）

    Args:
        exp_data: 实验数据字典
        repo: 仓库名

    Returns:
        dict: 性能指标字典
    """
    is_parallel = exp_data.get('mode') == 'parallel'

    if is_parallel:
        perf_data = exp_data.get('foreground', {}).get('performance_metrics', {})
    else:
        perf_data = exp_data.get('performance_metrics', {})

    # 根据不同任务提取不同的性能指标
    if 'examples' in repo or 'pytorch_resnet_cifar10' in repo:
        # 图像分类任务
        return {
            'perf_test_accuracy': perf_data.get('test_accuracy'),
        }
    elif 'Person_reID' in repo:
        # Person ReID任务
        return {
            'perf_map': perf_data.get('map'),
            'perf_rank1': perf_data.get('rank1'),
            'perf_rank5': perf_data.get('rank5'),
        }
    elif 'VulBERTa' in repo:
        # 漏洞检测任务
        return {
            'perf_eval_loss': perf_data.get('eval_loss'),
        }
    elif 'bug-localization' in repo:
        # Bug定位任务
        return {
            'perf_top1_accuracy': perf_data.get('top1_accuracy'),
            'perf_top5_accuracy': perf_data.get('top5_accuracy'),
        }
    elif 'MRT-OAST' in repo:
        # MRT-OAST任务
        return {
            'perf_accuracy': perf_data.get('accuracy'),
            'perf_precision': perf_data.get('precision'),
            'perf_recall': perf_data.get('recall'),
        }
    else:
        # 未知任务，返回空
        return {}


def load_all_experiments(results_dir):
    """
    遍历所有session目录，加载experiment.json

    Args:
        results_dir: 结果目录路径

    Returns:
        list: 实验数据列表
    """
    experiments = []

    print("=" * 80)
    print("遍历实验JSON文件")
    print("=" * 80)

    session_dirs = sorted(results_dir.glob('run_*'))
    print(f"找到 {len(session_dirs)} 个session目录")

    for session_dir in session_dirs:
        if not session_dir.is_dir():
            continue

        for exp_dir in session_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name in ['__pycache__', '.git']:
                continue

            json_file = exp_dir / 'experiment.json'
            if not json_file.exists():
                continue

            try:
                with open(json_file) as f:
                    data = json.load(f)
                experiments.append(data)
            except Exception as e:
                print(f"  ⚠️ 跳过 {json_file}: {e}")

    print(f"✅ 成功加载 {len(experiments)} 个实验JSON文件")
    return experiments


def experiments_to_dataframe(experiments, models_config):
    """
    将实验列表转换为DataFrame，包含完整超参数

    Args:
        experiments: 实验JSON列表
        models_config: models_config.json配置

    Returns:
        pd.DataFrame: 包含所有实验的DataFrame（超参数完整）
    """
    print("\n" + "=" * 80)
    print("转换实验数据为DataFrame")
    print("=" * 80)

    rows = []
    stats = {
        'total': 0,
        'parallel': 0,
        'non_parallel': 0,
        'hyperparams_from_default': defaultdict(int),
        'hyperparams_from_recorded': defaultdict(int),
    }

    for exp in experiments:
        stats['total'] += 1
        is_parallel = exp.get('mode') == 'parallel'

        if is_parallel:
            stats['parallel'] += 1
            fg = exp.get('foreground', {})
            repo = fg.get('repository')
            model = fg.get('model')
        else:
            stats['non_parallel'] += 1
            repo = exp.get('repository')
            model = exp.get('model')

        # 提取完整超参数（核心功能）
        complete_params = extract_complete_hyperparams(exp, models_config)

        # 提取能耗指标
        energy_metrics = extract_energy_metrics(exp)

        # 提取性能指标
        perf_metrics = extract_performance_metrics(exp, repo)

        # 构建行数据
        row = {
            'experiment_id': exp['experiment_id'],
            'timestamp': exp['timestamp'],
            'repository': repo,
            'model': model,
            'mode': exp.get('mode', 'default'),
            'is_parallel': 1 if is_parallel else 0,
            **complete_params,  # ⭐ 完整超参数（记录值 + 默认值）
            **energy_metrics,
            **perf_metrics,
        }

        # 计算派生指标
        derived_metrics = calculate_derived_energy_metrics(row)
        row.update(derived_metrics)

        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"✅ 转换完成: {len(df)} 行数据")
    print(f"  - 并行模式: {stats['parallel']} 个")
    print(f"  - 非并行模式: {stats['non_parallel']} 个")

    return df


def handle_missing_values(df):
    """
    处理缺失值（v2.0：主要删除行，超参数应该已完整）

    策略:
    1. 删除性能指标全缺失的行
    2. 删除能耗指标全缺失的行
    3. 验证超参数完整性（理论上应该接近100%）

    Args:
        df: 原始DataFrame

    Returns:
        pd.DataFrame: 清理后的DataFrame
    """
    print("\n" + "=" * 80)
    print("处理缺失值")
    print("=" * 80)

    initial_rows = len(df)

    # 1. 删除性能指标全缺失的行
    perf_cols = [c for c in df.columns if c.startswith('perf_')]
    if perf_cols:
        df_clean = df[df[perf_cols].notna().any(axis=1)].copy()
        deleted_perf = initial_rows - len(df_clean)
        print(f"  删除性能全缺失: {deleted_perf} 行 ({deleted_perf/initial_rows*100:.2f}%)")
    else:
        df_clean = df.copy()
        print("  无性能指标列")

    # 2. 删除能耗指标全缺失的行
    energy_cols = [c for c in df_clean.columns if c.startswith('energy_')]
    if energy_cols:
        df_clean = df_clean[df_clean[energy_cols].notna().any(axis=1)].copy()
        deleted_energy = len(df) - len(df_clean)
        print(f"  删除能耗全缺失: {deleted_energy} 行 ({deleted_energy/len(df)*100:.2f}%)")
    else:
        print("  无能耗指标列")

    # 3. 验证超参数完整性
    hyperparam_cols = [c for c in df_clean.columns if
                       c.startswith('hyperparam_') or
                       c in ['training_duration', 'l2_regularization', 'seed']]

    print(f"\n超参数完整性验证 (共 {len(hyperparam_cols)} 列):")
    missing_counts = df_clean[hyperparam_cols].isnull().sum()

    any_missing = False
    for col in sorted(hyperparam_cols):
        count = missing_counts[col]
        if count > 0:
            any_missing = True
            print(f"  ⚠️ {col}: {count} 行缺失 ({count/len(df_clean)*100:.2f}%)")
        else:
            print(f"  ✅ {col}: 完整 (0 缺失)")

    if not any_missing:
        print("\n✅ 所有超参数列都完整（0缺失）！")
    else:
        print(f"\n⚠️ 仍有 {missing_counts.sum()} 个超参数缺失值")

    # 4. 检查派生指标缺失情况
    derived_cols = ['cpu_pkg_ratio', 'gpu_power_fluctuation', 'gpu_temp_fluctuation']
    existing_derived = [c for c in derived_cols if c in df_clean.columns]

    if existing_derived:
        print(f"\n派生指标完整性验证 (共 {len(existing_derived)} 列):")
        for col in existing_derived:
            count = df_clean[col].isnull().sum()
            print(f"  {col}: {count} 行缺失 ({count/len(df_clean)*100:.2f}%)")

    # 5. 总体统计
    total_deleted = initial_rows - len(df_clean)
    print(f"\n总计删除: {total_deleted} 行 ({total_deleted/initial_rows*100:.2f}%)")
    print(f"保留: {len(df_clean)} 行 ({len(df_clean)/initial_rows*100:.2f}%)")

    return df_clean


def analyze_data_quality(df):
    """
    分析数据质量

    Args:
        df: DataFrame

    Returns:
        dict: 质量统计
    """
    print("\n" + "=" * 80)
    print("数据质量分析")
    print("=" * 80)

    stats = {}

    # 1. 基础统计
    print(f"\n基础统计:")
    print(f"  总行数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")

    # 2. 缺失值统计
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_rate = missing_cells / total_cells * 100

    print(f"\n缺失值统计:")
    print(f"  总单元格数: {total_cells}")
    print(f"  缺失单元格数: {missing_cells}")
    print(f"  总体缺失率: {missing_rate:.2f}%")

    stats['total_missing_rate'] = missing_rate

    # 3. 完全无缺失行统计
    complete_rows = df.dropna().shape[0]
    complete_rate = complete_rows / len(df) * 100

    print(f"\n完全无缺失行:")
    print(f"  完全无缺失行数: {complete_rows}")
    print(f"  占比: {complete_rate:.2f}%")

    stats['complete_rows'] = complete_rows
    stats['complete_rate'] = complete_rate

    # 4. 按列类型统计缺失
    print(f"\n按列类型缺失统计:")

    col_groups = {
        '超参数': [c for c in df.columns if c.startswith('hyperparam_') or c in ['training_duration', 'l2_regularization', 'seed']],
        '能耗指标': [c for c in df.columns if c.startswith('energy_')],
        '性能指标': [c for c in df.columns if c.startswith('perf_')],
        '派生指标': [c for c in df.columns if c in ['cpu_pkg_ratio', 'gpu_power_fluctuation', 'gpu_temp_fluctuation']],
    }

    for group_name, cols in col_groups.items():
        if not cols:
            continue
        group_missing = df[cols].isnull().sum().sum()
        group_total = len(df) * len(cols)
        group_rate = group_missing / group_total * 100 if group_total > 0 else 0
        print(f"  {group_name}: {group_missing}/{group_total} ({group_rate:.2f}%)")

    # 5. 按模式统计
    print(f"\n按模式统计:")
    if 'is_parallel' in df.columns:
        parallel_count = df[df['is_parallel'] == 1].shape[0]
        non_parallel_count = df[df['is_parallel'] == 0].shape[0]
        print(f"  并行模式: {parallel_count} 行")
        print(f"  非并行模式: {non_parallel_count} 行")

    # 6. 按仓库统计
    print(f"\n按仓库统计:")
    if 'repository' in df.columns:
        repo_counts = df['repository'].value_counts()
        for repo, count in repo_counts.items():
            print(f"  {repo}: {count} 行")

    return stats


def save_dataframe(df, output_path):
    """
    保存DataFrame到CSV

    Args:
        df: DataFrame
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n✅ 数据已保存到: {output_path}")
    print(f"  - 行数: {len(df)}")
    print(f"  - 列数: {len(df.columns)}")


def main():
    """主函数"""
    print("=" * 80)
    print("能耗因果分析 - 数据重新提取 v2.0")
    print("=" * 80)
    print("方法: experiment.json + models_config.json 联合提取")
    print("=" * 80)

    # 1. 加载models_config.json
    print("\n阶段1: 加载配置")
    models_config = load_models_config()
    print(f"✅ 成功加载 models_config.json")
    print(f"  - 仓库数: {len(models_config['models'])}")

    # 2. 加载所有实验JSON
    print("\n阶段2: 加载实验数据")
    # 从analysis/scripts/运行，需要../../results
    results_dir = Path('../../results')
    if not results_dir.exists():
        results_dir = Path('../results')
    experiments = load_all_experiments(results_dir)

    # 3. 转换为DataFrame
    print("\n阶段3: 提取完整数据")
    df = experiments_to_dataframe(experiments, models_config)

    # 4. 处理缺失值
    print("\n阶段4: 处理缺失值")
    df_clean = handle_missing_values(df)

    # 5. 分析数据质量
    print("\n阶段5: 数据质量分析")
    stats = analyze_data_quality(df_clean)

    # 6. 保存完整数据
    print("\n阶段6: 保存数据")
    output_path = 'data/energy_research/raw/energy_data_extracted_v2.csv'
    save_dataframe(df_clean, output_path)

    # 7. 生成质量报告
    print("\n" + "=" * 80)
    print("数据提取完成总结")
    print("=" * 80)
    print(f"✅ 原始实验数: {len(experiments)}")
    print(f"✅ 有效数据行: {len(df_clean)}")
    print(f"✅ 总体缺失率: {stats.get('total_missing_rate', 0):.2f}%")
    print(f"✅ 完全无缺失行: {stats.get('complete_rows', 0)} ({stats.get('complete_rate', 0):.2f}%)")
    print(f"✅ 输出文件: {output_path}")

    # 8. 保存列名清单（用于阶段3分层分析）
    columns_info = {
        'all_columns': list(df_clean.columns),
        'hyperparam_columns': [c for c in df_clean.columns if c.startswith('hyperparam_') or c in ['training_duration', 'l2_regularization', 'seed']],
        'energy_columns': [c for c in df_clean.columns if c.startswith('energy_')],
        'performance_columns': [c for c in df_clean.columns if c.startswith('perf_')],
        'derived_columns': [c for c in df_clean.columns if c in ['cpu_pkg_ratio', 'gpu_power_fluctuation', 'gpu_temp_fluctuation']],
    }

    columns_file = Path('data/energy_research/raw/extracted_columns_info.json')
    with open(columns_file, 'w', encoding='utf-8') as f:
        json.dump(columns_info, f, indent=2, ensure_ascii=False)

    print(f"✅ 列信息已保存: {columns_file}")

    return df_clean


if __name__ == '__main__':
    df = main()
    print("\n✅ 阶段2完成！")
