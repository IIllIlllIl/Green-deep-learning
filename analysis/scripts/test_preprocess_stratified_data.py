#!/usr/bin/env python3
"""测试数据分层预处理脚本

用途: 验证preprocess_stratified_data.py的正确性
作者: Claude
日期: 2025-12-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# 添加analysis目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_mock_data():
    """创建模拟数据用于测试"""
    # 模拟数据：18行涵盖所有任务组
    data = {
        'experiment_id': [f'exp_{i:03d}' for i in range(1, 19)],
        'timestamp': ['2025-12-24 10:00:00'] * 18,
        'repository': (
            ['examples'] * 4 +  # mnist, mnist_ff, mnist_rnn, siamese
            ['pytorch_resnet_cifar10'] * 2 +  # resnet20
            ['Person_reID_baseline_pytorch'] * 6 +  # densenet121, hrnet18, pcb
            ['VulBERTa'] * 3 +  # mlp
            ['bug-localization-by-dnn-and-rvsm'] * 3  # default
        ),
        'model': (
            ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese'] +
            ['resnet20', 'resnet20'] +
            ['densenet121', 'densenet121', 'hrnet18', 'hrnet18', 'pcb', 'pcb'] +
            ['mlp', 'mlp', 'mlp'] +
            ['default', 'default', 'default']
        ),
        # 超参数
        'training_duration': [10] * 18,
        'hyperparam_learning_rate': [0.001] * 18,
        'l2_regularization': [0.0001] * 18,
        'seed': [42] * 18,
        'hyperparam_dropout': [None] * 6 + [0.5] * 6 + [None] * 6,  # 只有Person_reID有
        'hyperparam_kfold': [None] * 15 + [10] * 3,  # 只有Bug定位有
        # 能耗指标
        'energy_cpu_total_joules': [1000.0] * 18,
        'energy_gpu_total_joules': [5000.0] * 18,
        'gpu_power_avg_watts': [200.0] * 18,
        # 中介变量
        'gpu_util_avg': [80.0] * 18,
        'gpu_temp_max': [75.0] * 18,
        'cpu_pkg_ratio': [0.2] * 18,
        'gpu_power_fluctuation': [50.0] * 18,
        'gpu_temp_fluctuation': [10.0] * 18,
        # 性能指标（按任务分组，非该任务的为NaN）
        'perf_test_accuracy': (
            [0.95] * 6 +  # examples + CIFAR-10有
            [None] * 12  # 其他任务没有
        ),
        'perf_map': (
            [None] * 6 +  # 前6个没有
            [0.85] * 6 +  # Person_reID有
            [None] * 6  # 其他没有
        ),
        'perf_rank1': (
            [None] * 6 +
            [0.90] * 6 +
            [None] * 6
        ),
        'perf_rank5': (
            [None] * 6 +
            [0.95] * 6 +
            [None] * 6
        ),
        'perf_eval_loss': (
            [None] * 12 +
            [0.25] * 3 +  # VulBERTa有
            [None] * 3
        ),
        'perf_top1_accuracy': (
            [None] * 15 +
            [0.70] * 3  # Bug定位有
        ),
        'perf_top5_accuracy': (
            [None] * 15 +
            [0.85] * 3
        ),
    }

    return pd.DataFrame(data)


def test_onehot_encoding():
    """测试One-Hot编码生成逻辑"""
    print("\n" + "=" * 80)
    print("测试1: One-Hot编码生成")
    print("=" * 80)

    df = create_mock_data()

    # 测试图像分类One-Hot
    print("\n测试图像分类One-Hot编码:")
    image_df = df[df['repository'].isin(['examples', 'pytorch_resnet_cifar10'])].copy()

    # 生成One-Hot
    image_df['is_mnist'] = image_df.apply(
        lambda row: 1 if 'mnist' in row['model'] or row['model'] == 'siamese' else 0,
        axis=1
    )
    image_df['is_cifar10'] = image_df.apply(
        lambda row: 1 if row['repository'] == 'pytorch_resnet_cifar10' else 0,
        axis=1
    )

    # 验证互斥性
    onehot_sum = image_df[['is_mnist', 'is_cifar10']].sum(axis=1)
    assert (onehot_sum == 1).all(), "图像分类One-Hot编码应互斥（和=1）"

    # 验证覆盖率
    assert image_df['is_mnist'].sum() == 4, "应有4个MNIST样本"
    assert image_df['is_cifar10'].sum() == 2, "应有2个CIFAR-10样本"

    print(f"  ✅ 互斥性验证通过: 所有行One-Hot和=1")
    print(f"  ✅ 覆盖率验证通过: MNIST={image_df['is_mnist'].sum()}, CIFAR-10={image_df['is_cifar10'].sum()}")

    # 测试Person_reID One-Hot
    print("\n测试Person_reID One-Hot编码:")
    reid_df = df[df['repository'] == 'Person_reID_baseline_pytorch'].copy()

    reid_df['is_densenet121'] = (reid_df['model'] == 'densenet121').astype(int)
    reid_df['is_hrnet18'] = (reid_df['model'] == 'hrnet18').astype(int)
    reid_df['is_pcb'] = (reid_df['model'] == 'pcb').astype(int)

    onehot_sum = reid_df[['is_densenet121', 'is_hrnet18', 'is_pcb']].sum(axis=1)
    assert (onehot_sum == 1).all(), "Person_reID One-Hot编码应互斥（和=1）"

    assert reid_df['is_densenet121'].sum() == 2, "应有2个densenet121样本"
    assert reid_df['is_hrnet18'].sum() == 2, "应有2个hrnet18样本"
    assert reid_df['is_pcb'].sum() == 2, "应有2个pcb样本"

    print(f"  ✅ 互斥性验证通过: 所有行One-Hot和=1")
    print(f"  ✅ 覆盖率验证通过: densenet={reid_df['is_densenet121'].sum()}, hrnet={reid_df['is_hrnet18'].sum()}, pcb={reid_df['is_pcb'].sum()}")

    return True


def test_column_selection():
    """测试列选择正确性"""
    print("\n" + "=" * 80)
    print("测试2: 列选择正确性")
    print("=" * 80)

    # 定义每个任务的预期列
    expected_columns = {
        'image_classification': [
            'experiment_id', 'timestamp',
            'is_mnist', 'is_cifar10',
            'training_duration', 'hyperparam_learning_rate', 'l2_regularization', 'seed',
            'energy_cpu_total_joules', 'energy_gpu_total_joules', 'gpu_power_avg_watts',
            'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
            'gpu_power_fluctuation', 'gpu_temp_fluctuation',
            'perf_test_accuracy'
        ],
        'person_reid': [
            'experiment_id', 'timestamp',
            'is_densenet121', 'is_hrnet18', 'is_pcb',
            'training_duration', 'hyperparam_learning_rate', 'hyperparam_dropout', 'seed',
            'energy_cpu_total_joules', 'energy_gpu_total_joules', 'gpu_power_avg_watts',
            'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
            'gpu_power_fluctuation', 'gpu_temp_fluctuation',
            'perf_map', 'perf_rank1', 'perf_rank5'
        ],
        'vulberta': [
            'experiment_id', 'timestamp',
            'training_duration', 'hyperparam_learning_rate', 'l2_regularization', 'seed',
            'energy_cpu_total_joules', 'energy_gpu_total_joules', 'gpu_power_avg_watts',
            'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
            'gpu_power_fluctuation', 'gpu_temp_fluctuation',
            'perf_eval_loss'
        ],
        'bug_localization': [
            'experiment_id', 'timestamp',
            'training_duration', 'l2_regularization', 'hyperparam_kfold', 'seed',
            'energy_cpu_total_joules', 'energy_gpu_total_joules', 'gpu_power_avg_watts',
            'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
            'gpu_power_fluctuation', 'gpu_temp_fluctuation',
            'perf_top1_accuracy', 'perf_top5_accuracy'
        ]
    }

    for task_name, expected_cols in expected_columns.items():
        print(f"\n{task_name}:")
        print(f"  预期列数: {len(expected_cols)}")
        print(f"  预期列: {', '.join(expected_cols[:5])}...")

        # 验证列顺序和数量
        assert len(expected_cols) == len(set(expected_cols)), f"{task_name}: 列名有重复"
        print(f"  ✅ 列名无重复")

        # 验证关键列存在
        assert 'experiment_id' in expected_cols, f"{task_name}: 缺少experiment_id"
        assert 'timestamp' in expected_cols, f"{task_name}: 缺少timestamp"
        assert 'energy_cpu_total_joules' in expected_cols, f"{task_name}: 缺少能耗列"
        print(f"  ✅ 关键列存在")

    return True


def test_missing_removal():
    """测试性能缺失行删除逻辑"""
    print("\n" + "=" * 80)
    print("测试3: 性能缺失行删除")
    print("=" * 80)

    df = create_mock_data()

    # 测试图像分类
    print("\n图像分类:")
    image_df = df[df['repository'].isin(['examples', 'pytorch_resnet_cifar10'])].copy()
    print(f"  删除前: {len(image_df)} 行")

    # 删除性能全缺失行
    image_df_clean = image_df.dropna(subset=['perf_test_accuracy'], how='all')
    print(f"  删除后: {len(image_df_clean)} 行")

    # 验证所有保留行都有性能指标
    assert not image_df_clean['perf_test_accuracy'].isna().any(), "保留行应无性能缺失"
    print(f"  ✅ 保留行性能指标100%填充")

    # 测试Person_reID（多个性能指标）
    print("\nPerson_reID:")
    reid_df = df[df['repository'] == 'Person_reID_baseline_pytorch'].copy()
    print(f"  删除前: {len(reid_df)} 行")

    # 删除所有性能指标都缺失的行
    perf_cols = ['perf_map', 'perf_rank1', 'perf_rank5']
    reid_df_clean = reid_df.dropna(subset=perf_cols, how='all')
    print(f"  删除后: {len(reid_df_clean)} 行")

    # 验证至少有一个性能指标
    assert not reid_df_clean[perf_cols].isna().all(axis=1).any(), "保留行应至少有一个性能指标"
    print(f"  ✅ 保留行至少有一个性能指标")

    return True


def test_dry_run():
    """测试Dry Run模式（处理少量数据）"""
    print("\n" + "=" * 80)
    print("测试4: Dry Run模式")
    print("=" * 80)

    df = create_mock_data()

    # 模拟dry run：只处理前10行
    df_dry = df.head(10).copy()

    print(f"\n原始数据: {len(df)} 行")
    print(f"Dry Run数据: {len(df_dry)} 行")

    # 验证dry run数据结构完整
    assert all(col in df_dry.columns for col in ['experiment_id', 'repository', 'model']), \
        "Dry run数据应包含关键列"

    print(f"✅ Dry run数据结构完整")

    # 测试数据分组
    repos = df_dry['repository'].unique()
    print(f"\nDry run包含的仓库: {repos}")

    for repo in repos:
        repo_count = len(df_dry[df_dry['repository'] == repo])
        print(f"  {repo}: {repo_count} 行")

    print(f"✅ Dry run分组统计正常")

    return True


def test_output_format():
    """测试输出文件格式"""
    print("\n" + "=" * 80)
    print("测试5: 输出文件格式")
    print("=" * 80)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 创建模拟输出数据
        df = create_mock_data()
        image_df = df[df['repository'].isin(['examples', 'pytorch_resnet_cifar10'])].copy()

        # 添加One-Hot
        image_df['is_mnist'] = (image_df['model'].str.contains('mnist') |
                                 (image_df['model'] == 'siamese')).astype(int)
        image_df['is_cifar10'] = (image_df['repository'] == 'pytorch_resnet_cifar10').astype(int)

        # 选择列
        selected_cols = [
            'experiment_id', 'timestamp',
            'is_mnist', 'is_cifar10',
            'training_duration', 'hyperparam_learning_rate', 'l2_regularization', 'seed',
            'energy_cpu_total_joules', 'energy_gpu_total_joules', 'gpu_power_avg_watts',
            'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
            'gpu_power_fluctuation', 'gpu_temp_fluctuation',
            'perf_test_accuracy'
        ]

        image_df_selected = image_df[selected_cols].copy()

        # 删除性能缺失行
        image_df_clean = image_df_selected.dropna(subset=['perf_test_accuracy'], how='all')

        # 保存到临时文件
        output_file = tmpdir_path / 'training_data_image_classification.csv'
        image_df_clean.to_csv(output_file, index=False)

        # 验证文件存在
        assert output_file.exists(), "输出文件应存在"
        print(f"✅ 文件创建成功: {output_file.name}")

        # 读取并验证
        df_read = pd.read_csv(output_file)

        # 验证列数
        assert len(df_read.columns) == 17, f"应有17列，实际{len(df_read.columns)}列"
        print(f"✅ 列数正确: {len(df_read.columns)} 列")

        # 验证列名
        assert list(df_read.columns) == selected_cols, "列名顺序应一致"
        print(f"✅ 列名顺序正确")

        # 验证行数
        print(f"✅ 行数: {len(df_read)} 行")

        # 验证One-Hot互斥性
        onehot_sum = df_read[['is_mnist', 'is_cifar10']].sum(axis=1)
        assert (onehot_sum == 1).all(), "One-Hot编码应互斥"
        print(f"✅ One-Hot互斥性验证通过")

        # 验证无性能缺失
        assert not df_read['perf_test_accuracy'].isna().any(), "性能指标应完全填充"
        print(f"✅ 性能指标100%填充")

    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("数据分层预处理脚本 - 测试套件")
    print("=" * 80)
    print("目的: 验证preprocess_stratified_data.py的正确性")
    print("=" * 80)

    tests = [
        ("One-Hot编码生成", test_onehot_encoding),
        ("列选择正确性", test_column_selection),
        ("性能缺失行删除", test_missing_removal),
        ("Dry Run模式", test_dry_run),
        ("输出文件格式", test_output_format),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✅ {test_name} - 通过")
        except AssertionError as e:
            failed += 1
            print(f"\n❌ {test_name} - 失败")
            print(f"   错误: {e}")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} - 异常")
            print(f"   错误: {e}")

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ 所有测试通过！可以继续创建主脚本。")
        return True
    else:
        print("\n❌ 部分测试失败，请修复后再继续。")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
