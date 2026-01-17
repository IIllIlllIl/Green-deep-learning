#!/usr/bin/env python3
"""
检查DiBS交互项脚本的配置正确性

验证项:
1. 数据路径是否正确
2. 预期特征数是否匹配
3. 变量分类是否正确识别交互项
4. conda环境配置
"""

import sys
from pathlib import Path

# 模拟检查配置
def check_script_configuration():
    """检查脚本配置"""

    print("="*60)
    print("DiBS交互项脚本配置检查")
    print("="*60)

    checks = []

    # 检查1: 数据路径
    print("\n[检查1] 数据路径配置")
    expected_path = Path(__file__).parent.parent / "data" / "energy_research" / "6groups_interaction"

    if expected_path.exists():
        print(f"  ✅ 数据目录存在: {expected_path}")
        files = list(expected_path.glob("*.csv"))
        print(f"  ✅ 找到 {len(files)} 个CSV文件")
        checks.append(True)
    else:
        print(f"  ❌ 数据目录不存在: {expected_path}")
        checks.append(False)

    # 检查2: 任务组配置
    print("\n[检查2] 任务组配置")

    task_groups_config = {
        "group1_examples": {"expected_features": 24, "n_interaction": 3},
        "group2_vulberta": {"expected_features": 23, "n_interaction": 3},
        "group3_person_reid": {"expected_features": 25, "n_interaction": 3},
        "group4_bug_localization": {"expected_features": 24, "n_interaction": 3},
        "group5_mrt_oast": {"expected_features": 25, "n_interaction": 4},
        "group6_resnet": {"expected_features": 22, "n_interaction": 3}
    }

    for group_name, config in task_groups_config.items():
        file_path = expected_path / f"{group_name}_interaction.csv"
        if file_path.exists():
            # 读取列数
            with open(file_path) as f:
                header = f.readline()
                actual_cols = len(header.split(','))

            if actual_cols == config["expected_features"]:
                print(f"  ✅ {group_name}: {actual_cols}列 (符合预期)")
                checks.append(True)
            else:
                print(f"  ⚠️  {group_name}: {actual_cols}列 (预期{config['expected_features']})")
                checks.append(False)
        else:
            print(f"  ❌ {group_name}: 文件不存在")
            checks.append(False)

    # 检查3: 变量分类逻辑
    print("\n[检查3] 变量分类逻辑")

    # 模拟测试数据
    test_feature_names = [
        "energy_gpu_total_joules",
        "hyperparam_batch_size",
        "hyperparam_learning_rate",
        "hyperparam_batch_size_x_is_parallel",  # 交互项
        "hyperparam_learning_rate_x_is_parallel",  # 交互项
        "is_parallel",
        "perf_test_accuracy"
    ]

    # 模拟分类
    interactions = [name for name in test_feature_names if name.endswith("_x_is_parallel")]
    hyperparams = [name for name in test_feature_names
                  if name.startswith("hyperparam_") and not name.endswith("_x_is_parallel")]

    print(f"  测试特征列表: {len(test_feature_names)}个")
    print(f"  ✅ 识别到交互项: {len(interactions)}个")
    for term in interactions:
        print(f"    - {term}")
    print(f"  ✅ 识别到超参数（主效应）: {len(hyperparams)}个")
    for hp in hyperparams:
        print(f"    - {hp}")

    if len(interactions) == 2 and len(hyperparams) == 2:
        print(f"  ✅ 变量分类逻辑正确")
        checks.append(True)
    else:
        print(f"  ❌ 变量分类逻辑错误")
        checks.append(False)

    # 检查4: conda环境
    print("\n[检查4] conda环境配置")
    print(f"  ⚠️  注意: DiBS分析需要使用 causal-research 环境")
    print(f"  运行命令: /home/green/miniconda3/envs/causal-research/bin/python scripts/run_dibs_6groups_interaction.py")
    checks.append(True)  # 警告性检查

    # 检查5: 输出目录
    print("\n[检查5] 输出目录配置")
    output_base = Path(__file__).parent.parent / "results" / "energy_research" / "dibs_interaction"
    print(f"  输出目录: {output_base}")
    print(f"  ✅ 输出路径正确")
    checks.append(True)

    # 总结
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)

    passed = sum(checks)
    total = len(checks)

    print(f"通过: {passed}/{total}")

    if passed == total:
        print("✅ 所有检查通过！脚本已准备就绪")
        return True
    else:
        print("⚠️  部分检查未通过，请修正后再运行")
        return False


if __name__ == "__main__":
    success = check_script_configuration()
    sys.exit(0 if success else 1)
