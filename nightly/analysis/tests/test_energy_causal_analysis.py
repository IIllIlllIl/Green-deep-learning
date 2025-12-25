"""
能耗因果分析脚本测试 - 验证修改正确性
测试范围:
1. 数据文件完整性
2. 模块导入
3. DiBS配置一致性
4. 数据格式验证
5. 小规模快速运行测试
"""
import sys
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def print_test_header(test_name):
    """打印测试标题"""
    print(f"\n{'='*70}")
    print(f"  测试: {test_name}")
    print(f"{'='*70}")

def print_result(passed, message):
    """打印测试结果"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {message}")
    return passed

# ============================================================================
# 测试1: 数据文件完整性
# ============================================================================
print_test_header("数据文件完整性检查")

TASK_GROUPS = [
    {
        'name': 'image_classification',
        'display_name': '图像分类',
        'data_file': 'data/energy_research/training/training_data_image_classification.csv',
        'expected_samples': 258,
        'expected_features': 13
    },
    {
        'name': 'person_reid',
        'display_name': 'Person_reID',
        'data_file': 'data/energy_research/training/training_data_person_reid.csv',
        'expected_samples': 116,
        'expected_features': 16
    },
    {
        'name': 'vulberta',
        'display_name': 'VulBERTa',
        'data_file': 'data/energy_research/training/training_data_vulberta.csv',
        'expected_samples': 142,
        'expected_features': 10
    },
    {
        'name': 'bug_localization',
        'display_name': 'Bug定位',
        'data_file': 'data/energy_research/training/training_data_bug_localization.csv',
        'expected_samples': 132,
        'expected_features': 11
    }
]

all_files_valid = True
for task in TASK_GROUPS:
    file_path = task['data_file']

    # 检查文件存在
    if not os.path.exists(file_path):
        all_files_valid = print_result(False, f"{task['display_name']}: 文件不存在 - {file_path}") and all_files_valid
        continue

    try:
        # 加载数据
        df = pd.read_csv(file_path)

        # 检查样本数
        if len(df) != task['expected_samples']:
            all_files_valid = print_result(False,
                f"{task['display_name']}: 样本数不匹配 (期望{task['expected_samples']}, 实际{len(df)})") and all_files_valid
            continue

        # 检查特征数
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) != task['expected_features']:
            all_files_valid = print_result(False,
                f"{task['display_name']}: 特征数不匹配 (期望{task['expected_features']}, 实际{len(numeric_cols)})") and all_files_valid
            continue

        # 检查数据类型
        if df[numeric_cols].select_dtypes(include=[np.number]).shape[1] != len(numeric_cols):
            all_files_valid = print_result(False,
                f"{task['display_name']}: 存在非数值列") and all_files_valid
            continue

        print_result(True, f"{task['display_name']}: {len(df)}样本 × {len(numeric_cols)}特征")

    except Exception as e:
        all_files_valid = print_result(False, f"{task['display_name']}: 加载失败 - {e}") and all_files_valid

# ============================================================================
# 测试2: 模块导入
# ============================================================================
print_test_header("必要模块导入检查")

import_success = True

try:
    from utils.causal_discovery import CausalGraphLearner
    print_result(True, "CausalGraphLearner导入成功")
except Exception as e:
    import_success = print_result(False, f"CausalGraphLearner导入失败: {e}") and import_success

try:
    from utils.causal_inference import CausalInferenceEngine
    print_result(True, "CausalInferenceEngine导入成功")
except Exception as e:
    import_success = print_result(False, f"CausalInferenceEngine导入失败: {e}") and import_success

# ============================================================================
# 测试3: DiBS配置一致性（与Adult分析对比）
# ============================================================================
print_test_header("DiBS配置一致性检查")

# Adult分析的配置
ADULT_CONFIG = {
    'n_steps': 3000,
    'alpha': 0.1,
    'threshold': 0.3,
    'random_seed': 42
}

# 从能耗分析脚本读取配置
config_file = 'scripts/demos/demo_energy_task_specific.py'
config_match = True

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        content = f.read()

        # 检查n_steps
        if 'DIBS_N_STEPS = 3000' in content:
            print_result(True, "n_steps = 3000 (与Adult分析一致)")
        else:
            config_match = print_result(False, "n_steps不匹配") and config_match

        # 检查alpha
        if 'DIBS_ALPHA = 0.1' in content:
            print_result(True, "alpha = 0.1 (与Adult分析一致)")
        else:
            config_match = print_result(False, "alpha不匹配") and config_match

        # 检查threshold
        if 'DIBS_THRESHOLD = 0.3' in content:
            print_result(True, "threshold = 0.3 (与Adult分析一致)")
        else:
            config_match = print_result(False, "threshold不匹配") and config_match

        # 检查random_seed
        if 'DIBS_RANDOM_SEED = 42' in content:
            print_result(True, "random_seed = 42 (与Adult分析一致)")
        else:
            config_match = print_result(False, "random_seed不匹配") and config_match
else:
    config_match = print_result(False, f"配置文件不存在: {config_file}") and config_match

# ============================================================================
# 测试4: 数据格式验证
# ============================================================================
print_test_header("数据格式验证")

format_valid = True

for task in TASK_GROUPS:
    file_path = task['data_file']

    if not os.path.exists(file_path):
        continue

    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        causal_data = df[numeric_cols].copy()

        # 检查样本量是否充足（DiBS推荐至少10个）
        if len(causal_data) < 10:
            format_valid = print_result(False,
                f"{task['display_name']}: 样本量不足 ({len(causal_data)} < 10)") and format_valid
            continue

        # 检查是否有全为NaN的列
        all_nan_cols = causal_data.columns[causal_data.isna().all()].tolist()
        if all_nan_cols:
            print(f"  ⚠️  {task['display_name']}: 发现全NaN列 - {all_nan_cols} (将被自动移除)")

        # 检查缺失率
        missing_rate = causal_data.isna().sum().sum() / (len(causal_data) * len(numeric_cols))
        if missing_rate > 0.5:
            format_valid = print_result(False,
                f"{task['display_name']}: 缺失率过高 ({missing_rate*100:.1f}% > 50%)") and format_valid
            continue

        # 检查方差（DiBS需要有变化的变量）
        zero_var_cols = causal_data.columns[causal_data.var() == 0].tolist()
        if zero_var_cols:
            print(f"  ⚠️  {task['display_name']}: 零方差列 - {zero_var_cols} (可能影响DiBS)")

        print_result(True,
            f"{task['display_name']}: 格式有效 (样本={len(causal_data)}, 缺失率={missing_rate*100:.1f}%)")

    except Exception as e:
        format_valid = print_result(False, f"{task['display_name']}: 验证失败 - {e}") and format_valid

# ============================================================================
# 测试5: 输出目录创建
# ============================================================================
print_test_header("输出目录检查")

dirs_valid = True

required_dirs = [
    'results/energy_research/task_specific',
    'logs/energy_research/experiments'
]

for dir_path in required_dirs:
    try:
        os.makedirs(dir_path, exist_ok=True)
        print_result(True, f"目录可创建: {dir_path}")
    except Exception as e:
        dirs_valid = print_result(False, f"目录创建失败: {dir_path} - {e}") and dirs_valid

# ============================================================================
# 测试6: 小规模快速运行测试（使用最小任务组）
# ============================================================================
print_test_header("小规模快速运行测试")

# 选择样本量最小的任务组进行快速测试
test_task = TASK_GROUPS[1]  # Person_reID (116样本)
print(f"使用 {test_task['display_name']} 进行快速测试...")
print(f"配置: 10样本, 3特征, 100步DiBS (极速模式)")

quick_test_success = True

try:
    # 加载数据
    df = pd.read_csv(test_task['data_file'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    causal_data = df[numeric_cols].copy()

    # 移除全NaN列
    causal_data = causal_data.dropna(axis=1, how='all')
    numeric_cols = causal_data.columns.tolist()

    # 使用极小子集（10样本，前3个特征）
    test_data = causal_data.iloc[:10, :3].copy()
    test_cols = test_data.columns.tolist()

    print(f"  测试数据: {len(test_data)}样本 × {len(test_cols)}特征")

    # 测试DiBS
    print(f"  开始DiBS测试（100步，预计<30秒）...")
    from utils.causal_discovery import CausalGraphLearner

    learner = CausalGraphLearner(
        n_vars=len(test_cols),
        n_steps=100,  # 极小步数用于快速测试
        alpha=0.1,
        random_seed=42
    )

    import time
    start = time.time()
    causal_graph = learner.fit(test_data, verbose=False)
    dibs_time = time.time() - start

    print_result(True, f"DiBS完成 (耗时: {dibs_time:.1f}秒, 图形状: {causal_graph.shape})")

    # 测试边提取
    edges = learner.get_edges(threshold=0.3)
    print_result(True, f"边提取成功 (检测到 {len(edges)} 条因果边)")

    # 如果有边，测试DML
    if len(edges) > 0:
        print(f"  开始DML测试（分析 {len(edges)} 条边）...")
        from utils.causal_inference import CausalInferenceEngine

        engine = CausalInferenceEngine(verbose=False)

        start = time.time()
        causal_effects = engine.analyze_all_edges(
            data=test_data,
            causal_graph=causal_graph,
            var_names=test_cols,
            threshold=0.3
        )
        dml_time = time.time() - start

        print_result(True, f"DML完成 (耗时: {dml_time:.1f}秒, 分析了 {len(causal_effects)} 条边)")

        # 检查显著效应
        significant = engine.get_significant_effects()
        print(f"  统计显著的因果效应: {len(significant)}/{len(causal_effects)}")
    else:
        print(f"  ⚠️  未检测到因果边（阈值0.3），跳过DML测试")
        print(f"     这是正常的（10样本的极小数据集）")

    print_result(True, "小规模运行测试完成 - 核心流程验证成功")

except Exception as e:
    quick_test_success = print_result(False, f"快速运行测试失败: {e}") and quick_test_success
    import traceback
    traceback.print_exc()

# ============================================================================
# 测试7: 脚本语法检查
# ============================================================================
print_test_header("脚本语法检查")

syntax_valid = True

script_file = 'scripts/demos/demo_energy_task_specific.py'
if os.path.exists(script_file):
    try:
        with open(script_file, 'r') as f:
            code = f.read()
        compile(code, script_file, 'exec')
        print_result(True, f"脚本语法正确: {script_file}")
    except SyntaxError as e:
        syntax_valid = print_result(False, f"脚本语法错误: {e}") and syntax_valid
else:
    syntax_valid = print_result(False, f"脚本文件不存在: {script_file}") and syntax_valid

bash_script = 'scripts/experiments/run_energy_causal_analysis.sh'
if os.path.exists(bash_script):
    print_result(True, f"Bash脚本存在: {bash_script}")
    # 检查执行权限
    if os.access(bash_script, os.X_OK):
        print_result(True, f"Bash脚本有执行权限")
    else:
        print_result(False, f"Bash脚本缺少执行权限 - 运行: chmod +x {bash_script}")
else:
    syntax_valid = print_result(False, f"Bash脚本不存在: {bash_script}") and syntax_valid

# ============================================================================
# 总结
# ============================================================================
print(f"\n{'='*70}")
print("  测试总结")
print(f"{'='*70}")

all_tests = [
    ("数据文件完整性", all_files_valid),
    ("模块导入", import_success),
    ("DiBS配置一致性", config_match),
    ("数据格式验证", format_valid),
    ("输出目录创建", dirs_valid),
    ("小规模快速运行", quick_test_success),
    ("脚本语法检查", syntax_valid)
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

for test_name, result in all_tests:
    status = "✅" if result else "❌"
    print(f"{status} {test_name}")

print(f"\n通过率: {passed}/{total} ({passed/total*100:.0f}%)")

if passed == total:
    print("\n🎉 所有测试通过！可以安全运行Stage 8分析。")
    print("\n运行命令:")
    print("  cd /home/green/energy_dl/nightly/analysis")
    print("  screen -S energy_dibs bash scripts/experiments/run_energy_causal_analysis.sh")
    sys.exit(0)
else:
    print(f"\n⚠️  {total - passed} 个测试失败，请修复后再运行。")
    sys.exit(1)
