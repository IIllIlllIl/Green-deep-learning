"""
因果分析配置参数 - 能耗数据专用

基于Adult数据集实验的成功经验和能耗数据的特点
最后更新：2025-12-22
"""

# ============================================================================
# 数据预处理配置
# ============================================================================

# 归一化方案（与Adult实验一致）
NORMALIZATION_METHOD = 'StandardScaler'  # 标准化：均值0，方差1

# 缺失值处理（推荐：删除）
MISSING_VALUE_STRATEGY = 'drop'  # 'drop': 删除含缺失值的行
                                  # 'impute': 插补（不推荐）

# 重复实验处理（推荐：保留所有）
DUPLICATE_STRATEGY = 'keep_all'  # 'keep_all': 保留所有重复
                                  # 'average': 平均重复实验
                                  # 'drop_duplicates': 去重

# 超参数填充率阈值（保留填充率>10%的超参数）
HYPERPARAM_FILL_RATE_THRESHOLD = 0.10  # 10%

# ============================================================================
# DiBS因果图学习配置
# ============================================================================

# DiBS优化迭代次数
DIBS_N_STEPS = 3000  # 3000步（快速版，Adult从5000降至3000仍成功）
                     # 5000步（标准版，与Adult实验一致）
                     # 10000步（高精度版，论文可能使用）

# DiBS稀疏性系数（控制图的稀疏程度）
DIBS_ALPHA = 0.1  # 0.05: 更密集的图
                  # 0.1: 标准（Adult实验）
                  # 0.2: 更稀疏

# 随机种子（确保可复现）
RANDOM_SEED = 42

# 边筛选阈值（只保留权重≥阈值的边）
EDGE_THRESHOLD = 0.3  # 0.2: 更宽松（更多边）
                      # 0.3: 标准（Adult发现6条边）
                      # 0.5: 更严格（更少边）

# ============================================================================
# DML因果推断配置
# ============================================================================

# 基学习器类型（用于E[Y|Z]和E[X|Z]估计）
DML_BASE_LEARNER = 'RandomForest'  # 'RandomForest': 随机森林（Adult实验）
                                    # 'GradientBoosting': 梯度提升
                                    # 'Linear': 线性回归

# 随机森林超参数（与Adult一致）
DML_RF_MAX_DEPTH = 3  # 最大深度（防止过拟合）
DML_RF_N_ESTIMATORS = 100  # 树的数量

# 交叉验证折数（DML的Cross-fitting）
DML_N_FOLDS = 2  # 2折（Adult实验，样本量小时推荐）
                 # 5折（样本量大时推荐）

# 置信水平
CONFIDENCE_LEVEL = 0.95  # 95%置信区间

# 统计显著性阈值
SIGNIFICANCE_THRESHOLD = 0.05  # p < 0.05判定为统计显著

# ============================================================================
# 任务分组配置
# ============================================================================

TASK_GROUPS = {
    'image_classification': {
        'name': '图像分类',
        'repositories': ['examples', 'pytorch_resnet_cifar10'],
        'perf_metric': 'perf_test_accuracy',
        'onehot_vars': ['is_mnist', 'is_cifar10'],  # 2个One-Hot变量
        'expected_samples': 185,
    },
    'person_reid': {
        'name': 'Person_reID检索',
        'repositories': ['Person_reID_baseline_pytorch'],
        'perf_metric': 'perf_map',
        'onehot_vars': ['is_densenet121', 'is_hrnet18', 'is_pcb'],  # 3个One-Hot变量
        'expected_samples': 93,
    },
    'vulberta': {
        'name': 'VulBERTa漏洞检测',
        'repositories': ['VulBERTa'],
        'perf_metric': 'perf_eval_loss',
        'onehot_vars': [],  # 无One-Hot（单模型）
        'expected_samples': 52,
    },
    'bug_localization': {
        'name': 'Bug定位',
        'repositories': ['bug-localization-by-dnn-and-rvsm'],
        'perf_metric': 'perf_top1_accuracy',
        'onehot_vars': [],  # 无One-Hot（单模型）
        'expected_samples': 40,
    },
}

# ============================================================================
# 变量选择配置
# ============================================================================

# 基础变量（所有任务组都包含）
BASE_VARIABLES = [
    # 能耗总量（2个）
    'energy_cpu_total_joules',
    'energy_gpu_total_joules',
]

# 能耗中介变量（5个）
MEDIATOR_VARIABLES = [
    'gpu_util_avg',              # GPU平均利用率
    'gpu_temp_max',              # GPU最高温度
    'cpu_pkg_ratio',             # CPU Package能耗比例
    'gpu_power_fluctuation',     # GPU功率波动
    'gpu_temp_fluctuation',      # GPU温度波动
]

# 超参数候选（根据填充率动态选择）
HYPERPARAM_CANDIDATES = [
    'hyperparam_learning_rate',
    'hyperparam_batch_size',
    'hyperparam_dropout',
    'hyperparam_seed',               # 随机种子
    'hyperparam_training_duration',  # 统一后：epochs + max_iter
    'hyperparam_l2_regularization',  # 统一后：weight_decay + alpha
]

# ============================================================================
# 输出配置
# ============================================================================

# 数据目录
DATA_DIR = 'data'

# 结果目录
RESULTS_DIR = 'results'

# 日志目录
LOGS_DIR = 'logs'

# 日志级别
VERBOSE = True  # True: 详细日志，False: 简洁日志

# 因果图保存格式
CAUSAL_GRAPH_FORMAT = 'npy'  # 'npy': NumPy数组（Adult实验）
                              # 'pkl': Pickle（Python对象）
                              # 'graphml': GraphML（可视化）

# 因果效应保存格式
CAUSAL_EFFECTS_FORMAT = 'csv'  # 'csv': CSV表格
                                # 'json': JSON对象
                                # 'markdown': Markdown报告

# ============================================================================
# 性能配置
# ============================================================================

# GPU设备
DEVICE = 'cuda'  # 'cuda': GPU加速（推荐）
                 # 'cpu': CPU（慢）

# 并行处理
N_JOBS = -1  # -1: 使用所有CPU核心
             # 1: 串行
             # N: 使用N个核心

# ============================================================================
# 高级配置（可选，不推荐修改）
# ============================================================================

# 是否添加交互项（不推荐，增加复杂度）
ADD_INTERACTION_TERMS = False

# 是否对能耗进行对数变换（可选，视数据分布而定）
LOG_TRANSFORM_ENERGY = False

# 异常值处理（可选）
OUTLIER_DETECTION = False  # True: 检测并移除异常值（3σ外）
                           # False: 保留所有数据

# 异常值阈值（标准差倍数）
OUTLIER_THRESHOLD = 3.0  # 3σ

# ============================================================================
# 验证配置
# ============================================================================

# 最小样本量（每个任务组）
MIN_SAMPLES_PER_GROUP = 10  # DiBS最低要求

# 警告阈值（样本量<20时发出警告）
WARNING_SAMPLES_THRESHOLD = 20

# ============================================================================
# 配置验证函数
# ============================================================================

def validate_config():
    """验证配置参数的合理性"""

    errors = []
    warnings = []

    # 验证DiBS参数
    if DIBS_N_STEPS < 1000:
        warnings.append(f"DiBS迭代次数较少 ({DIBS_N_STEPS})，可能影响收敛")
    if DIBS_ALPHA < 0 or DIBS_ALPHA > 1:
        errors.append(f"DiBS稀疏性系数无效 ({DIBS_ALPHA})，应在[0, 1]范围内")

    # 验证DML参数
    if DML_N_FOLDS < 2:
        errors.append(f"DML交叉验证折数无效 ({DML_N_FOLDS})，应≥2")
    if CONFIDENCE_LEVEL <= 0 or CONFIDENCE_LEVEL >= 1:
        errors.append(f"置信水平无效 ({CONFIDENCE_LEVEL})，应在(0, 1)范围内")

    # 验证任务组配置
    for task_name, task_config in TASK_GROUPS.items():
        if task_config['expected_samples'] < MIN_SAMPLES_PER_GROUP:
            errors.append(f"任务组 {task_name} 样本量过少 ({task_config['expected_samples']}) < {MIN_SAMPLES_PER_GROUP}")
        elif task_config['expected_samples'] < WARNING_SAMPLES_THRESHOLD:
            warnings.append(f"任务组 {task_name} 样本量较少 ({task_config['expected_samples']})，统计功效可能不足")

    # 打印结果
    if errors:
        print("❌ 配置错误：")
        for error in errors:
            print(f"  - {error}")
        return False

    if warnings:
        print("⚠️  配置警告：")
        for warning in warnings:
            print(f"  - {warning}")

    print("✅ 配置验证通过")
    return True


def print_config_summary():
    """打印配置摘要"""
    print("=" * 80)
    print("因果分析配置摘要")
    print("=" * 80)

    print("\n【数据预处理】")
    print(f"  归一化方法: {NORMALIZATION_METHOD}")
    print(f"  缺失值处理: {MISSING_VALUE_STRATEGY}")
    print(f"  重复实验处理: {DUPLICATE_STRATEGY}")
    print(f"  超参数填充率阈值: {HYPERPARAM_FILL_RATE_THRESHOLD:.1%}")

    print("\n【DiBS因果图学习】")
    print(f"  迭代次数: {DIBS_N_STEPS}")
    print(f"  稀疏性系数: {DIBS_ALPHA}")
    print(f"  边筛选阈值: {EDGE_THRESHOLD}")
    print(f"  随机种子: {RANDOM_SEED}")

    print("\n【DML因果推断】")
    print(f"  基学习器: {DML_BASE_LEARNER}")
    print(f"  交叉验证折数: {DML_N_FOLDS}")
    print(f"  置信水平: {CONFIDENCE_LEVEL:.0%}")
    print(f"  显著性阈值: {SIGNIFICANCE_THRESHOLD}")

    print("\n【任务分组】")
    print(f"  任务组数量: {len(TASK_GROUPS)}")
    total_samples = sum(cfg['expected_samples'] for cfg in TASK_GROUPS.values())
    print(f"  总样本量: {total_samples}")
    for task_name, task_config in TASK_GROUPS.items():
        onehot_count = len(task_config['onehot_vars'])
        print(f"    - {task_config['name']}: {task_config['expected_samples']}样本, {onehot_count}个One-Hot变量")

    print("\n【输出配置】")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"  日志级别: {'详细' if VERBOSE else '简洁'}")
    print(f"  因果图格式: {CAUSAL_GRAPH_FORMAT}")
    print(f"  因果效应格式: {CAUSAL_EFFECTS_FORMAT}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # 验证配置
    if validate_config():
        # 打印配置摘要
        print_config_summary()
