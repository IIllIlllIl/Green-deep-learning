# 阶段3-5详细实施方案

**日期**: 2025-12-24
**版本**: v1.0
**基于**: 阶段2完成成果（energy_data_extracted_v2.csv, 418行×34列）

---

## 概述

本文档详细规划阶段3-5的实施方案，从数据分层保存到DiBS因果分析验证的完整流程。

**关键原则** ⭐⭐⭐:
- **测试驱动**: 每个脚本必须编写对应的测试文件
- **Dry Run优先**: 先在少量数据上验证逻辑，再全量执行
- **增量开发**: 一次完成一个阶段，验证通过后再进入下一阶段

---

## 阶段3：数据分层与保存

### 3.1 目标

将418行统一数据按任务分层，生成4个训练数据文件，每个文件只保留任务相关的性能指标和变量。

### 3.2 输入与输出

**输入**:
- `data/energy_research/raw/energy_data_extracted_v2.csv` (418行×34列)
- `../../mutation/models_config.json` (仓库和模型定义)

**输出**:
```
analysis/data/energy_research/processed/
├── training_data_image_classification.csv  (~116行×17列)
├── training_data_person_reid.csv          (~69行×17列)
├── training_data_vulberta.csv             (~96行×14列)
└── training_data_bug_localization.csv     (~91行×14列)
```

**输出列清单（按任务组）**:

#### 图像分类组 (17列)
```python
columns = [
    # 实验标识 (2列)
    'experiment_id', 'timestamp',

    # One-Hot编码 (2列) - 控制数据集异质性
    'is_mnist', 'is_cifar10',

    # 超参数 (4列)
    'training_duration', 'hyperparam_learning_rate',
    'l2_regularization', 'seed',

    # 能耗指标 (3列)
    'energy_cpu_total_joules', 'energy_gpu_total_joules',
    'gpu_power_avg_watts',

    # 中介变量 (5列)
    'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
    'gpu_power_fluctuation', 'gpu_temp_fluctuation',

    # 性能指标 (1列)
    'perf_test_accuracy'
]
```

#### Person_reID组 (17列)
```python
columns = [
    # 实验标识 (2列)
    'experiment_id', 'timestamp',

    # One-Hot编码 (3列) - 控制模型异质性
    'is_densenet121', 'is_hrnet18', 'is_pcb',

    # 超参数 (4列)
    'training_duration', 'hyperparam_learning_rate',
    'hyperparam_dropout', 'seed',

    # 能耗指标 (3列)
    'energy_cpu_total_joules', 'energy_gpu_total_joules',
    'gpu_power_avg_watts',

    # 中介变量 (5列)
    'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
    'gpu_power_fluctuation', 'gpu_temp_fluctuation',

    # 性能指标 (3列)
    'perf_map', 'perf_rank1', 'perf_rank5'
]
```

#### VulBERTa组 (14列)
```python
columns = [
    # 实验标识 (2列)
    'experiment_id', 'timestamp',

    # 超参数 (4列)
    'training_duration', 'hyperparam_learning_rate',
    'l2_regularization', 'seed',

    # 能耗指标 (3列)
    'energy_cpu_total_joules', 'energy_gpu_total_joules',
    'gpu_power_avg_watts',

    # 中介变量 (5列)
    'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
    'gpu_power_fluctuation', 'gpu_temp_fluctuation',

    # 性能指标 (1列)
    'perf_eval_loss'
]
```

#### Bug定位组 (14列)
```python
columns = [
    # 实验标识 (2列)
    'experiment_id', 'timestamp',

    # 超参数 (4列)
    'training_duration', 'l2_regularization',
    'hyperparam_kfold', 'seed',

    # 能耗指标 (3列)
    'energy_cpu_total_joules', 'energy_gpu_total_joules',
    'gpu_power_avg_watts',

    # 中介变量 (5列)
    'gpu_util_avg', 'gpu_temp_max', 'cpu_pkg_ratio',
    'gpu_power_fluctuation', 'gpu_temp_fluctuation',

    # 性能指标 (2列)
    'perf_top1_accuracy', 'perf_top5_accuracy'
]
```

### 3.3 核心逻辑伪代码

```python
def preprocess_stratified_data():
    """
    数据分层预处理主函数

    步骤：
    1. 加载v2数据
    2. 按仓库分组
    3. 为每组创建One-Hot编码
    4. 选择任务相关列
    5. 删除性能全缺失行
    6. 保存分层文件
    """

    # 1. 加载数据
    df = pd.read_csv('data/energy_research/raw/energy_data_extracted_v2.csv')

    # 2. 定义任务组映射
    task_groups = {
        'image_classification': {
            'repos': ['examples', 'pytorch_resnet_cifar10'],
            'models': {
                'examples': ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese'],
                'pytorch_resnet_cifar10': ['resnet20']
            },
            'performance_cols': ['perf_test_accuracy'],
            'onehot_logic': lambda row: {
                'is_mnist': 1 if 'mnist' in row['model'] else 0,
                'is_cifar10': 1 if row['repository'] == 'pytorch_resnet_cifar10' else 0
            }
        },
        'person_reid': {
            'repos': ['Person_reID_baseline_pytorch'],
            'models': ['densenet121', 'hrnet18', 'pcb'],
            'performance_cols': ['perf_map', 'perf_rank1', 'perf_rank5'],
            'onehot_logic': lambda row: {
                'is_densenet121': 1 if row['model'] == 'densenet121' else 0,
                'is_hrnet18': 1 if row['model'] == 'hrnet18' else 0,
                'is_pcb': 1 if row['model'] == 'pcb' else 0
            }
        },
        'vulberta': {
            'repos': ['VulBERTa'],
            'models': ['mlp'],
            'performance_cols': ['perf_eval_loss'],
            'onehot_logic': None  # 单模型，无需One-Hot
        },
        'bug_localization': {
            'repos': ['bug-localization-by-dnn-and-rvsm'],
            'models': ['default'],
            'performance_cols': ['perf_top1_accuracy', 'perf_top5_accuracy'],
            'onehot_logic': None  # 单模型，无需One-Hot
        }
    }

    # 3. 处理每个任务组
    for task_name, config in task_groups.items():
        # 筛选仓库
        mask = df['repository'].isin(config['repos'])
        task_df = df[mask].copy()

        # 添加One-Hot编码
        if config['onehot_logic']:
            onehot_cols = task_df.apply(config['onehot_logic'], axis=1, result_type='expand')
            task_df = pd.concat([task_df, onehot_cols], axis=1)

        # 删除性能全缺失行
        perf_cols = config['performance_cols']
        task_df = task_df.dropna(subset=perf_cols, how='all')

        # 选择任务相关列
        selected_cols = get_task_columns(task_name, config)
        task_df = task_df[selected_cols]

        # 保存
        output_path = f'data/energy_research/processed/training_data_{task_name}.csv'
        task_df.to_csv(output_path, index=False)

        print(f"✅ {task_name}: {len(task_df)} 行，{len(selected_cols)} 列")
```

### 3.4 实现计划

#### 步骤1: 创建主脚本 (40分钟)

**文件**: `analysis/scripts/preprocess_stratified_data.py`

**功能模块**:
1. `load_extracted_data()` - 加载v2数据
2. `create_onehot_encoding()` - 生成One-Hot列
3. `select_task_columns()` - 选择任务相关列
4. `remove_missing_performance()` - 删除性能缺失行
5. `save_stratified_data()` - 保存分层文件
6. `generate_summary_report()` - 生成汇总报告

#### 步骤2: 创建测试脚本 (20分钟)

**文件**: `analysis/scripts/test_preprocess_stratified_data.py`

**测试用例**:
1. `test_onehot_encoding()` - 验证One-Hot生成逻辑
2. `test_column_selection()` - 验证列选择正确性
3. `test_missing_removal()` - 验证缺失值删除
4. `test_dry_run()` - Dry run前10行数据
5. `test_output_format()` - 验证输出文件格式

**测试数据**: 使用前10行数据创建mock输入

#### 步骤3: Dry Run验证 (10分钟)

```bash
# 运行dry run（只处理前20行）
cd analysis/scripts
conda run -n fairness python3 preprocess_stratified_data.py --dry-run --limit 20

# 检查输出
ls -lh ../data/energy_research/processed/
head -5 ../data/energy_research/processed/training_data_image_classification.csv
```

**验证标准**:
- ✅ 4个CSV文件成功生成
- ✅ 列数符合预期（14-17列）
- ✅ One-Hot编码正确（互斥，和=1）
- ✅ 无性能全缺失行

#### 步骤4: 全量执行 (5分钟)

```bash
# 运行全量处理
conda run -n fairness python3 preprocess_stratified_data.py --output-dir ../data/energy_research/processed/

# 生成汇总报告
conda run -n fairness python3 preprocess_stratified_data.py --summary
```

**预期输出**:
```
✅ 图像分类: 116行 × 17列
✅ Person_reID: 69行 × 17列
✅ VulBERTa: 96行 × 14列
✅ Bug定位: 91行 × 14列

总计: 372行有效数据（删除46行性能全缺失）
```

### 3.5 质量验证检查清单

```python
# 运行验证脚本
conda run -n fairness python3 scripts/verify_stratified_data.py

# 验证项目：
# 1. 行数正确性
assert len(image_df) + len(reid_df) + len(vul_df) + len(bug_df) == 372

# 2. 列名正确性
assert set(image_df.columns) == expected_image_columns

# 3. One-Hot互斥性
assert (image_df[['is_mnist', 'is_cifar10']].sum(axis=1) == 1).all()

# 4. 无性能缺失
assert not image_df['perf_test_accuracy'].isna().any()

# 5. 可计算相关矩阵
corr = image_df.drop(['experiment_id', 'timestamp'], axis=1).corr()
assert not corr.isna().any().any()
```

---

## 阶段4：数据质量验证

### 4.1 目标

验证分层数据满足DiBS因果分析的所有前提条件。

### 4.2 验证维度

#### 维度1: 缺失值统计

```python
def check_missing_rates():
    """
    检查每个任务组的缺失率

    目标：
    - 超参数列：0%缺失
    - 能耗列：0%缺失
    - 性能列：0%缺失
    - 中介变量列：<5%缺失（可接受）
    """
    for task in tasks:
        df = load_task_data(task)

        for col in df.columns:
            missing_rate = df[col].isna().sum() / len(df) * 100

            if 'hyperparam' in col or 'training_duration' in col:
                assert missing_rate == 0, f"{task}.{col}: 超参数不应缺失"

            if 'energy' in col:
                assert missing_rate == 0, f"{task}.{col}: 能耗不应缺失"

            if 'perf_' in col:
                assert missing_rate == 0, f"{task}.{col}: 性能不应缺失"

            if 'gpu_' in col or 'cpu_' in col:
                assert missing_rate < 5, f"{task}.{col}: 中介变量缺失率应<5%"
```

#### 维度2: 完全无缺失行比例

```python
def check_complete_rows():
    """
    检查完全无缺失行的比例

    目标（v3.0方案）：
    - 图像分类组：>90%
    - Person_reID组：>90%
    - VulBERTa组：>80%
    - Bug定位组：>80%
    """
    for task in tasks:
        df = load_task_data(task)
        complete_rows = df.dropna()
        complete_rate = len(complete_rows) / len(df) * 100

        print(f"{task}: {len(complete_rows)}/{len(df)} ({complete_rate:.1f}% 完全无缺失)")
```

#### 维度3: 相关矩阵可计算性

```python
def check_correlation_matrix():
    """
    检查相关矩阵是否可计算

    目标：
    - 相关矩阵无nan值
    - 相关矩阵有对角线全为1
    - 相关矩阵在[-1, 1]范围内
    """
    for task in tasks:
        df = load_task_data(task)

        # 移除非数值列
        numeric_df = df.select_dtypes(include=[np.number])

        # 计算相关矩阵
        corr = numeric_df.corr()

        # 验证
        assert not corr.isna().any().any(), f"{task}: 相关矩阵包含nan"
        assert (np.diag(corr) == 1).all(), f"{task}: 对角线应为1"
        assert (corr >= -1).all().all() and (corr <= 1).all().all(), f"{task}: 相关系数应在[-1,1]"

        print(f"✅ {task}: 相关矩阵可计算，形状={corr.shape}")
```

#### 维度4: 数值范围合理性

```python
def check_numeric_ranges():
    """
    检查数值列的范围是否合理

    目标：
    - 能耗列：>0
    - GPU温度：20-110°C
    - GPU功率：0-600W
    - 准确率：0-1或0-100
    """
    ranges = {
        'energy_cpu_total_joules': (0, 1e9),
        'energy_gpu_total_joules': (0, 1e9),
        'gpu_temp_max': (20, 110),
        'gpu_power_avg_watts': (0, 600),
        'perf_test_accuracy': (0, 100),
        'perf_map': (0, 1),
    }

    for task in tasks:
        df = load_task_data(task)

        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                assert len(out_of_range) == 0, f"{task}.{col}: {len(out_of_range)} 个值超出范围"
```

#### 维度5: One-Hot编码正确性

```python
def check_onehot_encoding():
    """
    检查One-Hot编码是否正确

    目标：
    - 每行One-Hot列的和=1（互斥）
    - 只包含0和1
    - 覆盖所有样本
    """
    # 图像分类
    image_df = load_task_data('image_classification')
    onehot_sum = image_df[['is_mnist', 'is_cifar10']].sum(axis=1)
    assert (onehot_sum == 1).all(), "图像分类: One-Hot和应为1"

    # Person_reID
    reid_df = load_task_data('person_reid')
    onehot_sum = reid_df[['is_densenet121', 'is_hrnet18', 'is_pcb']].sum(axis=1)
    assert (onehot_sum == 1).all(), "Person_reID: One-Hot和应为1"
```

### 4.3 实现计划

#### 步骤1: 创建验证脚本 (30分钟)

**文件**: `analysis/scripts/verify_stratified_data_quality.py`

**功能模块**:
1. `check_missing_rates()` - 缺失率检查
2. `check_complete_rows()` - 完全行检查
3. `check_correlation_matrix()` - 相关矩阵检查
4. `check_numeric_ranges()` - 数值范围检查
5. `check_onehot_encoding()` - One-Hot检查
6. `generate_quality_report()` - 生成质量报告

#### 步骤2: 运行验证 (5分钟)

```bash
cd analysis/scripts
conda run -n fairness python3 verify_stratified_data_quality.py \
  --data-dir ../data/energy_research/processed/ \
  --output-report ../docs/reports/STRATIFIED_DATA_QUALITY_REPORT.md
```

#### 步骤3: 解读报告

**预期输出**: `STRATIFIED_DATA_QUALITY_REPORT.md`

```markdown
# 分层数据质量验证报告

## 验证摘要

| 任务组 | 样本数 | 列数 | 完全无缺失行 | 相关矩阵 |
|--------|--------|------|-------------|---------|
| 图像分类 | 116 | 17 | 105 (90.5%) | ✅ 可计算 |
| Person_reID | 69 | 17 | 63 (91.3%) | ✅ 可计算 |
| VulBERTa | 96 | 14 | 80 (83.3%) | ✅ 可计算 |
| Bug定位 | 91 | 14 | 75 (82.4%) | ✅ 可计算 |

## 详细检查结果

### ✅ 缺失率检查
- 超参数列：0%缺失 ✅
- 能耗列：0%缺失 ✅
- 性能列：0%缺失 ✅
- 中介变量列：<2%缺失 ✅

### ✅ 相关矩阵检查
- 所有任务组的相关矩阵可计算 ✅
- 无nan值 ✅
- 数值范围[-1, 1] ✅

### ✅ One-Hot编码检查
- 图像分类：互斥性100% ✅
- Person_reID：互斥性100% ✅

### ✅ 数值范围检查
- 能耗指标：范围合理 ✅
- 温度指标：范围合理 ✅
- 性能指标：范围合理 ✅

## 结论

✅ **数据质量验证通过，可以进入阶段5（DiBS分析）**
```

---

## 阶段5：DiBS因果分析验证

### 5.1 目标

运行DiBS因果图学习，验证新数据能够成功发现因果边，对比v1.0（Adult, 0边）和v3.0（能耗分层, 预期3-8边/任务）的改进。

### 5.2 分析配置

#### DiBS超参数

```python
# 参考Adult成功经验
dibs_config = {
    'n_particles': 20,          # 粒子数
    'n_steps': 1000,           # 优化步数
    'optimizer': 'adam',        # 优化器
    'learning_rate': 0.005,     # 学习率
    'temperature': 1.0,         # 温度参数
    'alpha_linear': 0.05,       # DAG正则化
    'verbose': True             # 打印进度
}
```

#### 运行模式

```python
# 模式1: 串行运行（稳定，适合调试）
for task in tasks:
    run_dibs_analysis(task, dibs_config)

# 模式2: 并行运行（快速，适合生产）
from multiprocessing import Pool
with Pool(4) as p:
    p.map(run_dibs_task, [(task, dibs_config) for task in tasks])
```

### 5.3 实现计划

#### 步骤1: 创建分析脚本 (40分钟)

**文件**: `analysis/scripts/run_stratified_dibs_analysis.py`

**功能模块**:
1. `load_task_data()` - 加载任务数据
2. `prepare_dibs_input()` - 准备DiBS输入矩阵
3. `run_dibs()` - 运行DiBS优化
4. `extract_causal_edges()` - 提取因果边
5. `save_results()` - 保存结果
6. `generate_analysis_report()` - 生成分析报告

**核心逻辑**:

```python
def run_stratified_dibs_analysis(task_name, config):
    """
    运行单个任务组的DiBS分析

    步骤：
    1. 加载数据并删除标识列
    2. 标准化数值列
    3. 运行DiBS学习因果图
    4. 筛选高置信度因果边（posterior > 0.5）
    5. 保存结果
    """

    # 1. 加载数据
    df = pd.read_csv(f'data/energy_research/processed/training_data_{task_name}.csv')

    # 2. 移除标识列
    df = df.drop(['experiment_id', 'timestamp'], axis=1)

    # 3. 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # 4. 运行DiBS
    from utils.causal_discovery import run_dibs
    result = run_dibs(
        data=X,
        variable_names=list(df.columns),
        n_particles=config['n_particles'],
        n_steps=config['n_steps'],
        verbose=config['verbose']
    )

    # 5. 提取因果边
    adjacency_matrix = result['adjacency_matrix']
    edges = []
    n_vars = len(df.columns)
    for i in range(n_vars):
        for j in range(n_vars):
            posterior = adjacency_matrix[i, j]
            if posterior > 0.5:  # 高置信度阈值
                edges.append({
                    'source': df.columns[i],
                    'target': df.columns[j],
                    'posterior': posterior
                })

    # 6. 保存结果
    output_dir = f'results/energy_research/task_specific/{task_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # 保存因果图
    np.save(f'{output_dir}/causal_graph.npy', adjacency_matrix)

    # 保存因果边
    import pickle
    with open(f'{output_dir}/causal_edges.pkl', 'wb') as f:
        pickle.dump(edges, f)

    print(f"✅ {task_name}: 发现 {len(edges)} 条因果边")

    return edges
```

#### 步骤2: 创建测试脚本 (20分钟)

**文件**: `analysis/scripts/test_stratified_dibs.py`

**测试用例**:
1. `test_data_loading()` - 测试数据加载
2. `test_standardization()` - 测试标准化
3. `test_dibs_dry_run()` - Dry run with 少量步数
4. `test_edge_extraction()` - 测试边提取逻辑
5. `test_result_saving()` - 测试结果保存

**Dry Run配置**:
```python
dry_run_config = {
    'n_particles': 5,      # 少量粒子
    'n_steps': 10,         # 少量步数
    'verbose': True
}
```

#### 步骤3: Dry Run验证 (15分钟)

```bash
# 测试单个任务组（图像分类，116样本）
cd analysis/scripts
conda run -n fairness python3 run_stratified_dibs_analysis.py \
  --task image_classification \
  --n-particles 5 \
  --n-steps 10 \
  --dry-run

# 检查是否能成功完成
# 预期：无报错，生成mock结果文件
```

**验证标准**:
- ✅ DiBS优化无报错
- ✅ 生成adjacency_matrix
- ✅ 可以提取因果边
- ✅ 结果文件可保存

#### 步骤4: 全量执行 (60分钟，可并行)

```bash
# 方式1: 串行运行（稳定）
cd analysis/scripts
nohup bash run_all_stratified_dibs.sh > ../logs/energy_research/stratified_dibs_20251224.log 2>&1 &

# 方式2: 并行运行（快速）
conda run -n fairness python3 run_stratified_dibs_analysis.py --all-tasks --parallel

# 监控进度
tail -f ../logs/energy_research/stratified_dibs_20251224.log
```

**预期运行时间**:
- 图像分类 (116样本, 17列): ~15分钟
- Person_reID (69样本, 17列): ~10分钟
- VulBERTa (96样本, 14列): ~20分钟
- Bug定位 (91样本, 14列): ~15分钟

**总计**: 约60分钟（串行），约20分钟（并行）

### 5.4 结果分析

#### 预期结果（基于v3.0方案）

| 任务组 | 样本数 | 变量数 | 预期因果边数 | 关键因果路径 |
|--------|--------|--------|------------|------------|
| 图像分类 | 116 | 17 | 3-8条 | learning_rate → energy, training_duration → accuracy |
| Person_reID | 69 | 17 | 3-6条 | dropout → mAP, learning_rate → energy |
| VulBERTa | 96 | 14 | 2-5条 | learning_rate → eval_loss, weight_decay → energy |
| Bug定位 | 91 | 14 | 2-5条 | kfold → top1_accuracy, max_iter → energy |

#### 对比v1.0改进

| 维度 | v1.0 (Adult) | v3.0 (能耗分层) | 改进 |
|------|-------------|----------------|------|
| **样本量** | 10个 | 69-116个/任务 | **7-11倍** |
| **变量数** | 15个 | 14-17个/任务 | 优化选择 |
| **因果边数** | **0条** | 预期3-8条/任务 | **质的飞跃** |
| **相关矩阵** | 包含nan（失败） | 完全可计算 | ✅ |
| **One-Hot编码** | 无 | 有（控制异质性） | ✅ |
| **数据质量** | 32-100%缺失 | <2%缺失 | ✅ |

#### 生成对比报告

**文件**: `analysis/docs/reports/DIBS_V1_VS_V3_COMPARISON_REPORT.md`

```markdown
# DiBS v1.0 vs v3.0 对比报告

## 执行摘要

v3.0方案通过数据重提取、分层分析、One-Hot编码，成功解决v1.0的0因果边问题。

## 关键改进

### 1. 数据质量提升 ⭐⭐⭐
- v1.0: 超参数32-100%缺失 → DiBS计算失败
- v3.0: 超参数100%填充 → DiBS可计算

### 2. 样本量增加 ⭐⭐⭐
- v1.0: 10个配置 → 统计功效不足
- v3.0: 69-116个/任务 → 统计功效充足

### 3. 异质性控制 ⭐⭐
- v1.0: 混合不同模型/数据集 → DiBS混淆基线差异
- v3.0: One-Hot编码 → DiBS区分真实因果关系

### 4. 任务特定优化 ⭐⭐
- v1.0: 全局分析 → 丢失任务特定模式
- v3.0: 分层分析 → 发现任务特定因果路径

## 因果发现结果

[详细列出每个任务组发现的因果边]

## 结论

v3.0方案成功实现能耗数据的因果分析，为超参数优化提供因果指导。
```

### 5.5 后续DML分析（可选）

如果DiBS成功发现因果边，可以继续运行DML估计因果效应：

```bash
# 运行DML因果推断
cd analysis/scripts
conda run -n fairness python3 run_stratified_dml_analysis.py \
  --task image_classification \
  --causal-graph results/energy_research/task_specific/image_classification/causal_graph.npy
```

**DML输出**: 每条因果边的ATE（平均因果效应）、置信区间、p值

---

## 时间估算与进度跟踪

### 总时间估算

| 阶段 | 子任务 | 预估时间 | 优先级 |
|------|--------|---------|--------|
| **阶段3** | 创建预处理脚本 | 40分钟 | 🔴 P0 |
| | 创建测试脚本 | 20分钟 | 🔴 P0 |
| | Dry Run验证 | 10分钟 | 🔴 P0 |
| | 全量执行 | 5分钟 | 🔴 P0 |
| **阶段4** | 创建验证脚本 | 30分钟 | 🟠 P1 |
| | 运行验证 | 5分钟 | 🟠 P1 |
| | 解读报告 | 10分钟 | 🟠 P1 |
| **阶段5** | 创建分析脚本 | 40分钟 | 🟡 P2 |
| | 创建测试脚本 | 20分钟 | 🟡 P2 |
| | Dry Run验证 | 15分钟 | 🟡 P2 |
| | 全量执行 | 60分钟 | 🟡 P2 |
| | 结果分析 | 30分钟 | 🟡 P2 |

**总计**: 约4.5-5小时（含Dry Run和测试）

### 进度跟踪清单

```markdown
- [ ] 阶段3.1: preprocess_stratified_data.py
- [ ] 阶段3.2: test_preprocess_stratified_data.py
- [ ] 阶段3.3: Dry Run通过
- [ ] 阶段3.4: 生成4个分层文件
- [ ] 阶段4.1: verify_stratified_data_quality.py
- [ ] 阶段4.2: 生成质量报告
- [ ] 阶段4.3: 所有验证通过
- [ ] 阶段5.1: run_stratified_dibs_analysis.py
- [ ] 阶段5.2: test_stratified_dibs.py
- [ ] 阶段5.3: Dry Run通过
- [ ] 阶段5.4: 全量执行完成
- [ ] 阶段5.5: 生成对比报告
```

---

## 风险与应对

### 风险1: DiBS优化超时

**症状**: DiBS运行>30分钟无进展

**原因**: 样本量大或变量数多

**应对**:
1. 减少n_particles (20 → 10)
2. 减少n_steps (1000 → 500)
3. 移除低方差变量（如seed）

### 风险2: 仍然发现0因果边

**症状**: DiBS完成但posterior全部<0.5

**原因**:
- 数据线性关系弱
- 变量选择不当
- 样本量仍不足

**应对**:
1. 检查相关矩阵（是否有显著相关）
2. 尝试降低posterior阈值（0.5 → 0.3）
3. 增加样本量（合并并行/非并行数据）

### 风险3: One-Hot编码导致共线性

**症状**: 相关矩阵出现nan或接近1的相关

**原因**: One-Hot列完全线性相关

**应对**:
1. 移除一个One-Hot列（n-1编码）
2. 检查VIF（方差膨胀因子）
3. 使用PCA降维

### 风险4: 内存不足

**症状**: DiBS运行时OOM

**原因**: 大样本量 × 高变量数

**应对**:
1. 减少n_particles
2. 使用GPU加速（如可用）
3. 分批处理

---

## 附录：关键代码示例

### A. One-Hot编码实现

```python
def create_onehot_image_classification(row):
    """
    图像分类One-Hot编码

    规则：
    - MNIST系列 (mnist, mnist_ff, mnist_rnn, siamese) → is_mnist=1
    - CIFAR-10 (resnet20) → is_cifar10=1
    """
    is_mnist = 1 if 'mnist' in row['model'] or row['model'] == 'siamese' else 0
    is_cifar10 = 1 if row['repository'] == 'pytorch_resnet_cifar10' else 0

    return pd.Series({'is_mnist': is_mnist, 'is_cifar10': is_cifar10})

def create_onehot_person_reid(row):
    """
    Person_reID One-Hot编码

    规则：
    - densenet121 → is_densenet121=1
    - hrnet18 → is_hrnet18=1
    - pcb → is_pcb=1
    """
    return pd.Series({
        'is_densenet121': 1 if row['model'] == 'densenet121' else 0,
        'is_hrnet18': 1 if row['model'] == 'hrnet18' else 0,
        'is_pcb': 1 if row['model'] == 'pcb' else 0
    })
```

### B. 相关矩阵可视化

```python
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_correlation_matrix(task_name):
    """生成相关矩阵热力图"""
    df = pd.read_csv(f'data/energy_research/processed/training_data_{task_name}.csv')

    # 移除标识列
    numeric_df = df.select_dtypes(include=[np.number])

    # 计算相关矩阵
    corr = numeric_df.corr()

    # 绘图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title(f'Correlation Matrix - {task_name}')
    plt.tight_layout()
    plt.savefig(f'results/energy_research/task_specific/{task_name}/correlation_matrix.png', dpi=300)
    plt.close()
```

### C. DiBS结果可视化

```python
import networkx as nx

def visualize_causal_graph(task_name, threshold=0.5):
    """生成因果图可视化"""
    # 加载邻接矩阵
    adj_matrix = np.load(f'results/energy_research/task_specific/{task_name}/causal_graph.npy')

    # 加载变量名
    df = pd.read_csv(f'data/energy_research/processed/training_data_{task_name}.csv')
    var_names = [c for c in df.columns if c not in ['experiment_id', 'timestamp']]

    # 创建有向图
    G = nx.DiGraph()

    for i, source in enumerate(var_names):
        for j, target in enumerate(var_names):
            posterior = adj_matrix[i, j]
            if posterior > threshold:
                G.add_edge(source, target, weight=posterior)

    # 绘图
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # 节点分组着色
    node_colors = []
    for node in G.nodes():
        if 'hyperparam' in node or node in ['training_duration', 'l2_regularization', 'seed']:
            node_colors.append('lightblue')  # 超参数
        elif 'energy' in node or 'power' in node:
            node_colors.append('lightcoral')  # 能耗
        elif 'perf_' in node:
            node_colors.append('lightgreen')  # 性能
        else:
            node_colors.append('lightyellow')  # 中介变量

    nx.draw(G, pos, node_color=node_colors, with_labels=True,
            node_size=2000, font_size=8, arrows=True, arrowsize=15)

    plt.title(f'Causal Graph - {task_name} (threshold={threshold})')
    plt.tight_layout()
    plt.savefig(f'results/energy_research/task_specific/{task_name}/causal_graph.png', dpi=300)
    plt.close()
```

---

**文档版本**: v1.0
**创建时间**: 2025-12-24
**预估完成时间**: 4.5-5小时（含测试和Dry Run）
**下一步**: 等待用户确认后执行阶段3
