# 正确的6分组数据设计方案

**日期**: 2026-01-15
**数据源**: data.csv (970行，818条可用数据)
**设计原则**: 将共用超参数和性能指标的模型分为一组，保留所有可用数据

---

## 📋 分组设计

基于数据分析，按照共用特征将模型分为以下6组：

### Group 1: 图像分类-小型模型组 (examples)
**模型**: mnist, mnist_ff, mnist_rnn, siamese
**样本数**: 304行
**共同特征**:
- 超参数: `hyperparam_batch_size`, `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`
- 性能指标: `perf_test_accuracy`, `perf_test_loss`(部分)
- 特点: 小型图像分类模型，共享batch_size参数

### Group 2: 代码漏洞检测组 (VulBERTa)
**模型**: mlp
**样本数**: 72行
**共同特征**:
- 超参数: `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`, `hyperparam_weight_decay`
- 性能指标: `perf_eval_loss`, `perf_final_training_loss`, `perf_eval_samples_per_second`
- 特点: NLP任务，独特的性能评估指标

### Group 3: 行人重识别组 (Person_reID)
**模型**: densenet121, hrnet18, pcb
**样本数**: 206行
**共同特征**:
- 超参数: `hyperparam_dropout`, `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`
- 性能指标: `perf_map`, `perf_rank1`, `perf_rank5`
- 特点: 检索任务，使用dropout，多个排序性能指标

### Group 4: 缺陷定位组 (bug-localization)
**模型**: default
**样本数**: 90行
**共同特征**:
- 超参数: `hyperparam_alpha`, `hyperparam_kfold`, `hyperparam_max_iter`, `hyperparam_seed`
- 性能指标: `perf_top1_accuracy`, `perf_top5_accuracy`, `perf_top10_accuracy`, `perf_top20_accuracy`
- 特点: 使用scikit-learn，独特的超参数集(alpha, kfold)

### Group 5: 多目标优化组 (MRT-OAST)
**模型**: default
**样本数**: 72行
**共同特征**:
- 超参数: `hyperparam_dropout`, `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`, `hyperparam_weight_decay`
- 性能指标: `perf_accuracy`, `perf_precision`, `perf_recall`
- 特点: 多目标性能指标，同时使用dropout和weight_decay

### Group 6: 图像分类-ResNet组 (pytorch_resnet_cifar10)
**模型**: resnet20
**样本数**: 74行
**共同特征**:
- 超参数: `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`, `hyperparam_weight_decay`
- 性能指标: `perf_best_val_accuracy`, `perf_test_accuracy`
- 特点: 大型模型，使用weight_decay正则化

---

## 📊 数据保留预期

| 组别 | 模型数 | 原始可用数据 | 预期保留 | 保留率 |
|------|--------|-------------|----------|--------|
| Group 1 | 4 | 304 | 304 | 100% |
| Group 2 | 1 | 72 | 72 | 100% |
| Group 3 | 3 | 206 | 206 | 100% |
| Group 4 | 1 | 90 | 90 | 100% |
| Group 5 | 1 | 72 | 72 | 100% |
| Group 6 | 1 | 74 | 74 | 100% |
| **总计** | 11 | 818 | 818 | 100% |

**关键优势**: 保留所有818条可用数据，无数据损失！

---

## 🛠️ 实现方案

### 1. 数据生成脚本结构

```python
def generate_6groups_data(input_file='data/data.csv', output_dir='analysis/data/energy_research/dibs_6groups'):
    """
    生成6组DiBS分析数据
    核心原则：保留所有可用数据，只选择每组实际使用的列
    """

    # 定义6组配置
    GROUP_CONFIGS = {
        'group1_image_classification_small': {
            'name': '图像分类-小型模型组',
            'repos': ['examples'],
            'models': ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese'],
            'hyperparams': ['hyperparam_batch_size', 'hyperparam_epochs',
                          'hyperparam_learning_rate', 'hyperparam_seed'],
            'performance': ['perf_test_accuracy', 'perf_test_loss']
        },
        'group2_vulberta': {
            'name': '代码漏洞检测组',
            'repos': ['VulBERTa'],
            'models': ['mlp'],
            'hyperparams': ['hyperparam_epochs', 'hyperparam_learning_rate',
                          'hyperparam_seed', 'hyperparam_weight_decay'],
            'performance': ['perf_eval_loss', 'perf_final_training_loss',
                          'perf_eval_samples_per_second']
        },
        # ... 其他4组
    }

    # 读取数据
    df = pd.read_csv(input_file)

    # 筛选可用数据
    df_usable = df[
        (df['training_success'] == True) &
        (~df[[col for col in df.columns if col.startswith('energy_')]].isnull().all(axis=1)) &
        (~df[[col for col in df.columns if col.startswith('perf_')]].isnull().all(axis=1))
    ]

    results = {}
    for group_id, config in GROUP_CONFIGS.items():
        # 筛选该组的数据
        group_mask = (
            df_usable['repository'].isin(config['repos']) &
            df_usable['model'].isin(config['models'])
        )
        group_df = df_usable[group_mask].copy()

        # 选择该组需要的列
        meta_cols = ['experiment_id', 'timestamp', 'repository', 'model']
        energy_cols = [col for col in df.columns if col.startswith('energy_')]
        control_cols = ['duration_seconds', 'is_parallel', 'num_mutated_params']

        # 只选择该组实际使用的超参数和性能指标
        selected_cols = (meta_cols + energy_cols + control_cols +
                        config['hyperparams'] + config['performance'])

        # 确保所有列都存在
        selected_cols = [col for col in selected_cols if col in group_df.columns]

        # 生成该组数据
        group_data = group_df[selected_cols]

        # 只删除在选定列中全为空的行
        group_data_clean = group_data.dropna(how='all', subset=config['hyperparams'] + config['performance'])

        results[group_id] = group_data_clean

    return results
```

### 2. 关键实现点

1. **不设置缺失率阈值** - 保留所有非全空的数据
2. **按组选择列** - 每组只包含该组实际使用的特征
3. **保留部分缺失数据** - 只要有部分特征非空就保留
4. **统一处理能耗和控制变量** - 所有组都包含这些通用特征

### 3. 数据验证

```python
def validate_group_data(group_data, group_name):
    """验证组数据质量"""
    print(f"\n{group_name}:")
    print(f"  样本数: {len(group_data)}")
    print(f"  特征数: {len(group_data.columns)}")

    # 检查关键列的完整性
    for col in group_data.columns:
        if col.startswith('hyperparam_') or col.startswith('perf_'):
            non_null = group_data[col].notna().sum()
            print(f"  {col}: {non_null}/{len(group_data)} ({non_null/len(group_data)*100:.1f}%)")
```

---

## 📈 预期成果

1. **数据保留率**: 100% (818/818条)
2. **每组数据质量**: 组内特征高度一致，适合DiBS分析
3. **特征清晰度**: 每组只包含相关特征，避免噪声
4. **可解释性**: 分组逻辑清晰，便于解释因果关系

---

## ⚠️ 注意事项

1. **不要使用缺失率阈值** - 这是之前的错误理解
2. **保留所有可用数据** - 即使某些特征有缺失
3. **按实际使用分组** - 基于模型实际使用的特征，而不是理论设计
4. **验证数据完整性** - 确保没有意外丢失数据

---

**设计者**: Claude
**创建日期**: 2026-01-15
**状态**: 待评审和实施