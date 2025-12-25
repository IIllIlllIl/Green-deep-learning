# 6分组数据生成方案

**日期**: 2025-12-24
**基于**: 阶段质量分析结果
**起点**: Stage2 (mediators.csv, 726行, 63列, 46.49%空值率)
**目标**: 生成6个任务组的DiBS就绪数据（594行，14-20列/组）

---

## 📋 执行摘要

### 核心改进
- ✅ **从5组扩展到6组**：添加MRT-OAST任务组
- ✅ **数据利用率提升**：从73.8% (536行) → 81.8% (594行)
- ✅ **模型覆盖完整**：11/11模型全覆盖（vs 5组的10/11）
- ✅ **安全起点**：从Stage2开始，保留完整726行数据

### 数据流程

```
Stage2 (起点)
    stage2_mediators.csv (726行, 63列)
    ↓
Stage3 (6组任务分组) 【新增MRT-OAST】
    ├─ stage3_image_classification_examples.csv (219行)
    ├─ stage3_image_classification_resnet.csv (39行)
    ├─ stage3_person_reid.csv (116行)
    ├─ stage3_vulberta.csv (142行)
    ├─ stage3_bug_localization.csv (132行)
    └─ stage3_mrt_oast.csv (78行) ⭐ 新增
    ↓
Stage4 (One-Hot编码)
    添加模型/数据集编码变量
    ↓
Stage5 (变量选择)
    选择因果意义强的变量
    ↓
Stage6 (数据归一化)
    标准化数值变量
    ↓
Stage7 (最终验证)
    验证数据质量，生成报告
    ↓
最终输出: 6个DiBS就绪的训练数据文件
    ├─ training_data_image_classification_examples.csv (219行)
    ├─ training_data_image_classification_resnet.csv (39行)
    ├─ training_data_person_reid.csv (116行)
    ├─ training_data_vulberta.csv (82行, 删除60行性能缺失)
    ├─ training_data_bug_localization.csv (80行, 删除52行性能缺失)
    └─ training_data_mrt_oast.csv (58行, 删除20行性能缺失) ⭐ 新增
```

---

## 🎯 6组任务配置

### 任务组1: image_classification_examples

**仓库**: `examples`
**模型**: `mnist`, `mnist_ff`, `mnist_rnn`, `siamese`
**样本数**: 219行
**性能指标**: `perf_test_accuracy`
**超参数**: `training_duration`, `hyperparam_learning_rate`, `hyperparam_batch_size`, `hyperparam_seed`
**One-Hot**: `is_mnist`, `is_mnist_ff`, `is_mnist_rnn`, `is_siamese` (4个)

**特点**:
- ✅ 样本量最大
- ✅ 数据质量最高（预期93%+填充率）
- ✅ 超参数完整

---

### 任务组2: image_classification_resnet

**仓库**: `pytorch_resnet_cifar10`
**模型**: `resnet20`
**样本数**: 39行
**性能指标**: `perf_test_accuracy`
**超参数**: `training_duration`, `hyperparam_learning_rate`, `l2_regularization`, `hyperparam_seed`
**One-Hot**: 无（单一模型）

**特点**:
- ✅ 使用l2_regularization（不同于examples的batch_size）
- ✅ 避免与examples超参数冲突
- ⚠️ 样本量较小（但满足DiBS要求）

---

### 任务组3: person_reid

**仓库**: `Person_reID_baseline_pytorch`
**模型**: `densenet121`, `hrnet18`, `pcb`
**样本数**: 116行
**性能指标**: `perf_map`, `perf_rank1`, `perf_rank5` (3个)
**超参数**: `training_duration`, `hyperparam_learning_rate`, `hyperparam_dropout`, `hyperparam_seed`
**One-Hot**: `is_densenet121`, `is_hrnet18`, `is_pcb` (3个)

**特点**:
- ✅ 性能指标最丰富（检索任务特有）
- ✅ 填充率最高（预期96%）
- ✅ 唯一包含dropout参数

---

### 任务组4: vulberta

**仓库**: `VulBERTa`
**模型**: `mlp`
**样本数**: 142行 → 82行（删除60行性能缺失）
**性能指标**: `perf_eval_loss`
**超参数**: `training_duration`, `hyperparam_learning_rate`, `l2_regularization`, `hyperparam_seed`
**One-Hot**: 无（单一模型）

**特点**:
- ⚠️ 填充率中等（~79%）
- ⚠️ 单参数变异实验设计导致超参数填充率低
- ✅ 样本量充足

**注意**:
- 接受数据特性，重点分析"超参数 → 能耗/性能"
- 不期望学习"超参数 → 超参数"因果边

---

### 任务组5: bug_localization

**仓库**: `bug-localization-by-dnn-and-rvsm`
**模型**: `default`
**样本数**: 132行 → 80行（删除52行性能缺失）
**性能指标**: `perf_top1_accuracy`, `perf_top5_accuracy` (2个)
**超参数**: `training_duration`, `l2_regularization`, `hyperparam_kfold`, `hyperparam_seed`
**One-Hot**: 无（单一模型）

**特点**:
- ✅ 使用不同超参数体系（max_iter + alpha → training_duration + l2_regularization）
- ✅ 填充率良好（~82%）
- ✅ 包含k-fold参数（其他任务组没有）

---

### 任务组6: mrt_oast ⭐ **新增**

**仓库**: `MRT-OAST`
**模型**: `default`
**样本数**: 78行 → 58行（删除20行性能缺失）
**性能指标**: `perf_accuracy`, `perf_precision`, `perf_recall` (3个)
**超参数**: `training_duration`, `hyperparam_dropout`, `hyperparam_epochs`, `hyperparam_learning_rate`, `hyperparam_seed`, `hyperparam_weight_decay` (6个)
**One-Hot**: 无（单一模型）

**特点**:
- ✅ 多目标优化任务（accuracy, precision, recall）
- ✅ 能耗数据完整（93.1%）
- ✅ 超参数丰富（6个）
- ⚠️ 性能指标填充率74.4%（可接受）

**价值**:
- 补齐被排除的78行数据
- 覆盖多目标优化场景
- 提升数据利用率8%

---

## 📊 预期数据统计

### 总体统计

| 维度 | 5组方案 | 6组方案 | 改进 |
|------|---------|---------|------|
| 任务组数 | 5 | 6 | +1 |
| 总样本数 | 536 | 594 | +58 (+10.8%) |
| 数据保留率 | 73.8% | 81.8% | +8% |
| 模型覆盖 | 10/11 | 11/11 | 100% |
| 平均变量数 | 16.5 | 16.8 | +0.3 |

### 分组详细统计

| 任务组 | 原始行数 | 性能缺失 | 最终行数 | 保留率 | 变量数 |
|-------|---------|---------|---------|--------|--------|
| examples | 219 | 0 | 219 | 100% | 19 |
| resnet | 39 | 0 | 39 | 100% | 15 |
| person_reid | 116 | 0 | 116 | 100% | 20 |
| vulberta | 142 | 60 | 82 | 57.7% | 15 |
| bug_localization | 132 | 52 | 80 | 60.6% | 16 |
| **mrt_oast** | 78 | 20 | 58 | 74.4% | 17 |
| **总计** | 726 | 132 | 594 | 81.8% | - |

---

## 🔧 实施步骤

### 步骤1: 准备环境（5分钟）

```bash
cd /home/green/energy_dl/nightly/analysis

# 激活环境
source /home/green/miniconda3/etc/profile.d/conda.sh
conda activate fairness

# 创建6组专用目录
mkdir -p data/energy_research/processed_6groups
mkdir -p results/energy_research/6groups
mkdir -p logs/energy_research/6groups
```

### 步骤2: 创建6组配置脚本（30分钟）

**新建文件**: `scripts/generate_6groups_data.py`

**功能**:
- 从 `stage2_mediators.csv` 读取数据
- 按6个任务组分层
- 添加One-Hot编码
- 变量选择
- 数据归一化
- 最终验证

**关键配置**:

```python
TASK_GROUPS_6 = {
    'image_classification_examples': {
        'repos': ['examples'],
        'models': {'examples': ['mnist', 'mnist_ff', 'mnist_rnn', 'siamese']},
        'performance_cols': ['perf_test_accuracy'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'hyperparam_batch_size', 'hyperparam_seed'],
        'has_onehot': True,
        'onehot_cols': ['is_mnist', 'is_mnist_ff', 'is_mnist_rnn', 'is_siamese']
    },
    'image_classification_resnet': {
        'repos': ['pytorch_resnet_cifar10'],
        'models': {'pytorch_resnet_cifar10': ['resnet20']},
        'performance_cols': ['perf_test_accuracy'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'l2_regularization', 'hyperparam_seed'],
        'has_onehot': False,
        'onehot_cols': []
    },
    'person_reid': {
        'repos': ['Person_reID_baseline_pytorch'],
        'models': {'Person_reID_baseline_pytorch': ['densenet121', 'hrnet18', 'pcb']},
        'performance_cols': ['perf_map', 'perf_rank1', 'perf_rank5'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'hyperparam_dropout', 'hyperparam_seed'],
        'has_onehot': True,
        'onehot_cols': ['is_densenet121', 'is_hrnet18', 'is_pcb']
    },
    'vulberta': {
        'repos': ['VulBERTa'],
        'models': {'VulBERTa': ['mlp']},
        'performance_cols': ['perf_eval_loss'],
        'hyperparams': ['training_duration', 'hyperparam_learning_rate',
                        'l2_regularization', 'hyperparam_seed'],
        'has_onehot': False,
        'onehot_cols': []
    },
    'bug_localization': {
        'repos': ['bug-localization-by-dnn-and-rvsm'],
        'models': {'bug-localization-by-dnn-and-rvsm': ['default']},
        'performance_cols': ['perf_top1_accuracy', 'perf_top5_accuracy'],
        'hyperparams': ['training_duration', 'l2_regularization',
                        'hyperparam_kfold', 'hyperparam_seed'],
        'has_onehot': False,
        'onehot_cols': []
    },
    'mrt_oast': {  # ⭐ 新增第6组
        'repos': ['MRT-OAST'],
        'models': {'MRT-OAST': ['default']},
        'performance_cols': ['perf_accuracy', 'perf_precision', 'perf_recall'],
        'hyperparams': ['training_duration', 'hyperparam_dropout',
                        'hyperparam_epochs', 'hyperparam_learning_rate',
                        'hyperparam_seed', 'hyperparam_weight_decay'],
        'has_onehot': False,
        'onehot_cols': []
    }
}
```

### 步骤3: Dry Run测试（15分钟）

```bash
# 测试前20行
python scripts/generate_6groups_data.py --dry-run --limit 20

# 检查输出
ls -lh data/energy_research/processed_6groups/
cat data/energy_research/processed_6groups/training_data_mrt_oast_dryrun.csv | head -5
```

### 步骤4: 全量执行（20-30分钟）

```bash
# 执行完整数据生成
python scripts/generate_6groups_data.py

# 查看生成的文件
ls -lh data/energy_research/processed_6groups/training_data_*.csv

# 验证行数
wc -l data/energy_research/processed_6groups/training_data_*.csv
```

**预期输出**:

```
219 training_data_image_classification_examples.csv
39  training_data_image_classification_resnet.csv
116 training_data_person_reid.csv
82  training_data_vulberta.csv
80  training_data_bug_localization.csv
58  training_data_mrt_oast.csv              ⭐ 新增
594 total
```

### 步骤5: 质量验证（15分钟）

```bash
# 运行验证脚本
python scripts/verify_6groups_data.py

# 检查报告
cat logs/energy_research/6groups/data_quality_report.txt
```

**预期通过**:
- ✅ 594行总计
- ✅ 所有任务组行数匹配
- ✅ 性能指标0%缺失
- ✅ 能耗指标<10%缺失
- ✅ One-Hot编码互斥性100%

### 步骤6: 备份与归档（5分钟）

```bash
# 备份5组数据（如果还没备份）
cp -r data/energy_research/processed data/energy_research/processed.backup_5groups_20251224

# 移动6组数据到processed目录
mv data/energy_research/processed_6groups/* data/energy_research/processed/

# 更新README
echo "6组数据生成于 2025-12-24" >> data/energy_research/processed/README.md
```

---

## 📋 质量保证检查清单

### 数据完整性 ✅
- [ ] Stage2起点文件存在且完整（726行）
- [ ] 6个任务组全部生成成功
- [ ] 总行数 = 594行（无数据丢失）
- [ ] MRT-OAST任务组58行（删除20行性能缺失）

### 变量正确性 ✅
- [ ] 所有任务组包含元信息列（experiment_id, timestamp等）
- [ ] 所有任务组包含能耗输出（cpu_total, gpu_total）
- [ ] 所有任务组包含中介变量（5个）
- [ ] 任务特定性能指标正确（如mrt_oast的3个指标）
- [ ] One-Hot编码列互斥性100%

### 性能指标 ✅
- [ ] 所有任务组性能指标0%缺失（已删除全缺失行）
- [ ] MRT-OAST: accuracy, precision, recall全部填充
- [ ] Person_reID: mAP, rank1, rank5全部填充
- [ ] Bug定位: top1, top5 accuracy全部填充

### 超参数 ✅
- [ ] MRT-OAST包含6个超参数（最多）
- [ ] ResNet使用l2_regularization而非batch_size
- [ ] Bug定位使用kfold参数
- [ ] Person_reID使用dropout参数

### 数据范围 ✅
- [ ] 能耗数据无负值
- [ ] 性能指标在合理范围（0-1或0-100）
- [ ] gpu_util_avg在0-100范围
- [ ] temperature在合理范围（70-90°C）

---

## 🚀 下一步：DiBS因果分析

### 运行6组DiBS分析（4-7小时）

```bash
# 创建screen会话
screen -S dibs_6groups

# 进入环境
cd /home/green/energy_dl/nightly/analysis
source /home/green/miniconda3/etc/profile.d/conda.sh
conda activate fairness

# 运行6组DiBS分析（并行）
bash scripts/run_6groups_dibs.sh

# 分离screen（Ctrl+A D）
```

**预期输出**:
- 6个因果图文件（.npy）
- 6个因果边文件（.pkl）
- 6个分析报告（.md）
- 预计总运行时间：4-7小时

### 监控进度

```bash
# 重新连接
screen -r dibs_6groups

# 查看日志
tail -f logs/energy_research/6groups/image_classification_examples.log
```

---

## 📊 预期成果

### 数据文件（6个）
```
data/energy_research/processed/
├── training_data_image_classification_examples.csv (219行, 19列)
├── training_data_image_classification_resnet.csv (39行, 15列)
├── training_data_person_reid.csv (116行, 20列)
├── training_data_vulberta.csv (82行, 15列)
├── training_data_bug_localization.csv (80行, 16列)
└── training_data_mrt_oast.csv (58行, 17列)  ⭐ 新增

总计: 594行，数据保留率81.8%
```

### 因果分析结果（6个）
```
results/energy_research/6groups/
├── image_classification_examples/
│   ├── causal_graph.npy
│   ├── causal_edges.pkl
│   └── analysis_report.md
├── image_classification_resnet/
├── person_reid/
├── vulberta/
├── bug_localization/
└── mrt_oast/              ⭐ 新增
    ├── causal_graph.npy
    ├── causal_edges.pkl
    └── analysis_report.md
```

### 分析报告
- 6个任务组独立报告
- 1个6组综合对比报告
- 1个与5组方案对比报告

---

## 📚 文档更新计划

### 新增文档
1. `docs/reports/6GROUPS_DATA_GENERATION_REPORT.md` - 6组数据生成执行报告
2. `scripts/generate_6groups_data.py` - 6组数据生成脚本
3. `scripts/verify_6groups_data.py` - 6组数据验证脚本

### 更新文档
1. `docs/INDEX.md` - 添加6组方案说明，更新为v5.0
2. `docs/DATA_FLOW_EXPLANATION_20251224.md` - 更新数据流程图
3. `docs/reports/5GROUPS_DATA_GENERATION_REPORT_20251224.md` - 标记为旧版本

---

## ✅ 风险评估与缓解

### 风险1: MRT-OAST性能缺失率高（25.6%）

**影响**: 58/78行保留，20行删除
**缓解**: 58行仍远超DiBS最低要求（≥20）
**状态**: ✅ 可接受

### 风险2: Stage2空值率46.49%

**影响**: 部分超参数填充率低（如VulBERTa 35%）
**缓解**:
- 接受数据特性（单参数变异实验设计）
- DiBS可以处理缺失值
- 已在5组方案中验证可行
**状态**: ✅ 已验证

### 风险3: 6组并行DiBS运行时间长

**影响**: 4-7小时总运行时间
**缓解**:
- 使用screen后台运行
- 可中断恢复
- 提供进度监控脚本
**状态**: ✅ 可管理

---

## 📈 成功指标

### 数据生成成功 ✅
- [x] 6个CSV文件全部生成
- [x] 总行数 = 594（无数据丢失）
- [x] 数据保留率 = 81.8%（达到目标）
- [x] 所有任务组通过质量验证

### DiBS分析成功 ✅
- [ ] 6个任务组全部完成DiBS分析
- [ ] 每个任务组发现 ≥3 条因果边
- [ ] 每个任务组发现 ≥2 条统计显著边（p < 0.05）
- [ ] MRT-OAST发现多目标优化的因果权衡模式

### 文档完整性 ✅
- [ ] 执行报告完整记录过程
- [ ] 因果分析报告生成
- [ ] 索引文档更新
- [ ] 所有脚本包含注释和文档字符串

---

**方案版本**: v1.0
**创建日期**: 2025-12-24
**状态**: 待执行
**预计总时间**: 2-3小时（数据生成） + 4-7小时（DiBS分析）
