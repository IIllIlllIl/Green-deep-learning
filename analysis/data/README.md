# Analysis数据目录说明

**最后更新**: 2025-12-23
**目录结构版本**: v2.2 - 数据纠正（使用data.csv，56列精简格式）⭐

---

## 📁 目录结构

```
data/
├── paper_replication/      # 论文复现数据（ASE 2023）
│   ├── adult_training_data.csv          # Adult数据集训练数据
│   ├── demo_training_data.csv           # 演示数据
│   └── large_scale_training_data.csv    # 大规模测试数据
│
└── energy_research/        # 能耗研究数据（主项目）
    ├── raw/                # 原始数据
    ├── processed/          # 预处理后的数据
    └── experiments/        # 按实验编号组织的数据
```

---

## 📊 数据集说明

### 1. paper_replication/ - 论文复现数据

**用途**: 复现ASE 2023论文《Causality-Aided Trade-off Analysis for Machine Learning Fairness》

**数据集**:

| 文件名 | 数据集 | 样本数 | 特征数 | 说明 |
|--------|--------|--------|--------|------|
| `adult_training_data.csv` | Adult Income | 10 | 10 | 2方法 × 5 alpha值 |
| `demo_training_data.csv` | 演示数据 | 5 | 8 | 演示DiBS+DML流程 |
| `large_scale_training_data.csv` | 大规模测试 | 100 | 10 | 测试DiBS可扩展性 |

**关键特征**:
- 输入变量: `fairness_method`, `alpha` (公平性参数)
- 输出变量: `Tr_Acc`, `Tr_F1`, `Te_Acc`, `Te_F1`, `Tr_Fair`, `Te_Fair`
- 研究目标: 识别公平性-准确率权衡的因果关系

**相关文档**:
- [Adult完整分析报告](../docs/reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md)
- [代码流程详解](../docs/CODE_WORKFLOW_EXPLAINED.md)

---

### 2. energy_research/ - 能耗研究数据

**用途**: 分析训练超参数对深度学习能耗和性能的因果影响

**数据来源**: 主项目 `../results/data.csv`（726个实验，经过处理的精简数据）⭐

**最新更新**: 2025-12-23 - 纠正数据加载（使用data.csv而非raw_data.csv）

#### 2.1 raw/ - 原始数据

**存放内容**:
- 从主项目提取的**经过处理的精简数据**（data.csv）
- **来源**: `/home/green/energy_dl/nightly/results/data.csv` (726个实验)

**重要说明** ⚠️:
- 此目录名为"raw"但存放的是主项目**已处理**的data.csv（非raw_data.csv）
- data.csv是从raw_data.csv精简而来，删除了32个fg_*详细列，适合因果分析使用

**当前文件**:
- `energy_data_original.csv` (296KB) - 主项目data.csv的副本（2025-12-23纠正）⭐
- `energy_data_original.csv.backup_54col_20251222` (276KB) - 旧版本备份（54列，676行）
- `energy_data_original.csv.WRONG_raw_data_87col_20251223` (321KB) - 错误加载的raw_data.csv备份（87列，已废弃）

**data.csv数据格式**:
- **行数**: 727行（含表头）= 726个有效实验
- **列数**: 56列（精简格式）
- **关键列分类**:

  **元信息 (10列)**:
  - `experiment_id`, `timestamp`, `repository`, `model`, `mode`
  - `is_parallel`, `training_success`, `duration_seconds`, `retries`, `mutated_param`

  **超参数 (9列)**:
  - `hyperparam_alpha`, `hyperparam_batch_size`, `hyperparam_dropout`
  - `hyperparam_epochs`, `hyperparam_kfold`, `hyperparam_learning_rate`
  - `hyperparam_max_iter`, `hyperparam_seed`, `hyperparam_weight_decay`

  **能耗 (11列)**:
  - CPU能耗: `energy_cpu_pkg_joules`, `energy_cpu_ram_joules`, `energy_cpu_total_joules`
  - GPU能耗: `energy_gpu_total_joules`, `energy_gpu_avg/max/min_watts`
  - GPU监控: `energy_gpu_util_avg/max_percent`, `energy_gpu_temp_avg/max_celsius`

  **性能 (16列)**:
  - 通用: `perf_accuracy`, `perf_test_accuracy`, `perf_best_val_accuracy`
  - 损失: `perf_test_loss`, `perf_final_training_loss`, `perf_eval_loss`
  - Person_reID: `perf_map`, `perf_rank1/5`, `perf_precision`, `perf_recall`
  - Bug定位: `perf_top1/5/10/20_accuracy`
  - 其他: `perf_eval_samples_per_second`

  **后台任务信息 (4列)**:
  - `bg_repository`, `bg_model`, `bg_note`, `bg_log_directory`

  **前景任务元信息 (3列)**:
  - `fg_duration_seconds`, `fg_retries`, `fg_error_message`

  **其他信息 (3列)**:
  - `error_message`, `experiment_source`, `num_mutated_params`

**data.csv vs raw_data.csv对比**:

| 特性 | data.csv (当前使用) | raw_data.csv (未使用) |
|------|-------------------|---------------------|
| **列数** | 56列 | 87列 |
| **数据处理** | ✅ 已处理（精简） | ❌ 未处理（完整） |
| **fg_*详细列** | ❌ 已删除（32列） | ✅ 包含（32列） |
| **is_parallel列** | ✅ 有 | ❌ 无 |
| **适用场景** | 因果分析（推荐）⭐ | 数据存档、深度分析 |

**样本量分布**:

| Repository | 样本数 | 占比 |
|------------|--------|------|
| examples (MNIST系列) | 219 | 30.2% |
| VulBERTa | 142 | 19.6% |
| bug-localization-by-dnn-and-rvsm | 132 | 18.2% |
| Person_reID_baseline_pytorch | 116 | 16.0% |
| MRT-OAST | 78 | 10.7% |
| pytorch_resnet_cifar10 | 39 | 5.4% |
| **总计** | **726** | **100%** |

**关键优势** ✅:
- ✅ **简洁**: 56列 vs 87列，减少36%列数
- ✅ **易用**: `is_parallel`列直接标识并行模式
- ✅ **完整**: 包含因果分析所需的所有关键变量
- ✅ **避免混淆**: 删除fg_详细列，避免与基础列产生歧义

**注意事项** ⚠️:
- ⚠️ **列名**: 主项目使用`repository`而非`repo`作为仓库列名
- ⚠️ **数据语义**: 并行模式下，基础列（`hyperparam_*`, `energy_*`, `perf_*`）通常表示前景任务数据
- 📊 **纠正报告**: 查看 [数据加载纠正报告](../docs/reports/DATA_LOADING_CORRECTION_REPORT_20251223.md)

#### 2.2 processed/ - 预处理数据

**存放内容**:
- 缺失值处理后的数据
- 变量类型转换后的数据（DiBS要求数值型）
- 标准化/归一化后的数据

**预处理步骤**:
1. 删除缺失能耗或性能数据的样本
2. One-Hot编码分类变量（mode, optimizer等）
3. 标准化连续变量（可选）
4. 生成交互项特征（可选）

#### 2.3 experiments/ - 实验数据

**存放内容**: 按实验设计组织的数据集

**示例实验**:
- `exp001_lr_bs_energy/` - 学习率和批量大小对能耗的影响
- `exp002_parallel_mode_comparison/` - 并行/非并行模式对比
- `exp003_model_architecture/` - 模型架构的因果分析

**实验数据格式**:
```
experiments/
└── exp001_lr_bs_energy/
    ├── README.md          # 实验说明
    ├── design.json        # 实验设计参数
    ├── train_data.csv     # 训练数据
    └── metadata.json      # 元数据（样本量、变量列表等）
```

---

## 🔄 数据隔离原则

**为什么隔离**:
1. **防止混淆**: 论文复现数据（Adult等）与能耗研究数据（主项目）有不同的研究目标
2. **易于管理**: 各自独立的数据、结果、日志目录
3. **可复现性**: 论文复现结果保持独立，方便验证

**隔离策略**:
- 论文复现数据 → `paper_replication/`
- 能耗研究数据 → `energy_research/`
- 各自独立的结果目录（`results/`）和日志目录（`logs/`）

---

## 📖 使用指南

### 查看论文复现数据

```bash
# Adult数据集
cd data/paper_replication/
head -5 adult_training_data.csv
```

### 使用能耗研究数据

```bash
# 1. 提取主项目数据（首次使用）
python3 scripts/convert_energy_data.py

# 2. 查看原始数据
cd data/energy_research/raw/
ls -lh

# 3. 预处理数据
python3 scripts/preprocess_energy_data.py

# 4. 查看预处理结果
cd data/energy_research/processed/
head -5 processed_energy_data.csv
```

---

## ⚠️ 注意事项

1. **不要混合数据**: 论文复现和能耗研究数据应保持独立
2. **保留原始数据**: `raw/` 目录中的原始数据不要修改，所有处理在 `processed/` 中进行
3. **实验数据版本控制**: 每个实验使用独立子目录，包含完整的元数据
4. **大文件管理**: 对于 > 100MB 的数据文件，考虑使用Git LFS或外部存储

---

## 📚 相关文档

- [数据迁移指南](../docs/MIGRATION_GUIDE.md) - 如何将新数据集应用到因果分析
- [文档总索引](../docs/INDEX.md) - 所有文档的索引
- [项目README](../README.md) - analysis模块总体介绍

---

**维护者**: Analysis模块维护团队
**联系方式**: 查看项目根目录CLAUDE.md
