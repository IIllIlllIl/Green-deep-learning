# Analysis数据目录说明

**最后更新**: 2025-12-22
**目录结构版本**: v2.0

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

**数据来源**: 主项目 `../results/raw_data.csv`（676个实验，91.1%有效数据）

#### 2.1 raw/ - 原始数据

**存放内容**:
- 从主项目提取的原始CSV数据
- 未经处理的实验记录
- **来源**: `/home/green/energy_dl/nightly/results/data.csv` (676个实验)

**当前文件**:
- `energy_data_original.csv` (276KB) - 主项目data.csv的副本（2025-12-22复制）

**数据格式**:
- **行数**: 677行（含表头）= 676个有效实验
- **列数**: 54列
- **关键列**:
  - 超参数: `learning_rate`, `batch_size`, `epochs`, `dropout`, `weight_decay`
  - 模式: `is_parallel` (Boolean), `mode` (字符串)
  - 模型: `repository`, `model`
  - 能耗: `energy_cpu_*`, `energy_gpu_*`（12列）
  - 性能: `perf_*`（14列）
  - 元数据: `experiment_id`, `timestamp`, `mutated_param`, 等

**注意**:
- 此文件是精简版（54列），不包含并行模式下的前景任务详细数据（`fg_*`字段）
- 如需完整数据（87列），使用 `raw_data.csv`（包含`fg_*`和`bg_*`字段）

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
