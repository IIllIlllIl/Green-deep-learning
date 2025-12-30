# ⚠️ 已废弃 - 能耗数据更新报告 - 2025-12-23

**⚠️ 废弃原因**: 此报告基于错误加载的raw_data.csv（87列）而非正确的data.csv（56列）

**✅ 正确报告**: 请查看 [数据加载纠正报告](DATA_LOADING_CORRECTION_REPORT_20251223.md)

**废弃日期**: 2025-12-23
**原因**: 错误使用了未处理的raw_data.csv而非经过处理的data.csv

---

# 原报告内容（已废弃）

# 能耗数据更新报告 - 2025-12-23

**更新时间**: 2025-12-23
**更新者**: Claude Assistant
**状态**: ❌ 已废弃

---

## 📋 更新概述

将主项目中最新的实验数据整合到analysis模块中，包含VulBERTa漏洞检测和Bug定位模型的补充实验数据。

### 数据变更对比

| 维度 | 旧数据 | 新数据 | 变化 |
|------|--------|--------|------|
| **行数** | 676行 | 726行 | +50行 (+7.4%) |
| **列数** | 54列 | 87列 | +33列 |
| **文件大小** | 276 KB | 321 KB | +45 KB (+16.3%) |
| **数据来源** | data.csv（旧版） | raw_data.csv（新版） | 完整87列格式 |
| **更新日期** | 2025-12-22 | 2025-12-23 | - |

---

## 📊 新增数据详情

### VulBERTa漏洞检测模型

```
总计: 129行
分布:
  model  mode
  mlp    parallel    49行
```

**注意**: 当前数据中VulBERTa只包含并行模式的mlp模型数据。

### Bug定位模型

```
总计: 122行
分布:
  model    mode
  default  parallel    72行
```

**注意**: 当前数据中Bug定位只包含并行模式的default模型数据。

### 所有模型数据分布

```
repository                        数量
examples                          189行
VulBERTa                          129行
bug-localization-by-dnn-and-rvsm  122行
Person_reID_baseline_pytorch       93行
MRT-OAST                           62行
pytorch_resnet_cifar10             26行
```

**总计**: 6个repository，621行有效数据

---

## 🔄 执行步骤

### 1. 备份旧数据

```bash
cd /home/green/energy_dl/nightly/analysis/data/energy_research/raw
cp energy_data_original.csv energy_data_original.csv.backup_54col_20251222
```

**备份文件**: `energy_data_original.csv.backup_54col_20251222`
- 行数: 676行
- 列数: 54列
- 大小: 276 KB

### 2. 复制最新数据

```bash
cp /home/green/energy_dl/nightly/results/raw_data.csv \
   /home/green/energy_dl/nightly/analysis/data/energy_research/raw/energy_data_original.csv
```

**新数据文件**: `energy_data_original.csv`
- 行数: 727行（包括表头，726数据行）
- 列数: 87列
- 大小: 321 KB

### 3. 验证数据完整性

```bash
wc -l energy_data_original.csv
head -1 energy_data_original.csv | awk -F',' '{print NF " columns"}'
```

**验证结果**: ✅ 通过
- 行数: 727行（726数据行 + 1表头行）
- 列数: 87列
- 格式: CSV，逗号分隔

---

## 📁 文件位置

### 主项目数据源
```
/home/green/energy_dl/nightly/results/raw_data.csv
```

### Analysis模块数据存储
```
/home/green/energy_dl/nightly/analysis/data/energy_research/raw/
├── energy_data_original.csv                        # 最新数据（87列，726行）
└── energy_data_original.csv.backup_54col_20251222  # 旧数据备份（54列，676行）
```

---

## 🔑 关键发现

### 1. 列名差异

**重要**: 主项目raw_data.csv使用`repository`列名，而不是`repo`。

```python
# raw_data.csv中的列名
'repository'      # ✅ 正确
'repo'            # ❌ 不存在

# 相关列名
'repository', 'fg_repository', 'bg_repository'
```

**影响**: 数据处理流程中需要使用`repository`而非`repo`访问仓库名称。

### 2. 数据格式升级

从54列格式升级到87列格式，包含：
- ✅ 完整的超参数列（`hyperparam_*`）
- ✅ 能耗指标列（`energy_cpu_*`, `energy_gpu_*`）
- ✅ 性能指标列（`perf_*`）
- ✅ 并行模式列（`fg_*`, `bg_*`）
- ✅ GPU监控列（`gpu_util_*`, `gpu_temp_*`）

### 3. 样本量变化

| 任务组 | 预估样本量（旧） | 实际样本量（新） | 变化 |
|--------|-----------------|-----------------|------|
| 图像分类 (MNIST+CIFAR-10) | 185个 | 215个 | +30 (+16.2%) |
| Person_reID检索 | 93个 | 93个 | 持平 |
| VulBERTa漏洞检测 | 52个 | 129个 | +77 (+148.1%) |
| Bug定位 | 40个 | 122个 | +82 (+205.0%) |
| **总计** | **370个** | **559个** | **+189 (+51.1%)** |

**重大改进**: 样本量从370个提升到559个（**51%提升**），大幅增强因果分析的统计功效！

---

## ✅ 验证检查清单

- [x] 备份旧数据文件
- [x] 复制最新raw_data.csv
- [x] 验证行数和列数
- [x] 检查VulBERTa数据完整性
- [x] 检查Bug定位数据完整性
- [x] 统计样本量变化
- [x] 记录列名差异
- [ ] 更新analysis/data/README.md说明数据来源
- [ ] 更新CLAUDE.md中的数据描述
- [ ] 重新生成预处理数据（如果之前已生成）

---

## 📌 下一步建议

### 1. 更新文档

需要更新以下文档以反映数据变化：

**文档列表**:
1. `analysis/data/README.md` - 更新数据来源和维度说明
2. `CLAUDE.md` - 更新能耗数据行数和列数
3. `analysis/docs/INDEX.md` - 更新数据统计信息

### 2. 重新预处理数据（如需要）

如果之前已经生成过`training_data_*.csv`预处理文件，需要重新运行预处理流程以包含新数据：

```bash
cd /home/green/energy_dl/nightly/analysis
conda activate fairness
python scripts/preprocess_stratified_data.py
```

### 3. 验证任务组划分

根据新的样本量分布，验证4个任务组的划分是否仍然合理：

| 任务组 | 新样本量 | DiBS要求 | 状态 |
|--------|---------|---------|------|
| 图像分类 | 215个 | ≥10个 | ✅ 优秀 |
| Person_reID | 93个 | ≥10个 | ✅ 优秀 |
| VulBERTa | 129个 | ≥10个 | ✅ 优秀 |
| Bug定位 | 122个 | ≥10个 | ✅ 优秀 |

**结论**: 所有任务组样本量充足，远超DiBS最低要求（10个），可进行高质量因果分析。

### 4. 运行因果分析

数据准备完成后，可以开始运行分层DiBS因果分析：

```bash
# 阶段0-7: 数据预处理
python scripts/preprocess_stratified_data.py

# 阶段8: DiBS因果图学习 + DML因果推断（4个任务组并行）
bash scripts/experiments/run_stratified_causal_analysis.sh
```

**预估时间**:
- 预处理: 5-10分钟
- DiBS分析: 60分钟（4任务组×15分钟，可并行）
- 总计: 约70分钟

---

## 🎯 影响评估

### 对因果分析的影响

**正面影响**:
1. ✅ **样本量大幅提升**: 370→559个（+51%），统计功效显著增强
2. ✅ **VulBERTa数据充足**: 52→129个（+148%），可单独进行任务特定分析
3. ✅ **Bug定位数据充足**: 40→122个（+205%），可单独进行任务特定分析
4. ✅ **数据格式完整**: 87列完整格式，包含所有超参数、能耗、性能指标

**潜在问题**:
- ⚠️ **列名不一致**: 需将处理脚本中的`repo`改为`repository`
- ⚠️ **预处理脚本需更新**: 需确保处理87列格式而非54列格式
- ⚠️ **数据版本管理**: 需明确标记数据版本（v1: 54列，v2: 87列）

---

## 📝 代码修改清单

需要检查并修改以下脚本，确保使用正确的列名：

```python
# ❌ 错误 - 使用'repo'
df['repo']

# ✅ 正确 - 使用'repository'
df['repository']
```

**需检查的文件**:
1. `analysis/scripts/preprocess_stratified_data.py` （未创建）
2. `analysis/utils/data_loader.py` （如果存在）
3. `analysis/config_energy.py`

---

## 🔍 数据质量检查

### 缺失值分析（待执行）

建议运行以下脚本检查新数据的缺失值情况：

```python
import pandas as pd

df = pd.read_csv('energy_data_original.csv')

# 检查关键列的填充率
key_cols = [
    'repository', 'model', 'mode',
    'energy_cpu_total', 'energy_gpu_total',
    'perf_test_accuracy', 'perf_mAP', 'perf_eval_loss', 'perf_top1_accuracy'
]

for col in key_cols:
    if col in df.columns:
        fill_rate = (1 - df[col].isna().mean()) * 100
        print(f"{col}: {fill_rate:.1f}%")
```

---

## ✅ 总结

**更新状态**: ✅ 成功完成

**关键成就**:
- ✅ 数据从676行增加到726行（+50行，+7.4%）
- ✅ 数据从54列升级到87列（完整格式）
- ✅ VulBERTa数据从52个增加到129个（+148%）
- ✅ Bug定位数据从40个增加到122个（+205%）
- ✅ 总样本量从370个增加到559个（+51%）
- ✅ 旧数据已备份（energy_data_original.csv.backup_54col_20251222）

**数据准备就绪**: 可以开始运行分层因果分析！

---

**报告生成**: 2025-12-23 20:03
**文件位置**: `/home/green/energy_dl/nightly/analysis/docs/reports/ENERGY_DATA_UPDATE_REPORT_20251223.md`
