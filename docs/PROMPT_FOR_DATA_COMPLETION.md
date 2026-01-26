# 下一个对话：数据补完任务 Prompt

**创建日期**: 2026-01-26
**适用场景**: 下一次启动Claude Code时使用
**优先级**: 高

---

## 📋 任务概述

**目标**: 补完能耗数据分析项目的缺失数据

**当前状态**:
- 数据完整性: 95.1% (795/836条有效数据)
- 已完成ATE集成实施���2026-01-26）
- 准备进行因果推断分析

**核心问题**:
- 剩余4.9%的数据缺失（41个实验）
- 需要补完这些数据以进行完整的因果分析

---

## 🗂️ 关键文件路径

### 数据文件

```
/home/green/energy_dl/nightly/
├── data/
│   ├── raw_data.csv                  # 主数据文件 (87列, 1225行)
│   ├── data.csv                      # 精简数据文件 (56列, 971行)
│   ├── backups/                      # 数据备份目录
│   └── recoverable_energy_data.json  # 可恢复数据清单
```

### 核心脚本

```
/home/green/energy_dl/nightly/
├── tools/data_management/
│   ├── validate_raw_data.py          # 验证数据完整性
│   ├── analyze_missing_energy_data.py # 分析缺失数据
│   ├── verify_recoverable_data.py    # 验证数据可恢复性
│   ├── repair_missing_energy_data.py # 修复缺失数据
│   ├── append_session_to_raw_data.py # 追加新实验数据
│   └── create_unified_data_csv.py    # 创建统一数据文件
│
├── analysis/utils/
│   └── causal_inference.py           # 因果推��引擎（刚完成）
│
└── analysis/tests/
    └── test_ctf_style_ate.py         # ATE功能测试（刚完成）
```

### 关键文档

```
/home/green/energy_dl/nightly/docs/
├── results_reports/
│   ├── DATA_REPAIR_REPORT_20260104.md           # 数据修复报告
│   └── DATA_USABILITY_SUMMARY_20260113.md       # 数据可用性总结
│
├── guides/
│   └── APPEND_SESSION_TO_RAW_DATA_GUIDE.md      # 数据追加指南
│
├── DATA_USAGE_GUIDE.md                          # 数据使用指南
│
└── current_plans/
    ├── ATE_INTEGRATION_COMPLETION_REPORT_20260126.md  # ATE实施完成报告
    ├── CTF_STYLE_ATE_QUICK_START_20260126.md          # 快速使用指南
    └── ATE_PROJECT_STATUS_20260126.md                 # 项目状态
```

### 实验数据目录

```
/home/green/energy_dl/nightly/archives/
└── experiments/
    └── [YYYY-MM-DD]/
        ├── [model_name]/
        │   ├── experiment.json          # 实验配置和结果
        │   ├── foreground.log           # 前台日志
        │   ├── background.log           # 后台日志
        │   └── metrics.csv              # 性能指标
```

---

## 🎯 具体任务

### 任务1: 识别缺失数据（1-2小时）

**目标**: 找出所有缺失的数据及其原因

**步骤**:
```bash
cd /home/green/energy_dl/nightly

# 1. 验证当前数据完整性
python3 tools/data_management/validate_raw_data.py

# 2. 分析缺失数据详情
python3 tools/data_management/analyze_missing_energy_data.py

# 3. 检查数据可恢复性
python3 tools/data_management/verify_recoverable_data.py
```

**预期输出**:
- 缺失数据的实验ID列表
- 缺失数据的原因分类
- 可恢复的数据清单

### 任务2: 补完缺失数据（2-4小时）

**目标**: 从实验文件中恢复缺失的数据

**情况A: 数据存在但未提取**
```bash
# 使用修复脚本
python3 tools/data_management/repair_missing_energy_data.py

# 验证修复结果
python3 tools/data_management/validate_raw_data.py
```

**情况B: 需要重新运行实验**
```bash
# 查看需要重新运行的实验
cat data/missing_experiments_list.txt

# 使用mutation.py重新运行
python3 mutation.py --config <config_file> --repository <repo> --model <model>
```

**情况C: 数据永久丢失**
- 标记为不可恢复
- 更新数据质量报告
- 评估对分析的影响

### 任务3: 验证数据完整性（1小时）

**目标**: 确保所有数据完整且一致

```bash
# 1. 验证修复后的数据
python3 tools/data_management/validate_raw_data.py

# 2. 对比raw_data.csv和data.csv
python3 tools/data_management/compare_data_vs_raw_data.py

# 3. 检查数据质量
python3 tools/data_management/independent_quality_assessment.py
```

**成功标准**:
- 数据完整性 ≥ 98%
- 无关键能耗指标缺失
- 数据格式一致

### 任务4: 追加新实验数据（如有）（1-2小时）

**目标**: 如果有新的实验，追加到数据集

```bash
# 参考指南
cat docs/guides/APPEND_SESSION_TO_RAW_DATA_GUIDE.md

# 运行追加脚本
python3 tools/data_management/append_session_to_raw_data.py \
    --experiment-dir <experiment_directory> \
    --output data/raw_data.csv
```

---

## 📊 数据结构说明

### raw_data.csv 结构（87列）

**关键列**:
```csv
experiment_id          # 实验唯一ID
timestamp              # 时间戳
repository             # 代码仓库
model                  # 模型名称
is_parallel            # 是否并行模式 (0=非并行, 1=并行)

# 性能指标 (perf_*)
perf_accuracy
perf_test_accuracy
perf_training_time
...

# 能耗指标 - 非并行模式 (energy_*)
energy_cpu_total_joules
energy_gpu_avg_watts
energy_gpu_total_joules
...

# 能耗指标 - 并行模式 (fg_*)
fg_duration_seconds
fg_energy_cpu_total_joules
fg_energy_gpu_total_joules
...
```

**注意事项**:
- 并行模式使用 `fg_*` 前缀字段
- 非并行模式使用 `energy_*` 前缀字段
- 部分实验可能同时有两组数据

### data.csv 结构（56列）

**特点**:
- 精简版本，移除了部分冗余字段
- 统一了并行/非并行字段的命名
- 添加了 `is_parallel` 列便于区分

---

## 🔧 常用工具和命令

### 数据验证

```bash
# 快速检查完整性
python3 tools/data_management/validate_raw_data.py

# 查看数据统计
head -5 data/raw_data.csv
wc -l data/raw_data.csv

# 查看缺失值
cd analysis
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('../data/raw_data.csv')
missing = df.isnull().sum()
print(missing[missing > 0])
EOF
```

### 数据修复

```bash
# 修复缺失的能耗数据
python3 tools/data_management/repair_missing_energy_data.py \
    --input data/raw_data.csv \
    --output data/raw_data_repaired.csv \
    --experiment-root /home/green/energy_dl/nightly/archives/experiments

# 备份原数据
cp data/raw_data.csv data/backups/raw_data_before_repair.csv
```

### 数据追加

```bash
# 追加单个实验
python3 tools/data_management/append_session_to_raw_data.py \
    --experiment-dir archives/experiments/2026-01-26/ResNet \
    --output data/raw_data.csv

# 追加整个会话
python3 tools/data_management/append_session_to_raw_data.py \
    --session-dir archives/experiments/2026-01-26 \
    --output data/raw_data.csv
```

---

## 📈 当前数据状态

### 数据完整性（截至2026-01-26）

```
总实验数: 970个（含header，实际969条数据）
├─ 完全可用: 577条 (59.5%)
│  └─ 训练成功 + 有能耗数据 + 有性能指标
├─ 仅有能耗数据: 251条 (25.9%)
│  └─ 训练失败但有能耗记录
└─ 其他情况: 141条 (14.6%)
   └─ 数据不完整或缺失

能耗数据可用性: 828/969 (85.4%)
```

### 已知问题

1. **部分实验缺失能耗数据**（141条）
   - 主要是并行模式的实验
   - 需要从experiment.json中提取

2. **数据格式不一致**
   - raw_data.csv: 87列（包含fg_前缀字段）
   - data.csv: 56列（统一字段）

3. **实验ID可能重复**
   - 需要使用复合键：experiment_id + timestamp

---

## ✅ 验收标准

### 数据完整性
- [ ] 原始数据完整性 ≥ 98%
- [ ] 能耗数据完整性 ≥ 95%
- [ ] 无关键指标缺失

### 数据质量
- [ ] 无重复记录（或正确处理重复）
- [ ] 数据类型正确
- [ ] 数值范围合理
- [ ] 时间戳一致

### 文档更新
- [ ] 更新数据质量报告
- [ ] 记录修复过程
- [ ] 更新CLAUDE.md中的状态

### 测试验证
- [ ] 运行数据验证脚本通过
- [ ] 运行ATE分析无错误
- [ ] 因果推断结果合理

---

## 🚨 常见问题和解决方案

### Q1: 数据文件很大，如何高效处理？

**解决方案**:
```python
# 使用chunksize分块读取
import pandas as pd

chunk_size = 1000
for chunk in pd.read_csv('data/raw_data.csv', chunksize=chunk_size):
    process(chunk)
```

### Q2: 如何处理重复的实验ID？

**解决方案**:
```python
# 使用复合键
df['composite_key'] = df['experiment_id'] + '|' + df['timestamp'].astype(str)

# 去重
df = df.drop_duplicates(subset=['composite_key'], keep='last')
```

### Q3: 能耗数据提取失败怎么办？

**解决方案**:
```bash
# 1. 检查experiment.json是否存在
ls archives/experiments/*/experiment.json

# 2. 检查JSON格式
cat archives/experiments/2026-01-XX/YYY/experiment.json | jq .

# 3. 手动提取单个实验
python3 << 'EOF'
import json
with open('path/to/experiment.json') as f:
    data = json.load(f)
    print(data.get('energy_metrics', {}))
EOF
```

### Q4: 如何快速验证修复效果？

**解决方案**:
```bash
# 运行完整验证套件
cd /home/green/energy_dl/nightly
python3 tools/data_management/validate_raw_data.py \
    && python3 tools/data_management/compare_data_vs_raw_data.py \
    && echo "✅ 验证通过"
```

---

## 📞 参考资源

### 内部文档
1. **数据使用指南**: `docs/DATA_USAGE_GUIDE.md`
   - 详细的数据格式说明
   - 字段含义和单位

2. **数据修复报告**: `docs/results_reports/DATA_REPAIR_REPORT_20260104.md`
   - 之前的修复经验
   - 常见问题和解决方案

3. **数据追加指南**: `docs/guides/APPEND_SESSION_TO_RAW_DATA_GUIDE.md`
   - 如何追加新实验数据
   - 注意事项和最佳实践

### 代码示例
1. **数据验证脚本**: `tools/data_management/validate_raw_data.py`
   - 验证逻辑参考

2. **数据修复脚本**: `tools/data_management/repair_missing_energy_data.py`
   - 修复方法参考

3. **因果推断引擎**: `analysis/utils/causal_inference.py`
   - 数据使用示例

### 相关链接
- 项目README: `README.md`
- CLAUDE指南: `CLAUDE.md`
- ATE实施报告: `docs/current_plans/ATE_INTEGRATION_COMPLETION_REPORT_20260126.md`

---

## 🎯 快速开始 Checklist

**首次进入项目时**:
- [ ] 阅读 `CLAUDE.md`（5分钟快速指南）
- [ ] 查看 `docs/DATA_USAGE_GUIDE.md`（数据使用必读）
- [ ] 运行 `tools/quick_health_check.sh`（健康检查）
- [ ] 检查 `data/raw_data.csv` 状态

**开始数据补完任务前**:
- [ ] 理解数据结构（87列含义）
- [ ] 运行数据验证脚本
- [ ] 识别缺失数据类型
- [ ] 准备实验数据路径

**任务完成后**:
- [ ] 运行完整验证套件
- [ ] 更新数据质量报告
- [ ] 备份修复后的数据
- [ ] 测试因果推断功能

---

## 📝 任务日志模板

```markdown
## 数据补完任务执行日志

**日期**: YYYY-MM-DD
**执行者**: [Your Name]
**任务**: 数据补完

### 进度记录

- [ ] 任务1: 识别缺失数据
  - 开始时间:
  - 完成时间:
  - 结果:

- [ ] 任务2: 补完缺失数据
  - 开始时间:
  - 完成时间:
  - 结果:

- [ ] 任务3: 验证数据完整性
  - 开始时间:
  - 完成时间:
  - 结果:

### 数据统计

- 修复前完整性: XX%
- 修复后完整性: XX%
- 新增数据条数: XX
- 修复数据条数: XX

### 问题和解决方案

1. **问题描述**:
   - 解决方案:
   - 耗时:

2. **问题描述**:
   - 解决方案:
   - 耗时:

### 验收确认

- [ ] 数据完整性达标
- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] 备份已完成
```

---

**Prompt版本**: 1.0
**创建日期**: 2026-01-26
**适用项目**: Energy DL Nightly Analysis
**状态**: ✅ 就绪
