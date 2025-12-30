# 项目结构重组方案

**日期**: 2025-12-21
**目标**: 整理项目结构，消除冗余，提高可维护性

---

## 📋 当前问题分析

### 1. 根目录混乱
```
问题:
- 8个Python脚本分散在根目录
- 5个日志文件在根目录
- 2个Shell脚本在根目录
- 混合了demo、测试、工具脚本

影响:
- 难以快速找到需要的脚本
- 新用户不知道从哪里开始
- 维护困难
```

### 2. 文档重复
```
问题:
docs/reports/ 有26个报告文档
  - STAGE1_2开头的文档有5个（重复）
  - PROJECT_STATUS开头的文档有3个（重复）
  - REPLICATION相关文档有2个（重复）

影响:
- 信息分散，难以找到最新版本
- 浪费存储空间
- 维护负担重
```

### 3. 临时文件未归档
```
问题:
- *.log 文件散落在根目录
- *.txt 状态文件在根目录
- 没有统一的日志管理

影响:
- 根目录看起来杂乱
- 难以追踪历史运行记录
```

---

## 🎯 重组方案

### 方案1: 创建scripts目录

**目的**: 集中管理所有可执行脚本

**结构**:
```
scripts/
├── demos/              # 演示脚本
│   ├── demo_quick_run.py
│   ├── demo_large_scale.py
│   ├── demo_adult_dataset.py
│   └── demo_adult_full_analysis.py
├── experiments/        # 完整实验脚本（带后台运行支持）
│   └── run_adult_analysis.sh
├── utils/              # 工具脚本
│   ├── monitor_progress.sh
│   └── activate_env.sh
└── testing/            # 测试脚本
    ├── test_dibs_quick.py
    └── run_tests.py
```

**保留在根目录**:
```
/
├── config.py           # 配置文件
├── README.md           # 主文档
└── requirements.txt    # 依赖
```

### 方案2: 整合文档

**docs/reports/ 整合方案**:

**保留（最新/最完整的）**:
```
reports/
├── ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md    # 最新Adult完整分析
├── ADULT_DATASET_VALIDATION_REPORT.md          # Adult验证报告
├── LARGE_SCALE_EXPERIMENT_REPORT.md            # 大规模实验
├── REPLICATION_EVALUATION.md                   # 复现评估（最全）
└── TEST_VALIDATION_REPORT.md                   # 测试验证
```

**归档（过时的阶段性报告）**:
```
reports/archives/
├── STAGE1_2_COMPLETE_REPORT.md
├── STAGE1_2_FINAL_REPORT.md
├── STAGE1_2_PROGRESS_REPORT.md
├── STAGE1_2_SUMMARY.md
├── STAGE1_COMPLETION_REPORT.md
├── PROJECT_STATUS_REPORT.md
├── PROJECT_STATUS_SUMMARY.md
├── PROGRESS_UPDATE.md
├── PAPER_COMPARISON_REPORT.md
├── GPU_TEST_REPORT.md
├── CODE_REVIEW_REPORT.md
├── DELIVERY_CHECKLIST.md
└── TASK_COMPLETION_SUMMARY.md
```

**删除（重复的）**:
- 无（移动到archives即可）

**docs/guides/ 整合方案**:

**保留**:
```
guides/
├── ENVIRONMENT_SETUP.md         # 环境配置
├── QUICK_START.md              # 快速开始（合并REPLICATION_QUICK_START）
├── USAGE_GUIDE.md              # 使用指南（合并USAGE_GUIDE_FOR_NEW_RESEARCH）
└── IMPROVEMENT_GUIDE.md         # 改进指南
```

**删除/合并**:
- REPLICATION_QUICK_START.md → 合并到 QUICK_START.md
- USAGE_GUIDE_FOR_NEW_RESEARCH.md → 合并到 USAGE_GUIDE.md
- FULL_REPLICATION_PLAN.md → 归档（已完成）
- QUICK_SUMMARY.md → 归档（信息已过时）
- DOCUMENTATION_INDEX.md → 重新生成

**顶层文档（docs/）**:
```
docs/
├── CODE_WORKFLOW_EXPLAINED.md   # 代码流程说明
├── MIGRATION_GUIDE.md           # 迁移指南
└── INDEX.md                     # 文档索引（新建）
```

### 方案3: 创建logs目录

**目的**: 统一管理日志文件

**结构**:
```
logs/
├── experiments/        # 实验日志
│   ├── adult_full_analysis_20251221_163516.log
│   ├── adult_dataset_run.log
│   └── large_scale_run.log
├── demos/             # 演示日志
│   └── demo_output.log
└── status/            # 状态文件
    └── adult_analysis_status.txt
```

---

## 📝 执行步骤

### 步骤1: 创建目录结构
```bash
mkdir -p scripts/{demos,experiments,utils,testing}
mkdir -p logs/{experiments,demos,status}
mkdir -p docs/reports/archives
```

### 步骤2: 移动脚本文件
```bash
# Demo脚本
mv demo_*.py scripts/demos/

# 实验脚本
mv run_adult_analysis.sh scripts/experiments/

# 工具脚本
mv monitor_progress.sh activate_env.sh scripts/utils/

# 测试脚本
mv test_dibs_quick.py run_tests.py scripts/testing/
```

### 步骤3: 移动日志文件
```bash
# 实验日志
mv adult_full_analysis_*.log adult_dataset_run.log large_scale_run.log logs/experiments/

# Demo日志
mv demo_output.log logs/demos/

# 状态文件
mv adult_analysis_status.txt logs/status/
```

### 步骤4: 整理文档
```bash
# 归档过时报告
cd docs/reports
mv STAGE1_*.md PROJECT_STATUS*.md PROGRESS_UPDATE.md archives/
mv PAPER_COMPARISON_REPORT.md GPU_TEST_REPORT.md archives/
mv CODE_REVIEW_REPORT.md DELIVERY_CHECKLIST.md archives/
mv TASK_COMPLETION_SUMMARY.md archives/

# 归档过时指南
cd ../guides
mv FULL_REPLICATION_PLAN.md QUICK_SUMMARY.md ../reports/archives/
```

### 步骤5: 创建新文档
- 创建 docs/INDEX.md (文档总索引)
- 合并 guides 中的重复文档
- 更新 README.md 反映新结构

---

## 🎯 预期结果

### 最终目录结构
```
/home/green/energy_dl/analysis/
├── data/                    # 数据文件
├── docs/                    # 文档
│   ├── guides/             # 指南（4个）
│   ├── reports/            # 报告（5个活跃 + archives/）
│   ├── CODE_WORKFLOW_EXPLAINED.md
│   ├── MIGRATION_GUIDE.md
│   └── INDEX.md
├── logs/                    # 日志（新建）
│   ├── experiments/
│   ├── demos/
│   └── status/
├── results/                 # 结果文件
├── scripts/                 # 脚本（新建）
│   ├── demos/
│   ├── experiments/
│   ├── utils/
│   └── testing/
├── tests/                   # 单元测试
├── utils/                   # 核心模块
├── config.py               # 配置
├── README.md               # 主文档
└── requirements.txt        # 依赖

根目录文件数量: 3个（从15+个减少）
文档数量: 活跃10个 + 归档13个（从26个整理）
```

---

## ✅ 验证清单

重组完成后需要验证：

```
□ 根目录只有3个文件（config.py, README.md, requirements.txt）
□ 所有脚本都在 scripts/ 目录下
□ 所有日志都在 logs/ 目录下
□ docs/reports/ 只有5个活跃报告
□ docs/guides/ 只有4个指南
□ 创建了 docs/INDEX.md 总索引
□ 更新了 README.md 反映新结构
□ 所有脚本的import路径仍然正确
□ 测试套件仍然能运行
```

---

**批准后执行此方案**
