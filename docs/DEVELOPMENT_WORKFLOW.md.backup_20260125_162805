# 开发工作流程规范

**版本**: v1.0.0
**最后更新**: 2026-01-10
**适用范围**: Energy DL 项目所有开发任务

---

## 📋 概述

本文档定义了 Energy DL 项目的标准开发工作流程，确保每个任务都经过充分的测试和验证。

**核心原则**: 测试先行 → 小范围验证 → 全量执行 → 独立检查

---

## 🎯 任务执行流程

### 每次执行任务前必须遵循的4个步骤

```
┌─────────────────────────────────────────────────────────────┐
│  步骤1: 理解与规划                                           │
│  ├─ 确认当前项目进度和状态                                   │
│  ├─ 阅读相关规范和文档                                       │
│  ├─ 列出详细的任务步骤                                       │
│  └─ 使用 TodoWrite 创建任务清单                              │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤2: 开发与检查                                           │
│  ├─ ✅ 先编写测试脚本 (test_*.py)                            │
│  ├─ ✅ 开发主功能脚本                                        │
│  ├─ ✅ 先执行 Dry Run（10-20行数据）                         │
│  ├─ ✅ 验证 Dry Run 结果                                     │
│  ├─ ✅ 验证通过后再全量执行                                  │
│  └─ ✅ 运行全量测试                                          │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤3: 独立验证                                             │
│  ├─ 启动独立 Subagent 进行客观检查                           │
│  ├─ Subagent 编写测试并验证数据质量                          │
│  ├─ 审查 Subagent 的验证报告                                 │
│  └─ 根据反馈修复问题（如有）                                 │
└─────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤4: 维护与归档                                           │
│  ├─ 更新相关文档                                             │
│  ├─ 归档旧文件到 archives/ 或 tools/legacy/                  │
│  ├─ 更新版本号和变更日志                                     │
│  └─ 使用中文回答用户                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 详细说明

### 步骤1: 理解与规划

#### 检查清单
- [ ] **阅读 CLAUDE.md** - 了解当前项目状态和主要任务
- [ ] **查看 TodoWrite** - 确认是否有待完成任务
- [ ] **阅读相关规范** - JSON配置、脚本开发等
- [ ] **创建任务清单** - 使用 TodoWrite 列出所有子任务

#### 示例
```bash
# 查看项目状态
cat CLAUDE.md | head -30

# 查看当前任务
# (TodoWrite 工具会显示待办事项)

# 规划任务步骤
# 1. 分析需求
# 2. 设计方案
# 3. 编写测试
# 4. 实现功能
# 5. Dry Run 验证
# 6. 全量执行
# 7. 独立验证
```

---

### 步骤2: 开发与检查

#### 2.1 测试先行原则

**❌ 错误做法**：先写功能，后补测试
```python
# main.py
def process_data(df):
    # 直接实现功能
    ...
```

**✅ 正确做法**：先写测试，再写功能
```python
# tests/test_process_data.py
import unittest

class TestProcessData(unittest.TestCase):
    def test_basic_processing(self):
        df = create_sample_data()
        result = process_data(df)
        self.assertEqual(len(result), 10)

    def test_empty_input(self):
        df = pd.DataFrame()
        result = process_data(df)
        self.assertEqual(len(result), 0)

# 然后再写 main.py
```

#### 2.2 Dry Run 验证

**原则**: 任何数据处理脚本都必须先在小数据集上验证

```bash
# ❌ 错误做法：直接全量执行
python3 script.py --input data/raw_data.csv --output result.csv

# ✅ 正确做法：Dry Run → 检查 → 全量执行

# 步骤1: Dry Run（前10行）
python3 script.py --input data/raw_data.csv --output result_test.csv --dry-run --limit 10

# 步骤2: 检查 Dry Run 结果
head -20 result_test.csv
python3 -c "import pandas as pd; df=pd.read_csv('result_test.csv'); print(df.info()); print(df.head())"

# 步骤3: 验证数据正确性
python3 verify_result.py --file result_test.csv --expected-rows 10

# 步骤4: 全量执行（验证通过后）
python3 script.py --input data/raw_data.csv --output result.csv
```

#### 2.3 全量测试

```bash
# 运行所有单元测试
python3 -m pytest tests/unit/ -v

# 运行特定测试
python3 tests/test_process_data.py

# 验证数据完整性
python3 tools/data_management/validate_raw_data.py
```

---

### 步骤3: 独立验证

**为什么需要独立验证？**
- 🎯 避免确认偏差（自己检查自己的工作容易忽略问题）
- 🎯 独立视角发现潜在遗漏
- 🎯 提供第二层质量保证

**何时使用独立 Subagent？**
- ✅ 数据处理后验证
- ✅ 测试编写
- ✅ 数据质量检查
- ✅ 实验结果分析
- ✅ 配置验证

**详细指南**: 参见 [INDEPENDENT_VALIDATION_GUIDE.md](INDEPENDENT_VALIDATION_GUIDE.md)

---

### 步骤4: 维护与归档

#### 文档更新检查清单
- [ ] 更新 CLAUDE.md（如涉及核心功能）
- [ ] 更新相关专题文档
- [ ] 添加到 SCRIPTS_QUICKREF.md（如是新脚本）
- [ ] 更新版本号和变更日志

#### 归档规则

**一次性任务脚本** → `tools/legacy/completed_*/`
```bash
# 示例：数据修复脚本（任务完成后）
mv tools/data_management/repair_missing_energy_data.py \
   tools/legacy/completed_data_tasks_20260110/
```

**历史实验结果** → `archives/runs/`
```bash
# 示例：旧的实验运行结果
mv results/run_20251201_143022 archives/runs/
```

**过时文档** → `docs/archived/`
```bash
# 示例：旧版方案文档
mv docs/OLD_PLAN.md docs/archived/OLD_PLAN_archived_20260110.md
```

---

## 🚨 常见错误与避免方法

### 错误1: 跳过测试直接全量执行

**后果**:
- 数据损坏风险高
- 错误发现时间延后
- 修复成本增加

**避免方法**:
- 强制要求 Dry Run
- 代码审查检查测试覆盖

### 错误2: 自我验证不客观

**后果**:
- 确认偏差导致遗漏问题
- 边界条件未充分测试

**避免方法**:
- 使用独立 Subagent 验证
- 要求生成验证报告

### 错误3: 忘记更新文档

**后果**:
- 知识流失
- 后续开发者困惑

**避免方法**:
- 在 TodoWrite 中添加"更新文档"任务
- 代码审查时检查文档更新

---

## 📚 相关文档

- [脚本开发规范](SCRIPT_DEV_STANDARDS.md) - 详细的脚本编写标准
- [独立验证规范](INDEPENDENT_VALIDATION_GUIDE.md) - Subagent 验证指南
- [JSON配置规范](JSON_CONFIG_WRITING_STANDARDS.md) - 配置文件书写标准
- [CLAUDE.md](../CLAUDE.md) - 项目快速指南

---

## 🔄 版本历史

### v1.0.0 (2026-01-10)
- 初始版本
- 从 CLAUDE.md 拆分独立
- 添加详细流程图和示例

---

**维护者**: Green
**文档类型**: 开发规范
**强制执行**: 是 ⭐⭐⭐
