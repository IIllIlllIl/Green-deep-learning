# 文件结构整理方案

**创建日期**: 2026-02-01
**目标**: 整理从1月中旬开始错位摆放的文件，使其符合项目结构规范
**状态**: 待确认

---

## 📊 问题概述

从1月中旬开始，在根目录（`analysis/`）产生了大量文件，未按照项目规范放置在正确的子目录中。

### 发现的错位文件统计

- **总文件数**: 13个（不含README.md）
- **测试文件**: 4个（应放在tests/）
- **脚本文件**: 3个（应放在scripts/或废弃）
- **文档文件**: 6个（应放在docs/或archived/）

---

## 📋 详细整理清单

### 1. 测试文件（4个）→ `tests/`

| 文件名 | 类型 | 正确位置 | 说明 |
|--------|------|----------|------|
| `test_tradeoff_simple.py` | 测试 | `tests/` | 权衡检测简单测试 |
| `test_tradeoff_logic.py` | 测试 | `tests/` | 权衡检测逻辑测试 |
| `test_tradeoff_optimization.py` | 测试 | `tests/` | 权衡优化测试 |
| `test_ctf_alignment.py` | 测试 | `tests/` | CTF对齐测试 |

**操作**: 移动到 `tests/unit/` 或 `tests/integration/`

---

### 2. 脚本文件（3个）→ 分类处理

| 文件名 | 类型 | 正确位置 | 说明 |
|--------|------|----------|------|
| `check_correlation.py` | 诊断脚本 | `tools/legacy/` | 临时诊断脚本，已完成任务 |
| `diagnose_ate_issue.py` | 诊断脚本 | `tools/legacy/` | 临时诊断脚本，已完成任务 |
| `check_tradeoff_sources.py` | 检查脚本 | `tools/legacy/` | 临时检查脚本，已完成任务 |

**操作**: 移动到 `tools/legacy/`（作为历史记录保留）

---

### 3. 文档文件（6个）→ 分类处理

#### 3.1 报告类文档 → `docs/reports/`

| 文件名 | 正确位置 | 说明 |
|--------|----------|------|
| `TRADEOFF_OPTIMIZATION_ACCEPTANCE_REPORT.md` | `docs/reports/` | 权衡优化验收报告 |
| `TRADEOFF_OPTIMIZATION_FINAL_SUMMARY.md` | `docs/reports/` | 权衡优化最终总结 |
| `DIBS_GLOBAL_STD_VS_INTERACTION_COMPARISON_REPORT.md` | `docs/reports/` | DIBS对比报告 |
| `DIBS_METHODS_COMPARISON_SUMMARY.md` | `docs/reports/` | DIBS方法比较总结 |

**操作**: 移动到 `docs/reports/`（如果不存在则创建）

#### 3.2 临时提示文档 → `docs/archived/`

| 文件名 | 正确位置 | 说明 |
|--------|----------|------|
| `NEXT_CONVERSATION_PROMPT.md` | `docs/archived/` | 临时对话提示，已完成 |

**操作**: 移动到 `docs/archived/`（作为历史记录）

---

### 4. 特殊文件

| 文件名 | 位置 | 说明 |
|--------|------|------|
| `README.md` | 保持不变 | 在正确位置 |

**操作**: 不移动

---

## 🗂️ 目标目录结构

整理后的目录结构应符合以下规范：

```
analysis/
├── docs/
│   ├── reports/              # 新建：存放各种报告
│   │   ├── TRADEOFF_OPTIMIZATION_ACCEPTANCE_REPORT.md
│   │   ├── TRADEOFF_OPTIMIZATION_FINAL_SUMMARY.md
│   │   ├── DIBS_GLOBAL_STD_VS_INTERACTION_COMPARISON_REPORT.md
│   │   └── DIBS_METHODS_COMPARISON_SUMMARY.md
│   ├── archived/             # 已存在：存放过时文档
│   │   └── NEXT_CONVERSATION_PROMPT.md
│   ├── current_plans/        # 已存在
│   ├── guides/               # 已存在
│   └── technical_reference/  # 已存在
├── tests/
│   ├── unit/                 # 已存在：单元测试
│   │   ├── test_tradeoff_logic.py
│   │   └── test_ctf_alignment.py
│   └── integration/          # 已存在：集成测试
│       ├── test_tradeoff_simple.py
│       └── test_tradeoff_optimization.py
├── tools/
│   └── legacy/               # 已存在：历史脚本归档
│       ├── check_correlation.py
│       ├── diagnose_ate_issue.py
│       └── check_tradeoff_sources.py
├── scripts/                  # 已存在：正式执行脚本
├── utils/                    # 已存在：工具模块
└── [其他目录保持不变]
```

---

## 📝 执行步骤

### 步骤1：创建必要目录
```bash
mkdir -p docs/reports
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tools/legacy
```

### 步骤2：移动文件（按类别）

#### 2.1 测试文件
```bash
# 单元测试
mv test_tradeoff_logic.py tests/unit/
mv test_ctf_alignment.py tests/unit/

# 集成测试
mv test_tradeoff_simple.py tests/integration/
mv test_tradeoff_optimization.py tests/integration/
```

#### 2.2 脚本文件
```bash
mv check_correlation.py tools/legacy/
mv diagnose_ate_issue.py tools/legacy/
mv check_tradeoff_sources.py tools/legacy/
```

#### 2.3 文档文件
```bash
# 报告
mv TRADEOFF_OPTIMIZATION_ACCEPTANCE_REPORT.md docs/reports/
mv TRADEOFF_OPTIMIZATION_FINAL_SUMMARY.md docs/reports/
mv DIBS_GLOBAL_STD_VS_INTERACTION_COMPARISON_REPORT.md docs/reports/
mv DIBS_METHODS_COMPARISON_SUMMARY.md docs/reports/

# 临时文档
mv NEXT_CONVERSATION_PROMPT.md docs/archived/
```

### 步骤3：验证整理结果
```bash
# 检查根目录（应只有README.md）
ls -1 *.py *.md 2>/dev/null

# 验证文件已正确移动
ls -la tests/unit/
ls -la tests/integration/
ls -la tools/legacy/
ls -la docs/reports/
```

### 步骤4：更新相关引用
需要更新引用这些文件的其他文件（如果有）

---

## ⚠️ 注意事项

1. **不要删除文件**: 所有文件都应该移动到合适位置，而不是删除
2. **保留历史**: 临时脚本归档到`tools/legacy/`，便于追溯
3. **检查引用**: 移动后需要检查是否有其他文件引用了这些文件
4. **Git追踪**: 使用`git mv`而不是`mv`，保持Git历史

---

## 🔍 验证清单

完成整理后，验证：

- [ ] 根目录只有README.md（和极少数必要的配置文件）
- [ ] 所有测试文件都在tests/下
- [ ] 所有报告都在docs/reports/下
- [ ] 所有临时脚本都在tools/legacy/下
- [ ] 没有文件丢失
- [ ] 可以正常运行测试

---

## 📊 整理统计

| 类别 | 移动前 | 移动后 |
|------|--------|--------|
| 根目录文件 | 13个 | 1个（README.md） |
| tests/ | 原有 | +4个 |
| docs/reports/ | 0个 | +4个 |
| docs/archived/ | 原有 | +1个 |
| tools/legacy/ | 原有 | +3个 |

---

## 🚀 下一步

确认方案后，将：
1. 创建整理脚本
2. 执行文件移动
3. 验证整理结果
4. 更新相关文档

---

**请确认以上方案是否可行，我将等待您的确认后再执行整理操作。**
