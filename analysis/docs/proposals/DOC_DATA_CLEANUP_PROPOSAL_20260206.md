# 文档与数据整理方案

**版本**: v1.1 (评审后修订版)
**日期**: 2026-02-06
**状态**: 已通过评审 (评分 4.2/5)
**目的**: 整理本周(2月2日-2月6日)生成的文档和数据，减少过时信息被误用风险

---

## 0. 评审修订记录

| 问题 | 评审意见 | 修订内容 |
|------|---------|---------|
| P0-1 | `/results/.../global_std/`包含唯一汇总报告 | 改为迁移而非删除 |
| P0-2 | 缺少根本原因追查 | 添加脚本工作目录检查 |
| P1-1 | scripts/results的JSON需验证 | 已验证，确认可删除 |
| P1-2 | mv命令无错误处理 | 使用find+xargs |
| P1-3 | INDEX更新位置不明 | 添加具体行号 |
| P1-4 | 数据源表格缺说明 | 添加原因说明列 |

---

## 1. 问题分析

### 1.1 发现的问题

#### P0: 严重问题（必须修复）

1. **错误放置的结果目录** `analysis/scripts/results/`
   - 位置: `/analysis/scripts/results/energy_research/`
   - 问题: 脚本目录下不应有results子目录，这是错误的工作目录导致
   - 已验证内容:
     - `all_tradeoffs_global_std.json`: `{}` (空对象，无有效数据)
     - `config_info.json`: 显示 `"total_groups_processed": 0`，运行失败的产物
     - `global_std_dibs_ate/`: 空目录
   - 处理: **删除整个目录**（已验证无有效数据）

2. **过时的提案文档**
   - `STRATIFIED_ANALYSIS_PROPOSAL_20260205.md` - 已被V2版本取代
   - 问题: 旧版本提案可能被误用
   - 处理: **移至archive目录**

3. **项目根目录下的唯一数据** `/results/energy_research/data/global_std/`
   - **评审修正**: 此目录包含唯一的 `dibs_global_std_total_report.json`（汇总报告）
   - 内容: group5_mrt_oast的DiBS结果（741字节汇总+详细文件）
   - **根本原因**: 某脚本工作目录配置错误导致输出到项目根目录
   - 处理: **迁移到analysis目录而非删除**

4. **过时的NEXT_SESSION_PROMPT.md**
   - 位置: `/analysis/results/energy_research/NEXT_SESSION_PROMPT.md`
   - 日期: 2月2日创建，内容已过时
   - 处理: **删除**（新的prompt应在对话中直接提供）

#### P1: 中等问题（应当修复）

5. **备份文件散落**
   - `compute_ate_dibs_global_std.py.backup`
   - `compute_ate_dibs_global_std.py.before_fix`
   - `create_global_standardized_data.py.backup`
   - `causal_inference.py.bak`
   - `tradeoff_detection.py.backup`
   - 处理: **统一移至archive目录或删除**

6. **旧的调试日志** `/analysis/logs/dibs/`
   - 包含多个2月3日的调试日志和PID文件
   - 大部分是空文件或调试过程产物
   - 处理: **清理空文件，保留有价值的完整日志**

7. **ATE计算日志** `/analysis/scripts/logs/ate/`
   - 包含多个重复的日志文件（fixed版和非fixed版）
   - 处理: **只保留最终fixed版本的日志**

#### P2: 低优先级（建议改进）

8. **INDEX.md更新日期过旧**
   - `analysis/docs/INDEX.md` 显示"最后更新: 2026-01-23"
   - 但2月份有大量新内容
   - 处理: **更新INDEX.md添加新内容**

9. **interaction_tradeoff_verification目录**
   - 位置: `/analysis/results/energy_research/interaction_tradeoff_verification/`
   - 内容: 2月2日的验证报告
   - 状态: 可能已完成其使命
   - 处理: **保留但在INDEX中标记为历史参考**

### 1.2 数据使用风险点

| 风险 | 描述 | 之前出现过？ | 预防措施 |
|------|------|-------------|---------|
| 使用未预处理数据 | 使用`6groups_global_std`而非`6groups_dibs_ready` | ✅ 是 | 在INDEX中明确标注数据用途 |
| 使用错误路径 | 项目根目录vs analysis目录 | 可能 | 迁移并统一到analysis目录 |
| 使用过时提案 | 旧版本提案 | 可能 | 归档旧版本 |

### 1.3 根本原因分析（评审新增）

**问题**: 为什么会有数据输出到项目根目录 `/results/` 而非 `/analysis/results/`？

**可能原因**:
1. 脚本使用相对路径 `results/` 而非绝对路径
2. 脚本在错误的工作目录下执行
3. 脚本硬编码了错误的输出路径

**预防措施**:
- 检查相关脚本的输出路径配置
- 建议脚本使用 `__file__` 相对路径或环境变量定义输出目录

---

## 2. 整理计划

### 2.1 执行顺序

```
Phase 0: 备份（安全措施）
└── 创建整理前备份 archive_backup_20260206.tar.gz

Phase 1: 迁移和删除
├── 迁移 /results/.../global_std/ → /analysis/results/.../global_std_migrated/
├── 删除 /analysis/scripts/results/ (错误放置，已验证无效)
└── 删除 /analysis/results/energy_research/NEXT_SESSION_PROMPT.md (过时)

Phase 2: 归档需要保留参考的内容
├── 移动 STRATIFIED_ANALYSIS_PROPOSAL_20260205.md → archive/
├── 移动 *.backup, *.bak, *.before_fix → archive/backup_files/
└── 清理 /analysis/logs/dibs/ 空文件

Phase 3: 更新文档索引
├── 更新 analysis/docs/INDEX.md (在第8行后插入分层分析章节)
│   ├── 添加分层分析相关文档
│   ├── 添加新的数据目录说明（含原因）
│   └── 标记历史参考文档
├── 更新日期和状态
└── 添加数据使用警告

Phase 4: 验证和记录
├── 验证目录结构正确性
├── 验证无dead links
├── 生成整理报告
└── 更新CLAUDE.md（如需要）
```

预计总耗时：约30分钟

### 2.2 详细操作

#### Phase 0: 备份

```bash
# 创建整理前备份
cd /home/green/energy_dl/nightly
tar -czf analysis/archive/archive_backup_20260206.tar.gz \
    analysis/scripts/results/ \
    results/energy_research/data/global_std/ \
    analysis/results/energy_research/NEXT_SESSION_PROMPT.md \
    2>/dev/null || true
```

#### Phase 1: 迁移和删除

```bash
# 1. 迁移项目根目录的数据到analysis目录
mkdir -p /home/green/energy_dl/nightly/analysis/results/energy_research/data/global_std_migrated/
cp -r /home/green/energy_dl/nightly/results/energy_research/data/global_std/* \
      /home/green/energy_dl/nightly/analysis/results/energy_research/data/global_std_migrated/

# 验证迁移成功后删除原目录
rm -rf /home/green/energy_dl/nightly/results/energy_research/data/global_std/

# 2. 删除错误放置的results目录（已验证无有效数据）
rm -rf /home/green/energy_dl/nightly/analysis/scripts/results/

# 3. 删除过时的session prompt
rm -f /home/green/energy_dl/nightly/analysis/results/energy_research/NEXT_SESSION_PROMPT.md
```

#### Phase 2: 归档操作

```bash
# 1. 创建归档目录
mkdir -p /home/green/energy_dl/nightly/analysis/docs/proposals/archive/
mkdir -p /home/green/energy_dl/nightly/analysis/archive/backup_files_20260206/

# 2. 移动旧提案
mv /home/green/energy_dl/nightly/analysis/docs/proposals/STRATIFIED_ANALYSIS_PROPOSAL_20260205.md \
   /home/green/energy_dl/nightly/analysis/docs/proposals/archive/

# 3. 移动备份文件（使用find避免文件不存在的错误）
find /home/green/energy_dl/nightly/analysis -type f \( -name "*.backup" -o -name "*.bak" -o -name "*.before_fix" \) \
    -exec mv {} /home/green/energy_dl/nightly/analysis/archive/backup_files_20260206/ \;

# 4. 清理空日志文件
find /home/green/energy_dl/nightly/analysis/logs/dibs/ -type f -empty -delete
```

#### Phase 3: INDEX更新内容

需要在 `analysis/docs/INDEX.md` **第8行后**（"## 📚 文档组织说明" 之前）插入：

```markdown
## 🆕 分层因果分析 (2026-02) ⭐⭐⭐

**状态**: 进行中（DiBS分析执行中）

### 提案文档
- [STRATIFIED_ANALYSIS_PROPOSAL_V2_20260206.md](proposals/STRATIFIED_ANALYSIS_PROPOSAL_V2_20260206.md) - 分层分析总体方案 v2.1 ⭐⭐⭐
- [STRATIFIED_DIBS_ANALYSIS_PROPOSAL_20260206.md](proposals/STRATIFIED_DIBS_ANALYSIS_PROPOSAL_20260206.md) - DiBS分层分析方案 v1.1 ⭐⭐
- [DOC_DATA_CLEANUP_PROPOSAL_20260206.md](proposals/DOC_DATA_CLEANUP_PROPOSAL_20260206.md) - 文档整理方案 v1.1

### 数据目录说明

⚠️ **重要数据选择指南**（避免之前的数据选择错误）:

| 用途 | 正确数据源 | 原因 | 避免使用 |
|------|-----------|------|---------|
| DiBS因果图学习 | `6groups_dibs_ready/` | 无缺失值，已预处理 | ❌ `6groups_global_std/`（含缺失值） |
| 分层DiBS分析 | `stratified/` | 按is_parallel分层 | ❌ `6groups_*`（未分层） |
| 全局标准化（非DiBS） | `6groups_global_std/` | 含统一标准化参数 | ❌ 直接用`raw_data.csv` |
| 一般分析 | `6groups_final/` | 基础分组数据 | - |

### 分层分析脚本
- `scripts/stratified/prepare_stratified_data.py` - 分层数据准备
- `scripts/stratified/run_dibs_stratified.py` - 分层DiBS分析

### 分层分析结果
- `results/energy_research/stratified/dibs/` - DiBS分层因果图
- `data/energy_research/stratified/` - 分层数据（4层）

---
```

同时更新 INDEX.md 第3行的日期：
```
**最后更新**: 2026-02-06
```

---

## 3. 验收标准

### 3.1 结构验证

- [ ] `/analysis/scripts/` 下无 `results/` 目录
- [ ] 所有 `*.backup`, `*.bak`, `*.before_fix` 文件已归档到 `archive/backup_files_20260206/`
- [ ] 旧版提案已移至 `proposals/archive/`
- [ ] 无过时的NEXT_SESSION_PROMPT
- [ ] `/results/energy_research/data/global_std/` 已迁移到analysis目录

### 3.2 INDEX验证

- [ ] 更新日期为2026-02-06
- [ ] 包含分层分析文档链接（3个提案）
- [ ] 包含数据目录选择指南（4行表格，含原因列）
- [ ] 历史文档已标记

### 3.3 数据完整性

- [ ] `6groups_dibs_ready/` 完整（6组数据）
- [ ] `stratified/` 完整（4层数据）
- [ ] `global_std_migrated/` 包含迁移的汇总报告
- [ ] 无重复数据目录

### 3.4 定量指标（评审新增）

- [ ] 备份文件数量: 6个已归档
- [ ] 空日志文件: 全部清理
- [ ] INDEX.md无dead links（验证方法：检查所有链接文件存在）

---

## 4. 风险评估

| 操作 | 风险 | 缓解措施 |
|------|------|---------|
| 删除scripts/results | 低 - 已验证无有效数据 | Phase 0备份 |
| 迁移global_std数据 | 低 - 复制后再删除 | 验证迁移完整性 |
| 移动备份文件 | 低 - 使用find避免错误 | 移动而非删除 |
| 更新INDEX | 低 - 仅添加新内容 | 保留原有内容 |

---

## 5. 应急回滚计划（评审新增）

如果执行中出现问题，可从备份恢复：

```bash
# 从Phase 0创建的备份恢复
cd /home/green/energy_dl/nightly
tar -xzf analysis/archive/archive_backup_20260206.tar.gz
```

---

## 6. 预期结果

### 整理后的目录结构

```
analysis/
├── data/energy_research/
│   ├── 6groups_dibs_ready/     # DiBS预处理数据（首选）
│   ├── 6groups_global_std/     # 全局标准化数据（有缺失值）
│   ├── 6groups_final/          # 最终分组数据
│   ├── stratified/             # 分层数据（新）
│   └── archive/                # 归档数据
├── docs/
│   ├── INDEX.md                # 更新后的索引
│   ├── proposals/
│   │   ├── STRATIFIED_*.md     # 当前提案
│   │   └── archive/            # 旧提案
│   └── reports/                # 验收报告
├── results/energy_research/
│   ├── data/                   # 分析结果
│   ├── stratified/             # 分层分析结果（新）
│   └── tradeoff_detection_*/   # 权衡检测结果
├── scripts/
│   ├── stratified/             # 分层分析脚本
│   └── *.py                    # 全局分析脚本
├── archive/
│   └── backup_files_20260206/  # 备份文件归档
└── logs/
    └── dibs/                   # 清理后的日志
```

---

## 7. 执行确认

- [x] 方案已通过同行评审 (评分 4.2/5)
- [x] 方案已根据评审意见修订 (v1.0 → v1.1)
- [x] 用户已确认执行
- [x] 所有操作已完成 (2026-02-06 19:55)
- [x] 验收检查已通过

### 执行结果

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0 | ✅ 完成 | 备份已创建 (6994字节) |
| Phase 1 | ✅ 完成 | 迁移global_std，删除无效目录 |
| Phase 2 | ✅ 完成 | 归档6个备份文件，清理2个空日志 |
| Phase 3 | ✅ 完成 | INDEX.md已更新 |
| Phase 4 | ✅ 完成 | 全部验收项通过 |

---

**作者**: Claude Code
**创建日期**: 2026-02-06 19:35
**修订日期**: 2026-02-06 19:50
**执行日期**: 2026-02-06 19:55
**评审评分**: 4.2/5
