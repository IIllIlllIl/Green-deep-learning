# analysis目录归档方案

**日期**: 2026-02-10
**范围**: 仅限 `analysis/` 目录下的数据、结果和脚本
**归档位置**: `analysis/archive/archived_20260210/`

---

## 1. 保留文件（最新可用的）

### 1.1 数据文件 ✅ KEEP

| 目录 | 用途 | 创建日期 | 状态 |
|------|------|---------|------|
| `data/energy_research/6groups_global_std/` | **全局标准化数据**（删除0列+超参数合并） | 2026-01-30 | ✅ 保留 |
| `data/energy_research/6groups_dibs_ready/` | DiBS预处理数据 | 2026-01-30 | ✅ 保留 |
| `data/energy_research/6groups_dibs_ready_v1_backup/` | 预处理备份 | 2026-02-10 | ✅ 保留 |

### 1.2 分析结果 ✅ KEEP

| 目录 | 用途 | 更新日期 | 状态 |
|------|------|---------|------|
| `results/energy_research/data/global_std/` | **DiBS因果图结果**（6组） | 2026-02-10 | ✅ 保留 |
| `results/energy_research/data/global_std_dibs_ate/` | **ATE计算结果**（6组） | 2026-02-03 | ✅ 保留 |
| `results/energy_research/tradeoff_detection_global_std/` | **权衡检测结果**（61个权衡） | 2026-02-10 | ✅ 保留 |
| `results/energy_research/rq_analysis/` | 研究问题分析+可视化 | 2026-02-07 | ✅ 保留 |
| `results/energy_research/archive/` | 已有归档 | - | ✅ 保留 |

### 1.3 分析脚本 ✅ KEEP

| 脚本 | 用途 | 更新日期 | 状态 |
|------|------|---------|------|
| `scripts/run_dibs_6groups_global_std.py` | DiBS训练（13000步） | 2026-01-30 | ✅ 保留 |
| `scripts/validate_dibs_results.py` | DiBS结果验证 | 2026-01-06 | ✅ 保留 |
| `scripts/compute_ate_dibs_global_std.py` | ATE计算（DML） | 2026-02-03 | ✅ 保留 |
| `scripts/run_algorithm1_tradeoff_detection_global_std.py` | 权衡检测 | 2026-02-03 | ✅ 保留 |
| `scripts/preprocess_for_dibs_global_std.py` | DiBS数据预处理 | 2026-01-30 | ✅ 保留 |
| `scripts/visualize_dibs_causal_graphs.py` | 可视化（需更新路径） | 2026-01-06 | ✅ 保留 |

---

## 2. 归档文件（旧版本）

### 2.1 数据文件 📦 ARCHIVE

| 目录 | 用途 | 创建日期 | 归档原因 |
|------|------|---------|---------|
| `data/energy_research/6groups_final/` | 旧版最终数据 | 2026-01-17 | 已被global_std替代 |
| `data/energy_research/6groups_interaction/` | 交互项版本 | 2026-01-17 | 未使用 |
| `data/energy_research/stratified/` | 分层采样数据 | 2026-02-07 | 实验性数据 |
| `data/energy_research/archive/` | 旧归档 | - | 移至新归档目录 |

### 2.2 分析结果 📦 ARCHIVE

| 目录 | 用途 | 创建日期 | 归档原因 |
|------|------|---------|---------|
| `results/energy_research/archived_data/` | 旧版归档数据 | 2026-02-08 | 整合到新归档 |
| `results/energy_research/interaction_tradeoff_verification/` | 交互项权衡验证 | 2026-02-03 | 已被global_std替代 |
| `results/energy_research/tradeoff_detection_interaction_based/` | 基于交互项的权衡 | 2026-02-02 | 已被global_std替代 |
| `results/energy_research/stratified/` | 分层采样结果 | 2026-02-07 | 实验性结果 |
| `results/energy_research/reports/` | 旧版报告 | 2026-02-04 | 可能过期 |

### 2.3 分析脚本 📦 ARCHIVE（待确认）

需要归档的旧版脚本（不在保留列表中的）：

| 类别 | 数量 | 处理方式 |
|------|------|---------|
| 旧版DiBS脚本 | ~5个 | 归档到 `archive/scripts/` |
| 旧版ATE脚本 | ~3个 | 归档到 `archive/scripts/` |
| 旧版权衡脚本 | ~2个 | 归档到 `archive/scripts/` |
| 其他工具脚本 | ~10个 | 归档到 `archive/scripts/` |

**详细脚本列表见**: 附录A

---

## 3. 归档执行计划

### 3.1 归档目录结构

```
analysis/archive/archived_20260210/
├── data/
│   ├── 6groups_final/
│   ├── 6groups_interaction/
│   ├── stratified/
│   └── archive/
├── results/
│   ├── archived_data/
│   ├── interaction_tradeoff_verification/
│   ├── tradeoff_detection_interaction_based/
│   ├── stratified/
│   └── reports/
└── scripts/
    ├── old_dibs/
    ├── old_ate/
    ├── old_tradeoff/
    └── utils/
```

### 3.2 执行步骤

1. **创建归档目录**
   ```bash
   mkdir -p archive/archived_20260210/{data,results,scripts}
   ```

2. **归档数据文件**
   ```bash
   mv data/energy_research/6groups_final archive/archived_20260210/data/
   mv data/energy_research/6groups_interaction archive/archived_20260210/data/
   mv data/energy_research/stratified archive/archived_20260210/data/
   # ...
   ```

3. **归档结果文件**
   ```bash
   mv results/energy_research/archived_data archive/archived_20260210/results/
   mv results/energy_research/interaction_tradeoff_verification archive/archived_20260210/results/
   # ...
   ```

4. **归档脚本文件**
   ```bash
   # 仅归档不在保留列表中的脚本
   # 见附录A
   ```

5. **生成归档清单**
   ```bash
   tree archive/archived_20260210 > archive/ARCHIVE_MANIFEST_20260210.txt
   ```

---

## 4. 风险控制

### 4.1 安全措施

- ✅ 使用 `mv` 而非 `rm`（可恢复）
- ✅ 保留原有目录结构
- ✅ 生成归档清单
- ✅ 创建归档说明文档

### 4.2 回滚方案

如需恢复归档文件：
```bash
# 恢复某个目录
mv archive/archived_20260210/data/6groups_final data/energy_research/
```

### 4.3 验证检查

归档后验证：
- [ ] 保留文件完整性（6组数据+结果）
- [ ] DiBS脚本可执行
- [ ] ATE脚本可执行
- [ ] 权衡检测脚本可执行
- [ ] 归档清单正确

---

## 5. 附录

### 附录A: 待归档脚本详细列表

**需要保留的脚本** (核心工作流):
```
scripts/run_dibs_6groups_global_std.py
scripts/validate_dibs_results.py
scripts/compute_ate_dibs_global_std.py
scripts/run_algorithm1_tradeoff_detection_global_std.py
scripts/preprocess_for_dibs_global_std.py
scripts/visualize_dibs_causal_graphs.py
```

**待归档的脚本** (扫描后确认):
- DiBS相关（旧版）: `run_dibs_*.py` (非global_std版本)
- ATE相关（旧版）: `compute_ate_*.py` (非global_std版本)
- 权衡相关（旧版）: `run_algorithm1_*.py` (非global_std版本)
- 工具脚本: `analyze_*.py`, `test_*.py`, `debug_*.py` 等

**详细列表**（需要扫描确认）:
- [ ] 待扫描 `scripts/` 目录生成完整列表

---

## 6. 执行确认

请审核以上方案并确认：

- [ ] 保留文件列表正确
- [ ] 归档文件列表正确
- [ ] 归档目录结构合理
- [ ] 安全措施充分

**确认后执行**:
```bash
bash archive_files.sh
```

---

**方案制定**: Claude Code
**日期**: 2026-02-10
**状态**: 待审核
