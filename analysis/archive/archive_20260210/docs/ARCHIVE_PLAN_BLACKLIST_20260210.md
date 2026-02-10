# analysis目录归档方案（黑名单策略）

**日期**: 2026-02-10
**策略**: 黑名单（只归档明确被替代的旧版本）
**归档位置**: `analysis/archive/archived_20260210/`
**原则**: 保守方案，只归档明确过时的文件，保留所有其他文件

---

## 策略说明

### 原方案（白名单）vs 新方案（黑名单）

| 方案 | 思路 | 风险 | 适用场景 |
|------|------|------|---------|
| 白名单 | 列出保留文件，归档其他 | ❌ 高（易误删依赖） | 明确知道所有需要保留的文件 |
| **黑名单** | 列出归档文件，保留其他 | ✅ 低（保守安全） | 依赖关系复杂，不确定所有依赖 |

**本方案采用黑名单策略**：
- ✅ 只归档明确被替代的旧版本
- ✅ 自动保留所有依赖（utils/等）
- ✅ 自动保留可能有用的工具脚本
- ✅ 降低误删风险

---

## 归档规则

### 归档条件（必须同时满足）

1. **有明确的替代版本**
   - 数据：有global_std版本替代
   - 结果：有global_std结果替代
   - 脚本：有global_std脚本替代

2. **创建时间较早**
   - 在最新版本之前创建

3. **不再使用**
   - 路径已过时
   - 功能已被新版本覆盖

---

## 1. 数据文件归档清单

### 明确被替代的数据

| 归档目录 | 替代目录 | 归档原因 |
|---------|---------|---------|
| `data/energy_research/6groups_final/` | `6groups_global_std/` | 旧版最终数据，已被全局标准化版本替代 |
| `data/energy_research/6groups_interaction/` | `6groups_global_std/` | 交互项版本，未采用 |

**保留**: 其他所有数据目录（包括stratified/实验数据、archive/旧归档等）

---

## 2. 结果文件归档清单

### 明确被替代的结果

| 归档目录 | 替代目录 | 归档原因 |
|---------|---------|---------|
| `results/energy_research/archived_data/` | `data/global_std/` | 旧版归档数据 |
| `results/energy_research/interaction_tradeoff_verification/` | `tradeoff_detection_global_std/` | 交互项权衡验证，已被global_std替代 |
| `results/energy_research/tradeoff_detection_interaction_based/` | `tradeoff_detection_global_std/` | 基于交互项的权衡检测，已被global_std替代 |

**保留**: 其他所有结果目录（包括stratified/、reports/等）

---

## 3. 脚本文件归档清单

### 明确被替代的脚本

#### DiBS脚本（有global_std版本替代）

| 归档脚本 | 替代脚本 | 归档原因 |
|---------|---------|---------|
| `scripts/run_dibs_6groups_final.py` | `run_dibs_6groups_global_std.py` | 旧版DiBS脚本 |
| `scripts/run_dibs_6groups_interaction.py` | `run_dibs_6groups_global_std.py` | 交互项版本 |
| `scripts/run_dibs_for_questions_2_3.py` | `run_dibs_6groups_global_std.py` | 旧版问题2-3脚本 |
| `scripts/run_dibs_on_new_6groups.py` | `run_dibs_6groups_global_std.py` | 旧版6组脚本 |

#### ATE脚本（有global_std版本替代）

| 归档脚本 | 替代脚本 | 归档原因 |
|---------|---------|---------|
| `scripts/compute_ate_for_whitelist.py` | `compute_ate_dibs_global_std.py` | 旧版白名单ATE |
| `scripts/compute_ate_whitelist.py` | `compute_ate_dibs_global_std.py` | 旧版白名单ATE |

#### 权衡检测脚本（有global_std版本替代）

| 归档脚本 | 替代脚本 | 归档原因 |
|---------|---------|---------|
| `scripts/run_algorithm1_tradeoff_detection.py` | `run_algorithm1_tradeoff_detection_global_std.py` | 旧版权衡检测 |
| `scripts/verify_interaction_tradeoffs.py` | `run_algorithm1_tradeoff_detection_global_std.py` | 交互项验证 |

#### 数据生成脚本（已生成完成，不再需要）

| 归档脚本 | 归档原因 |
|---------|---------|
| `scripts/generate_6groups_final.py` | 数据已生成，不再使用 |
| `scripts/create_global_standardized_data.py` | 数据已生成，不再使用 |

**保留**: 其他所有脚本（包括utils/、工具脚本、诊断脚本、配置脚本等）

---

## 4. 归档统计（黑名单方案）

| 类别 | 归档数量 | 保留数量 | 说明 |
|------|---------|---------|------|
| 数据目录 | 2 | 其他所有 | 只归档明确被替代的 |
| 结果目录 | 3 | 其他所有 | 只归档明确被替代的 |
| 脚本 | 10 | 其他所有(35+) | 只归档有新版本的 |
| **utils/目录** | **0** | **1(全部)** | **自动保留，不归档** |

**总计归档**: ~15项（vs 原方案的46项）

---

## 5. 执行命令

### 5.1 创建归档目录

```bash
mkdir -p archive/archived_20260210/{data,results,scripts/{dibs,ate,tradeoff,data_gen}}
```

### 5.2 归档数据文件

```bash
# 只归档2个明确被替代的数据目录
mv data/energy_research/6groups_final archive/archived_20260210/data/
mv data/energy_research/6groups_interaction archive/archived_20260210/data/
```

### 5.3 归档结果文件

```bash
# 只归档3个明确被替代的结果目录
mv results/energy_research/archived_data archive/archived_20260210/results/
mv results/energy_research/interaction_tradeoff_verification archive/archived_20260210/results/
mv results/energy_research/tradeoff_detection_interaction_based archive/archived_20260210/results/
```

### 5.4 归档脚本文件

```bash
# DiBS脚本（4个）
mv scripts/run_dibs_6groups_final.py archive/archived_20260210/scripts/dibs/
mv scripts/run_dibs_6groups_interaction.py archive/archived_20260210/scripts/dibs/
mv scripts/run_dibs_for_questions_2_3.py archive/archived_20260210/scripts/dibs/
mv scripts/run_dibs_on_new_6groups.py archive/archived_20260210/scripts/dibs/

# ATE脚本（2个）
mv scripts/compute_ate_for_whitelist.py archive/archived_20260210/scripts/ate/
mv scripts/compute_ate_whitelist.py archive/archived_20260210/scripts/ate/

# 权衡检测脚本（2个）
mv scripts/run_algorithm1_tradeoff_detection.py archive/archived_20260210/scripts/tradeoff/
mv scripts/verify_interaction_tradeoffs.py archive/archived_20260210/scripts/tradeoff/

# 数据生成脚本（2个）
mv scripts/generate_6groups_final.py archive/archived_20260210/scripts/data_gen/
mv scripts/create_global_standardized_data.py archive/archived_20260210/scripts/data_gen/
```

---

## 6. 自动保留的文件（无需操作）

以下文件将自动保留（不在归档清单中）：

### 数据目录
- ✅ `6groups_global_std/` - 当前使用
- ✅ `6groups_dibs_ready/` - 当前使用
- ✅ `6groups_dibs_ready_v1_backup/` - 备份
- ✅ `stratified/` - 实验数据
- ✅ `archive/` - 旧归档
- ✅ 其他所有数据目录

### 结果目录
- ✅ `data/global_std/` - DiBS结果
- ✅ `data/global_std_dibs_ate/` - ATE结果
- ✅ `tradeoff_detection_global_std/` - 权衡结果
- ✅ `rq_analysis/` - RQ分析
- ✅ `reports/` - 报告
- ✅ `stratified/` - 分层分析
- ✅ 其他所有结果目录

### 脚本
- ✅ **所有global_std脚本** - 当前使用
- ✅ **utils/目录** - 核心依赖
- ✅ **工具脚本** - 诊断、检查、分析脚本
- ✅ **配置脚本** - config.py等
- ✅ **RQ分析脚本** - rq1/rq2/rq3_analysis.py
- ✅ **其他所有脚本** (35+个)

---

## 7. 归档后验证

```bash
# 验证归档
echo "=== 归档验证 ==="
echo "数据归档: $(ls archive/archived_20260210/data/ | wc -l) 个目录"
echo "结果归档: $(ls archive/archived_20260210/results/ | wc -l) 个目录"
echo "脚本归档: $(find archive/archived_20260210/scripts -name "*.py" | wc -l) 个脚本"

# 验证保留
echo ""
echo "=== 保留验证 ==="
echo "数据保留: $(ls data/energy_research/ | wc -l) 个目录"
echo "结果保留: $(ls results/energy_research/ | wc -l) 个目录"
echo "脚本保留: $(ls scripts/*.py | wc -l) 个脚本"
echo "utils保留: $(ls utils/*.py | grep -v __pycache__ | wc -l) 个模块"
```

---

## 8. 优势对比

### 黑名单方案 vs 白名单方案

| 维度 | 白名单方案 | 黑名单方案（本方案） |
|------|-----------|-------------------|
| **归档数量** | 46项 | ~15项 |
| **风险等级** | 高（易误删依赖） | 低（保守安全） |
| **依赖处理** | 需要手动分析所有依赖 | 自动保留所有依赖 |
| **utils/处理** | 需要明确添加到保留列表 | 自动保留 |
| **工具脚本** | 需要逐一评估 | 自动保留 |
| **可恢复性** | 可恢复但复杂 | 简单可恢复 |
| **执行复杂度** | 高（需要仔细检查） | 低（明确清晰） |

---

## 9. 风险控制

### 9.1 安全措施

- ✅ 使用 `mv` 命令（可恢复）
- ✅ 只归档明确过时的文件
- ✅ 自动保留所有依赖
- ✅ 保守策略，宁可不归档

### 9.2 回滚方案

```bash
# 如需恢复某个目录
mv archive/archived_20260210/data/6groups_final data/energy_research/

# 如需恢复某个脚本
mv archive/archived_20260210/scripts/dibs/run_dibs_6groups_final.py scripts/
```

---

## 10. 执行检查清单

归档前确认：

- [ ] 理解黑名单策略（只归档明确过时的）
- [ ] 确认归档的数据目录正确（2个）
- [ ] 确认归档的结果目录正确（3个）
- [ ] 确认归档的脚本正确（10个）
- [ ] 理解utils/等依赖将自动保留
- [ ] 理解工具脚本将自动保留

---

## 11. 总结

**黑名单方案核心原则**：
- ✅ 只归档**明确被替代**的旧版本
- ✅ 自动保留所有依赖和工具脚本
- ✅ 保守安全，宁可多保留

**归档清单总结**：
- 数据：2个目录（6groups_final, 6groups_interaction）
- 结果：3个目录（archived_data, interaction相关2个）
- 脚本：10个（DiBS 4个, ATE 2个, 权衡 2个, 数据生成 2个）

**保留清单**：
- 其他所有文件自动保留（包括utils/、工具脚本等35+个脚本）

---

**方案制定**: Claude Code
**日期**: 2026-02-10
**状态**: ⏳ 待审核 - 黑名单策略（保守方案）
