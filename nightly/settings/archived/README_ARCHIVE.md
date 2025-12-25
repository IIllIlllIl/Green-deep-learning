# 归档配置文件说明

**归档日期**: 2025-12-03
**原因**: 配置优化 - 旧版统一runs_per_config被参数精确优化替代

---

## 归档内容

此目录包含已过时的stage配置文件和文档，已被优化版本替代。

### 归档的配置文件

| 文件 | 说明 | 问题 |
|------|------|------|
| `stage2_fast_models_parallel.json` | 旧Stage2配置 | 统一runs_per_config=6 |
| `stage3_medium_models_parallel.json` | 旧Stage3配置 | 统一runs_per_config=6 |
| `stage4_vulberta_parallel.json` | 旧Stage4配置 | 统一runs_per_config=6 |
| `stage5_densenet121_parallel.json` | 旧Stage5配置 | 统一runs_per_config=6 |
| `stage6_hrnet18_parallel.json` | 旧Stage6配置 | 统一runs_per_config=6 |
| `stage7_pcb_parallel.json` | 旧Stage7配置 | 统一runs_per_config=6 |

### 归档的文档

| 文件 | 说明 | 替代文档 |
|------|------|---------|
| `STAGED_EXECUTION_PLAN.md` | 旧分阶段执行计划 | `OPTIMIZED_CONFIG_REPORT.md` |
| `QUICK_REFERENCE.md` | 旧快速参考 | `QUICK_REFERENCE_OPTIMIZED.md` |

---

## 旧版问题分析

### 统一runs_per_config的问题

旧版所有配置使用 `runs_per_config: 6`，导致：

1. **过度运行**: 已有4个唯一值的参数只需1个，却运行6次
2. **资源浪费**: 大量实验被去重机制跳过
3. **时间浪费**: 预计390次尝试，实际只需105次
4. **效率低**: 资源利用率仅26.9%

**示例**:
```json
{
  "repo": "Person_reID_baseline_pytorch",
  "model": "hrnet18",
  "mode": "mutation",
  "mutate": ["learning_rate"],
  "runs_per_config": 6,  // ❌ 问题: learning_rate已有4个，只需1个新值，运行6次浪费5次
  "comment": "learning_rate: 当前4个"
}
```

---

## 新版优化方案

### 参数精确优化

新版为每个参数计算精确的runs_per_config:

```
runs_per_config = (5 - current_unique_count) + 1
```

**优化示例**:
```json
{
  "repo": "Person_reID_baseline_pytorch",
  "model": "hrnet18",
  "mode": "mutation",
  "mutate": ["learning_rate"],
  "runs_per_config": 2,  // ✅ 优化: 需要1个 + 1余量 = 2
  "comment": "learning_rate: 4→5个，需要1个"
}
```

### 优化效果对比

| 指标 | 旧版 | 新版 | 改善 |
|------|------|------|------|
| 总尝试次数 | 390 | 105-145 | -63% |
| 资源利用率 | 26.9% | >90% | +235% |
| 浪费GPU时间 | ~180h | <20h | -89% |
| 配置复杂度 | 低（统一） | 中（精确） | 值得 |

---

## 新版配置文件

新的优化配置文件位于 `settings/` 根目录:

1. `stage2_optimized_nonparallel_and_fast_parallel.json`
2. `stage3_optimized_mnist_ff_and_medium_parallel.json`
3. `stage4_optimized_vulberta_densenet121_parallel.json`
4. `stage5_optimized_hrnet18_parallel.json`
5. `stage6_optimized_pcb_parallel.json`

**详细文档**:
- `OPTIMIZED_CONFIG_REPORT.md` - 详细优化报告
- `QUICK_REFERENCE_OPTIMIZED.md` - 快速参考指南
- `EXECUTION_READY.md` - 执行准备清单

---

## 保留原因

这些旧配置文件被保留用于:
1. **历史参考**: 了解配置演进过程
2. **对比分析**: 理解优化前后的差异
3. **学习案例**: 展示为什么统一配置不是最优方案

**注意**: 不建议使用这些旧配置文件执行新实验。

---

**归档者**: Claude Code
**归档日期**: 2025-12-03
**状态**: 仅供参考，不建议使用
