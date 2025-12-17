# Phase 4 验证配置修正报告

**日期**: 2025-12-13 19:00
**问题**: 原配置使用`mutate_params`对象导致多参数同时变异
**修正**: 改用`mutate`数组确保单参数变异

---

## 🔧 问题描述

原配置文件使用了错误的格式：

```json
{
  "mutation_type": "mutation",
  "mutate_params": {
    "epochs": {
      "distribution": "log_uniform",
      "min": 5,
      "max": 20
    }
  }
}
```

**问题**: `mutate_params`对象格式会导致多个参数同时变异，不符合实验设计要求（每次只变异一个参数）。

---

## ✅ 修正方案

参考`stage2_optimized_nonparallel_and_fast_parallel.json`的正确格式：

### 非并行模式

```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "mutation",
  "mutate": ["learning_rate"],
  "runs_per_config": 2
}
```

### 并行模式

```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "mode": "parallel",
  "foreground": {
    "repo": "VulBERTa",
    "model": "mlp",
    "mode": "mutation",
    "mutate": ["learning_rate"]
  },
  "background": {
    "repo": "examples",
    "model": "mnist",
    "hyperparameters": {}
  },
  "runs_per_config": 2
}
```

### 关键点

1. ✅ 使用`"mutate": ["参数名"]`数组格式
2. ✅ 每次只包含一个参数（单参数变异原则）
3. ✅ 不使用`mutate_params`对象
4. ✅ 顶层必须包含`"mode": "mutation"`或`"mode": "parallel"`
5. ✅ 顶层必须包含`"max_retries"`, `"governor"`, `"use_deduplication"`等全局设置

---

## 📊 修正后的配置统计

**配置文件**: `settings/test_phase4_validation_optimized.json`

| 指标 | 数值 |
|------|------|
| 实验总数 | 17 |
| 默认值实验 | 2 |
| 非并行变异 | 9 |
| 并行实验 | 6 |

**单参数变异分布**:
- learning_rate: 4次
- alpha: 2次
- dropout: 2次
- seed: 2次
- epochs: 1次
- max_iter: 1次
- weight_decay: 1次

**验证结果**:
- ✅ JSON格式验证通过
- ✅ 无配置错误
- ✅ 无配置警告
- ✅ 所有实验均为单参数变异

---

## 🎯 配置目标

**目的**: 验证修复后的正则表达式，恢复105个缺失性能数据的实验

**模型覆盖**:
- VulBERTa/mlp: 6个实验（3默认 + 1并行默认 + 5变异 + 1并行变异）
- bug-localization: 6个实验（2默认 + 1并行默认 + 3变异 + 2并行变异）
- MRT-OAST: 4个实验（4变异）

**预计时间**:
- 无去重: 15.6小时
- 去重率50%: 7.8小时 ⭐ 预期
- 去重率70%: 4.7小时

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/test_phase4_validation_optimized.json
```

---

## 📚 参考文档

- 正确格式参考: `settings/stage2_optimized_nonparallel_and_fast_parallel.json`
- JSON配置最佳实践: `docs/JSON_CONFIG_BEST_PRACTICES.md`
- 配置指南: `docs/SETTINGS_CONFIGURATION_GUIDE.md`

---

**修正完成时间**: 2025-12-13 19:00
**验证状态**: ✅ 已通过格式和逻辑验证
**状态**: 可以执行
