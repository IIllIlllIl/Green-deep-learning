# MRT-OAST 作为第6组的可行性分析

**分析日期**: 2025-12-24
**数据来源**: stage2_mediators.csv

---

## 基础统计

- 总行数: 78 行
- 有效行数（删除性能全缺失）: 58 行
- 删除行数: 20 行
- DiBS要求: ✅ 充足（58 ≥ 20）

## 可行性评估

✅ 数据量 ≥ 20
✅ 性能指标 ≥ 1
✅ 能耗数据完整
✅ 超参数 ≥ 3

**结论**: ✅ 可以作为第6组

## 推荐配置

```python
'mrt_oast': {
    'repos': ['MRT-OAST'],
    'models': {'MRT-OAST': ['default']},
    'performance_cols': ['perf_accuracy', 'perf_precision', 'perf_recall'],
    'hyperparams': ['training_duration'] + ['hyperparam_dropout', 'hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_seed', 'hyperparam_weight_decay'],
    'has_onehot': False,
    'onehot_cols': []
}
```

**预期数据量**: 58 行
