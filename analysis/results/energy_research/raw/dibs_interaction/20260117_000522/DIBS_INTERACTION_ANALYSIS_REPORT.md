# 6分组交互项数据DiBS因果分析报告

**分析日期**: 2026-01-17 01:03:07
**数据源**: analysis/data/energy_research/6groups_interaction/
**数据类型**: 标准化 + 交互项 (超参数 × is_parallel)
**任务组数**: 6个
**DiBS配置**: alpha=0.05, beta=0.1 ⭐, particles=20, steps=5000

---

## 📊 总体统计

- **成功任务组**: 6/6
- **总耗时**: 57.8分钟 (0.96小时)
- **平均耗时**: 9.6分钟/组

## 📋 任务组详细结果

| 任务组 | 状态 | 耗时(分) | 样本数 | 特征数 | 超参数 | 交互项⭐ | 性能 | 能耗 | 强边(>0.3) |
|--------|------|---------|-------|-------|--------|----------|------|------|----------|
| examples（图像分类-小型） | ✅ 成功 | 15.8 | 304 | 23 | 3 | 3 | 1 | 4 | 65 |
| VulBERTa（代码漏洞检测） | ✅ 成功 | 6.7 | 72 | 21 | 3 | 3 | 3 | 4 | 60 |
| Person_reID（行人重识别） | ✅ 成功 | 12.5 | 206 | 24 | 3 | 3 | 3 | 4 | 72 |
| bug-localization（缺陷定位） | ✅ 成功 | 8.3 | 90 | 23 | 3 | 3 | 4 | 4 | 59 |
| MRT-OAST（缺陷定位） | ✅ 成功 | 7.7 | 72 | 24 | 4 | 4 | 3 | 4 | 78 |
| pytorch_resnet（图像分类-ResNe | ✅ 成功 | 6.7 | 74 | 21 | 3 | 3 | 2 | 4 | 42 |

## 🎯 研究问题1：超参数对能耗的影响（包括调节效应）

### 总体发现

- **主效应（超参数→能耗）**: 2条
- **⭐ 调节效应（交互项→能耗）**: 5条
- **间接路径（超参数→中介→能耗）**: 3条
- **总因果路径**: 10条

### ⭐ 调节效应分析 (Top 10)

| 任务组 | 交互项 | 能耗指标 | 调节强度 | 主效应强度 |
|--------|--------|----------|---------|----------|
| Person_reID（行人重识别） | hyperparam_epochs_x_is_parallel | energy_gpu_total_joules | 0.4000 | 0.3500 |
| examples（图像分类-小型） | hyperparam_batch_size_x_is_parallel | energy_cpu_total_joules | 0.3500 | 0.0000 |
| examples（图像分类-小型） | hyperparam_batch_size_x_is_parallel | energy_gpu_total_joules | 0.3000 | 0.0000 |
| bug-localization（缺陷定 | hyperparam_kfold_x_is_parallel | energy_gpu_total_joules | 0.3000 | 0.1000 |
| pytorch_resnet（图像分类- | hyperparam_epochs_x_is_parallel | energy_cpu_pkg_joules | 0.3000 | 0.3000 |

## 🔄 研究问题2：能耗-性能权衡关系

（与原始分析相同，未受交互项影响）

## 🔬 研究问题3：中介效应路径

（与原始分析相同，未受交互项影响）

## 💡 结论与下一步

✅ DiBS成功在6/6个任务组上完成因果发现。

### 交互项方案关键发现

1. DiBS能够识别交互项（超参数 × is_parallel）对能耗的影响
2. 调节效应揭示了并行模式如何改变超参数的因果作用
3. 主效应和调节效应可以同时存在，提供完整的因果图景


---

**报告生成时间**: 2026-01-17 01:03:07
**使用脚本**: run_dibs_6groups_interaction.py
**方法学参考**: docs/INTERACTION_TERMS_TRANSFORMATION_PLAN.md
