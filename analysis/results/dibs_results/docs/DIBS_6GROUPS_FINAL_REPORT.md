# 6分组数据DiBS因果分析报告

**分析日期**: 2026-01-16 01:31:27
**数据源**: analysis/data/energy_research/6groups_final/ (818条记录)
**任务组数**: 6个
**DiBS配置**: alpha=0.05, beta=0.1 ⭐, particles=20, steps=5000
**配置来源**: DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md (最优配置)

---

## 📊 总体统计

- **成功任务组**: 6/6
- **总耗时**: 48.1分钟 (0.80小时)
- **平均耗时**: 8.0分钟/组

## 📋 任务组详细结果

| 任务组 | 状态 | 耗时(分) | 样本数 | 特征数 | 超参数 | 性能 | 能耗 | 中介 | 强边(>0.3) | 总边(>0.01) |
|--------|------|---------|-------|-------|--------|------|------|------|-----------|------------|
| examples（图像分类-小型） | ✅ 成功 | 14.4 | 304 | 20 | 4 | 1 | 4 | 7 | 135 | 230 |
| VulBERTa（代码漏洞检测） | ✅ 成功 | 5.2 | 72 | 18 | 4 | 3 | 4 | 6 | 100 | 185 |
| Person_reID（行人重识别） | ✅ 成功 | 10.6 | 206 | 21 | 4 | 3 | 4 | 7 | 139 | 277 |
| bug-localization（缺陷定位） | ✅ 成功 | 6.6 | 90 | 20 | 4 | 4 | 4 | 7 | 143 | 215 |
| MRT-OAST（缺陷定位） | ✅ 成功 | 6.1 | 72 | 20 | 5 | 3 | 4 | 7 | 98 | 232 |
| pytorch_resnet（图像分类-ResNe | ✅ 成功 | 5.2 | 74 | 18 | 4 | 2 | 4 | 7 | 97 | 200 |

## 🎯 研究问题1：超参数对能耗的影响

### 总体发现

- **直接因果边（超参数→能耗）**: 57条
- **间接路径（超参数→中介→能耗）**: 133条
- **总因果路径**: 190条

### 超参数→能耗直接效应 (Top 10)

| 任务组 | 超参数 | 能耗指标 | 强度 |
|--------|--------|----------|------|
| VulBERTa（代码漏洞检测） | hyperparam_learning_rate | energy_gpu_total_joules | 1.0000 |
| VulBERTa（代码漏洞检测） | hyperparam_epochs | energy_cpu_ram_joules | 1.0000 |
| VulBERTa（代码漏洞检测） | hyperparam_epochs | energy_gpu_total_joules | 1.0000 |
| VulBERTa（代码漏洞检测） | hyperparam_seed | energy_gpu_total_joules | 1.0000 |
| VulBERTa（代码漏洞检测） | hyperparam_l2_regularization | energy_gpu_total_joules | 1.0000 |
| Person_reID（行人重识别） | hyperparam_dropout | energy_cpu_ram_joules | 1.0000 |
| Person_reID（行人重识别） | hyperparam_dropout | energy_gpu_total_joules | 1.0000 |
| Person_reID（行人重识别） | hyperparam_learning_rate | energy_cpu_ram_joules | 1.0000 |
| Person_reID（行人重识别） | hyperparam_learning_rate | energy_gpu_total_joules | 1.0000 |
| Person_reID（行人重识别） | hyperparam_epochs | energy_cpu_ram_joules | 1.0000 |

## 🔄 研究问题2：能耗-性能权衡关系

### 总体发现

- **直接因果边（性能→能耗）**: 46条
- **直接因果边（能耗→性能）**: 0条
- **共同超参数**: 8个（同时影响能耗和性能）
- **中介权衡路径**: 200条

## 🔬 研究问题3：中介效应路径

### 总体发现

- **中介路径（超参数→中介→能耗）**: 133条
- **中介路径（超参数→中介→性能）**: 15条
- **多步路径（≥4节点）**: 278条
- **总中介路径**: 426条

## 💡 结论与下一步

✅ DiBS成功在6/6个任务组上完成因果发现。

### 下一步建议

1. 使用回归分析量化DiBS发现的因果边强度
2. 对中介路径进行Sobel检验验证
3. 生成因果图可视化
4. 撰写研究发现报告

---

**报告生成时间**: 2026-01-16 01:31:27
**使用脚本**: run_dibs_6groups_final.py
**参考文档**: DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md
