# 能耗数据集方法对比报告

**生成时间**: 2025-12-26 20:23:33

---

## 测试方法摘要

| 方法 | 运行时间 | 成功 | 关键发现 |
|------|----------|------|----------|
| Correlation Analysis | 0.0秒 | ✅ | 强相关对: 5 |
| Regression Analysis | 0.4秒 | ✅ | 分析了3个目标 |
| Partial Correlation | 0.1秒 | ✅ | 强相关对: 0 |
| PC Algorithm | 0.0秒 | ❌ | 失败 |
| Mutual Information | 0.1秒 | ✅ | 完成 |

---

## Correlation Analysis

### 统计摘要

- Pearson平均: 0.232
- Pearson最大: 0.931
- 强相关对(|r|>0.7): 5

### Top 10相关对

1. energy_gpu_avg_watts <-> gpu_temp_max: 0.931
2. energy_cpu_total_joules <-> energy_gpu_total_joules: 0.907
3. energy_gpu_avg_watts <-> gpu_util_avg: 0.842
4. gpu_util_avg <-> gpu_temp_max: 0.801
5. gpu_power_fluctuation <-> gpu_temp_fluctuation: 0.773
6. energy_gpu_total_joules <-> gpu_util_avg: 0.680
7. is_mnist_ff <-> perf_test_accuracy: -0.668
8. is_siamese <-> energy_gpu_total_joules: 0.648
9. is_mnist_rnn <-> cpu_pkg_ratio: 0.569
10. energy_cpu_total_joules <-> gpu_util_avg: 0.553

---

## Regression Analysis

### 目标: energy_cpu_total_joules

- 线性回归 R²: 0.827
- 随机森林 R²: 0.985

### 目标: energy_gpu_total_joules

- 线性回归 R²: 0.797
- 随机森林 R²: 0.986

### 目标: energy_gpu_avg_watts

- 线性回归 R²: 0.976
- 随机森林 R²: 0.999


---

## Partial Correlation

### 统计摘要

- Pearson平均: 0.000
- Pearson最大: 0.000
- 强相关对(|r|>0.7): 0

### Top 10相关对


---

## PC Algorithm

- 失败原因: causal-learn not installed

---

## Mutual Information


---

## 方法推荐

基于测试结果，推荐使用的方法按优先级排序：

1. **Correlation Analysis** ⭐⭐⭐⭐⭐
   - 原因: 快速、直观、易解释

2. **Partial Correlation** ⭐⭐⭐⭐⭐
   - 原因: 快速、直观、易解释

3. **Regression Analysis** ⭐⭐⭐⭐
   - 原因: 提供特征重要性，可预测

4. **Mutual Information** ⭐⭐⭐
   - 原因: 捕捉非线性关系

5. **PC Algorithm** ⭐
   - 原因: 未成功

