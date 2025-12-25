# 因果分析数据处理方案总结

**版本**: v3.0
**日期**: 2025-12-22
**状态**: ✅ 方案确认完成，进入实施阶段

---

## 📋 核心决策

### 最终方案：4个任务组 + One-Hot编码

| 任务组 | 样本量 | 性能指标 | One-Hot变量 | 变量总数 |
|-------|-------|---------|------------|---------|
| **图像分类** | **185个** | perf_test_accuracy | `is_mnist`, `is_cifar10` (2个) | **15个** |
| **Person_reID检索** | 93个 | perf_map | `is_densenet121`, `is_hrnet18`, `is_pcb` (3个) | **16个** |
| **VulBERTa漏洞检测** | 52个 | perf_eval_loss | 无 | **13个** |
| **Bug定位** | 40个 | perf_top1_accuracy | 无 | **13个** |

**总有效样本**: 370个

---

## 🎯 关键改进

### 1. 合并MNIST和CIFAR-10 ⭐⭐⭐

**原因**：
- ✅ 都使用`perf_test_accuracy`，语义相同
- ✅ 样本量提升：26个（仅CIFAR-10）→ 185个（合并后）**提升7倍**
- ✅ 更高的DiBS统计功效

**挑战**：
- ⚠️ 性能分布差异大：MNIST波动30.7%，CIFAR-10仅0.46%

**解决方案**：
- ✅ 添加2个One-Hot变量（`is_mnist`, `is_cifar10`）控制异质性

### 2. One-Hot编码（控制混淆） ⭐⭐⭐

**作用**：避免DiBS将数据集/模型的基线差异误判为因果关系

**示例**（无One-Hot的问题）：
```
DiBS可能学到：learning_rate → test_accuracy (ATE = 0.15)
但实际原因：MNIST用更高learning_rate + MNIST准确率基线更低
→ 混淆了"数据集差异"和"learning_rate因果效应"
```

**示例**（有One-Hot）：
```
正确的因果图：
  is_mnist → learning_rate  （MNIST倾向用更高学习率）
  is_mnist → test_accuracy   （MNIST的准确率基线）
  learning_rate → test_accuracy  （控制数据集后的真实因果效应）
```

### 3. 动态变量选择（避免稀疏变量）

**规则**：只保留填充率 > 10% 的超参数

**效果**：
- 图像分类：保留3个超参数（learning_rate 53%, batch_size 49%, training_duration 55%）
- Person_reID：保留3个超参数（learning_rate 60%, dropout 60%, training_duration 61%）
- VulBERTa：保留2个超参数（training_duration 27%, l2_regularization 27%）
- Bug定位：保留2个超参数（training_duration 25%, l2_regularization 25%）

---

## 📊 完整变量集（每个任务组）

### 超参数（2-5个，动态选择）
1. `hyperparam_learning_rate` - 学习率
2. `hyperparam_batch_size` - 批次大小
3. `hyperparam_dropout` - Dropout比例
4. `hyperparam_seed` - 随机种子
5. `hyperparam_training_duration` ✅ **新**：统一epochs和max_iter
6. `hyperparam_l2_regularization` ✅ **新**：统一weight_decay和alpha

### 能耗总量（2个）
6. `energy_cpu_total_joules` - CPU总能耗
7. `energy_gpu_total_joules` - GPU总能耗

### 能耗中介变量（5个）✅ **新增**
8. `gpu_util_avg` - GPU平均利用率（%）
9. `gpu_temp_max` - GPU最高温度（°C）
10. `cpu_pkg_ratio` - CPU Package能耗比例
11. `gpu_power_fluctuation` - GPU功率波动（max - min，单位W）
12. `gpu_temp_fluctuation` - GPU温度波动（max - avg，单位°C）

### 性能指标（1个，任务特定）
13. `perf_test_accuracy` / `perf_map` / `perf_eval_loss` / `perf_top1_accuracy`

### One-Hot编码（0-3个）✅ **新增**
14-16. `is_mnist`, `is_cifar10` / `is_densenet121`, `is_hrnet18`, `is_pcb`

**变量总数**: 13-16个（取决于超参数填充率和One-Hot数量）

---

## 🔧 实施步骤

### 阶段1: 数据预处理 ⏳

**脚本**: `analysis/scripts/preprocess_stratified_data.py`

**输出**:
```
analysis/data/training_data_image_classification.csv  (185行, ~15列)
analysis/data/training_data_person_reid.csv           (93行, ~16列)
analysis/data/training_data_vulberta.csv              (52行, ~13列)
analysis/data/training_data_bug_localization.csv      (40行, ~13列)
```

### 阶段2: DiBS因果图学习 ⏳

**预估时间**: ~60分钟（4个任务组）
- 图像分类: ~30分钟（185样本，15变量）
- Person_reID: ~15分钟（93样本，16变量）
- VulBERTa: ~8分钟（52样本，13变量）
- Bug定位: ~6分钟（40样本，13变量）

### 阶段3: DML因果推断 ⏳

**输出**: 每条因果边的ATE、置信区间、p值

### 阶段4: 报告生成 ⏳

**输出**:
- 综合报告：跨任务共性发现
- 任务特定报告：4个任务组各1份

---

## 📈 预期因果发现

### 跨任务通用模式

**超参数 → 能耗**：
- `learning_rate → gpu_util_avg → energy_gpu_total` （学习率影响GPU利用率）
- `batch_size → gpu_util_avg → energy_gpu_total` （批次大小影响GPU利用率）
- `training_duration → energy_cpu_total, energy_gpu_total` （训练时长直接影响能耗）

**超参数 → 性能**：
- `learning_rate → perf_*` （学习率影响性能）
- `dropout → perf_*` （正则化影响性能）

**One-Hot → 其他变量**（基线差异，不可干预）：
- `is_mnist → test_accuracy` （MNIST的准确率基线）
- `is_mnist → learning_rate` （MNIST倾向用不同的超参数）
- `is_densenet121 → mAP` （不同模型的mAP基线）

### 任务特定模式

**图像分类**：
- `is_mnist → gpu_temp_fluctuation` （MNIST训练简单，温度波动小）

**Person_reID**：
- `is_pcb → energy_gpu_total` （PCB模型计算量大，能耗高）

---

## ⚠️ 注意事项

### 1. One-Hot变量的因果解释

**正确解释**：
- ✅ `ATE(is_mnist → test_accuracy) = -9.4%` → MNIST的准确率基线比CIFAR-10低9.4%
- ✅ 这是**基线差异**，不可干预（不能"把MNIST变成CIFAR-10"）

**错误解释**：
- ❌ "将is_mnist从0改为1可以降低9.4%准确率" → 无意义

**用途**：
- ✅ 控制混淆，使DiBS正确识别超参数的因果效应
- ✅ 解释任务间的基线差异

### 2. 超参数填充率差异

不同任务组的超参数填充率差异大：
- **图像分类**：learning_rate 53%, batch_size 49%（较好）
- **Bug定位**：learning_rate 0%（完全缺失）

**影响**：
- Bug定位组只能分析training_duration和l2_regularization的因果效应
- 无法发现learning_rate的因果模式

### 3. 数据保留数量

**4个任务组 vs 5个任务组**：
- 数据保留数量：370个（完全相同）
- 区别：仅在于分组方式（合并MNIST+CIFAR-10 vs 分开）

---

## 📚 相关文档

- [VARIABLE_EXPANSION_PLAN.md](./VARIABLE_EXPANSION_PLAN.md) - **完整方案详解** ⭐⭐⭐
- [COLUMN_USAGE_ANALYSIS.md](./COLUMN_USAGE_ANALYSIS.md) - 原始列使用率分析
- [ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md](./ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md) - DiBS基线分析（v1.0）
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - 数据迁移指南

---

**维护者**: Green
**最后更新**: 2025-12-22
**下一步**: 实施阶段1 - 编写预处理脚本 `preprocess_stratified_data.py`
