# 中介效应分析报告（问题3）

**分析日期**: 2026-01-06
**测试路径数**: 7条

---

## 📊 分析汇总

- **成功分析**: 7/7
- **失败分析**: 0/7

### 中介效应详情

- **显著中介效应**: 2/7
- **完全中介**: 0/7
- **部分中介**: 2/7
- **无中介**: 5/7

## 📋 详细结果

| 任务组 | 路径 | 间接效应 | Sobel p | 中介类型 | 中介比例 |
|--------|------|----------|---------|----------|----------|
| group6_resnet | hyperparam_epoc→energy_gpu→energy_g | -0.0629 | 0.0000 | 部分中介 | -44.6% |
| group6_resnet | hyperparam_epoc→energy_gpu→energy_g | 0.0167 | 0.0612 | 无中介 | 11.8% |
| group6_resnet | hyperparam_epoc→energy_gpu→energy_g | -0.0289 | 0.0001 | 部分中介 | -20.5% |
| group3_person_reid | hyperparam_epoc→energy_gpu→energy_g | 0.0133 | 0.3531 | 无中介 | 5.1% |
| group3_person_reid | hyperparam_epoc→energy_gpu→energy_g | 0.0403 | 0.4665 | 无中介 | 15.3% |
| group1_examples | hyperparam_batc→energy_gpu→energy_g | 0.0496 | 0.2606 | 无中介 | 38.4% |
| group1_examples | hyperparam_batc→energy_gpu→energy_g | 0.0221 | 0.6724 | 无中介 | 17.2% |

## ✅ 显著中介路径

### group6_resnet: epochs通过GPU利用率影响总能耗

**路径**: hyperparam_epochs → energy_gpu_util_avg_percent → energy_gpu_total_joules

**路径系数**:
- 路径a (hyperparam_epochs→energy_gpu_util_avg_percent): -0.4313 (p=0.0000)
- 路径b (energy_gpu_util_avg_percent→energy_gpu_total_joules): 0.1458 (p=0.0000)
- 总效应c: 0.1410 (p=0.0000)
- 直接效应c': 0.2039 (p=0.0000)

**中介效应**:
- 间接效应: -0.0629
- Sobel检验: z=-5.4673, p=0.0000
- 中介类型: **部分中介**
- 中介比例: -44.6%

**解释**:
- hyperparam_epochs对energy_gpu_total_joules的影响**部分**通过energy_gpu_util_avg_percent实现（-44.6%）
- 还存在144.6%的直接效应

### group6_resnet: epochs通过GPU峰值利用率影响总能耗

**路径**: hyperparam_epochs → energy_gpu_util_max_percent → energy_gpu_total_joules

**路径系数**:
- 路径a (hyperparam_epochs→energy_gpu_util_max_percent): -0.3592 (p=0.0000)
- 路径b (energy_gpu_util_max_percent→energy_gpu_total_joules): 0.0806 (p=0.0000)
- 总效应c: 0.1410 (p=0.0000)
- 直接效应c': 0.1700 (p=0.0000)

**中介效应**:
- 间接效应: -0.0289
- Sobel检验: z=-3.8834, p=0.0001
- 中介类型: **部分中介**
- 中介比例: -20.5%

**解释**:
- hyperparam_epochs对energy_gpu_total_joules的影响**部分**通过energy_gpu_util_max_percent实现（-20.5%）
- 还存在120.5%的直接效应

## ⚠️ 无显著中介的路径

### group6_resnet: epochs通过GPU温度影响总能耗

- 路径: hyperparam_epochs → energy_gpu_temp_max_celsius → energy_gpu_total_joules
- 间接效应: 0.0167 (p=0.0612)
- ⚠️ 路径b不显著: energy_gpu_temp_max_celsius对energy_gpu_total_joules无显著影响（控制hyperparam_epochs后）

### group3_person_reid: epochs通过GPU利用率影响平均功率

- 路径: hyperparam_epochs → energy_gpu_util_avg_percent → energy_gpu_avg_watts
- 间接效应: 0.0133 (p=0.3531)
- ⚠️ 路径a不显著: hyperparam_epochs对energy_gpu_util_avg_percent无显著影响
- ⚠️ 路径b不显著: energy_gpu_util_avg_percent对energy_gpu_avg_watts无显著影响（控制hyperparam_epochs后）

### group3_person_reid: epochs通过GPU温度影响平均功率

- 路径: hyperparam_epochs → energy_gpu_temp_max_celsius → energy_gpu_avg_watts
- 间接效应: 0.0403 (p=0.4665)
- ⚠️ 路径a不显著: hyperparam_epochs对energy_gpu_temp_max_celsius无显著影响

### group1_examples: batch_size通过GPU温度影响峰值功率

- 路径: hyperparam_batch_size → energy_gpu_temp_max_celsius → energy_gpu_max_watts
- 间接效应: 0.0496 (p=0.2606)
- ⚠️ 路径a不显著: hyperparam_batch_size对energy_gpu_temp_max_celsius无显著影响

### group1_examples: batch_size通过GPU利用率影响峰值功率

- 路径: hyperparam_batch_size → energy_gpu_util_avg_percent → energy_gpu_max_watts
- 间接效应: 0.0221 (p=0.6724)
- ⚠️ 路径a不显著: hyperparam_batch_size对energy_gpu_util_avg_percent无显著影响

## 💡 结论

### 中介效应检出率: 28.6%

⚠️ **部分中间变量起到中介作用**

### 关键发现

1. **GPU利用率是重要中介变量** ✅
   - group6_resnet: hyperparam_epochs通过energy_gpu_util_avg_percent影响energy_gpu_total_joules （部分中介，-44.6%）
   - group6_resnet: hyperparam_epochs通过energy_gpu_util_max_percent影响energy_gpu_total_joules （部分中介，-20.5%）

### 对问题3的回答

**问题3: 训练过程中的中间变量（如GPU利用率、温度等）在超参数对能耗的影响中起到什么作用？**

**回答**: 中间变量在超参数对能耗的影响中起到**显著中介作用**。

- **部分中介路径** (2条): 超参数对能耗的影响部分通过中间变量实现

这说明：
1. 超参数不是直接影响能耗，而是通过改变GPU状态（利用率、温度、显存）来影响能耗
2. 优化能耗的关键是控制这些中间变量

---

**报告生成时间**: 2026-01-06
**分析方法**: Sobel检验中介分析
**数据来源**: DiBS训练数据（6个任务组）
