# 回归分析验证DiBS发现报告

**分析日期**: 2026-01-05
**验证边数**: 5条

---

## 📊 验证汇总

- **成功验证**: 5/5
- **失败验证**: 0/5

### 成功验证的边详情

- **回归显著（p<0.05）**: 5/5
- **方向与DiBS一致**: 4/5
- **完全验证（显著+方向一致）**: 4/5 ✅

## 📋 详细验证结果

| 任务组 | 边 | DiBS强度 | 回归系数 | p值 | R² | 验证状态 |
|--------|-----|---------|---------|-----|-----|----------|
| group1_examples | hyperparam_batch_size → energy | 0.200 | 0.128980 | 0.0318 | 0.0887 | ✅ 一致且显著 |
| group3_person_reid | hyperparam_epochs → energy_gpu | 0.300 | 0.263041 | 0.0028 | 0.1558 | ✅ 一致且显著 |
| group3_person_reid | hyperparam_epochs → energy_gpu | 0.400 | -0.371361 | 0.0000 | 0.5045 | ❌ 方向不一致 |
| group6_resnet | hyperparam_epochs → energy_gpu | 0.300 | 0.141047 | 0.0000 | 0.9972 | ✅ 一致且显著 |
| group6_resnet | hyperparam_epochs → energy_gpu | 0.150 | 0.436504 | 0.0000 | 0.9177 | ✅ 一致且显著 |

## ✅ 完全验证的因果边（显著+方向一致）

### group1_examples: hyperparam_batch_size → energy_gpu_max_watts

**DiBS发现**:
- 边强度: 0.200
- 预期方向: positive

**回归验证**:
- 回归系数: 0.128980
- 标准误: 0.059751
- t值: 2.1586
- p值: 0.031808 *
- 95%置信区间: [0.011314, 0.246646]
- R²: 0.0887
- 样本数: 259

**解释**:
- 批量大小（batch_size）每增加1，energy_gpu_max_watts增加约0.13单位
- 该因果关系在统计上显著（p=0.0318 < 0.05）
- 模型解释了8.9%的能耗变化

### group3_person_reid: hyperparam_epochs → energy_gpu_avg_watts

**DiBS发现**:
- 边强度: 0.300
- 预期方向: positive

**回归验证**:
- 回归系数: 0.263041
- 标准误: 0.086532
- t值: 3.0398
- p值: 0.002816 **
- 95%置信区间: [0.091995, 0.434087]
- R²: 0.1558
- 样本数: 146

**解释**:
- 训练轮数（epochs）每增加1，energy_gpu_avg_watts增加约0.26单位
- 该因果关系在统计上显著（p=0.0028 < 0.05）
- 模型解释了15.6%的能耗变化

### group6_resnet: hyperparam_epochs → energy_gpu_total_joules

**DiBS发现**:
- 边强度: 0.300
- 预期方向: positive

**回归验证**:
- 回归系数: 0.141047
- 标准误: 0.008354
- t值: 16.8838
- p值: 0.000000 ***
- 95%置信区间: [0.124231, 0.157863]
- R²: 0.9972
- 样本数: 49

**解释**:
- 训练轮数（epochs）每增加1，energy_gpu_total_joules增加约0.14单位
- 该因果关系在统计上显著（p=0.0000 < 0.05）
- 模型解释了99.7%的能耗变化

### group6_resnet: hyperparam_epochs → energy_gpu_avg_watts

**DiBS发现**:
- 边强度: 0.150
- 预期方向: positive

**回归验证**:
- 回归系数: 0.436504
- 标准误: 0.045569
- t值: 9.5790
- p值: 0.000000 ***
- 95%置信区间: [0.344779, 0.528230]
- R²: 0.9177
- 样本数: 49

**解释**:
- 训练轮数（epochs）每增加1，energy_gpu_avg_watts增加约0.44单位
- 该因果关系在统计上显著（p=0.0000 < 0.05）
- 模型解释了91.8%的能耗变化

## ⚠️ 需要进一步分析的边

### group3_person_reid: hyperparam_epochs → energy_gpu_min_watts

- DiBS强度: 0.400
- 回归系数: -0.371361 (p=0.0000)
- ⚠️ **方向不一致**: DiBS预期positive，回归发现negative

## 💡 结论

### DiBS验证率: 80.0%

✅ **DiBS发现高度可信**: 80%以上的边通过回归验证

### 关键发现

1. **epochs是能耗的主要驱动因素** ✅
   - group3_person_reid: 每增加1个epoch，energy_gpu_avg_watts增加0.26单位 (p=0.0028)
   - group6_resnet: 每增加1个epoch，energy_gpu_total_joules增加0.14单位 (p=0.0000)
   - group6_resnet: 每增加1个epoch，energy_gpu_avg_watts增加0.44单位 (p=0.0000)

2. **batch_size影响GPU峰值功率** ✅
   - group1_examples: 每增加1个batch_size，energy_gpu_max_watts增加0.13单位 (p=0.0318)

### 后续建议

1. 对完全验证的边，可以在论文中作为**强证据**引用
2. 对不一致的边，建议使用因果森林或工具变量法进一步验证
3. 结合DiBS（因果发现）+ 回归（因果量化）的双重证据，结论更可信

---

**报告生成时间**: 2026-01-05
**数据来源**: DiBS分析结果 + 原始DiBS训练数据
