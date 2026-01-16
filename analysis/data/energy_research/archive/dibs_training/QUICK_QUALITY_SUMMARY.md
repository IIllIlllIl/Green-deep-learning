# DiBS数据质量快速总结

**评估日期**: 2026-01-15

## 总体评级: ✅ 优秀

- **5/6 组已准备就绪**，可直接用于DiBS因果图学习
- **总样本数**: 423个高质量训练样本
- **数据完整性**: 100%（所有组零缺失值）
- **特征质量**: 16-19个特征/组，覆盖超参数、能耗、性能

---

## 快速评级表

| 组别 | 样本量 | 超参数 | DiBS就绪 | 评级 | 建议用途 |
|------|--------|--------|----------|------|---------|
| **examples**（图像分类-小型） | 126 | 4个 | ✅ | 优秀 | 问题1/2/3 |
| **Person_reID**（行人重识别） | 118 | 4个 | ✅ | 优秀 | 问题1/2/3 |
| **pytorch_resnet**（ResNet） | 41 | 4个 | ✅ | 良好 | 问题1/2 |
| **MRT-OAST**（缺陷定位） | 46 | 0个 | ✅ | 良好 | 仅问题2 |
| **bug-localization**（缺陷定位） | 40 | 0个 | ✅ | 良好 | 仅问题2 |
| **VulBERTa**（漏洞检测） | 52 | 0个 | ❌ | 需清理 | - |

---

## 研究问题适用性

### 问题1: 超参数对能耗的影响 ⭐⭐⭐

**推荐3个组（305样本，12超参数）**:

1. **examples（图像分类-小型）** - 126样本 ⭐ 最佳
   - 4个超参数: batch_size, epochs, learning_rate, seed
   - 超参数多样性高（42个不同batch_size值）
   - 样本量充足，特征完整

2. **Person_reID（行人重识别）** - 118样本 ⭐ 最佳
   - 4个超参数: dropout, epochs, learning_rate, seed
   - 超参数多样性极高（dropout 81%, seed 95%唯一值比率）
   - 最多特征（19个），3个性能指标

3. **pytorch_resnet（图像分类-ResNet）** - 41样本
   - 4个超参数: epochs, learning_rate, seed, weight_decay
   - 超参数多样性高（learning_rate 90%唯一值比率）
   - 样本量略少但充分

### 问题2: 能耗和性能的权衡关系 ⭐⭐⭐

**所有5个就绪组都可用（377样本）**:
- 覆盖4种任务类型：图像分类、行人重识别、缺陷定位
- 11个能耗指标：CPU、GPU功耗和温度
- 多样的性能指标：准确率、loss、precision/recall等

### 问题3: 中间变量的中介效应 ⭐⭐

**推荐2个高质量组（244样本）**:

1. **examples（图像分类-小型）** - 最完整
   - 4超参数 → 11能耗指标 → 1性能指标
   - 清晰的因果链路径

2. **Person_reID（行人重识别）** - 最丰富
   - 4超参数 → 11能耗指标 → 3性能指标
   - 可分析多个性能维度的中介效应

---

## 关键发现

### ✅ 优势

1. **数据完整性优秀**: 所有6组零缺失值（100%完整）
2. **样本量充分**: 5组满足DiBS要求（≥30），2组优秀（≥50）
3. **超参数多样性高**: 3组包含4个超参数，唯一值比率高
4. **特征覆盖全面**: 16-19特征/组，包含超参数、能耗、性能
5. **任务类型多样**: 覆盖图像分类、行人重识别、缺陷定位等

### ⚠️ 限制

1. **3组缺少超参数**: VulBERTa、bug-localization、MRT-OAST
   - 只能用于能耗-性能关系分析（问题2）
   - 无法研究超参数的因果影响（问题1/3）

2. **3组样本量<50**: pytorch_resnet(41), bug-localization(40), MRT-OAST(46)
   - 满足DiBS最低要求（≥30）
   - 但可能在因果图学习时稳定性略差
   - 建议使用交叉验证或bootstrap评估不确定性

3. **1组需要清理**: VulBERTa
   - 存在1个常数特征（energy_gpu_util_max_percent = 100.0）
   - 需要在DiBS训练前移除该特征

---

## 使用建议

### 立即可用（无需任何处理）

**对于问题1（超参数影响）**:
```python
# 直接加载并使用
import pandas as pd

# 最佳选择：examples组（样本量最大，超参数多样性高）
df = pd.read_csv('group1_examples.csv')

# 标准化后可直接用于DiBS
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 输入到DiBS模型...
```

**对于问题2（能耗-性能权衡）**:
- 可以合并5个就绪组（377样本）进行综合分析
- 或单独分析每组以比较不同任务类型

### 需要简单清理

**VulBERTa组（如果使用）**:
```python
# 移除常数特征
df = pd.read_csv('group2_vulberta.csv')
df_clean = df.drop(columns=['energy_gpu_util_max_percent'])

# 现在可以使用（仅问题2，无超参数）
```

### 提高稳定性（可选）

**对于小样本组（<50）**:
```python
# 方案A: 使用k-fold交叉验证
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(df):
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    # 在train_data上训练DiBS，在val_data上验证

# 方案B: 使用bootstrap评估不确定性
from sklearn.utils import resample

n_bootstrap = 100
results = []
for i in range(n_bootstrap):
    df_boot = resample(df, n_samples=len(df), random_state=i)
    # 在df_boot上训练DiBS
    results.append(learned_graph)

# 分析results的稳定性
```

---

## 数据质量细节

### 超参数多样性（3个组）

**examples组（126样本）**:
- batch_size: 42个唯一值，范围[19, 10000]
- epochs: 11个唯一值，范围[5, 15]
- learning_rate: 28个唯一值，范围[0.0056, 0.0184]
- seed: 28个唯一值，范围[1, 9809]

**Person_reID组（118样本）**:
- dropout: 96个唯一值（81%），范围[0.30, 0.59]
- epochs: 45个唯一值（38%），范围[31, 90]
- learning_rate: 96个唯一值（81%），范围[0.025, 0.075]
- seed: 112个唯一值（95%），范围[3, 9974]

**pytorch_resnet组（41样本）**:
- epochs: 32个唯一值（78%），范围[108, 297]
- learning_rate: 37个唯一值（90%），范围[0.051, 0.135]
- seed: 40个唯一值（98%），范围[409, 9992]
- weight_decay: 37个唯一值（90%），范围[0.000011, 0.00066]

### 能耗指标（所有组）

**11个能耗指标**（一致性极好）:
- CPU: pkg_joules, ram_joules, total_joules
- GPU功耗: avg_watts, max_watts, min_watts, total_joules
- GPU温度: temp_avg_celsius, temp_max_celsius
- GPU利用率: util_avg_percent, util_max_percent

### 性能指标（各组不同）

- **examples**: test_accuracy
- **Person_reID**: map, rank1, rank5
- **pytorch_resnet**: best_val_accuracy, test_accuracy
- **VulBERTa**: eval_loss, final_training_loss, eval_samples_per_second
- **bug-localization**: top1/5/10/20_accuracy
- **MRT-OAST**: accuracy, precision, recall

---

## 行动建议

### 立即开始（推荐）

1. **问题1分析**: 使用examples组（126样本，4超参数）
   - 样本量最大，超参数多样性高
   - 数据质量最优，特征完整
   - 可作为pilot study验证DiBS方法

2. **问题2分析**: 使用所有5个就绪组（377样本）
   - 覆盖多种任务类型
   - 样本量充足，结果更稳健
   - 可以比较不同任务的能耗-性能权衡

### 后续扩展（可选）

1. **增加样本量**: 对于<50样本的组
   - pytorch_resnet(41), bug-localization(40), MRT-OAST(46)
   - 可考虑收集更多实验数据

2. **补充超参数**: 对于无超参数的组
   - VulBERTa, bug-localization, MRT-OAST
   - 当前只能分析能耗-性能关系

3. **清理VulBERTa**: 移除常数特征后可用于问题2

---

## 结论

**总体质量: 优秀** ✅

- 数据已准备就绪，**可立即开始DiBS分析**
- 推荐优先使用examples组（问题1）和所有5组（问题2）
- 数据质量高，特征完整，超参数多样性好
- 仅需对VulBERTa组做简单清理（移除1个常数特征）

**预期分析效果**:
- 问题1（超参数影响）: 可基于3组（305样本）得到稳健结果
- 问题2（能耗-性能权衡）: 可基于5组（377样本）得到综合结论
- 问题3（中介效应）: 可基于2组（244样本）探索因果机制

**下一步**: 开始DiBS因果图学习，建议从examples组开始（质量最优，样本最多）

---

**详细报告**: 参见 `DATA_QUALITY_ASSESSMENT_20260115.md`

**生成时间**: 2026-01-15 17:08
