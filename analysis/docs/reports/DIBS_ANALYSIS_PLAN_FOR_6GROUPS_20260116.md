# DiBS因果分析方案 - 6分组数据

**文档版本**: v1.0
**创建日期**: 2026-01-16
**数据版本**: 6groups_final (818条记录, 100%利用率)
**状态**: 📋 方案设计完成，待执行

---

## 📑 目录

1. [执行摘要](#执行摘要)
2. [数据准备情况](#数据准备情况)
3. [DiBS环境和代码分析](#dibs环境和代码分析)
4. [现有脚本分析](#现有脚本分析)
5. [推荐方案](#推荐方案)
6. [执行步骤](#执行步骤)
7. [预期结果](#预期结果)
8. [注意事项](#注意事项)

---

## 执行摘要

### 任务目标

使用DiBS (Differentiable Bayesian Structure Learning) 对6分组数据进行因果图学习，回答三个核心研究问题：

1. **问题1**: 超参数对能耗的影响（方向和大小）
2. **问题2**: 能耗和性能之间的权衡关系
3. **问题3**: 中间变量的中介效应

### 当前状态

- ✅ 数据已准备完成 (818条记录, 100%利用率)
- ✅ DiBS环境已配置 (`causal-research` conda环境)
- ✅ DiBS脚本已存在且经过参数调优
- ✅ 数据质量验证完成 (99.5%评分)
- ⏳ 待执行DiBS分析

### 推荐方案

**方案选择**: 使用现有的 `run_dibs_on_new_6groups.py` 脚本，但需要更新数据路径和配置

**预计耗时**: 6组 × 10-15分钟/组 = 1-1.5小时

---

##数据准备情况

### 6分组数据概览

| 分组 | Repository | 模型数 | 数据行数 | 列数 | 数据质量 |
|------|-----------|--------|---------|------|---------|
| group1_examples | examples | 4 | 304 | 21 | 优秀 (4.3%缺失) |
| group2_vulberta | VulBERTa | 1 | 72 | 20 | 良好 (12.9%缺失) |
| group3_person_reid | Person_reID | 3 | 206 | 22 | 优秀 (3.8%缺失) |
| group4_bug_localization | bug-localization | 1 | 90 | 21 | 良好 (12.5%缺失) |
| group5_mrt_oast | MRT-OAST | 1 | 72 | 21 | 良好 (8.5%缺失, 16.7%性能缺失) |
| group6_resnet | pytorch_resnet | 1 | 74 | 19 | 优秀 (5.8%缺失) |
| **总计** | - | **11** | **818** | - | **优秀 (99.5%总评)** |

### 数据位置

```
/home/green/energy_dl/nightly/analysis/data/energy_research/6groups_final/
├── group1_examples.csv           # 304行 × 21列
├── group2_vulberta.csv           # 72行 × 20列
├── group3_person_reid.csv        # 206行 × 22列
├── group4_bug_localization.csv   # 90行 × 21列
├── group5_mrt_oast.csv           # 72行 × 21列
└── group6_resnet.csv             # 74行 × 19列
```

### 数据特点

**优势**:
- ✅ 100%数据利用率 (818/818)
- ✅ 无重复记录 (timestamp唯一)
- ✅ 模型变量One-hot n-1编码正确
- ✅ L2正则化语义合并完成
- ✅ 能耗数据100%完整

**注意事项**:
- ⚠️ Group 5有16.7%性能数据缺失 (仍可用，60/72条可用)
- ⚠️ 部分能耗指标有缺失值（约7.2%平均缺失率）

**DiBS兼容性**:
- ⚠️ **DiBS要求零缺失值** - 需要在分析前处理缺失值
- ✅ 已包含11个能耗指标
- ✅ 超参数、性能指标、中介变量清晰分类

---

## DiBS环境和代码分析

### 1. Conda环境

**环境名称**: `causal-research`
**位置**: `/home/green/miniconda3/envs/causal-research`

**激活方式**:
```bash
conda activate causal-research
```

或使用完整路径：
```bash
/home/green/miniconda3/envs/causal-research/bin/python script.py
```

**⚠️ 重要**: base环境没有安装DiBS，必须使用causal-research环境！

### 2. DiBS核心模块

**位置**: `analysis/utils/causal_discovery.py`

**类**: `CausalGraphLearner`

**关键参数**:
```python
CausalGraphLearner(
    n_vars=21,                      # 变量数（根据分组）
    alpha=0.05,                     # DAG惩罚参数（稀疏性）
    beta=0.1,                       # 无环约束惩罚斜率
    n_particles=20,                 # 粒子数（推荐值）
    tau=1.0,                        # Gumbel-softmax温度
    n_steps=5000,                   # 迭代次数
    n_grad_mc_samples=128,          # MC梯度样本数
    n_acyclicity_mc_samples=32,     # 无环性MC样本数
    random_seed=42
)
```

**输入要求**:
- DataFrame格式数据
- **零缺失值**（DiBS会崩溃）
- 无常量特征（标准差>0）

**输出**:
- `np.ndarray`: 因果图邻接矩阵 (n_vars × n_vars)
- `graph[i,j]`: 变量i→j的因果边强度 (0-1)

### 3. 参数调优结果

根据 `docs/reports/DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md`，最优配置为：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `alpha_linear` | 0.05 | DiBS默认值，效果良好 |
| `beta_linear` | 0.1 | **关键** - 低无环约束，允许更多边探索 |
| `n_particles` | 20 | 最佳性价比（10太少，50太慢） |
| `n_steps` | 5000 | 足够收敛 |
| `tau` | 1.0 | Gumbel-softmax温度 |

**成功案例**:
- 2026-01-05在问题2和3的数据上成功发现因果边
- beta=0.1时发现70+条强边（threshold>0.3）

---

## 现有脚本分析

### 主脚本: `run_dibs_on_new_6groups.py`

**位置**: `/home/green/energy_dl/nightly/analysis/scripts/run_dibs_on_new_6groups.py`

**功能**:
1. ✅ 加载6个分组数据
2. ✅ 自动分类变量（超参数、性能、能耗、中介）
3. ✅ 运行DiBS因果发现
4. ✅ 提取3个研究问题的证据
5. ✅ 生成总结报告

**关键函数**:

| 函数 | 功能 |
|------|------|
| `load_task_group_data()` | 加载并验证数据 |
| `classify_variables()` | 变量分类 |
| `run_dibs_analysis()` | 执行DiBS分析 |
| `extract_research_question_1_evidence()` | 提取问题1证据（超参数→能耗） |
| `extract_research_question_2_evidence()` | 提取问题2证据（能耗-性能权衡） |
| `extract_research_question_3_evidence()` | 提取问题3证据（中介效应） |
| `generate_summary_report()` | 生成Markdown报告 |

**当前配置**:

```python
OPTIMAL_CONFIG = {
    "alpha_linear": 0.05,
    "beta_linear": 0.1,          # ⭐ 关键参数
    "n_particles": 20,
    "tau": 1.0,
    "n_steps": 5000,
    "n_grad_mc_samples": 128,
    "n_acyclicity_mc_samples": 32
}
```

**问题**:
1. ⚠️ 脚本期望数据在 `data/energy_research/dibs_training/` 目录
2. ⚠️ 实际数据在 `data/energy_research/6groups_final/` 目录
3. ⚠️ 预期数据量与实际不符（脚本中hardcode的expected_samples不正确）

---

## 推荐方案

### 方案A: 修复现有脚本（推荐） ⭐

**优势**:
- ✅ 复用已验证的DiBS配置
- ✅ 复用证据提取逻辑
- ✅ 自动生成报告

**需要修改**:
1. 更新数据路径: `dibs_training/` → `6groups_final/`
2. 更新expected_samples和expected_features
3. 添加缺失值处理逻辑

**估计工作量**: 30分钟修改 + 1-1.5小时执行

### 方案B: 简化快速分析

创建一个精简版脚本，只执行DiBS学习，不做复杂的证据提取。

**优势**:
- ✅ 快速验证DiBS是否能运行
- ✅ 获得原始因果图矩阵

**劣势**:
- ❌ 需要手动提取研究问题证据
- ❌ 需要手动生成报告

**估计工作量**: 15分钟编写 + 1小时执行 + 2小时手动分析

### 方案C: 分批渐进式分析

先在1-2个小数据集上验证，再扩展到全部6组。

**优势**:
- ✅ 降低风险
- ✅ 快速发现问题

**劣势**:
- ❌ 总耗时更长
- ❌ 需要多次调整

---

## 推荐执行方案: 方案A（修复现有脚本）

### 步骤1: 修复数据路径和配置（预计15分钟）

**修改文件**: `scripts/run_dibs_on_new_6groups.py`

**修改1 - 数据路径 (line 100)**:
```python
# 原代码
data_dir = Path(__file__).parent.parent / "data" / "energy_research" / "dibs_training"

# 修改为
data_dir = Path(__file__).parent.parent / "data" / "energy_research" / "6groups_final"
```

**修改2 - 更新预期数据量 (lines 42-85)**:
```python
TASK_GROUPS = [
    {
        "id": "group1_examples",
        "name": "examples（图像分类-小型）",
        "csv_file": "group1_examples.csv",
        "expected_samples": 304,      # 修改
        "expected_features": 21       # 修改
    },
    {
        "id": "group2_vulberta",
        "name": "VulBERTa（代码漏洞检测）",
        "csv_file": "group2_vulberta.csv",
        "expected_samples": 72,       # 修改
        "expected_features": 20       # 修改
    },
    {
        "id": "group3_person_reid",
        "name": "Person_reID（行人重识别）",
        "csv_file": "group3_person_reid.csv",
        "expected_samples": 206,      # 修改
        "expected_features": 22       # 修改
    },
    {
        "id": "group4_bug_localization",
        "name": "bug-localization（缺陷定位）",
        "csv_file": "group4_bug_localization.csv",
        "expected_samples": 90,       # 修改
        "expected_features": 21       # 修改
    },
    {
        "id": "group5_mrt_oast",
        "name": "MRT-OAST（缺陷定位）",
        "csv_file": "group5_mrt_oast.csv",
        "expected_samples": 72,       # 修改
        "expected_features": 21       # 修改
    },
    {
        "id": "group6_resnet",
        "name": "pytorch_resnet（图像分类-ResNet）",
        "csv_file": "group6_resnet.csv",
        "expected_samples": 74,       # 修改
        "expected_features": 19       # 修改
    }
]
```

**修改3 - 添加缺失值处理 (在load_task_group_data函数中，line 118之后)**:
```python
# 检查缺失值
missing_count = df.isnull().sum().sum()
if missing_count > 0:
    print(f"  ⚠️ 警告: 发现 {missing_count} 个缺失值！")
    print(f"  处理方式: 使用列均值填充...")

    # 对每一列使用均值填充
    for col in df.columns:
        if df[col].isnull().any():
            missing_before = df[col].isnull().sum()
            df[col] = df[col].fillna(df[col].mean())
            print(f"    - {col}: {missing_before}个缺失值已填充")
```

### 步骤2: 执行DiBS分析（预计1-1.5小时）

**执行命令**:
```bash
cd /home/green/energy_dl/nightly/analysis

# 激活causal-research环境
conda activate causal-research

# 运行DiBS分析
python scripts/run_dibs_on_new_6groups.py
```

**预期输出**:
```
================================================================================
新6分组数据DiBS因果分析
================================================================================
开始时间: 2026-01-16 XX:XX:XX
数据源: data.csv (2026-01-15生成)
任务组数: 6
DiBS配置: alpha=0.05, beta=0.1, particles=20
================================================================================

输出目录: /home/green/energy_dl/nightly/analysis/results/energy_research/new_6groups_dibs/20260116_XXXXXX

================================================================================
进度: 1/6
================================================================================

加载数据: examples（图像分类-小型）
  数据规模: 304行 × 21列
  预期规模: 304行 × 21列

================================================================================
任务组: group1_examples - examples（图像分类-小型）
================================================================================

变量分类:
  超参数: 4个
    - hyperparam_batch_size
    - hyperparam_learning_rate
    - hyperparam_epochs
    - hyperparam_seed
  性能指标: 1个
    - perf_test_accuracy
  能耗指标: 3个
    - energy_cpu_pkg_joules
    - energy_cpu_ram_joules
    - energy_cpu_total_joules
  中介变量: 8个
    - energy_gpu_avg_watts
    - energy_gpu_max_watts
    ... (其他能耗相关指标)

执行DiBS因果发现...
  alpha_linear: 0.05
  beta_linear: 0.1
  n_particles: 20
  n_steps: 5000

正在运行DiBS算法（这可能需要几分钟）...

✅ DiBS执行成功！耗时: 12.3分钟

因果图统计:
  最小值: 0.000001
  最大值: 0.892341
  平均值: 0.045123
  标准差: 0.123456

边数统计:
  >0.01: 234条
  >0.1:  56条
  >0.3:  12条 ⭐ 强边
  >0.5:  3条

提取研究问题1证据（超参数→能耗）...
  直接边（超参数→能耗）: 5条
  间接路径（超参数→中介→能耗）: 8条

提取研究问题2证据（能耗-性能权衡）...
  直接边（性能→能耗）: 2条
  直接边（能耗→性能）: 1条
  共同超参数: 3个
  中介权衡路径: 4条

提取研究问题3证据（中介效应）...
  超参数→中介→能耗: 8条
  超参数→中介→性能: 3条
  多步路径: 2条

✅ 因果图矩阵已保存: .../group1_examples_causal_graph.npy
✅ 分析结果已保存: .../group1_examples_result.json

... (重复6次，每组约10-15分钟)

================================================================================
生成总结报告...
================================================================================

✅ 总结报告已保存: .../NEW_6GROUPS_DIBS_ANALYSIS_REPORT.md

================================================================================
✅ DiBS分析完成！
================================================================================
  成功任务组: 6/6
  结果目录: .../results/energy_research/new_6groups_dibs/20260116_XXXXXX
  总结报告: .../NEW_6GROUPS_DIBS_ANALYSIS_REPORT.md
================================================================================

结束时间: 2026-01-16 XX:XX:XX
```

### 步骤3: 检查结果（预计15分钟）

**输出文件结构**:
```
results/energy_research/new_6groups_dibs/20260116_XXXXXX/
├── NEW_6GROUPS_DIBS_ANALYSIS_REPORT.md    # 总结报告 ⭐
├── group1_examples_causal_graph.npy       # 因果图矩阵
├── group1_examples_feature_names.json     # 特征名称
├── group1_examples_result.json            # 详细结果（JSON）
├── group2_vulberta_causal_graph.npy
├── group2_vulberta_feature_names.json
├── group2_vulberta_result.json
... (每组3个文件)
```

**检查清单**:
- [ ] 所有6组都成功执行（无错误）
- [ ] 每组都有因果图矩阵文件（.npy）
- [ ] 总结报告已生成
- [ ] 报告中包含3个研究问题的证据

### 步骤4: 可视化因果图（可选，预计30分钟）

如果想要可视化因果图：

```python
from analysis.utils.causal_discovery import visualize_causal_graph
import numpy as np
import json

# 读取因果图和特征名称
graph = np.load('results/.../group1_examples_causal_graph.npy')
with open('results/.../group1_examples_feature_names.json') as f:
    feature_names = json.load(f)

# 可视化（只显示强边）
visualize_causal_graph(
    graph,
    feature_names,
    output_path='group1_causal_graph.png',
    threshold=0.3,    # 只显示强边
    top_k=20          # 只显示top 20边
)
```

---

## 预期结果

### 研究问题1: 超参数对能耗的影响

**预期发现**:
- **直接效应**: 10-30条超参数→能耗的直接因果边
  - 例如: `hyperparam_batch_size → energy_cpu_total_joules`
  - 例如: `hyperparam_learning_rate → energy_gpu_total_joules`

- **间接效应**: 20-40条超参数→中介→能耗的路径
  - 例如: `hyperparam_batch_size → energy_gpu_util_avg_percent → energy_gpu_total_joules`

**关键超参数** (预计影响最大):
1. `batch_size` - 影响内存和计算负载
2. `learning_rate` - 影响收敛速度
3. `epochs` - 直接决定训练时长

### 研究问题2: 能耗-性能权衡关系

**预期发现**:
- **权衡类型1**: 共同超参数（同时影响能耗和性能）
  - 例如: `batch_size` ↑ → `energy` ↑, `accuracy` ↑

- **权衡类型2**: 直接因果边（性能↔能耗）
  - 可能发现少量 `perf_accuracy → energy_*` 边
  - 或者 `energy_* → perf_accuracy` 边

- **权衡类型3**: 通过中介变量的间接权衡
  - 例如: `perf_accuracy → energy_gpu_util → energy_total`

### 研究问题3: 中介效应路径

**预期发现**:
- **GPU利用率中介**: 20-30条路径
  - `hyperparam_* → energy_gpu_util_* → energy_gpu_*`

- **GPU温度中介**: 10-20条路径
  - `hyperparam_* → energy_gpu_temp_* → energy_gpu_*`

- **功率中介**: 10-20条路径
  - `hyperparam_* → energy_*_watts → energy_*_joules`

**中介类型**:
- **完全中介**: 直接效应≈0，间接效应显著
- **部分中介**: 直接和间接效应都显著

---

## 注意事项

### 1. 环境问题 ⚠️⚠️⚠️

**最常见错误**: 在base环境运行DiBS

```bash
# ❌ 错误 - 会报错 "ModuleNotFoundError: No module named 'dibs'"
python scripts/run_dibs_on_new_6groups.py

# ✅ 正确
conda activate causal-research
python scripts/run_dibs_on_new_6groups.py
```

### 2. 缺失值问题 ⚠️⚠️

**DiBS对缺失值零容忍**！必须提前处理。

**推荐策略**:
- 能耗指标: 使用列均值填充
- 超参数: 不应该有缺失（如果有，检查数据生成脚本）
- 性能指标: 使用列均值填充或删除行

**Group 5特殊处理**:
- 有12行性能数据缺失
- 建议: 删除这12行或单独分析Group 5

### 3. 常量特征问题 ⚠️

**症状**: DiBS崩溃并报错 "singular matrix"

**原因**: 某列的所有值相同（标准差=0）

**解决**: `run_dibs_on_new_6groups.py` 已包含自动检测和移除逻辑 (lines 123-136)

### 4. 内存问题

**症状**: `Out of Memory` 错误

**原因**:
- n_particles太大
- n_steps太大
- 数据集太大

**解决**:
- 减少n_particles (20 → 10)
- 减少n_steps (5000 → 3000)
- 分批执行（一次只运行1-2组）

### 5. 收敛问题

**症状**: DiBS运行完成，但发现的边很少（<5条）

**原因**:
- alpha太大（图太稀疏）
- beta太大（无环约束太强）
- n_steps太少（未收敛）

**解决**:
- 降低alpha (0.05 → 0.01)
- 降低beta (0.1 → 0.05)
- 增加n_steps (5000 → 10000)

### 6. 执行时间

**预期耗时**:
- 小数据集 (72-90行): 5-8分钟/组
- 中等数据集 (206行): 10-12分钟/组
- 大数据集 (304行): 15-20分钟/组

**总耗时**: 约1-1.5小时（6组串行执行）

**加速方案** (可选):
- 使用GPU加速（如果DiBS支持CUDA）
- 并行执行6组（需要修改脚本）
- 减少n_particles或n_steps

---

## 快速执行指南（TL;DR）

### 一键执行命令

```bash
# 1. 进入分析目录
cd /home/green/energy_dl/nightly/analysis

# 2. 激活conda环境
conda activate causal-research

# 3. 修改脚本（如上述步骤1所述）
nano scripts/run_dibs_on_new_6groups.py

# 4. 执行DiBS分析
python scripts/run_dibs_on_new_6groups.py

# 5. 查看结果
ls -lh results/energy_research/new_6groups_dibs/
cat results/energy_research/new_6groups_dibs/*/NEW_6GROUPS_DIBS_ANALYSIS_REPORT.md
```

### 故障排查快速指南

| 问题 | 快速解决 |
|------|---------|
| `ModuleNotFoundError: No module named 'dibs'` | 检查是否激活了`causal-research`环境 |
| `ValueError: data contains NaN` | 添加缺失值填充代码 (见步骤1修改3) |
| `LinAlgError: singular matrix` | 检查并移除常量特征 (脚本已包含) |
| `OutOfMemoryError` | 减少`n_particles`或`n_steps` |
| 执行时间过长 (>30分钟/组) | 正常，耐心等待；或减少`n_steps` |
| 发现的边很少 (<5条) | 降低`alpha`和`beta` |

---

## 后续步骤

DiBS分析完成后：

1. **验证因果边** (1-2天)
   - 使用线性回归验证DiBS发现的因果边
   - 计算回归系数和p值
   - 与DiBS边强度对比

2. **中介效应分析** (1-2天)
   - 对中介路径进行Sobel检验
   - 计算间接效应大小
   - 区分完全中介和部分中介

3. **撰写研究报告** (2-3天)
   - 整合DiBS结果和回归验证
   - 回答3个研究问题
   - 生成图表和可视化

4. **论文撰写** (1-2周)
   - Method章节: 描述DiBS方法
   - Results章节: 展示因果发现结果
   - Discussion章节: 解释研究发现

---

**文档作者**: Claude
**最后更新**: 2026-01-16
**参考文档**:
- [6GROUPS_DATA_VALIDATION_REPORT_20260115.md](6GROUPS_DATA_VALIDATION_REPORT_20260115.md)
- [DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md](/home/green/energy_dl/nightly/analysis/docs/reports/DIBS_PARAMETER_TUNING_SUCCESS_REPORT_20260105.md)
- [QUESTIONS_2_3_DIBS_COMPLETE_REPORT_20260105.md](/home/green/energy_dl/nightly/analysis/docs/reports/QUESTIONS_2_3_DIBS_COMPLETE_REPORT_20260105.md)
