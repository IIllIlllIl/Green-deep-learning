# Analysis结果目录说明

**最后更新**: 2025-12-22
**目录结构版本**: v2.0

---

## 📁 目录结构

```
results/
├── paper_replication/          # 论文复现结果（ASE 2023）
│   ├── adult_causal_graph.npy          # Adult因果图邻接矩阵
│   ├── adult_causal_edges.pkl          # Adult筛选后的因果边
│   ├── adult_data_checkpoint.pkl       # Adult训练检查点（36MB）
│   ├── causal_graph.npy                # 演示因果图
│   └── large_scale_causal_graph.npy    # 大规模测试因果图
│
└── energy_research/            # 能耗研究结果（主项目）
    ├── experiment_001/         # 按实验编号组织
    ├── experiment_002/
    └── ...
```

---

## 📊 结果文件说明

### 1. paper_replication/ - 论文复现结果

**用途**: 存储ASE 2023论文复现的因果分析结果

**文件类型**:

| 文件名 | 类型 | 大小 | 说明 |
|--------|------|------|------|
| `*_causal_graph.npy` | NumPy数组 | ~1.6KB | DiBS学习的因果图邻接矩阵 (P×P) |
| `*_causal_edges.pkl` | Pickle | ~294B | 筛选后的因果边列表（置信度 > 0.7）|
| `*_data_checkpoint.pkl` | Pickle | ~36MB | DiBS训练检查点（包含中间状态）|

**Adult数据集结果概览**:
```python
# 因果图维度
adult_causal_graph.npy: (10, 10)  # 10个变量的DAG

# 筛选后的因果边
adult_causal_edges.pkl:
[
    ('Tr_F1', 'Te_Acc', -0.052),   # 训练F1 → 测试准确率（负效应）
    ('Tr_Fair', 'Te_Fair', 0.234),  # 训练公平性 → 测试公平性（正效应）
    ...
]
```

**查看方法**:
```python
import numpy as np
import pickle

# 读取因果图
G = np.load('results/paper_replication/adult_causal_graph.npy')
print(f"因果图维度: {G.shape}")
print(f"因果边数量: {(G > 0.7).sum()}")

# 读取因果边
with open('results/paper_replication/adult_causal_edges.pkl', 'rb') as f:
    edges = pickle.load(f)
print(f"高置信度因果边: {len(edges)}条")
```

**相关文档**:
- [Adult完整分析报告](../docs/reports/ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md)
- [复现评估报告](../docs/reports/REPLICATION_EVALUATION.md)

---

### 2. energy_research/ - 能耗研究结果

**用途**: 存储主项目能耗研究的因果分析结果

**组织方式**: 按实验编号创建独立子目录

#### 实验目录结构

```
energy_research/
└── experiment_001_lr_bs_energy/
    ├── README.md                    # 实验说明
    ├── config.json                  # 实验配置
    ├── causal_graph.npy            # 因果图
    ├── causal_edges.pkl            # 因果边
    ├── ate_results.csv             # 平均因果效应
    ├── tradeoff_analysis.json      # 权衡检测结果
    ├── visualizations/             # 可视化图表
    │   ├── causal_graph.png
    │   ├── ate_barplot.png
    │   └── tradeoff_heatmap.png
    └── logs/
        ├── dibs_training.log       # DiBS训练日志
        └── dml_inference.log       # DML推断日志
```

#### 标准输出文件

**1. causal_graph.npy** - 因果图邻接矩阵

```python
# 格式: (P, P) NumPy数组
# 元素: G[i,j] = 边i→j的后验概率 (0~1)
# 解读: G[i,j] > 0.7 表示高置信度因果边
```

**2. causal_edges.pkl** - 筛选后的因果边

```python
# 格式: List[Tuple[str, str, float, float, float]]
# 示例:
[
    ('learning_rate', 'energy_gpu', 0.85, 0.023, 0.001),
    # (cause, effect, edge_prob, ATE, p_value)
]
```

**3. ate_results.csv** - 平均因果效应

```csv
cause,effect,ATE,std_error,p_value,ci_lower,ci_upper,significant
learning_rate,energy_gpu,15.3,2.1,0.001,11.2,19.4,True
batch_size,energy_cpu,5.2,1.8,0.042,1.7,8.7,True
```

**4. tradeoff_analysis.json** - 权衡检测结果

```json
{
  "tradeoffs": [
    {
      "common_cause": "learning_rate",
      "outcome1": "energy_gpu",
      "outcome2": "test_acc",
      "ate1": 15.3,
      "ate2": -0.05,
      "direction": "opposite",
      "strength": "moderate"
    }
  ],
  "summary": {
    "total_tradeoffs": 2,
    "strong_tradeoffs": 1
  }
}
```

---

## 🔄 结果版本管理

### 版本命名规范

```
experiment_{编号}_{简短描述}/
  ├── v1_20251222/         # 第一次运行（日期）
  ├── v2_20251223/         # 第二次运行（修正超参数）
  └── latest/              # 符号链接到最新版本
```

### 实验元数据

每个实验目录必须包含 `metadata.json`:

```json
{
  "experiment_id": "001",
  "experiment_name": "Learning Rate and Batch Size Effect on GPU Energy",
  "created_date": "2025-12-22",
  "updated_date": "2025-12-22",
  "data_source": "data/energy_research/raw/energy_data_v1.csv",
  "num_samples": 616,
  "variables": {
    "inputs": ["learning_rate", "batch_size", "epochs"],
    "outputs": ["energy_gpu_avg", "test_acc", "test_f1"]
  },
  "dibs_config": {
    "n_particles": 20,
    "n_steps": 10000,
    "edge_threshold": 0.7
  },
  "dml_config": {
    "model": "LinearDML",
    "cv_folds": 5
  },
  "status": "completed",
  "runtime_minutes": 61.4
}
```

---

## 📈 可视化规范

### 因果图可视化 (causal_graph.png)

- **格式**: PNG（300 DPI）
- **布局**: 层次化布局（输入变量在上，输出变量在下）
- **节点颜色**: 输入变量（蓝色），输出变量（橙色）
- **边颜色**: 正效应（绿色），负效应（红色）
- **边宽度**: 与因果效应绝对值成正比

### ATE柱状图 (ate_barplot.png)

- **格式**: PNG（300 DPI）
- **X轴**: 因果边（"cause → effect"）
- **Y轴**: 平均因果效应（ATE）
- **颜色**: 显著（p < 0.05，深色），不显著（浅色）
- **误差棒**: 95%置信区间

### 权衡热力图 (tradeoff_heatmap.png)

- **格式**: PNG（300 DPI）
- **矩阵**: 输出变量 × 输出变量
- **颜色**: 红色（权衡强），蓝色（协同），白色（无关）

---

## 🔍 结果查询

### 查看所有实验

```bash
cd results/energy_research/
ls -d experiment_*/
```

### 查看特定实验结果

```bash
cd results/energy_research/experiment_001/
cat README.md              # 实验说明
cat metadata.json          # 元数据
python3 -c "import pickle; edges = pickle.load(open('causal_edges.pkl', 'rb')); print(edges)"
```

### 对比多个实验

```python
import pandas as pd
import glob

# 读取所有ATE结果
ate_files = glob.glob('results/energy_research/*/ate_results.csv')
all_ates = pd.concat([pd.read_csv(f) for f in ate_files])

# 对比不同实验的因果效应
print(all_ates.groupby(['cause', 'effect'])['ATE'].describe())
```

---

## ⚠️ 注意事项

1. **大文件管理**:
   - 检查点文件（*.pkl）可能很大（>10MB）
   - 考虑定期清理不需要的检查点
   - 使用 `.gitignore` 排除大文件

2. **结果备份**:
   - 重要实验结果应及时备份
   - 使用版本控制管理元数据和配置文件

3. **结果复现**:
   - 每个实验必须包含完整的配置文件
   - 记录随机种子（如果使用）

4. **数据隐私**:
   - 确保结果文件不包含敏感信息
   - 公开结果前检查数据脱敏

---

## 🧹 清理策略

### 清理检查点文件

```bash
# 删除所有检查点文件（保留因果图和因果边）
find results/ -name "*_checkpoint.pkl" -delete
```

### 清理临时日志

```bash
# 删除30天前的日志文件
find results/*/logs/ -name "*.log" -mtime +30 -delete
```

### 归档旧实验

```bash
# 将旧实验移至归档目录
mkdir -p results/archived/
mv results/energy_research/experiment_001/ results/archived/
```

---

## 📚 相关文档

- [数据目录说明](../data/README.md) - 数据组织方式
- [迁移指南](../docs/MIGRATION_GUIDE.md) - 如何应用到新数据集
- [文档总索引](../docs/INDEX.md) - 所有文档的索引

---

**维护者**: Analysis模块维护团队
**联系方式**: 查看项目根目录CLAUDE.md
