# Stage 8: DiBS + DML因果分析 - Screen运行指南

**生成时间**: 2025-12-23
**状态**: 准备就绪
**预计时间**: 60-120分钟

---

## 🚀 快速开始（Screen后台运行）

### 方法1: 一键启动（推荐）

```bash
cd /home/green/energy_dl/nightly/analysis

# 启动screen会话并运行分析
screen -S energy_dibs -L -Logfile logs/energy_research/experiments/screen.log \
  bash scripts/experiments/run_energy_causal_analysis.sh

# 分离screen: 按 Ctrl+A 然后 D
```

### 方法2: 分步操作

```bash
# 1. 进入analysis目录
cd /home/green/energy_dl/nightly/analysis

# 2. 启动screen会话
screen -S energy_dibs

# 3. 运行分析脚本
bash scripts/experiments/run_energy_causal_analysis.sh

# 4. 分离screen
# 按 Ctrl+A，然后按 D
```

---

## 📊 监控进度

### 实时查看日志

```bash
# 查看最新日志
tail -f logs/energy_research/experiments/energy_causal_analysis_*.log

# 或查看进度文件
watch -n 10 cat logs/energy_research/experiments/dibs_progress.txt
```

### 检查状态

```bash
# 查看分析状态
cat logs/energy_research/experiments/analysis_status.txt

# 可能的状态:
# - RUNNING: 正在运行
# - SUCCESS: 成功完成
# - FAILED:X: 失败（退出码X）
```

### 重新连接Screen会话

```bash
# 列出所有screen会话
screen -ls

# 重新连接到energy_dibs会话
screen -r energy_dibs

# 如果会话已附加到其他终端，强制连接
screen -d -r energy_dibs
```

---

## 📈 预期输出

### 4个任务组（按优先级）

1. **图像分类** (258样本, 13特征) - 优先级1
   - 预计时间: 20-30分钟

2. **Person_reID** (116样本, 16特征) - 优先级2
   - 预计时间: 15-25分钟

3. **VulBERTa** (142样本, 10特征) - 优先级3
   - 预计时间: 10-20分钟

4. **Bug定位** (132样本, 11特征) - 优先级4
   - 预计时间: 10-20分钟

**总计**: 60-120分钟（取决于变量数和DiBS收敛速度）

### 生成文件

每个任务组生成3个文件：

```
results/energy_research/task_specific/
├── image_classification_causal_graph.npy      # 因果图邻接矩阵
├── image_classification_causal_edges.pkl      # 因果边列表
├── image_classification_causal_effects.csv    # DML因果效应
├── person_reid_causal_graph.npy
├── person_reid_causal_edges.pkl
├── person_reid_causal_effects.csv
├── vulberta_causal_graph.npy
├── vulberta_causal_edges.pkl
├── vulberta_causal_effects.csv
├── bug_localization_causal_graph.npy
├── bug_localization_causal_edges.pkl
├── bug_localization_causal_effects.csv
└── analysis_summary.txt                       # 总体摘要
```

---

## 🔍 查看结果

### 快速查看摘要

```bash
cat results/energy_research/task_specific/analysis_summary.txt
```

### 查看因果图

```bash
python3 -c "
import numpy as np
g = np.load('results/energy_research/task_specific/image_classification_causal_graph.npy')
print(f'因果图形状: {g.shape}')
print(f'非零边数: {(g > 0.3).sum()}')
"
```

### 查看因果效应

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('results/energy_research/task_specific/image_classification_causal_effects.csv')
print(df.head(10))
print(f'\n统计显著的因果效应: {(df[\"p_value\"] < 0.05).sum()}')
"
```

---

## ⚠️ 常见问题

### 1. Screen会话意外断开

**症状**: `screen -r` 显示 "There is no screen to be resumed"

**解决**:
```bash
# 检查分析是否仍在运行
ps aux | grep demo_energy_task_specific.py

# 查看最新日志
tail -50 logs/energy_research/experiments/energy_causal_analysis_*.log

# 查看状态
cat logs/energy_research/experiments/analysis_status.txt
```

### 2. 分析中途失败

**症状**: `analysis_status.txt` 显示 "FAILED:X"

**解决**:
```bash
# 查看错误日志
tail -100 logs/energy_research/experiments/energy_causal_analysis_*.log

# 检查哪个任务组失败
cat results/energy_research/task_specific/analysis_summary.txt

# 重新运行（会跳过已完成的任务组）
bash scripts/experiments/run_energy_causal_analysis.sh
```

### 3. DiBS运行时间过长

**正常情况**: DiBS学习需要15-30分钟/任务组，变量多或样本少时可能更久

**监控**:
```bash
# 查看当前任务进度
tail -20 logs/energy_research/experiments/energy_causal_analysis_*.log | grep "DiBS"
```

---

## 🎯 核心技术参数

### DiBS配置（与Adult分析保持一致）

- **迭代次数**: 3000步（优化版，Adult分析: 5000→3000，速度提升>97%）
- **Alpha**: 0.1（稀疏性惩罚）
- **阈值**: 0.3（因果边置信度）
- **随机种子**: 42（可复现）

### DML配置

- **显著性水平**: p < 0.05
- **置信区间**: 95%
- **方法**: Double Machine Learning（消除混淆偏差）

---

## 📝 完成后操作

### 1. 验证完整性

```bash
# 检查是否所有4个任务组都成功
ls -lh results/energy_research/task_specific/*.csv | wc -l
# 应该输出 4（每个任务组1个causal_effects.csv）

# 查看摘要
cat results/energy_research/task_specific/analysis_summary.txt
```

### 2. 更新文档

完成后需要更新以下文档（在下一个对话中）：

- [ ] `analysis/docs/reports/VARIABLE_EXPANSION_PLAN.md` - 更新Stage 8状态
- [ ] `analysis/docs/INDEX.md` - 添加2025-12-23里程碑
- [ ] 创建 `analysis/docs/STAGE8_EXECUTION_REPORT.md` - 详细执行报告

### 3. 备份结果

```bash
# 创建结果备份
cd /home/green/energy_dl/nightly/analysis
tar -czf results_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  results/energy_research/task_specific/ \
  logs/energy_research/experiments/
```

---

## 📞 下一步

完成后，在下一个对话中可以：

1. **讨论因果发现**: 分析每个任务组发现的因果边
2. **对比分析**: 跨任务组的共性和差异
3. **生成报告**: 创建详细的Stage 8执行报告
4. **规划权衡检测**: 基于因果效应识别"能耗 vs 性能"权衡

---

**文档版本**: v1.0
**生成者**: Claude Code
**基于**: Adult数据集成功经验（61.4分钟，6条因果边，4条显著）
