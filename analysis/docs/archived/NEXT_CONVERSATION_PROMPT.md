# 下一个对话：DiBS因果分析 - 已准备好运行

## 🎯 状态概要
已完成：数据全局标准化 + ATE重新计算（6组）
待执行：运行DiBS因果发现（6分组，全局标准化数据）

### ✅ 已完成
| 任务 | 关键输出 |
|------|----------|
| 全局标准化数据 | `data/energy_research/6groups_global_std/` |
| DiBS就绪数据 | `data/energy_research/6groups_dibs_ready/` |
| 全局标准化ATE | `results/energy_research/data/global_std_ate/` (6组) |
| 参数对比 | `docs/DIBS_PARAMETER_COMPARISON.md` (与论文一致) |
| 数据验证 | ✅ subagent验证通过 |

### 🚀 当前任务：运行DiBS
- **状态**: 准备就绪，容忍长时间运行
- **数据**: 6组全局标准化数据（49特征，818总样本）
- **配置**: 优化参数（基于论文对比）
- **预期时间**: 每组2-4小时（CPU运行，共12-24小时）

## 📁 关键文件
### 数据
- `data/energy_research/6groups_dibs_ready/` (DiBS就绪数据)
- `data/energy_research/6groups_global_std/` (原始标准化数据)

### 结果
- `results/energy_research/data/global_std_ate/` (6组ATE，已完成)
- `results/energy_research/data/sensitivity_analysis/` (敏感性)

### 脚本
- `scripts/run_dibs_6groups_global_std.py` (DiBS运行脚本)
- `scripts/compute_ate_global_std.py` (ATE计算脚本)

## ⚙️ DiBS参数（与论文对比）
| 参数 | 当前值 | 论文对比 | 说明 |
|------|--------|----------|------|
| alpha_linear | 0.05 | ✅ 一致 | 论文默认值 |
| beta_linear | 0.1 | ⚠️ 调整 | 降低约束，基于实验优化 |
| n_particles | 20 | ✅ 合适 | 中等规模（49特征） |
| n_steps | 5000 | ✅ 充足 | 论文建议3000-10000步 |

详细对比见: `docs/DIBS_PARAMETER_COMPARISON.md`

## 🚀 执行命令

### 1. 激活环境
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate causal-research
```

### 2. 运行DiBS（串行推荐）
```bash
for group in group1_examples group2_vulberta group3_person_reid group4_bug_localization group5_mrt_oast group6_resnet; do
    echo "运行 $group DiBS (5000步)..."
    python scripts/run_dibs_6groups_global_std.py --group $group --n-steps 5000 --verbose
done
```

### 3. 验证结果
```bash
# 检查DiBS结果
python -c "
import glob, json
for f in glob.glob('results/energy_research/data/global_std/*/dibs_summary.json'):
    data = json.load(open(f))
    group = f.split('/')[-2]
    print(f'{group}: {data[\"strong_edge_percentage\"]:.1f}%强边 ({data[\"strong_edge_count\"]}条)')
"
```

### 4. 提取白名单边（可选）
```bash
python scripts/extract_dibs_edges_to_csv.py --source global_std
```

## ⚠️ 注意事项
1. **时间**: 每组2-4小时（CPU），共12-24小时
2. **内存**: 需要>8GB内存
3. **监控**: 脚本每1000步输出进度
4. **恢复**: 中断后可从上次完成的组继续
5. **质量**: 期望强边比例10-30%

## 🎯 研究目标
- 发现跨6组模型一致的因果模式
- 支持ATE跨组可比性（全局标准化已解决）
- 分析seed对能耗鲁棒性的影响

**开始**: 运行DiBS脚本（串行或并行）
**完成**: 验证结果，提取新白名单（可选）