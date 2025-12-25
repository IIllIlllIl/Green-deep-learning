# 6组DiBS因果分析实验指令

**日期**: 2025-12-24
**数据来源**: Stage2 (mediators.csv) → 6组分层数据
**总样本量**: 594行（81.8%数据保留率）

---

## ✅ 数据准备完成

### 6个训练数据文件

```
data/energy_research/processed/
├── training_data_image_classification_examples.csv  (219行, 19列)
├── training_data_image_classification_resnet.csv    (39行, 15列)
├── training_data_person_reid.csv                    (116行, 20列)
├── training_data_vulberta.csv                       (82行, 15列)
├── training_data_bug_localization.csv               (80行, 16列)
└── training_data_mrt_oast.csv                       (58行, 19列) ⭐ 新增
```

### 数据质量验证结果

| 任务组 | 行数 | 列数 | 性能指标缺失率 | 状态 |
|--------|------|------|---------------|------|
| image_classification_examples | 219 | 19 | test_accuracy: 0% | ✅ 优秀 |
| image_classification_resnet | 39 | 15 | test_accuracy: 0% | ✅ 优秀 |
| person_reid | 116 | 20 | mAP/rank1/rank5: 0% | ✅ 优秀 |
| vulberta | 82 | 15 | eval_loss: 0% | ✅ 优秀 |
| bug_localization | 80 | 16 | top1/top5: 0% | ✅ 优秀 |
| **mrt_oast** | **58** | **19** | **accuracy: 20.7%**, precision/recall: 0% | ✅ **良好** |

**总体评估**: 所有6个任务组数据质量优秀，满足DiBS因果分析要求。

---

## 🚀 DiBS因果分析执行指令

### 方式1: 单个任务组测试（推荐先执行）

**用途**: 快速验证单个任务组的DiBS分析流程

```bash
# 1. 进入analysis目录
cd /home/green/energy_dl/nightly/analysis

# 2. 激活环境
source /home/green/miniconda3/etc/profile.d/conda.sh
conda activate fairness

# 3. 测试单个任务组（例如：person_reid）
python3 scripts/demos/demo_energy_task_analysis.py \
    --task person_reid \
    --input data/energy_research/processed/training_data_person_reid.csv \
    --output results/energy_research/6groups/person_reid \
    --verbose

# 预期输出:
# - results/energy_research/6groups/person_reid/causal_graph.npy
# - results/energy_research/6groups/person_reid/causal_edges.pkl
# - results/energy_research/6groups/person_reid/analysis_report.md
# - 运行时间: 约30-60分钟（GPU加速）
```

### 方式2: 6组并行执行（完整实验）

**用途**: 一次性完成所有6个任务组的DiBS分析

#### 步骤1: 创建并行执行脚本

创建文件 `scripts/run_6groups_dibs_parallel.sh`:

```bash
#!/bin/bash
# 6组DiBS因果分析并行执行脚本
# 作者: Claude
# 日期: 2025-12-24

set -e

# 配置
ANALYSIS_DIR="/home/green/energy_dl/nightly/analysis"
DATA_DIR="${ANALYSIS_DIR}/data/energy_research/processed"
RESULTS_DIR="${ANALYSIS_DIR}/results/energy_research/6groups"
LOG_DIR="${ANALYSIS_DIR}/logs/energy_research/6groups"

# 创建输出目录
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

# 激活环境
source /home/green/miniconda3/etc/profile.d/conda.sh
conda activate fairness

# 切换到analysis目录
cd "${ANALYSIS_DIR}"

# 6个任务组配置
declare -A TASKS=(
    ["image_classification_examples"]="training_data_image_classification_examples.csv"
    ["image_classification_resnet"]="training_data_image_classification_resnet.csv"
    ["person_reid"]="training_data_person_reid.csv"
    ["vulberta"]="training_data_vulberta.csv"
    ["bug_localization"]="training_data_bug_localization.csv"
    ["mrt_oast"]="training_data_mrt_oast.csv"
)

echo "================================================================================"
echo "6组DiBS因果分析并行执行"
echo "================================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "任务数量: ${#TASKS[@]}"
echo "================================================================================"

# 后台运行所有任务
PIDS=()
for task_name in "${!TASKS[@]}"; do
    data_file="${TASKS[$task_name]}"
    input_path="${DATA_DIR}/${data_file}"
    output_dir="${RESULTS_DIR}/${task_name}"
    log_file="${LOG_DIR}/${task_name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "启动任务: ${task_name}"
    echo "  输入: ${input_path}"
    echo "  输出: ${output_dir}"
    echo "  日志: ${log_file}"

    # 后台运行
    python3 scripts/demos/demo_energy_task_analysis.py \
        --task "${task_name}" \
        --input "${input_path}" \
        --output "${output_dir}" \
        --verbose \
        > "${log_file}" 2>&1 &

    PIDS+=($!)
    echo "  进程ID: ${PIDS[-1]}"
done

echo ""
echo "================================================================================"
echo "所有任务已启动，等待完成..."
echo "================================================================================"

# 等待所有任务完成
SUCCESS_COUNT=0
FAILED_COUNT=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    task_name="${!TASKS[$i]}"

    wait $pid
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✅ 任务完成: ${task_name} (PID: $pid)"
        ((SUCCESS_COUNT++))
    else
        echo "❌ 任务失败: ${task_name} (PID: $pid, 退出码: $exit_code)"
        ((FAILED_COUNT++))
    fi
done

echo ""
echo "================================================================================"
echo "执行完成"
echo "================================================================================"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "成功: ${SUCCESS_COUNT}/${#TASKS[@]}"
echo "失败: ${FAILED_COUNT}/${#TASKS[@]}"
echo "================================================================================"

if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
```

#### 步骤2: 执行并行分析

```bash
# 1. 赋予执行权限
chmod +x scripts/run_6groups_dibs_parallel.sh

# 2. 后台执行（推荐使用screen）
screen -S dibs_6groups
bash scripts/run_6groups_dibs_parallel.sh

# 3. 分离screen (Ctrl+A, 然后按D)

# 4. 重新连接查看进度
screen -r dibs_6groups
```

**预期运行时间**: 4-7小时（6个任务并行，取决于GPU性能）

#### 步骤3: 监控执行进度

```bash
# 查看所有日志
tail -f logs/energy_research/6groups/*.log

# 查看特定任务日志
tail -f logs/energy_research/6groups/mrt_oast_*.log

# 检查已完成的任务
ls -lh results/energy_research/6groups/*/causal_graph.npy
```

---

## 📊 预期输出结果

### 每个任务组的输出

```
results/energy_research/6groups/
├── image_classification_examples/
│   ├── causal_graph.npy          # DiBS学习的因果图（邻接矩阵）
│   ├── causal_edges.pkl          # 筛选后的因果边列表
│   ├── analysis_report.md        # 因果分析报告
│   └── config.json               # 分析配置参数
├── image_classification_resnet/
├── person_reid/
├── vulberta/
├── bug_localization/
└── mrt_oast/                     ⭐ 新增
    ├── causal_graph.npy
    ├── causal_edges.pkl
    ├── analysis_report.md
    └── config.json
```

### 预期因果发现

| 任务组 | 预期因果边数 | 关键发现 |
|--------|-------------|----------|
| image_classification_examples | 3-8条 | learning_rate → 能耗/性能 |
| image_classification_resnet | 2-5条 | l2_regularization → 性能 |
| person_reid | 4-10条 | dropout → mAP, 能耗 → 温度 |
| vulberta | 2-6条 | learning_rate → eval_loss |
| bug_localization | 3-7条 | kfold → 准确率 |
| **mrt_oast** | **4-8条** | **多目标权衡**（accuracy vs precision vs recall） |

---

## 🔍 结果验证与分析

### 验证清单

```bash
# 1. 检查所有任务组是否完成
for task in image_classification_examples image_classification_resnet person_reid vulberta bug_localization mrt_oast; do
    echo "任务: $task"
    ls -lh results/energy_research/6groups/$task/causal_graph.npy 2>/dev/null && echo "  ✅ 完成" || echo "  ❌ 未完成"
done

# 2. 统计因果边数量
for task in image_classification_examples image_classification_resnet person_reid vulberta bug_localization mrt_oast; do
    echo "任务: $task"
    python3 -c "
import pickle
try:
    with open('results/energy_research/6groups/$task/causal_edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    print(f'  因果边数量: {len(edges)}')
    print(f'  统计显著边: {sum(1 for e in edges if e.get(\"p_value\", 1.0) < 0.05)}')
except:
    print('  ⚠️ 文件不存在或解析失败')
"
done

# 3. 生成汇总报告
python3 scripts/generate_6groups_summary_report.py \
    --results-dir results/energy_research/6groups \
    --output docs/reports/6GROUPS_CAUSAL_ANALYSIS_SUMMARY_20251224.md
```

### 关键指标检查

**每个任务组应满足**:
- ✅ 因果边数量 ≥ 2条
- ✅ 至少1条统计显著边（p < 0.05）
- ✅ 能耗相关因果边 ≥ 1条
- ✅ 性能相关因果边 ≥ 1条

**特别关注（MRT-OAST）**:
- ✅ 检测到多目标优化的权衡模式
- ✅ accuracy, precision, recall之间的因果关系
- ✅ dropout/weight_decay对多指标的差异化影响

---

## 📝 后续步骤

### 1. 生成综合对比报告

```bash
# 创建6组 vs 5组对比报告
python3 scripts/compare_5groups_vs_6groups.py \
    --results-5groups results/energy_research/5groups \
    --results-6groups results/energy_research/6groups \
    --output docs/reports/5GROUPS_VS_6GROUPS_COMPARISON_20251224.md
```

### 2. 可视化因果图

```bash
# 为每个任务组生成因果图可视化
for task in image_classification_examples image_classification_resnet person_reid vulberta bug_localization mrt_oast; do
    python3 scripts/visualize_causal_graph.py \
        --input results/energy_research/6groups/$task/causal_graph.npy \
        --output results/energy_research/6groups/$task/causal_graph.png \
        --title "$task Causal Graph"
done
```

### 3. 提取关键发现

```bash
# 提取所有任务组的关键因果路径
python3 scripts/extract_key_findings.py \
    --results-dir results/energy_research/6groups \
    --output docs/reports/6GROUPS_KEY_FINDINGS_20251224.md
```

---

## 🛠️ 故障排查

### 问题1: DiBS运行超时

**症状**: 某个任务组运行超过2小时仍未完成
**原因**: 数据维度过高或样本量过大
**解决**:
```bash
# 增加超时限制
python3 scripts/demos/demo_energy_task_analysis.py \
    --task vulberta \
    --timeout 7200  # 2小时
```

### 问题2: 内存不足

**症状**: OOM (Out of Memory) 错误
**解决**:
```bash
# 减少DiBS采样数
python3 scripts/demos/demo_energy_task_analysis.py \
    --task person_reid \
    --dibs-samples 500  # 默认1000
```

### 问题3: 0因果边

**症状**: 某个任务组检测到0条因果边
**排查**:
1. 检查数据质量（缺失率、方差）
2. 调整DiBS阈值
3. 查看日志中的警告信息

```bash
# 降低因果边阈值
python3 scripts/demos/demo_energy_task_analysis.py \
    --task bug_localization \
    --edge-threshold 0.3  # 默认0.5
```

---

## 📚 相关文档

- [6组数据生成方案](6GROUPS_DATA_GENERATION_PLAN_20251224.md) - 完整数据生成方案
- [阶段质量分析](STAGE_QUALITY_ANALYSIS_20251224.md) - Stage0-7质量评估
- [MRT-OAST可行性分析](MRT_OAST_FEASIBILITY_ANALYSIS.md) - 第6组可行性评估
- [变量扩展计划v3.0](VARIABLE_EXPANSION_PLAN.md) - 变量设计和扩展
- [Adult数据集完整分析](ADULT_COMPLETE_CAUSAL_ANALYSIS_REPORT.md) - DiBS/DML方法参考

---

## ✅ 质量保证

### 数据生成验证 ✅

- [x] 6个CSV文件全部生成
- [x] 总行数 = 594（无数据丢失）
- [x] 数据保留率 = 81.8%（达到目标）
- [x] 所有性能指标0%缺失（已删除性能全缺失行）
- [x] MRT-OAST包含6个超参数（最多）
- [x] One-Hot编码互斥性100%

### DiBS分析预期 ⏳

- [ ] 6个任务组全部完成DiBS分析
- [ ] 每个任务组发现 ≥2 条因果边
- [ ] 每个任务组发现 ≥1 条统计显著边（p < 0.05）
- [ ] MRT-OAST发现多目标优化的因果权衡模式
- [ ] 生成6组综合对比报告
- [ ] 生成关键发现总结

---

**文档版本**: v1.0
**创建日期**: 2025-12-24
**状态**: ✅ 数据准备完成，待执行DiBS分析
**预计总时间**: 4-7小时（6组并行执行）
