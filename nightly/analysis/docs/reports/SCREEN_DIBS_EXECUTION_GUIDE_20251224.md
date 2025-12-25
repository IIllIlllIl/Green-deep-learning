# Screen中运行DiBS因果分析指令

**日期**: 2025-12-24
**用途**: 在后台screen会话中运行DiBS因果分析
**预计时间**: 4-7小时（6组并行）

---

## 🎯 推荐方案：分步执行

为了确保稳定性和可监控性，建议按以下顺序执行：

### 方案A：单任务测试 → 6组并行（推荐） ⭐⭐⭐

**优势**：
- 先验证单个任务组流程正确
- 确认环境配置无误
- 预估实际运行时间
- 降低大规模并行风险

---

## 📋 执行步骤

### 步骤1: 单任务测试（MRT-OAST）

#### 1.1 创建screen会话

```bash
# 创建一个名为dibs_test的screen会话
screen -S dibs_test
```

**说明**：
- `-S dibs_test`：指定会话名称为"dibs_test"
- 进入screen后会看到新的shell提示符

#### 1.2 在screen中激活环境并运行测试

```bash
# 进入analysis目录
cd /home/green/energy_dl/nightly/analysis

# 激活conda环境
source /home/green/miniconda3/etc/profile.d/conda.sh
conda activate fairness

# 验证环境
echo "Python版本: $(python3 --version)"
echo "当前目录: $(pwd)"
echo "环境: $CONDA_DEFAULT_ENV"

# 运行MRT-OAST单任务测试（新增的第6组）
echo "=================================================="
echo "开始DiBS分析: MRT-OAST"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

python3 scripts/demos/demo_energy_task_analysis.py \
    --task mrt_oast \
    --input data/energy_research/processed/training_data_mrt_oast.csv \
    --output results/energy_research/6groups/mrt_oast \
    --verbose 2>&1 | tee logs/energy_research/6groups/mrt_oast_test_$(date +%Y%m%d_%H%M%S).log

echo "=================================================="
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
```

**预期输出**：
- 运行时间：30-60分钟
- 输出文件：causal_graph.npy, causal_edges.pkl, analysis_report.md
- 预期因果边：4-8条

#### 1.3 分离screen会话

```
方法1: 按键组合
按 Ctrl+A，然后按 D

方法2: 命令方式
Ctrl+A，然后输入 :detach
```

**说明**：
- 分离后DiBS继续在后台运行
- 你可以安全关闭SSH连接
- screen会话保持运行状态

#### 1.4 重新连接查看进度

```bash
# 列出所有screen会话
screen -ls

# 重新连接到dibs_test会话
screen -r dibs_test

# 如果有多个会话，使用完整ID
screen -r 12345.dibs_test  # 12345是进程ID
```

#### 1.5 监控测试结果

```bash
# 查看实时日志（在另一个终端）
tail -f logs/energy_research/6groups/mrt_oast_test_*.log

# 检查输出文件
ls -lh results/energy_research/6groups/mrt_oast/

# 验证因果图生成
python3 -c "
import numpy as np
import pickle

try:
    # 加载因果图
    graph = np.load('results/energy_research/6groups/mrt_oast/causal_graph.npy')
    print(f'✅ 因果图加载成功: {graph.shape}')

    # 加载因果边
    with open('results/energy_research/6groups/mrt_oast/causal_edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    print(f'✅ 因果边数量: {len(edges)}')
    print(f'✅ 统计显著边: {sum(1 for e in edges if e.get(\"p_value\", 1.0) < 0.05)}')

    # 显示前3条边
    for i, edge in enumerate(edges[:3], 1):
        print(f'{i}. {edge.get(\"source\")} → {edge.get(\"target\")}, ATE={edge.get(\"ate\", 0):.4f}')
except Exception as e:
    print(f'❌ 错误: {e}')
"
```

#### 1.6 测试完成后操作

```bash
# 1. 在screen会话中查看结果
ls -lh results/energy_research/6groups/mrt_oast/

# 2. 退出screen会话（测试完成）
exit

# 3. 验证测试成功
echo "MRT-OAST测试结果:"
ls -lh results/energy_research/6groups/mrt_oast/causal_graph.npy
```

---

### 步骤2: 6组并行执行

**前提条件**：
- ✅ 步骤1单任务测试成功
- ✅ 确认环境配置正确
- ✅ 了解预期运行时间

#### 2.1 创建6组并行执行脚本

首先创建并行执行脚本：

```bash
cd /home/green/energy_dl/nightly/analysis/scripts
vi run_6groups_dibs_parallel.sh
```

粘贴以下内容：

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
task_array=("${!TASKS[@]}")

for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    task_name="${task_array[$i]}"

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

保存并赋予执行权限：

```bash
chmod +x run_6groups_dibs_parallel.sh
```

#### 2.2 创建screen会话并运行

```bash
# 创建6组并行执行的screen会话
screen -S dibs_6groups
```

在screen中执行：

```bash
# 进入scripts目录
cd /home/green/energy_dl/nightly/analysis/scripts

# 运行并行脚本
bash run_6groups_dibs_parallel.sh
```

**说明**：
- 6个任务组同时后台运行
- 所有输出重定向到日志文件
- screen会话保持运行直到所有任务完成

#### 2.3 分离screen会话

```
按 Ctrl+A，然后按 D
```

**说明**：
- 6个DiBS任务继续在后台运行
- 预计4-7小时完成全部任务
- 可以安全关闭SSH连接

#### 2.4 监控运行进度

**方法1: 查看所有日志**

```bash
# 实时查看所有日志（在新终端）
tail -f /home/green/energy_dl/nightly/analysis/logs/energy_research/6groups/*.log
```

**方法2: 查看特定任务**

```bash
# 查看MRT-OAST进度
tail -f /home/green/energy_dl/nightly/analysis/logs/energy_research/6groups/mrt_oast_*.log

# 查看Person_reID进度
tail -f /home/green/energy_dl/nightly/analysis/logs/energy_research/6groups/person_reid_*.log
```

**方法3: 检查已完成的任务**

```bash
# 检查因果图文件
ls -lh /home/green/energy_dl/nightly/analysis/results/energy_research/6groups/*/causal_graph.npy

# 统计已完成任务数
find /home/green/energy_dl/nightly/analysis/results/energy_research/6groups -name "causal_graph.npy" | wc -l
```

**方法4: 实时监控脚本**

```bash
# 创建监控脚本
cat > /home/green/energy_dl/nightly/analysis/scripts/monitor_6groups_progress.sh << 'EOF'
#!/bin/bash

RESULTS_DIR="/home/green/energy_dl/nightly/analysis/results/energy_research/6groups"
TASKS=("image_classification_examples" "image_classification_resnet" "person_reid" "vulberta" "bug_localization" "mrt_oast")

while true; do
    clear
    echo "================================================================================"
    echo "6组DiBS因果分析进度监控"
    echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"

    completed=0
    for task in "${TASKS[@]}"; do
        if [ -f "${RESULTS_DIR}/${task}/causal_graph.npy" ]; then
            echo "✅ ${task}"
            ((completed++))
        else
            echo "⏳ ${task}"
        fi
    done

    echo ""
    echo "进度: ${completed}/${#TASKS[@]} 任务完成"
    echo "================================================================================"

    if [ $completed -eq ${#TASKS[@]} ]; then
        echo "🎉 所有任务完成！"
        break
    fi

    sleep 30
done
EOF

chmod +x /home/green/energy_dl/nightly/analysis/scripts/monitor_6groups_progress.sh

# 运行监控
bash /home/green/energy_dl/nightly/analysis/scripts/monitor_6groups_progress.sh
```

#### 2.5 重新连接screen查看结果

```bash
# 重新连接到6组并行会话
screen -r dibs_6groups

# 如果所有任务完成，会看到执行总结
# 如果仍在运行，会看到实时输出
```

---

## 🔍 验证和结果检查

### 验证所有任务完成

```bash
cd /home/green/energy_dl/nightly/analysis

# 检查所有6个任务组
for task in image_classification_examples image_classification_resnet person_reid vulberta bug_localization mrt_oast; do
    echo "============================================"
    echo "任务: $task"

    if [ -f "results/energy_research/6groups/$task/causal_graph.npy" ]; then
        echo "  ✅ 因果图已生成"

        # 显示文件信息
        ls -lh "results/energy_research/6groups/$task/"

        # 统计因果边
        python3 << PYEOF
import pickle
try:
    with open('results/energy_research/6groups/$task/causal_edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    print(f"  因果边数量: {len(edges)}")
    sig_edges = sum(1 for e in edges if e.get('p_value', 1.0) < 0.05)
    print(f"  统计显著边: {sig_edges}")
except Exception as e:
    print(f"  ⚠️ 无法读取因果边: {e}")
PYEOF
    else
        echo "  ❌ 未完成"
    fi
    echo ""
done
```

### 生成汇总报告

```bash
# 统计总结
echo "================================================================================"
echo "6组DiBS因果分析总结"
echo "================================================================================"

total_tasks=6
completed_tasks=$(find results/energy_research/6groups -name "causal_graph.npy" | wc -l)

echo "已完成任务: ${completed_tasks}/${total_tasks}"

if [ $completed_tasks -eq $total_tasks ]; then
    echo "✅ 所有任务已完成"

    # 生成详细报告
    python3 << 'PYEOF'
import pickle
import numpy as np
from pathlib import Path

tasks = [
    'image_classification_examples',
    'image_classification_resnet',
    'person_reid',
    'vulberta',
    'bug_localization',
    'mrt_oast'
]

print("\n详细结果:")
print("-" * 80)

total_edges = 0
total_sig_edges = 0

for task in tasks:
    edges_file = f'results/energy_research/6groups/{task}/causal_edges.pkl'

    try:
        with open(edges_file, 'rb') as f:
            edges = pickle.load(f)

        sig_edges = sum(1 for e in edges if e.get('p_value', 1.0) < 0.05)
        total_edges += len(edges)
        total_sig_edges += sig_edges

        print(f"{task}:")
        print(f"  因果边: {len(edges)}, 显著边: {sig_edges}")

    except Exception as e:
        print(f"{task}: ❌ 错误 - {e}")

print("-" * 80)
print(f"总计: {total_edges} 条因果边, {total_sig_edges} 条统计显著边")
PYEOF
else
    echo "⚠️ 仍有任务未完成: $((total_tasks - completed_tasks)) 个"
fi

echo "================================================================================"
```

---

## 🛠️ 常见操作

### Screen基本命令

| 操作 | 命令 |
|------|------|
| 创建新会话 | `screen -S 会话名` |
| 列出所有会话 | `screen -ls` |
| 重新连接会话 | `screen -r 会话名` |
| 分离会话 | `Ctrl+A, D` |
| 终止会话 | 在会话中输入 `exit` |
| 强制终止会话 | `screen -X -S 会话名 quit` |
| 查看会话中的命令 | `screen -r 会话名` |

### 在Screen中的快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+A, D` | 分离会话 |
| `Ctrl+A, K` | 杀死当前窗口 |
| `Ctrl+A, [` | 进入复制模式（可滚动查看历史） |
| `Ctrl+A, ]` | 粘贴 |
| `Ctrl+A, ?` | 显示帮助 |

### 故障排查

**问题1: screen会话意外终止**

```bash
# 检查是否有core dump
ls -lh /home/green/core*

# 查看系统日志
dmesg | tail -50

# 检查磁盘空间
df -h

# 检查内存使用
free -h
```

**问题2: 任务运行时间过长**

```bash
# 检查GPU是否正常工作
nvidia-smi

# 查看CPU使用率
htop

# 检查是否有进程卡住
ps aux | grep python3 | grep demo_energy
```

**问题3: 无法重新连接screen**

```bash
# 查看所有screen会话
screen -ls

# 如果显示"(Detached)"
screen -r dibs_6groups

# 如果显示"(Attached)"，强制连接
screen -d -r dibs_6groups
```

---

## 📊 预期结果

### 完成后的目录结构

```
results/energy_research/6groups/
├── image_classification_examples/
│   ├── causal_graph.npy          # 因果图（邻接矩阵）
│   ├── causal_edges.pkl          # 因果边列表
│   ├── analysis_report.md        # 分析报告
│   └── config.json               # 配置参数
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

### 预期因果边统计

| 任务组 | 预期因果边 | 预期显著边 |
|--------|-----------|-----------|
| image_classification_examples | 3-8 | 2-5 |
| image_classification_resnet | 2-5 | 1-3 |
| person_reid | 4-10 | 2-6 |
| vulberta | 2-6 | 1-4 |
| bug_localization | 3-7 | 2-4 |
| **mrt_oast** | **4-8** | **2-5** |
| **总计** | **18-48** | **10-27** |

---

## 🎯 执行清单

### 单任务测试阶段 ✓

- [ ] 创建screen会话 (dibs_test)
- [ ] 激活conda环境
- [ ] 运行MRT-OAST单任务测试
- [ ] 分离screen会话
- [ ] 监控测试进度
- [ ] 验证测试结果
- [ ] 退出测试会话

### 6组并行执行阶段 ✓

- [ ] 创建并行执行脚本
- [ ] 赋予执行权限
- [ ] 创建screen会话 (dibs_6groups)
- [ ] 运行并行脚本
- [ ] 分离screen会话
- [ ] 定期检查进度（每1-2小时）
- [ ] 等待所有任务完成（4-7小时）
- [ ] 验证所有结果
- [ ] 生成汇总报告

---

**文档版本**: v1.0
**创建日期**: 2025-12-24
**预计总时间**:
- 单任务测试: 30-60分钟
- 6组并行: 4-7小时
- 总计: 约5-8小时
