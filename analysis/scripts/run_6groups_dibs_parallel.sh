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
task_array=()

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

    # 验证输入文件存在
    if [ ! -f "${input_path}" ]; then
        echo "  ❌ 错误: 输入文件不存在"
        continue
    fi

    # 后台运行
    python3 scripts/demos/demo_single_task_dibs.py \
        --task "${task_name}" \
        --input "${input_path}" \
        --output "${output_dir}" \
        --verbose \
        > "${log_file}" 2>&1 &

    PIDS+=($!)
    task_array+=("${task_name}")
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
    task_name="${task_array[$i]}"

    echo "等待任务: ${task_name} (PID: ${pid})..."
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
