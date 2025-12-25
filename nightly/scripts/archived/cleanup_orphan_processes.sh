#!/bin/bash
################################################################################
# GPU残留进程清理脚本
#
# 用途：清理2x mutation run遗留的孤儿进程
# 运行方式：sudo bash cleanup_orphan_processes.sh
################################################################################

set -e

echo "================================================================================"
echo "GPU残留进程清理脚本"
echo "================================================================================"
echo ""

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then
    echo "错误：此脚本需要root权限运行"
    echo "请使用：sudo bash $0"
    exit 1
fi

# 显示清理前状态
echo "清理前GPU状态："
echo "--------------------------------------------------------------------------------"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
echo ""

echo "清理前进程信息："
echo "--------------------------------------------------------------------------------"
ps -f -p 1970647,1970645 2>/dev/null || echo "进程不存在"
echo ""

# 确认清理
echo "将要清理以下进程："
echo "  - PID 1970647 (python, 占用 ~5.5GB GPU内存)"
echo "  - PID 1970645 (python, 父进程)"
echo ""
read -p "确认清理这些进程？(y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消清理"
    exit 0
fi

# 清理进程
echo ""
echo "开始清理..."
echo "--------------------------------------------------------------------------------"

# 首先尝试SIGTERM（优雅退出）
echo "[1/3] 尝试优雅终止 (SIGTERM)..."
kill -TERM 1970647 1970645 2>/dev/null || true
sleep 2

# 检查是否还在运行
if kill -0 1970647 2>/dev/null || kill -0 1970645 2>/dev/null; then
    echo "[2/3] 优雅终止失败，使用强制终止 (SIGKILL)..."
    kill -9 1970647 1970645 2>/dev/null || true
    sleep 1
else
    echo "[2/3] 优雅终止成功"
fi

# 验证清理结果
echo "[3/3] 验证清理结果..."
sleep 2

STILL_RUNNING=0
if kill -0 1970647 2>/dev/null; then
    echo "⚠️  警告：进程 1970647 仍在运行"
    STILL_RUNNING=1
fi

if kill -0 1970645 2>/dev/null; then
    echo "⚠️  警告：进程 1970645 仍在运行"
    STILL_RUNNING=1
fi

if [ $STILL_RUNNING -eq 0 ]; then
    echo "✓ 所有目标进程已成功终止"
else
    echo "✗ 部分进程终止失败，可能需要重启系统"
    exit 1
fi

echo ""
echo "清理后GPU状态："
echo "--------------------------------------------------------------------------------"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
echo ""

# 显示GPU内存使用情况
echo "GPU内存详情："
echo "--------------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
echo ""

# 检查是否还有其他可疑的w用户进程
echo "检查其他可能的残留进程："
echo "--------------------------------------------------------------------------------"
ps aux | grep -E "^w\s" | grep python | grep -v grep || echo "未发现其他w用户的Python进程"
echo ""

echo "================================================================================"
echo "清理完成！"
echo "================================================================================"
echo ""
echo "建议："
echo "  1. 运行 'nvidia-smi' 确认GPU内存已释放"
echo "  2. 等待10秒让GPU完全释放资源"
echo "  3. 运行GPU内存清理测试："
echo "     sudo -E python3 mutation.py -ec settings/gpu_memory_cleanup_test.json"
echo ""
