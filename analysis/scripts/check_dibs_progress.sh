#!/bin/bash
# DiBS分析进度监控脚本
# 创建时间: 2026-01-16

echo "========================================"
echo "DiBS分析进度监控"
echo "========================================"
echo ""

# 检查进程状态
echo "1. 进程状态:"
ps aux | grep "run_dibs_6groups_final" | grep -v grep
echo ""

# 检查日志文件
echo "2. 日志文件信息:"
if [ -f "dibs_run.log" ]; then
    echo "  文件大小: $(du -h dibs_run.log | cut -f1)"
    echo "  行数: $(wc -l dibs_run.log | cut -d' ' -f1)"
    echo ""
    echo "  最新30行:"
    tail -30 dibs_run.log
else
    echo "  ❌ 日志文件不存在"
fi
echo ""

# 检查输出目录
echo "3. 输出目录:"
latest_dir=$(ls -td results/energy_research/dibs_6groups_final/2026* 2>/dev/null | head -1)
if [ -n "$latest_dir" ]; then
    echo "  最新目录: $latest_dir"
    echo "  文件列表:"
    ls -lh "$latest_dir"
else
    echo "  ⚠️ 还没有结果目录"
fi
echo ""

echo "========================================"
echo "监控命令："
echo "  查看实时日志: tail -f dibs_run.log"
echo "  查看进程: ps aux | grep run_dibs_6groups_final"
echo "  重新运行监控: bash check_dibs_progress.sh"
echo "========================================"
