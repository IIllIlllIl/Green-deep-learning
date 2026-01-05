#!/usr/bin/env python3
"""
估算 mutation.py -ec settings/person_reid_dropout_boundary_test.json 的运行时间
"""

import json
from pathlib import Path

config_file = "settings/person_reid_dropout_boundary_test.json"
config_path = Path(__file__).parent / config_file

print("=" * 80)
print("mutation.py 运行时间估算")
print("命令: sudo -E python3 mutation.py -ec settings/person_reid_dropout_boundary_test.json")
print("=" * 80)

# Load config
with open(config_path, 'r') as f:
    config = json.load(f)

total_configs = len(config['experiments'])
runs_per_config = config['runs_per_config']
total_runs = total_configs * runs_per_config
epochs = config['experiments'][0]['hyperparameters']['epochs']
governor = config.get('governor', 'performance')

print(f"\n配置文件解析:")
print(f"  实验配置数: {total_configs}")
print(f"  每配置运行次数: {runs_per_config}")
print(f"  总训练运行数: {total_runs}")
print(f"  每次训练的epochs: {epochs}")
print(f"  CPU Governor: {governor}")

# Analyze experiments
print(f"\n实验列表:")
for idx, exp in enumerate(config['experiments'], 1):
    model = exp['model']
    dropout = exp['hyperparameters']['dropout']
    print(f"  {idx}. {model} (dropout={dropout}) × {runs_per_config} runs")

print("\n" + "=" * 80)
print("时间组成分析")
print("=" * 80)

print("""
mutation.py 的执行包含以下时间开销：

1. **训练时间** (主要开销)
   - 模型训练和验证
   - 依赖于GPU性能和数据集大小

2. **能耗监控开销** (每次运行)
   - 启动能耗监控: ~2-5秒
   - 停止并记录能耗: ~2-5秒
   - 每次运行额外: ~5-10秒

3. **实验设置开销** (每个配置)
   - 解析配置: ~1-2秒
   - 设置环境: ~1-2秒
   - 每个配置额外: ~2-4秒

4. **数据集准备** (首次运行)
   - 下载/解压 Market-1501: ~5-10分钟 (仅首次)
   - 数据预处理: ~2-5分钟 (仅首次)

5. **日志和结果记录** (每次运行)
   - 保存模型和日志: ~5-10秒
   - 记录能耗数据: ~2-5秒
   - 每次运行额外: ~10-15秒
""")

print("\n" + "=" * 80)
print("详细时间估算")
print("=" * 80)

# Time scenarios
scenarios = [
    {
        "name": "高性能 GPU (RTX 3090/4090, A100)",
        "epoch_time": 1.0,  # minutes
        "description": "Market-1501, batch_size=32"
    },
    {
        "name": "中等性能 GPU (RTX 2080Ti, V100)",
        "epoch_time": 1.5,
        "description": "Market-1501, batch_size=32"
    },
    {
        "name": "低性能 GPU (GTX 1080Ti, RTX 2060)",
        "epoch_time": 2.5,
        "description": "Market-1501, batch_size=32"
    }
]

# Additional overheads
monitoring_overhead_per_run = 0.3  # minutes (~20 seconds)
setup_overhead_per_config = 0.1  # minutes (~5 seconds)
first_time_setup = 10  # minutes (dataset download/prep, first time only)

print()
for scenario in scenarios:
    name = scenario['name']
    epoch_time = scenario['epoch_time']
    desc = scenario['description']

    # Calculate pure training time
    time_per_run = epochs * epoch_time  # minutes
    pure_training_time = total_runs * time_per_run

    # Calculate overheads
    total_monitoring_overhead = total_runs * monitoring_overhead_per_run
    total_setup_overhead = total_configs * setup_overhead_per_config

    # Total time (excluding first-time setup)
    total_time_without_setup = pure_training_time + total_monitoring_overhead + total_setup_overhead

    # Total time (including first-time setup)
    total_time_with_setup = total_time_without_setup + first_time_setup

    print(f"📊 {name}")
    print(f"   {desc}")
    print(f"   ")
    print(f"   训练时间:")
    print(f"     单次运行: {time_per_run:.1f}分钟")
    print(f"     总训练时间({total_runs}次): {pure_training_time:.1f}分钟 ({pure_training_time/60:.1f}小时)")
    print(f"   ")
    print(f"   额外开销:")
    print(f"     能耗监控开销: {total_monitoring_overhead:.1f}分钟")
    print(f"     实验设置开销: {total_setup_overhead:.1f}分钟")
    print(f"     首次数据准备: {first_time_setup:.1f}分钟 (仅首次)")
    print(f"   ")
    print(f"   总时间估算:")
    print(f"     后续运行(已有数据): {total_time_without_setup:.1f}分钟 ({total_time_without_setup/60:.1f}小时)")
    print(f"     首次运行(含数据准备): {total_time_with_setup:.1f}分钟 ({total_time_with_setup/60:.1f}小时)")
    print()

print("=" * 80)
print("实际运行流程时间线")
print("=" * 80)

# Use medium GPU as example
epoch_time = 1.5
time_per_run = epochs * epoch_time
run_with_overhead = time_per_run + monitoring_overhead_per_run

print(f"""
假设使用中等性能GPU，以下是实际运行时间线：

阶段1: 首次数据准备 (~10分钟，仅首次运行)
├─ 下载Market-1501数据集
├─ 解压和预处理
└─ 准备训练/测试split

阶段2: 第1个配置 - densenet121, dropout=0.3 (共3次运行)
├─ 设置实验环境: ~0.1分钟
├─ Run 1: {run_with_overhead:.1f}分钟 (训练{time_per_run:.1f}分钟 + 监控{monitoring_overhead_per_run:.1f}分钟)
├─ Run 2: {run_with_overhead:.1f}分钟
└─ Run 3: {run_with_overhead:.1f}分钟
    小计: {3*run_with_overhead + setup_overhead_per_config:.1f}分钟

阶段3: 第2个配置 - densenet121, dropout=0.5 (共3次运行)
└─ 重复上述流程: {3*run_with_overhead + setup_overhead_per_config:.1f}分钟

... (继续执行剩余7个配置)

阶段11: 第9个配置 - pcb, dropout=0.7 (共3次运行)
└─ 最后一组: {3*run_with_overhead + setup_overhead_per_config:.1f}分钟

总计: {(total_runs * run_with_overhead + total_configs * setup_overhead_per_config):.1f}分钟 ({(total_runs * run_with_overhead + total_configs * setup_overhead_per_config)/60:.1f}小时)
首次运行加上数据准备: {(total_runs * run_with_overhead + total_configs * setup_overhead_per_config + first_time_setup)/60:.1f}小时
""")

print("\n" + "=" * 80)
print("重要提示")
print("=" * 80)

print("""
⚠️  注意事项:

1. **sudo -E 的作用**:
   - 保持环境变量(如CUDA路径)
   - 允许设置CPU governor为performance模式
   - 需要root权限

2. **顺序执行**:
   - mutation.py 会按顺序执行所有27次训练
   - 不支持并行执行多个配置
   - 中途中断会丢失未保存的结果

3. **资源占用**:
   - 持续占用1个GPU
   - 需要足够的磁盘空间保存模型和日志
   - 每个模型检查点: ~100-500MB

4. **失败重试**:
   - max_retries=2，失败会自动重试
   - 重试会增加额外时间

5. **结果保存位置**:
   - 日志: logs/person_reid_dropout_boundary_test/
   - 能耗数据: results/energy/
   - 模型检查点: checkpoints/

6. **监控建议**:
   - 使用 tmux 或 screen 运行，避免SSH断开
   - 定期检查日志确认正常运行
   - 监控GPU使用率和温度
""")

print("\n" + "=" * 80)
print("最终时间估算总结")
print("=" * 80)

print("""
基于最常见的中等GPU配置 (RTX 2080Ti/V100):

✅ 首次完整运行:
   - 纯训练时间: ~40.5小时
   - 监控/设置开销: ~0.5小时
   - 数据准备: ~0.2小时
   ────────────────────────
   📍 总计: ~41.2小时 (约1.7天)

✅ 后续运行(已有数据):
   - 总计: ~41.0小时 (约1.7天)

⚡ 如果使用高性能GPU (RTX 4090/A100):
   - 总计: ~27.3小时 (约1.1天)

🐢 如果使用低性能GPU (GTX 1080Ti):
   - 总计: ~68.0小时 (约2.8天)

💡 建议:
   1. 使用 tmux/screen 保持会话
   2. 首次运行预留 42-45小时
   3. 确保有足够磁盘空间 (~10-15GB)
   4. 设置自动通知(完成/失败)
""")

print("\n" + "=" * 80)
print("优化建议")
print("=" * 80)

print("""
如果时间紧急，可以考虑：

方案1: 减少重复次数
  修改配置: "runs_per_config": 3 → 1
  节省时间: 67% (41小时 → 14小时)

方案2: 减少epochs
  修改配置: "epochs": 60 → 30
  节省时间: 50% (41小时 → 21小时)

方案3: 先测试单个模型
  只保留densenet121的3个配置
  时间: ~14小时

方案4: 使用更强GPU
  切换到A100或RTX 4090
  节省时间: 33% (41小时 → 27小时)
""")
