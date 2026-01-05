#!/usr/bin/env python3
"""
计算 Person_reID Dropout 边界值测试的预计运行时间
"""

import json
from pathlib import Path

config_file = "settings/person_reid_dropout_boundary_test.json"
config_path = Path(__file__).parent / config_file

with open(config_path, 'r') as f:
    config = json.load(f)

print("=" * 80)
print("Person_reID Dropout 边界值测试 - 运行时间估算")
print("=" * 80)

# Configuration details
total_configs = len(config['experiments'])
runs_per_config = config['runs_per_config']
total_runs = total_configs * runs_per_config

# Get epochs from first experiment
epochs = config['experiments'][0]['hyperparameters']['epochs']

print(f"\n配置详情:")
print(f"  总配置数: {total_configs}")
print(f"  每配置运行次数: {runs_per_config}")
print(f"  总运行次数: {total_runs}")
print(f"  每次运行的 epochs: {epochs}")

print("\n" + "=" * 80)
print("时间估算（基于不同的硬件配置）")
print("=" * 80)

# Different scenarios based on hardware
scenarios = [
    {
        "name": "高性能 GPU (RTX 3090/4090, A100)",
        "epoch_time": 1.0,  # minutes per epoch
        "description": "Market-1501数据集, batch_size=32"
    },
    {
        "name": "中等性能 GPU (RTX 2080Ti, V100)",
        "epoch_time": 1.5,  # minutes per epoch
        "description": "Market-1501数据集, batch_size=32"
    },
    {
        "name": "低性能 GPU (GTX 1080Ti, RTX 2060)",
        "epoch_time": 2.5,  # minutes per epoch
        "description": "Market-1501数据集, batch_size=32"
    },
    {
        "name": "CPU 训练（不推荐）",
        "epoch_time": 15.0,  # minutes per epoch
        "description": "非常慢，仅用于测试"
    }
]

print()
for scenario in scenarios:
    name = scenario['name']
    epoch_time = scenario['epoch_time']
    desc = scenario['description']

    # Calculate times
    time_per_run = epochs * epoch_time  # minutes
    total_time_minutes = total_runs * time_per_run
    total_time_hours = total_time_minutes / 60
    total_time_days = total_time_hours / 24

    print(f"📊 {name}")
    print(f"   {desc}")
    print(f"   每个epoch: ~{epoch_time:.1f}分钟")
    print(f"   单次运行({epochs} epochs): ~{time_per_run:.0f}分钟 ({time_per_run/60:.1f}小时)")
    print(f"   总运行时间({total_runs}次): ~{total_time_minutes:.0f}分钟 ({total_time_hours:.1f}小时 / {total_time_days:.1f}天)")
    print()

print("=" * 80)
print("并行运行策略（如果有多GPU）")
print("=" * 80)

print("""
如果有多个GPU可用，可以并行运行多个实验以加速：

假设使用中等性能GPU (1.5分钟/epoch):
- 单GPU顺序运行: ~40.5小时
- 2 GPU并行: ~20.3小时 (减少50%)
- 3 GPU并行: ~13.5小时 (减少67%)
- 4 GPU并行: ~10.1小时 (减少75%)

注意:
1. 每个GPU运行一个独立的实验
2. 需要修改runner脚本支持并行执行
3. 注意GPU显存限制（densenet121, hrnet18, pcb显存需求不同）
""")

print("=" * 80)
print("优化运行时间的建议")
print("=" * 80)

print("""
### 方案1: 减少 runs_per_config (快速验证)

将 runs_per_config 从 3 改为 1:
- 配置数: 9
- 总运行数: 9 (减少67%)
- 预计时间(中等GPU): ~13.5小时

优点: 快速获得初步结果
缺点: 无法计算标准差，可能受随机性影响

---

### 方案2: 减少 epochs (快速测试)

将 epochs 从 60 改为 30:
- 总运行数: 27
- 预计时间(中等GPU): ~20.3小时 (减少50%)

优点: 仍可计算统计指标
缺点: 模型可能未充分收敛

---

### 方案3: 先测试单个模型

仅测试 densenet121 (3个配置 × 3次):
- 总运行数: 9
- 预计时间(中等GPU): ~13.5小时

优点: 快速验证策略有效性
缺点: 结论可能不适用于其他模型

---

### 方案4: 逐步测试

第一阶段: 每个模型先跑 runs_per_config=1
- 9次运行, ~13.5小时
- 查看初步结果，决定是否继续

第二阶段: 对有意义的配置增加到3次
- 选择性重复，节省时间
""")

print("\n" + "=" * 80)
print("推荐运行策略")
print("=" * 80)

print("""
### 推荐: 两阶段策略

**阶段1 - 快速验证 (修改配置运行)**
修改 runs_per_config: 3 → 1
- 运行时间: ~13.5小时 (中等GPU)
- 目标: 快速了解dropout影响趋势

**阶段2 - 完整验证 (如果阶段1有意义)**
使用原配置 runs_per_config: 3
- 运行时间: ~40.5小时 (中等GPU)
- 目标: 获得统计显著的结果

这样可以避免浪费时间在无意义的实验上。
""")

print("\n" + "=" * 80)
print("预计开销总结")
print("=" * 80)

print("""
基于常见硬件配置的最可能场景:

✅ 推荐配置: 中等GPU (RTX 2080Ti/V100 级别)
   - 完整运行 (runs_per_config=3): ~40.5小时 (~1.7天)
   - 快速验证 (runs_per_config=1): ~13.5小时 (~0.6天)

⚡ 如果有多GPU:
   - 2-GPU并行: ~20小时
   - 3-GPU并行: ~14小时

💡 建议: 先运行 runs_per_config=1 进行快速验证 (~13.5小时)
""")
