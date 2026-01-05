#!/usr/bin/env python3
"""
验证 Person_reID dropout 边界值测试配置
"""

import json
from pathlib import Path

config_file = "settings/person_reid_dropout_boundary_test.json"
config_path = Path(__file__).parent / config_file

print("=" * 80)
print("Person_reID Dropout 边界值测试配置验证")
print("=" * 80)

# Load config
with open(config_path, 'r') as f:
    config = json.load(f)

print(f"\n实验名称: {config['experiment_name']}")
print(f"描述: {config['description']}")
print(f"每个配置运行次数: {config['runs_per_config']}")
print(f"总实验配置数: {len(config['experiments'])}")
print(f"总训练运行数: {len(config['experiments']) * config['runs_per_config']}")

print("\n" + "=" * 80)
print("实验设计分析")
print("=" * 80)

# Analyze experiments by model
models = {}
dropout_values = set()

for exp in config['experiments']:
    model = exp['model']
    dropout = exp['hyperparameters']['dropout']
    dropout_values.add(dropout)

    if model not in models:
        models[model] = []
    models[model].append(dropout)

print(f"\n测试的模型数量: {len(models)}")
print(f"测试的 dropout 值: {sorted(dropout_values)}")

for model, dropouts in models.items():
    print(f"\n{model}:")
    print(f"  测试 dropout 值: {sorted(dropouts)}")
    print(f"  配置数: {len(dropouts)}")
    print(f"  总运行数: {len(dropouts) * config['runs_per_config']}")

# Check if config covers the boundary values
print("\n" + "=" * 80)
print("边界值覆盖检查")
print("=" * 80)

target_dropouts = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"\n目标 dropout 值: {target_dropouts}")

for model in models:
    model_dropouts = sorted(models[model])
    missing = set(target_dropouts) - set(model_dropouts)

    if not missing:
        print(f"✅ {model}: 覆盖所有目标值")
    else:
        print(f"❌ {model}: 缺少 {sorted(missing)}")

print("\n" + "=" * 80)
print("实验参数一致性检查")
print("=" * 80)

# Check parameter consistency
epochs_set = set()
lr_set = set()
seed_set = set()

for exp in config['experiments']:
    params = exp['hyperparameters']
    epochs_set.add(params['epochs'])
    lr_set.add(params['learning_rate'])
    seed_set.add(params['seed'])

print(f"\nepochs 值: {epochs_set}")
print(f"learning_rate 值: {lr_set}")
print(f"seed 值: {seed_set}")

if len(epochs_set) == 1 and len(lr_set) == 1 and len(seed_set) == 1:
    print("\n✅ 所有实验使用相同的 epochs, learning_rate, seed")
    print("   这确保了 dropout 是唯一的变量，便于对比分析")
else:
    print("\n⚠️ 警告: 实验参数不一致，可能影响对比分析")

print("\n" + "=" * 80)
print("实验结果预期")
print("=" * 80)

print("""
## 实验目标

验证 dropout 在 [0.3, 0.7] 范围内对 Person_reID 模型性能的影响。

## 测试点

1. **dropout=0.3 (下边界)**
   - 较低的正则化
   - 可能在训练集上表现更好，但过拟合风险较高

2. **dropout=0.4**
   - 中低正则化

3. **dropout=0.5 (默认值)**
   - 中等正则化
   - 基线对比点

4. **dropout=0.6**
   - 中高正则化

5. **dropout=0.7 (上边界)**
   - 较高的正则化
   - 可能在测试集上表现更好，但训练可能较慢

## 预期结果

通过对比 5 个 dropout 值的性能曲线，可以回答：

1. **边界值是否合理？**
   - 如果 0.3 或 0.7 是最优值 → 需要扩展范围
   - 如果最优值在中间 → 范围合理

2. **性能曲线形状？**
   - U 型曲线：存在最优 dropout 值
   - 单调递减/递增：需要调整范围
   - 平坦：dropout 对该模型影响不大

3. **default±0.2 策略是否合适？**
   - 如果 [0.3, 0.7] 能覆盖大部分性能变化 → 合适
   - 如果边界值显著更好 → 需要扩展

## 评估指标

Person_reID 模型的主要指标（从 models_config.json）：
- Rank@1: 首位命中率
- Rank@5: 前五命中率
- mAP: 平均精度均值

## 数据收集

每个配置运行 3 次（runs_per_config: 3），可以：
- 计算均值和标准差
- 评估结果的稳定性
- 减少随机性影响
""")

print("\n" + "=" * 80)
print("建议的后续分析步骤")
print("=" * 80)

print("""
1. **运行实验**
   ```bash
   python nightly_runner.py settings/person_reid_dropout_boundary_test.json
   ```

2. **收集结果**
   - 提取每个配置的 Rank@1, Rank@5, mAP
   - 计算每个 dropout 值的平均性能（3次运行）

3. **可视化分析**
   - 绘制 dropout vs 性能曲线（3个模型分别绘制）
   - 标注默认值 0.5 的位置
   - 标注边界值 0.3 和 0.7

4. **决策建议**
   - 如果曲线在 [0.3, 0.7] 内有明显最优点 → default±0.2 策略合适
   - 如果 0.3 附近最优 → 考虑扩展到 [0.0, 0.5] 或 [0.2, 0.6]
   - 如果 0.7 附近最优 → 考虑扩展到 [0.5, 0.9]
   - 如果性能平坦 → dropout 影响较小，可以使用更宽范围

5. **对比其他策略**
   - 如需要，可以额外测试 [0.0, 0.7] 范围（添加 0.0, 0.1, 0.2）
   - 比较两种策略在相同计算预算下的探索效果
""")

print("\n" + "=" * 80)
print("配置摘要")
print("=" * 80)

print(f"""
✅ 配置文件: {config_file}
✅ 测试模型: {len(models)} 个 ({', '.join(sorted(models.keys()))})
✅ Dropout 值: {len(dropout_values)} 个 ({', '.join(map(str, sorted(dropout_values)))})
✅ 总配置数: {len(config['experiments'])}
✅ 总运行数: {len(config['experiments']) * config['runs_per_config']}
✅ 参数一致性: epochs={list(epochs_set)[0]}, lr={list(lr_set)[0]}, seed={list(seed_set)[0]}

估计运行时间:
- 假设每次运行 60 epochs ≈ 60-90 分钟（取决于数据集大小）
- 总运行时间 ≈ {len(config['experiments']) * config['runs_per_config']} × 75分钟 = {len(config['experiments']) * config['runs_per_config'] * 75 / 60:.1f} 小时

建议: 如果时间有限，可以先运行 runs_per_config=1 进行快速验证
""")
