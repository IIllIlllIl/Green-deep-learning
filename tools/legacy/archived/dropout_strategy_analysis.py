#!/usr/bin/env python3
"""
分析 dropout default±0.2 策略的合理性
"""

import json
from pathlib import Path

# Load current config
config_path = Path(__file__).parent / "mutation" / "models_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

print("=" * 80)
print("Dropout Default±0.2 策略分析")
print("=" * 80)

# Find all models with dropout
dropout_models = []
for repo, repo_config in config["models"].items():
    params = repo_config.get("supported_hyperparams", {})
    if "dropout" in params:
        dropout_config = params["dropout"]
        dropout_models.append({
            "repo": repo,
            "models": repo_config["models"],
            "default": dropout_config.get("default"),
            "current_range": dropout_config["range"],
            "flag": dropout_config.get("flag")
        })

print(f"\n找到 {len(dropout_models)} 个支持 dropout 的仓库\n")

# Analyze each model
for idx, model in enumerate(dropout_models, 1):
    print(f"{'─' * 80}")
    print(f"{idx}. {model['repo']}")
    print(f"{'─' * 80}")
    print(f"   模型列表: {', '.join(model['models'])}")
    print(f"   参数标志: {model['flag']}")
    print(f"   默认值: {model['default']}")
    print(f"   当前范围: {model['current_range']}")

    default = model['default']
    current_min, current_max = model['current_range']

    # Calculate default±0.2 range
    relative_min = max(0.0, default - 0.2)  # dropout不能小于0
    relative_max = min(1.0, default + 0.2)  # dropout不能大于1

    print(f"\n   default±0.2 策略:")
    print(f"   推荐范围: [{relative_min:.1f}, {relative_max:.1f}]")

    # Check if current range matches default±0.2
    if abs(current_min - relative_min) < 0.01 and abs(current_max - relative_max) < 0.01:
        print(f"   ✅ 当前范围符合 default±0.2")
    else:
        print(f"   ❌ 当前范围不符合 default±0.2")
        print(f"      差异: 最小值 {current_min} vs {relative_min}, 最大值 {current_max} vs {relative_max}")

    # Check if default is within current range
    if current_min <= default <= current_max:
        print(f"   ✅ 默认值在当前范围内")
    else:
        print(f"   ❌ 默认值超出当前范围!")

    # Analyze exploration space
    print(f"\n   变异空间分析:")
    if default > 0:
        current_down = (default - current_min) / default * 100
        current_up = (current_max - default) / default * 100
        relative_down = (default - relative_min) / default * 100
        relative_up = (relative_max - default) / default * 100

        print(f"   当前配置:")
        print(f"      向下空间: {current_down:>6.1f}% (减少至 {current_min})")
        print(f"      向上空间: {current_up:>6.1f}% (增加至 {current_max})")

        print(f"   default±0.2 配置:")
        print(f"      向下空间: {relative_down:>6.1f}% (减少至 {relative_min})")
        print(f"      向上空间: {relative_up:>6.1f}% (增加至 {relative_max})")

    print()

print("\n" + "=" * 80)
print("策略建议")
print("=" * 80)

print("""
## Default±0.2 策略优缺点

### 优点:
1. ✅ **对称探索**: 在默认值上下对称探索，公平地测试增加和减少的效果
2. ✅ **相对保守**: ±0.2 的绝对范围相对保守，不会偏离默认值太远
3. ✅ **易于理解**: 简单明了的规则，便于解释和复现

### 缺点:
1. ❌ **固定绝对值**: 对于不同默认值，±0.2 的相对意义不同
   - default=0.2: ±0.2 = ±100% (范围很大)
   - default=0.5: ±0.2 = ±40% (范围适中)
2. ❌ **可能限制探索**: 无法探索 default 之外更极端的值
   - 例如 default=0.5 时，无法探索 0.0 (无dropout) 的效果

---

## 对比其他策略

### 策略A: Default±0.2 (相对范围，对称)
```
MRT-OAST:      [0.0, 0.4] (default=0.2)  ✅ 对称，±100%
Person_reID:   [0.3, 0.7] (default=0.5)  ✅ 对称，±40%
```
**适用场景**: 微调优化，关注默认值附近的性能变化

---

### 策略B: [0.0, 0.7] (固定绝对范围，探索性)
```
MRT-OAST:      [0.0, 0.7] (default=0.2)  覆盖 0.0~3.5× default
Person_reID:   [0.0, 0.7] (default=0.5)  覆盖 0.0~1.4× default
```
**适用场景**: 探索性实验，寻找最优 dropout 值

---

### 策略C: Default×[0.5, 1.5] (相对倍数)
```
MRT-OAST:      [0.1, 0.3] (default=0.2)  ±50%
Person_reID:   [0.25, 0.75] (default=0.5) ±50%
```
**问题**: dropout 不适合用倍数，因为有上下界限 [0.0, 1.0]

---

## 推荐方案

### 方案1 (推荐): 混合策略
- **MRT-OAST**: [0.0, 0.4] (default±0.2) ✅ 已采用
- **Person_reID**: [0.3, 0.7] (default±0.2) ✅ 需修改

**优点**:
- 对称探索，易于比较
- 适合快速验证 dropout 影响
- 范围适中，不会太极端

**缺点**:
- 无法探索 Person_reID 的 no-dropout (0.0) 效果

---

### 方案2 (如需探索极端值): 扩展范围
- **MRT-OAST**: [0.0, 0.5] (允许探索到 2.5× default)
- **Person_reID**: [0.0, 0.7] (允许探索 no-dropout 和强正则化)

**优点**:
- 可以探索更广泛的 dropout 范围
- 能够测试 no-dropout baseline

**缺点**:
- 不对称，可能偏离原始默认值较远

---

## 最终建议

**如果目标是微调和对比默认值**: 使用 default±0.2
- MRT-OAST: [0.0, 0.4] ✅ 保持不变
- Person_reID: [0.3, 0.7] ✅ 修改为对称范围

**如果目标是探索最优值**: 使用 [0.0, 0.7] 统一范围
- 两个模型都使用 [0.0, 0.7]
- 可以找到真正的最优 dropout，而不受默认值限制
""")

print("\n" + "=" * 80)
print("配置文件修改建议")
print("=" * 80)

print("""
### 采用 default±0.2 策略，修改 Person_reID:

```json
"Person_reID_baseline_pytorch": {
  "supported_hyperparams": {
    "dropout": {
      "flag": "--droprate",
      "type": "float",
      "default": 0.5,
      "range": [0.3, 0.7],  // 从 [0.0, 0.4] 改为 [0.3, 0.7]
      "distribution": "uniform"
    }
  }
}
```

### 采用探索性策略，两个模型统一:

```json
// MRT-OAST
"dropout": {
  "flag": "--dropout",
  "type": "float",
  "default": 0.2,
  "range": [0.0, 0.7],  // 从 [0.0, 0.4] 改为 [0.0, 0.7]
  "distribution": "uniform"
}

// Person_reID_baseline_pytorch
"dropout": {
  "flag": "--droprate",
  "type": "float",
  "default": 0.5,
  "range": [0.0, 0.7],  // 从 [0.0, 0.4] 改为 [0.0, 0.7]
  "distribution": "uniform"
}
```
""")
