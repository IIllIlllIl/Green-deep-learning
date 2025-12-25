#!/usr/bin/env python3
"""
Dropout 变异范围分析和建议
"""

import json
from pathlib import Path

# Load current config
config_path = Path(__file__).parent / "mutation" / "models_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

print("=" * 80)
print("当前 Dropout 配置分析")
print("=" * 80)

# Analyze current dropout configurations
dropout_configs = []
for repo, repo_config in config["models"].items():
    params = repo_config.get("supported_hyperparams", {})
    if "dropout" in params:
        dropout_config = params["dropout"]
        default = dropout_config.get("default")
        range_val = dropout_config["range"]

        print(f"\n{repo}:")
        print(f"  默认值: {default}")
        print(f"  范围: {range_val}")

        if default is not None:
            if default < range_val[0] or default > range_val[1]:
                print(f"  ⚠️ 警告: 默认值不在范围内!")
            elif default == range_val[1]:
                print(f"  ⚠️ 警告: 默认值是最大值，无法向上探索!")
            elif default == range_val[0]:
                print(f"  ⚠️ 警告: 默认值是最小值，无法向下探索!")
            else:
                margin_low = (default - range_val[0]) / default * 100
                margin_high = (range_val[1] - default) / default * 100
                print(f"  向下空间: {margin_low:.1f}% ({default} → {range_val[0]})")
                print(f"  向上空间: {margin_high:.1f}% ({default} → {range_val[1]})")

        dropout_configs.append({
            "repo": repo,
            "default": default,
            "range": range_val
        })

print("\n\n" + "=" * 80)
print("Dropout 变异范围建议")
print("=" * 80)

print("""
## 统一变异范围建议

### 方案1: 固定绝对范围（推荐用于探索性实验）
```json
"dropout": {
  "range": [0.0, 0.7],
  "distribution": "uniform"
}
```
**优点**:
  - 覆盖整个常用 dropout 范围
  - 可以探索 no dropout (0.0) 的效果
  - 可以探索高 dropout (0.6-0.7) 的效果
  - 不依赖默认值，适用于所有模型

**适用模型**: 所有模型（通用方案）

---

### 方案2: 相对范围（推荐用于对比实验）
```json
// 对于 default=0.2
"dropout": {
  "default": 0.2,
  "range": [0.0, 0.4],  // default ± 0.2
  "distribution": "uniform"
}

// 对于 default=0.5
"dropout": {
  "default": 0.5,
  "range": [0.3, 0.7],  // default ± 0.2
  "distribution": "uniform"
}
```
**优点**:
  - 在默认值附近对称探索
  - 更容易看到参数变化的影响
  - 变异幅度相对保守

**适用场景**: 微调实验，对比默认值周围的性能

---

### 方案3: 分层范围（推荐用于系统性研究）
```json
// 低 dropout 模型 (default ≤ 0.3)
"dropout": {
  "range": [0.0, 0.5],
  "distribution": "uniform"
}

// 中 dropout 模型 (0.3 < default ≤ 0.5)
"dropout": {
  "range": [0.2, 0.7],
  "distribution": "uniform"
}

// 高 dropout 模型 (default > 0.5)
"dropout": {
  "range": [0.3, 0.8],
  "distribution": "uniform"
}
```
**优点**:
  - 根据模型特性调整范围
  - 避免探索不合理的区域
  - 更精细的控制

---

### 推荐方案总结

| 模型 | 当前默认值 | 当前范围 | 推荐范围（方案1） |
|------|-----------|---------|-----------------|
| MRT-OAST | 0.2 | [0.0, 0.4] | [0.0, 0.7] ✅ |
| Person_reID | 0.5 | [0.0, 0.4] ⚠️ | [0.0, 0.7] ✅ |

**修正后的统一配置**:
```json
"dropout": {
  "flag": "--dropout",      // 或 --droprate
  "type": "float",
  "default": 0.2,           // 或模型特定值
  "range": [0.0, 0.7],      // 统一范围
  "distribution": "uniform"
}
```

---

## Dropout 在深度学习中的典型范围

- **0.0**: 无 dropout（基线）
- **0.1-0.3**: 轻度正则化（BERT等预训练模型）
- **0.3-0.5**: 中度正则化（CNN、RNN常用）
- **0.5-0.7**: 强正则化（全连接层、过拟合场景）
- **>0.7**: 很少使用（可能导致欠拟合）

---

## 建议行动

**立即修正**: Person_reID_baseline_pytorch 的 dropout 范围
- 当前: [0.0, 0.4] (default=0.5) ❌
- 修正: [0.0, 0.7] (default=0.5) ✅
  或: [0.3, 0.7] (default=0.5) ✅

""")
