# 阶段1：CTF源码完整性验证

**执行日期**: 2026-01-25
**状态**: ✅ 完成

---

## 📊 源码验证结果

### CTF_original源码完整性

✅ **状态**: 完整且可用

### 目录结构

```
CTF_original/
├── src/
│   ├── inf.py (337行) ⭐ 核心因果推断和ATE计算
│   ├── collect.py (1418行) ⭐ 数据收集和处理
│   ├── load_data.py (300行)
│   ├── models.py (337行)
│   ├── discovery.py (198行)
│   ├── fairness/
│   │   ├── in_p.py (466行)
│   │   ├── pre_p.py (313行)
│   │   └── post_p.py (空文件)
│   └── utils.py
├── causal-graph/ (6个因果图)
├── human-eval/
└── requirement.txt
```

**Python文件总数**: 11个
**总代码行数**: 3663行

---

## 🔍 关键发现

### 1. ATE计算实现 (CTF_original/src/inf.py:78-97)

**核心函数**: `compute_ate(parent, child, data_df, ref_df, dg, T0, T1)`

**技术栈**:
- ✅ `econml.dml.LinearDML`
- ✅ `sklearn.ensemble.RandomForestRegressor`
- ✅ `networkx.DiGraph` 用于因果图

**关键特性**:
1. 自动识别混淆因素（从因果图的predecessors）
2. 支持参考数据集 (ref_df)
3. 支持显式指定T0/T1（对照/处理值）
4. 返回单一ATE值

### 2. Trade-off检测算法 (CTF_original/src/inf.py:280-330)

**算法流程**:
```
1. 定义rules字典 → 指标期望改善方向
2. 使用DoWhy CausalModel计算因果效应
3. 检测冲突: improve_A != improve_B
4. 寻找原因:
   - 找common ancestors
   - 对每个潜在原因计算ATE
   - 判断效应方向
5. 输出到CSV文件
```

**核心依赖**:
- `econml==0.14.0` ⭐⭐⭐
- `dowhy==0.9.1`
- `networkx==2.8.8`
- `scikit-learn==1.1.3`

### 3. 我们的复现代码对比

**文件位置**:
- `utils/tradeoff_detection.py` (13332字节)
- `utils/causal_inference.py` (13783字节)
- `utils/fairness_methods.py` (5720字节)

**现状评估**:
- ✅ TradeoffDetector类已实现
- ✅ sign函数机制已实现
- ❌ 缺少ATE计算（依赖外部输入）
- ❌ 缺少LinearDML直接调用
- ❌ 缺少原因寻找算法

---

## 📋 阶段1关键结论

1. **源码完整性**: CTF_original仓库完整，包含所有核心功能
2. **可复现性**: 高 - 依赖明确，算法清晰
3. **技术栈**: EconML + DoWhy + NetworkX
4. **我们的代码**: 部分实现，缺少关键功能（ATE计算、原因寻找）

---

## ⚠️ 关键风险识别

1. **依赖兼容性**: EconML版本需保持0.14.0
2. **因果图格式**: 需要NetworkX DiGraph
3. **参考数据集**: ref_df的构建方式未明确
4. **T0/T1选择**: 如何确定干预值

---

## 🎯 下一步行动

**阶段2**: 详细对比CTF源码与复现代码的差异
