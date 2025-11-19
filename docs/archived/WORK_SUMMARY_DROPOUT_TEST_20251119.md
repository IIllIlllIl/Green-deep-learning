# Dropout 边界测试工作总结

**日期**: 2025-11-19
**任务**: Person_reID Dropout 边界值测试设计与实现

---

## ✅ 完成的工作

### 1. 文件整理（符合项目规范）

所有文件已按项目规范整理到正确目录：

#### 📁 settings/ - 实验配置
- `person_reid_dropout_boundary_test.json` ✅
  - 9个配置（3个模型 × 3个dropout值）
  - 每配置运行3次
  - 总计27次训练

#### 📁 scripts/ - 分析和验证脚本
- `dropout_analysis.py` - dropout配置分析
- `dropout_strategy_analysis.py` - default±0.2策略分析
- `estimate_dropout_test_time.py` - 运行时间估算
- `estimate_mutation_runtime.py` - mutation.py运行时间估算
- `validate_dropout_boundary_config.py` - 配置验证脚本
- `validate_mutation_config.py` - 变异配置验证

#### 📁 tests/ - 测试文件
- `test_mutation_verification.py` - 变异方法验证测试

#### 📁 docs/ - 实验设计文档
- `DROPOUT_BOUNDARY_TEST_DESIGN.md` - 完整实验设计文档
- `README.md` - 已更新索引

---

## 📊 验证结果

### 配置文件验证 ✅

```
✅ person_reid_dropout_boundary_test.json
   - 总配置数: 9
   - 总运行数: 27
   - Dropout值: [0.3, 0.5, 0.7]
   - 模型: densenet121, hrnet18, pcb
   - 参数一致性: ✅ epochs=60, lr=0.05, seed=1334
```

### Mutation 框架验证 ✅

```
✅ mutation_validation_1x.json (88个实验)
✅ mutation_all_models_3x_dynamic.json (240个实验)
   - 100% 使用 mutation 框架动态生成
   - 自动排除默认值
   - 确保唯一性
```

### Dropout 配置分析 ✅

**当前配置问题**:
- MRT-OAST: default=0.2, range=[0.0, 0.4] ✅ 符合 default±0.2
- Person_reID: default=0.5, range=[0.0, 0.4] ❌ 不符合 default±0.2

**建议修正**:
```json
"Person_reID_baseline_pytorch": {
  "dropout": {
    "default": 0.5,
    "range": [0.3, 0.7],  // 从 [0.0, 0.4] 改为 [0.3, 0.7]
    "distribution": "uniform"
  }
}
```

---

## ⏱️ 运行时间估算

### person_reid_dropout_boundary_test.json

| GPU配置 | 完整运行(27次) | 快速验证(9次) |
|---------|---------------|--------------|
| 高性能GPU (RTX 4090) | 27.3小时 | 9.0小时 |
| 中等GPU (RTX 2080Ti) | **40.8小时** | **13.5小时** |
| 低性能GPU (GTX 1080Ti) | 67.8小时 | 22.5小时 |

**推荐策略**:
1. 先运行 runs_per_config=1 快速验证 (~13.5小时)
2. 如有意义，再运行完整版获取统计数据 (~40.8小时)

---

## 🎯 实验目标

### 核心问题

验证 Person_reID 模型使用 **default±0.2 dropout 策略** 是否合适：
- 测试边界值: 0.3 (下界), 0.5 (默认), 0.7 (上界)
- 评估指标: Rank@1, Rank@5, mAP
- 控制变量: 只变化 dropout，其他参数固定

### 预期结果

| 观察 | 结论 | 建议 |
|------|------|------|
| 0.5最优 | default±0.2合适 | 采用 [0.3, 0.7] ✅ |
| 0.3最优 | 下界需扩展 | 考虑 [0.0, 0.5] |
| 0.7最优 | 上界需扩展 | 考虑 [0.5, 0.9] |
| 三者相近 | dropout影响小 | 可用 [0.0, 0.7] |

---

## 🚀 运行命令

### 验证配置
```bash
# 验证配置格式
python3 scripts/validate_dropout_boundary_config.py

# 估算时间
python3 scripts/estimate_mutation_runtime.py
```

### 执行实验
```bash
# 完整运行 (推荐使用 tmux/screen)
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -ec settings/person_reid_dropout_boundary_test.json -g performance
```

### 快速验证
修改配置文件 `runs_per_config: 3 → 1`，然后运行

---

## 📂 项目目录结构

```
energy_dl/nightly/
├── docs/
│   ├── DROPOUT_BOUNDARY_TEST_DESIGN.md  🆕 实验设计文档
│   └── README.md                         ✏️  已更新索引
├── scripts/
│   ├── dropout_analysis.py              🆕 dropout配置分析
│   ├── dropout_strategy_analysis.py     🆕 策略分析
│   ├── estimate_dropout_test_time.py    🆕 时间估算
│   ├── estimate_mutation_runtime.py     🆕 mutation时间估算
│   ├── validate_dropout_boundary_config.py  🆕 配置验证
│   └── validate_mutation_config.py      🆕 变异配置验证
├── tests/
│   └── test_mutation_verification.py    🆕 变异方法测试
├── settings/
│   ├── person_reid_dropout_boundary_test.json  🆕 实验配置
│   ├── mutation_validation_1x.json      🆕 1次变异验证
│   └── mutation_all_models_3x_dynamic.json  🆕 3次变异完整配置
└── mutation/
    └── models_config.json               ⚠️  待修正dropout范围
```

---

## 📝 后续任务

### 立即可做
1. ✅ 文件已整理到规范目录
2. ✅ 配置文件已验证
3. ✅ 实验设计文档已创建
4. ✅ 文档索引已更新

### 等待决策
1. ⏳ 是否修正 `mutation/models_config.json` 中 Person_reID 的 dropout 范围
2. ⏳ 是否运行 dropout 边界测试实验

### 实验运行后
1. 分析实验结果
2. 根据结果决定是否修改配置
3. 更新超参数范围文档
4. 创建实验报告

---

## 🔗 相关文档

- [DROPOUT_BOUNDARY_TEST_DESIGN.md](../docs/DROPOUT_BOUNDARY_TEST_DESIGN.md) - 实验设计
- [MUTATION_RANGES_QUICK_REFERENCE.md](../docs/MUTATION_RANGES_QUICK_REFERENCE.md) - 超参数范围
- [DEFAULT_BASELINE_REPORT_20251118.md](../docs/DEFAULT_BASELINE_REPORT_20251118.md) - 基线测试报告

---

## 📌 关键发现

### 1. Dropout 配置问题
- Person_reID 当前配置: default=0.5, range=[0.0, 0.4]
- **问题**: 默认值超出范围
- **影响**: 所有变异值都小于默认值，无法探索更高 dropout

### 2. Default±0.2 策略
- **MRT-OAST**: 已符合（default=0.2, range=[0.0, 0.4]）
- **Person_reID**: 不符合（应为 [0.3, 0.7]）
- **相对意义**:
  - MRT-OAST: ±0.2 = ±100% 变化
  - Person_reID: ±0.2 = ±40% 变化

### 3. Mutation 框架验证
- ✅ 动态生成正常工作
- ✅ 自动排除默认值
- ✅ 确保变异唯一性
- ✅ 分布策略正确（log-uniform vs uniform）

---

**创建时间**: 2025-11-19
**状态**: ✅ 准备就绪，等待运行实验
**维护者**: Green
