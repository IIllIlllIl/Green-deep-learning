# 轮间去重机制更新总结 (v2.0)

**更新时间**: 2025-11-26
**版本**: 2.0 (简化版)
**状态**: ✅ 已测试并就绪

---

## 📋 更新内容

### 主要改进

✅ **简化数据源配置**
- **之前 (v1.0)**: 需要指定多个实验轮次的 CSV 文件
  ```json
  "historical_csvs": [
    "results/defualt/summary.csv",
    "results/mutation_1x/summary.csv",
    "results/mutation_2x_20251122_175401/summary_safe.csv"
  ]
  ```

- **现在 (v2.0)**: 只需指定汇总数据文件
  ```json
  "historical_csvs": [
    "results/summary_all.csv"
  ]
  ```

✅ **优势**
- 更简单：单一数据源
- 更快速：无需读取多个文件
- 更易维护：只需维护一个汇总文件
- 更可靠：数据源统一，避免不一致

---

## 🔄 修改的文件

### 1. 配置文件更新

**`settings/mutation_2x_supplement.json`**

```diff
  "use_deduplication": true,
  "historical_csvs": [
-   "results/defualt/summary.csv",
-   "results/mutation_1x/summary.csv",
-   "results/mutation_2x_20251122_175401/summary_safe.csv"
+   "results/summary_all.csv"
  ],
```

### 2. 测试脚本增强

**`tests/functional/test_runner_dedup_integration.py`**

新增测试用例：
- ✅ Test 6: 从 summary_all.csv 加载历史数据

测试结果更新：**5/5 → 6/6 通过** ✅

### 3. 文档更新

**新增**:
- `docs/DEDUPLICATION_USER_GUIDE.md` - 完整的去重机制使用指南

**更新**:
- `settings/MUTATION_2X_SUPPLEMENT_README.md` - 配置说明
- `docs/SUPPLEMENT_EXPERIMENTS_READY.md` - 工作总结

---

## 📊 数据对比

### 历史数据统计

| 数据源 | 实验数 | 唯一超参数 | 备注 |
|--------|--------|-----------|------|
| `results/summary_all.csv` | 211 | **177** | ✅ 推荐 (v2.0) |
| 三个单独 CSV | 231 | 189 | v1.0 方式 |

**说明**:
- summary_all.csv 包含 211 条记录��已过滤失败实验）
- 提取出 177 个唯一的超参数组合
- 比之前的 189 个少，因为 summary_all.csv 已经过滤掉了失败和重复数据

---

## 🧪 测试验证

### 运行测试

```bash
python3 tests/functional/test_runner_dedup_integration.py
```

### 测试结果

```
================================================================================
Test Summary
================================================================================
Total tests: 6
Passed: 6
Failed: 0

✓ All tests passed!

Configuration:
  - Deduplication enabled: Yes
  - Historical data source: results/summary_all.csv
================================================================================
```

### 测试覆盖

1. ✅ 配置文件加载
2. ✅ 历史 CSV 文件存在性检查（含文件大小和行数验证）
3. ✅ MutationRunner 初始化
4. ✅ 去重模块导入
5. ✅ 实验配置验证
6. ✅ **从 summary_all.csv 加载并处理 177 个唯一超参数**

---

## 💡 使用方法

### 启用去重机制

在实验���置文件中：

```json
{
  "experiment_name": "your_experiment",
  "use_deduplication": true,
  "historical_csvs": [
    "results/summary_all.csv"
  ],
  "experiments": [ /* ... */ ]
}
```

### 运行实验

```bash
# 直接运行
python3 -m mutation.runner settings/mutation_2x_supplement.json

# 或后台运行
nohup python3 -m mutation.runner settings/mutation_2x_supplement.json > run.log 2>&1 &
```

### 验证去重生效

```bash
# 查看日志中的去重信息
grep "Loaded.*historical mutations" results/*/logs/*.log

# 应该看到：
# "Loaded 177 historical mutations for deduplication"
```

---

## 🎯 何时触发去重？

### ✅ 推荐启用的场景

1. **运行新一轮变异实验**
   - 已有历史实验数据
   - 希望避免重复超参数
   - 例如：mutation_3x, mutation_4x

2. **补充实验**
   - 补充之前失败或缺失的实验
   - 确保新超参数不重复
   - 例如：当前的 mutation_2x_supplement

3. **扩展实验**
   - 增加更多实验探索超参数空间
   - 避免与历史重复

### ⚠️ 不需要启用的场景

1. **第一轮实验 (default)**
   - 没有历史数据可去重
   - 内置机制已避免与默认值重复

2. **独立测试**
   - 临时测试某个模型
   - 不关心与历史重复

3. **指定超参数测试**
   - 明确指定超参数值 (mode: "default")
   - 不需要随机生成

---

## 📈 工作流程

### 完整的实验流程（含去重）

```
1. 聚合历史数据
   ↓
   python3 scripts/aggregate_csvs.py
   ↓
   生成 results/summary_all.csv

2. 配置新实验
   ↓
   编辑实验配置文件:
   - use_deduplication: true
   - historical_csvs: ["results/summary_all.csv"]

3. 运行实验
   ↓
   python3 -m mutation.runner settings/your_config.json
   ↓
   自动加载 177 个历史超参数
   ↓
   生成新的唯一超参数

4. 实验完成后
   ↓
   重新聚合数据:
   python3 scripts/aggregate_csvs.py
   ↓
   更新 results/summary_all.csv
   ↓
   下次实验将使用更新后的历史数据
```

---

## 🔧 维护说明

### 定期更新 summary_all.csv

**何时更新**:
- 每次完成新一轮实验后
- 补充实验完成后
- 发现数据不一致时

**如何更新**:
```bash
python3 scripts/aggregate_csvs.py
```

**验证更新**:
```bash
# 检查行数
wc -l results/summary_all.csv

# 检查更新时间
ls -lh results/summary_all.csv
```

### 检查去重效果

```bash
# 运行去重分析
python3 scripts/analyze_duplicates.py

# 如果发现新的重复，说明：
# 1. 去重机制未启用，或
# 2. summary_all.csv 未及时更新
```

---

## 📚 相关文档

### 核心文档
- **使用指南**: `docs/DEDUPLICATION_USER_GUIDE.md` ⭐ 详细说明
- **机制设计**: `docs/INTER_ROUND_DEDUPLICATION.md`

### 实现文件
- **去重模块**: `mutation/dedup.py`
- **集成代码**: `mutation/runner.py` (lines 28, 797-853, 927, 1024)

### 测试和验证
- **集成测试**: `tests/functional/test_runner_dedup_integration.py`
- **单元测试**: `tests/unit/test_dedup_mechanism.py`

### 配置示例
- **补充实验**: `settings/mutation_2x_supplement.json`
- **配置说明**: `settings/MUTATION_2X_SUPPLEMENT_README.md`

---

## ✅ 更新完成检查清单

- [x] 配置文件更新 (historical_csvs 改为单文件)
- [x] 测试脚本增强 (新增 Test 6)
- [x] 测试通过 (6/6)
- [x] 文档更新 (用户指南、README、总结)
- [x] 验证去重机制工作正常 (加载 177 个唯一超参数)

---

## 🎉 总结

### v2.0 改进

| 方面 | v1.0 | v2.0 (当前) | 改进 |
|------|------|------------|------|
| 数据源数量 | 3 个文件 | 1 个文件 | ✅ 简化 67% |
| 配置复杂度 | 需要指定多个路径 | 单一路径 | ✅ 更简单 |
| 维护成本 | 需要追踪多个文件 | 只维护一个文件 | ✅ 更易维护 |
| 加载速度 | 读取 3 个文件 | 读取 1 个文件 | ✅ 更快 |
| 数据一致性 | 可能不同步 | 统一数据源 | ✅ 更可靠 |
| 唯一超参数数 | 189 个 | 177 个 | ✅ 更精确 |

### 关键数据

- **数据源**: `results/summary_all.csv` (单一文件)
- **实验记录**: 211 条
- **唯一超参数**: 177 个
- **测试状态**: 6/6 通过 ✅
- **就绪状态**: ✅ Ready to Run

### 下一步

**立即可用**:
```bash
python3 -m mutation.runner settings/mutation_2x_supplement.json
```

系统���自动：
1. 加载 `results/summary_all.csv`
2. 提取 177 个唯一超参数
3. 避免生成重复组合
4. 确保每个实验的超参数都是新的

---

**版本**: 2.0
**更新时间**: 2025-11-26
**维护者**: Mutation-Based Training Energy Profiler Team
**状态**: ✅ 生产就绪
