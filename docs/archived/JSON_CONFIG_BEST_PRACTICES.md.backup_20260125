# JSON实验配置文件编写指南

**版本**: v1.0
**创建日期**: 2025-12-07
**重要性**: ⭐⭐⭐⭐⭐ 必读文档

---

## 📚 目录

1. [核心概念](#核心概念)
2. [常见错误与正确示例](#常见错误与正确示例)
3. [配置模板](#配置模板)
4. [最佳实践](#最佳实践)
5. [验证清单](#验证清单)
6. [故障排查](#故障排查)

---

## 🎯 核心概念

### `runs_per_config` 的语义

**定义**: 该配置项运行的次数

**常见误解**: ❌ "每个参数运行N次"
**正确理解**: ✅ "这个配置项运行N次"

### `mutate_params` 的语义

**定义**: 每次运行时同时变异的参数列表

**重要规则**:
- 列表中的所有参数会在**每次运行时同时变异**
- 如果想要"每个参数独立运行N次"，需要创建多个配置项

---

## ⚠️ 常见错误与正确示例

### 错误示例 1：多参数配置项（最常见错误）

```json
{
  "repo": "VulBERTa",
  "model": "mlp",
  "runs_per_config": 7,
  "mutate_params": ["epochs", "learning_rate", "seed", "weight_decay"]
}
```

**问题分析**:
- ❌ 预期：4参数 × 7次 = 28个实验
- ✅ 实际：7个实验（每个实验同时变异4个参数）
- ⚠️ 结果：缺失75%的实验，且无法分析单参数影响

**实际行为**:
```python
# generate_mutations(num_mutations=7, mutate_params=[4个参数])
# 生成7个mutations，每个mutation长这样：
{
  "epochs": 15,           # 同时变异
  "learning_rate": 0.001,  # 同时变异
  "seed": 4287,           # 同时变异
  "weight_decay": 0.0001  # 同时变异
}
```

### 正确示例 1：单参数配置项（推荐）

```json
[
  {
    "repo": "VulBERTa",
    "model": "mlp",
    "runs_per_config": 7,
    "mutate_params": ["epochs"],  // ✅ 只包含1个参数
    "comment": "仅变异epochs参数"
  },
  {
    "repo": "VulBERTa",
    "model": "mlp",
    "runs_per_config": 7,
    "mutate_params": ["learning_rate"],  // ✅ 另一个参数独立配置
    "comment": "仅变异learning_rate参数"
  },
  {
    "repo": "VulBERTa",
    "model": "mlp",
    "runs_per_config": 7,
    "mutate_params": ["seed"],
    "comment": "仅变异seed参数"
  },
  {
    "repo": "VulBERTa",
    "model": "mlp",
    "runs_per_config": 7,
    "mutate_params": ["weight_decay"],
    "comment": "仅变异weight_decay参数"
  }
]
```

**结果**:
- ✅ 预期：4个配置项 × 7次/配置 = 28个实验
- ✅ 实际：28个实验（每个参数独立7次）
- ✅ 可以分析每个参数对能耗/性能的独立影响

### 错误示例 2：键名错误

```json
{
  "repository": "examples",  // ❌ 错误：应为 "repo"
  "model": "mnist",
  "mutate_params": ["epochs"]
}
```

**问题**: `runner.py`会报 `KeyError: 'repo'`

**正确写法**:
```json
{
  "repo": "examples",  // ✅ 正确
  "model": "mnist",
  "mutate_params": ["epochs"]
}
```

### 错误示例 3：并行模式结构错误

```json
{
  "mode": "parallel",
  "repo": "examples",  // ❌ 错误：并行模式不直接指定repo
  "model": "mnist",
  "mutate_params": ["epochs"]
}
```

**正确写法**:
```json
{
  "mode": "parallel",
  "foreground": {  // ✅ 使用 foreground/background 结构
    "repo": "examples",
    "model": "mnist",
    "mode": "mutation",
    "mutate": ["epochs"]  // ✅ 并行模式使用 "mutate"
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {}
  },
  "runs_per_config": 7
}
```

---

## 📋 配置模板

### 模板1：非并行单参数变异（最常用）

```json
{
  "comment": "阶段描述 - 模型列表",
  "version": "4.7.1",
  "created": "YYYY-MM-DD",
  "estimated_duration_hours": XX.X,
  "estimated_experiments": XX,
  "target_completion": {
    "parameters": [
      "repo/model: N参数 × M次"
    ],
    "goal": "实验目标描述"
  },
  "use_deduplication": true,
  "historical_csvs": ["results/summary_all.csv"],
  "experiments": [
    {
      "repo": "repository_name",
      "model": "model_name",
      "baseline_runs": 0,
      "runs_per_config": 7,
      "mutate_params": ["parameter_name"],  // ⭐ 只包含1个参数
      "mode": "nonparallel",
      "comment": "仅变异parameter_name参数"
    }
  ]
}
```

### 模板2：并行模式单参数变异

```json
{
  "mode": "parallel",
  "foreground": {
    "repo": "examples",
    "model": "mnist",
    "mode": "mutation",
    "mutate": ["epochs"],  // ⭐ 只包含1个参数
    "comment": "仅变异epochs参数"
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {}
  },
  "runs_per_config": 5
}
```

### 模板3：默认模式（指定超参数值）

```json
{
  "repo": "examples",
  "model": "mnist",
  "mode": "default",
  "hyperparameters": {
    "epochs": 10,
    "learning_rate": 0.01
  },
  "runs_per_config": 3
}
```

---

## ✅ 最佳实践

### 1. 单参数原则（Single Parameter Principle）

**规则**: 每个配置项只变异一个参数

**理由**:
- ✅ 可以分析单个参数的独立影响
- ✅ 符合控制变量法的科学实验设计
- ✅ 便于理解和维护
- ✅ 避免混合变异带来的分析困难

**例外情况**:
- 如果明确需要研究参数之间的交互效应（极少见）
- 在这种情况下，应在comment中明确说明意图

### 2. 配置项命名规范

```json
{
  "comment": "仅变异parameter_name参数"  // ⭐ 明确说明意图
}
```

### 3. 估算实验数公式

**非并行模式**:
```
estimated_experiments = sum(每个配置项的runs_per_config)
```

**示例**:
```json
{
  "estimated_experiments": 28,  // 4个配置项 × 7次/配置 = 28
  "experiments": [
    {"runs_per_config": 7},  // 配置1
    {"runs_per_config": 7},  // 配置2
    {"runs_per_config": 7},  // 配置3
    {"runs_per_config": 7}   // 配置4
  ]
}
```

### 4. 去重配置最佳实践

```json
{
  "use_deduplication": true,  // ⭐ 始终启用
  "historical_csvs": ["results/summary_all.csv"]  // ⭐ 包含历史数据
}
```

**注意**:
- 去重基于超参数值+模式（parallel/nonparallel）
- 混合变异实验不会阻止单参数实验（参数值不同）
- runs_per_config应略大于目标唯一值数量（考虑去重）

### 5. 版本控制

```json
{
  "version": "4.7.1",
  "created": "2025-12-07",
  "comment": "Stage8: ... [已修复: 拆分多参数配置项]"
}
```

**建议**:
- 每次修改配置文件时更新version
- 在comment中记录重要变更
- 使用Git管理配置文件历史

---

## 📝 验证清单

在提交配置文件前，检查以下项目：

### 必须项 ✓

- [ ] 所有`mutate_params`/`mutate`只包含1个参数（除非明确需要多参数）
- [ ] 使用正确的键名：`"repo"`（不是`"repository"`）
- [ ] 并行模式使用`foreground`/`background`结构
- [ ] `estimated_experiments` = sum(runs_per_config)
- [ ] 启用去重：`"use_deduplication": true`
- [ ] 包含历史数据：`"historical_csvs": ["results/summary_all.csv"]`

### 推荐项 ⭐

- [ ] 每个配置项有清晰的`comment`
- [ ] 顶层有`version`和`created`字段
- [ ] 有`target_completion`说明实验目标
- [ ] JSON格式有效（使用`python -m json.tool`验证）

### 高级检查 🔍

- [ ] 计算总预估时间是否合理
- [ ] 检查模型名称和参数名称拼写
- [ ] 考虑去重率，runs_per_config是否足够
- [ ] 检查是否与现有实验重复

---

## 🐛 故障排查

### 问题1：实验数量远少于预期

**症状**: 预期200个实验，实际只有50个

**可能原因**: 多参数配置项

**检查方法**:
```bash
python3 -c "
import json
with open('settings/your_config.json') as f:
    cfg = json.load(f)
for exp in cfg['experiments']:
    params = exp.get('mutate_params', exp.get('foreground', {}).get('mutate', []))
    if len(params) > 1:
        print(f'⚠️  多参数配置: {exp.get(\"repo\")}/{exp.get(\"model\")} - {params}')
"
```

**解决方案**: 使用`scripts/fix_stage_configs.py`自动拆分

### 问题2：KeyError: 'repo'

**症状**: 运行时报错`KeyError: 'repo'`

**原因**: 使用了错误的键名`"repository"`

**解决方案**: 全局替换`"repository"` → `"repo"`

### 问题3：去重率过高/过低

**症状**:
- 过高：runs_per_config=10，只生成1个实验
- 过低：runs_per_config=5，生成5个实验但都是重复

**原因**:
- 过高：历史数据已包含该参数的所有值
- 过低：参数范围太窄或random seed问题

**解决方案**:
- 检查历史数据中该参数的唯一值数量
- 适当增加runs_per_config（推荐：目标值+2）
- 检查参数范围配置是否合理

### 问题4：estimated_experiments不匹配

**症状**: 配置文件中的estimated_experiments与实际运行数量不符

**检查脚本**:
```python
import json

with open('settings/your_config.json') as f:
    cfg = json.load(f)

actual = sum(exp.get('runs_per_config', cfg.get('runs_per_config', 1))
             for exp in cfg['experiments'])
estimated = cfg.get('estimated_experiments', 0)

if actual != estimated:
    print(f'⚠️  不匹配: estimated={estimated}, actual={actual}')
else:
    print(f'✓ 匹配: {actual}个实验')
```

---

## 📚 参考示例

### 优秀配置示例

1. **Stage2** (`settings/stage2_optimized_nonparallel_and_fast_parallel.json`)
   - ✅ 所有配置项都是单参数
   - ✅ 注释清晰
   - ✅ 估算准确

2. **Stage14** (`settings/stage14_stage7_8_supplement.json`)
   - ✅ 包含详细的分析数据
   - ✅ 明确说明补充原因
   - ✅ 完整的completion_status

### 错误配置示例（已归档）

1. **Stage7 v4.6.0** (`settings/stage7_nonparallel_fast_models.json.bak`)
   - ❌ 多参数配置项
   - ❌ 导致75.9%实验缺失

2. **Stage8 v4.6.0** (`settings/stage8_nonparallel_medium_slow_models.json.bak`)
   - ❌ 多参数配置项
   - ❌ 导致75.0%实验缺失

---

## 🔗 相关文档

- [实验配置指南](SETTINGS_CONFIGURATION_GUIDE.md) - JSON字段详细说明
- [Stage7-13配置Bug分析](../results_reports/STAGE7_13_CONFIG_BUG_ANALYSIS.md) - 问题详细分析
- [模型配置参考](../../config/models_config.json) - 支持的模型和参数

---

## 📞 获取帮助

如果遇到配置问题：

1. 查阅本文档的"故障排查"章节
2. 使用`scripts/fix_stage_configs.py --dry-run`检查配置
3. 参考已验证的优秀配置示例（Stage2）
4. 在配置文件中添加详细的comment说明意图

---

**维护者**: Green
**文档版本**: 1.0
**最后更新**: 2025-12-07

> **重要提示**: 在创建新配置文件时，请始终遵循"单参数原则"。如果不确定，参考Stage2配置作为模板。
