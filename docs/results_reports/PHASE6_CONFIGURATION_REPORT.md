# Phase 6 配置报告

**创建日期**: 2025-12-15
**版本**: v4.7.6-phase6
**目标**: VulBERTa/mlp完全补齐（非并行+并行模式）

---

## 执行摘要

**数据安全性验证**: ✅ 通过
- 总行数: 584
- 训练成功率: 86.0% (502/584)
- 能耗数据完整性: 83.6% (488/584)
- 性能数据完整性: 64.4% (376/584)
- 复合键唯一性: 100% (584/584)
- **结论**: raw_data.csv数据完整且可靠

**实验差距分析**: ✅ 完成
- 总缺口: 90个实验，56.98小时
- 主要缺口:
  - VulBERTa/mlp: 40个实验，41.72小时 ⭐
  - bug-localization: 40个实验，10.90小时
  - MRT-OAST: 10个实验，4.36小时
- 当前达标: 17/90 (18.9%)

**Phase 6配置**: ✅ 创建并验证
- **配置文件**: `settings/phase6_vulberta_mlp_completion.json`
- **实验数**: 40个
- **预计运行时间**: 41.72小时 ✅ (36-42h目标范围内)
- **目标**: VulBERTa/mlp模型100%达标

---

## 重要修复：calculate_experiment_gap.py

### 问题发现
原脚本在统计实验完成情况时，**未正确处理并行模式实验**的前景（foreground）数据：
- 只检查了`training_success`和`perf_*`字段（非并行模式）
- 未检查`fg_training_success`和`fg_perf_*`字段（并行模式）

### 修复内容
```python
# 修复前（错误）
training_success = row.get('training_success', '').lower() == 'true'
has_perf = any(v for k, v in row.items() if k.startswith('perf_') and v)

# 修复后（正确）
if mode == 'parallel':
    repo = row.get('fg_repository', '')
    model = row.get('fg_model', '')
    training_success = row.get('fg_training_success', '').lower() == 'true'
    has_perf = any(v for k, v in row.items() if k.startswith('fg_perf_') and v)
    hyperparam_prefix = 'fg_hyperparam_'
else:
    repo = row.get('repository', '')
    model = row.get('model', '')
    training_success = row.get('training_success', '').lower() == 'true'
    has_perf = any(v for k, v in row.items() if k.startswith('perf_') and v)
    hyperparam_prefix = 'hyperparam_'
```

### 影响
修复前的错误统计导致：
- 并行模式实验被低估：显示只有28个参数-模式组合达标
- 修复后显示正确结果：**17个参数-模式组合达标**
- 实验缺口从133个减少到90个

---

## Phase 6 配置详情

### 配置方案选择

**考虑的方案**:
1. **方案1**: VulBERTa/mlp完全补齐（40实验，41.72h）⭐ **选中**
2. 方案2: VulBERTa/mlp非并行 + bug-localization + MRT-OAST（70实验，36.12h）
3. 方案3: VulBERTa/mlp并行 + bug-localization + MRT-OAST（70实验，36.12h）

**选择方案1的理由**:
1. **集中火力**: 完成单个模型，达标模型数+1
2. **配置简单**: 仅40个实验，8个配置项
3. **运行时间**: 41.72小时，接近上限但在36-42h范围内
4. **重点模型**: VulBERTa/mlp是漏洞检测研究的核心模型

### 实验配置

| 模式 | 参数 | 实验数 | 运行时间 |
|------|------|--------|----------|
| 非并行 | epochs | 5 | 5.22h |
| 非并行 | learning_rate | 5 | 5.22h |
| 非并行 | seed | 5 | 5.22h |
| 非并行 | weight_decay | 5 | 5.22h |
| 并行 | epochs | 5 | 5.22h |
| 并行 | learning_rate | 5 | 5.22h |
| 并行 | seed | 5 | 5.22h |
| 并行 | weight_decay | 5 | 5.22h |
| **总计** | **8个配置** | **40个** | **41.72h** |

### 背景模型
- 并行模式使用`examples/mnist_rnn`作为背景负载
- 轻量级模型，不干扰前景训练

---

## 预期成果

### 完成后数据质量
- VulBERTa/mlp非并行: 5/5参数达标（0→100%）
- VulBERTa/mlp并行: 5/5参数达标（0→100%）
- VulBERTa/mlp模型: **完全达标** ✅

### 项目进度提升
| 指标 | 当前 | 完成后 | 增长 |
|------|------|--------|------|
| 非并行模式达标 | 9/11 | 10/11 | +1 |
| 并行模式达标 | 8/11 | 9/11 | +1 |
| 总参数-模式组合达标 | 17/90 | 19/90 | +2 |
| 达标率 | 18.9% | 21.1% | +2.2% |

### 剩余工作
完成Phase 6后，剩余缺口：
- bug-localization: 40个实验，10.90小时
- MRT-OAST: 10个实验，4.36小时
- **总计**: 50个实验，15.26小时

---

## 执行命令

```bash
# Phase 6执行命令
sudo -E python3 mutation.py -ec settings/phase6_vulberta_mlp_completion.json

# 预计开始时间: 用户决定
# 预计完成时间: 开始后41.72小时
# 预计生成数据: 40行新实验数据（raw_data.csv: 584→624）
```

---

## 配置文件验证

### JSON格式验证
✅ 通过 - 配置文件语法正确

### 实验数验证
✅ 通过 - 预期40个实验，计算得40个实验

### 运行时间验证
✅ 通过 - 41.72小时，在36-42h目标范围内

### 去重设置验证
✅ 启用 - 使用`data/raw_data.csv`进行历史去重

### 配置标准验证
✅ 通过 - 符合JSON配置书写规范:
- 使用`"repo"`而非`"repository"`
- 使用`"mutate": ["参数"]`单参数变异格式
- 并行模式使用`foreground/background`结构
- 每个实验只变异一个参数

---

## 文件清单

### 新增文件
1. `settings/phase6_vulberta_mlp_completion.json` - Phase 6配置文件
2. `docs/results_reports/PHASE6_CONFIGURATION_REPORT.md` - 本报告

### 修改文件
1. `scripts/calculate_experiment_gap.py` - 修复并行模式数据判断逻辑

### 备份文件
1. `data/raw_data.csv.backup_before_fix_20251215_183412` - 修复前备份

---

## 注意事项

### 1. 数据安全
- ✅ 已创建修复前备份
- ✅ 复合键唯一性验证通过
- ✅ 所有31行不匹配数据已修复

### 2. 去重机制
- ✅ 配置启用去重功能
- ✅ 使用raw_data.csv作为历史数据源
- ✅ 基于复合键（experiment_id + timestamp）去重

### 3. 运行环境
- 需要sudo权限（能耗监控）
- 需要GPU（CUDA环境）
- 预留充足磁盘空间（~40GB for 40实验）

---

**报告生成**: 2025-12-15
**下一步**: 执行Phase 6配置，补齐VulBERTa/mlp模型
