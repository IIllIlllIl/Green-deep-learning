# DiBS因果边白名单设计方案

**版本**: v1.1
**日期**: 2026-01-17
**目的**: 定义合理的因果边类型，筛选出符合研究逻辑的因果关系
**更新**: 添加 mediator → performance 规则

---

## 1. 背景

从DiBS因果发现结果中，我们得到了大量的因果边（每组约529条边）。但并非所有边都是合理的，例如：

**不合理的边**:
- `性能 → 超参数` (第55行: test_accuracy → learning_rate) - 反因果方向
- `能耗 → 超参数` (第72行: gpu_temp_avg_celsius → learning_rate) - 结果影响原因
- `模型控制变量 → 超参数` (第81行: model_mnist_rnn → learning_rate) - 实验设计变量不应影响超参数

**合理的边**:
- `超参数 → 中间变量 → 能耗` (第32行: cpu_total_joules → cpu_pkg_joules) - 符合能耗生成机制
- `超参数 → 性能` (第58行: batch_size → test_accuracy) - 主效应
- `交互项 → 能耗` (第66行: batch_size_x_parallel → cpu_total_joules) - 调节效应

---

## 2. 变量分类回顾

根据CSV文件的 `source_category` 和 `target_category` 字段，我们有以下变量类型：

| 变量类型 | 说明 | 示例 |
|---------|------|------|
| `hyperparam` | 超参数 | batch_size, learning_rate, epochs, seed |
| `interaction` | 交互项（超参数×并行模式） | batch_size_x_is_parallel, epochs_x_is_parallel |
| `energy` | 能耗指标 | cpu_total_joules, gpu_total_joules, cpu_pkg_joules, cpu_ram_joules |
| `mediator` | 中间变量（功率、温度、利用率） | gpu_avg_watts, gpu_max_watts, gpu_temp_avg_celsius, gpu_util_avg_percent |
| `performance` | 性能指标 | test_accuracy |
| `control` | 模型控制变量 | model_mnist_ff, model_mnist_rnn, model_siamese |
| `mode` | 并行模式 | is_parallel |

---

## 3. 因果方向合理性分析

### 3.1 研究问题驱动的因果方向

我们的研究问题是：
- **Q1**: 超参数对能耗的影响
- **Q2**: 能耗和性能的权衡
- **Q3**: 中间变量的中介效应

**合理的因果链**:
```
超参数 → 中间变量 → 能耗
超参数 → 性能
交互项 → 能耗/性能  (调节效应)
```

### 3.2 不合理的因果方向

基于**时间先后**和**实验设计**，以下方向是不合理的：

1. **结果变量不能影响原因变量**
   - ❌ 能耗 → 超参数
   - ❌ 性能 → 超参数
   - ❌ 中间变量 → 超参数

2. **实验设计变量不能被影响**
   - ❌ 任何变量 → 模型控制变量
   - ❌ 任何变量 → 并行模式
   - ❌ 任何变量 → 超参数（除交互项）

3. **反直觉的因果关系**
   - ❌ 性能 → 能耗 (虽然可能存在相关，但因果方向应该是能耗配置影响性能)
   - ❌ 中间变量 → 超参数

4. **二元变量不应有自循环**
   - ❌ mode → mode (is_parallel是固定的实验设计变量)

---

## 4. 白名单定义

### 4.1 白名单规则（基于 source_category → target_category）

我们定义以下因果边为**合理**的：

#### **规则组1: 超参数的主效应**
| Source | Target | 合理性 | 研究问题 |
|--------|--------|--------|---------|
| hyperparam | energy | ✅ 合理 | Q1 - 主效应 |
| hyperparam | mediator | ✅ 合理 | Q1 - 通过中间变量 |
| hyperparam | performance | ✅ 合理 | Q2 - 性能影响 |

#### **规则组2: 交互项的调节效应**
| Source | Target | 合理性 | 研究问题 |
|--------|--------|--------|---------|
| interaction | energy | ✅ 合理 | Q1 - 调节效应 |
| interaction | mediator | ✅ 合理 | Q1 - 调节中间变量 |
| interaction | performance | ✅ 合理 | Q2 - 调节性能 |

#### **规则组3: 中间变量的中介效应**
| Source | Target | 合理性 | 研究问题 |
|--------|--------|--------|---------|
| mediator | energy | ✅ 合理 | Q3 - 中介效应（如 watts → joules） |
| mediator | mediator | ✅ 合理 | Q3 - 中间变量链（如 temp → watts） |
| mediator | performance | ✅ 合理 | Q2/Q3 - 中间变量影响性能（见注1） |
| energy | energy | ✅ 合理 | Q3 - 能耗分解（如 cpu_total → cpu_pkg） |

**注1**: `mediator → performance` 的研究逻辑说明
- **RQ2研究需求**: 参照论文方法，我们需要识别"直接或间接的超参数→性能因果链"
- **间接路径**: hyperparam → mediator → performance（通过中间变量如GPU功率、温度影响性能）
- **直接路径**: hyperparam → performance（超参数直接影响性能）
- **路径污染预防**: 白名单明确禁止 `energy → mediator`（第165行），因此不会产生 `energy → mediator → performance` 路径
- **研究价值**: 识别中间变量（如GPU温度过高）如何间接影响模型性能

#### **规则组4: 控制变量的影响**
| Source | Target | 合理性 | 研究问题 |
|--------|--------|--------|---------|
| control | energy | ✅ 合理 | 控制变量对能耗的影响 |
| control | mediator | ✅ 合理 | 控制变量对中间变量的影响 |
| control | performance | ✅ 合理 | 控制变量对性能的影响 |
| mode | energy | ✅ 合理 | 并行模式的主效应 |
| mode | mediator | ✅ 合理 | 并行模式对中间变量的影响 |
| mode | performance | ✅ 合理 | 并行模式对性能的影响 |

### 4.2 黑名单规则（不合理的因果方向）

以下因果边被**明确排除**：

#### **黑名单组1: 结果变量 → 原因变量（反因果）**
| Source | Target | 原因 |
|--------|--------|------|
| energy | hyperparam | ❌ 能耗不能改变超参数 |
| performance | hyperparam | ❌ 性能不能改变超参数 |
| mediator | hyperparam | ❌ 中间变量不能改变超参数 |

#### **黑名单组2: 任意变量 → 实验设计变量**
| Source | Target | 原因 |
|--------|--------|------|
| energy | control | ❌ 能耗不能改变模型选择 |
| performance | control | ❌ 性能不能改变模型选择 |
| mediator | control | ❌ 中间变量不能改变模型选择 |
| energy | mode | ❌ 能耗不能改变并行模式 |
| performance | mode | ❌ 性能不能改变并行模式 |
| mediator | mode | ❌ 中间变量不能改变并行模式 |

#### **黑名单组3: 自循环和无意义的边**
| Source | Target | 原因 |
|--------|--------|------|
| X | X | ❌ 自循环（变量对自身的影响） |
| hyperparam | hyperparam | ❌ 超参数之间无因果（独立设定） |

#### **黑名单组4: 反直觉的因果关系**
| Source | Target | 原因 |
|--------|--------|------|
| performance | energy | ❌ 性能不应影响能耗（应该是能耗配置影响性能） |
| performance | mediator | ❌ 性能不应影响中间变量（应该是中间变量影响性能） |

---

## 5. 白名单完整列表

### 5.1 白名单矩阵

以下是**允许的**因果边类型（source → target）：

|  | hyperparam | interaction | energy | mediator | performance | control | mode |
|---|------------|-------------|--------|----------|-------------|---------|------|
| **hyperparam** | ❌ | ⚠️ | ✅ Q1 | ✅ Q1 | ✅ Q2 | ❌ | ❌ |
| **interaction** | ⚠️ | ⚠️ | ✅ Q1 | ✅ Q1 | ✅ Q2 | ❌ | ❌ |
| **energy** | ❌ | ❌ | ✅ Q3 | ❌ | ❌ | ❌ | ❌ |
| **mediator** | ❌ | ❌ | ✅ Q3 | ✅ Q3 | ✅ Q2/Q3 | ❌ | ❌ |
| **performance** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **control** | ❌ | ❌ | ✅ | ✅ | ✅ | ⚠️ | ❌ |
| **mode** | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |

**图例**:
- ✅ = 允许（合理的因果方向）
- ❌ = 禁止（不合理的因果方向）
- ⚠️ = 特殊情况（需要额外判断）
- Q1/Q2/Q3 = 对应的研究问题

### 5.2 特殊情况说明

#### ⚠️ `hyperparam → interaction`
- **示例**: learning_rate → learning_rate_x_is_parallel
- **判断**: 通常不合理（交互项由超参数和mode派生）
- **例外**: 如果交互项定义为独立变量，可能合理
- **建议**: **禁止**（避免混淆）

#### ⚠️ `interaction → hyperparam`
- **示例**: batch_size_x_is_parallel → batch_size
- **判断**: 不合理（交互项不能改变基础超参数）
- **建议**: **禁止**

#### ⚠️ `interaction → interaction`
- **示例**: batch_size_x_parallel → epochs_x_parallel
- **判断**: 通常不合理（交互项应该是独立的调节效应）
- **建议**: **禁止**

#### ⚠️ `control → control`
- **示例**: model_mnist_ff → model_mnist_rnn
- **判断**: 不合理（模型选择是独立的实验设计变量）
- **建议**: **禁止**

---

## 6. 过滤后预期结果

### 6.1 预期保留边数量估算

根据 group1_examples 的数据样本（前100行）：

| 规则组 | 边类型 | 原始数量（估算） | 预期保留比例 |
|--------|--------|----------------|------------|
| 规则组1 | hyperparam → energy/mediator/performance | ~30条 | 100% 保留 |
| 规则组2 | interaction → energy/mediator/performance | ~50条 | 100% 保留 |
| 规则组3 | mediator → energy/mediator, energy → energy | ~80条 | 100% 保留 |
| 规则组4 | control/mode → energy/mediator/performance | ~40条 | 100% 保留 |
| **黑名单** | 反因果方向、自循环等 | ~329条 | **0% 保留** |

**预期**:
- 原始边数: 529条/组
- 白名单过滤后: **~210条/组** (约40%保留)
- 主要过滤: energy/performance/mediator → hyperparam, 自循环, control/mode作为target

### 6.2 示例：被过滤的边

| 行号 | Source | Target | 原因 |
|------|--------|--------|------|
| 55 | test_accuracy | learning_rate | 性能不能改变超参数 |
| 72 | gpu_temp_avg_celsius | learning_rate | 中间变量不能改变超参数 |
| 113 | test_accuracy | model_mnist_ff | 性能不能改变模型选择 |
| 373 | epochs_x_parallel | epochs_x_parallel | 自循环 |

### 6.3 示例：保留的边

| 行号 | Source | Target | 原因 |
|------|--------|--------|------|
| 58 | batch_size | test_accuracy | 超参数 → 性能（Q2主效应） |
| 66 | batch_size_x_parallel | cpu_total_joules | 交互项 → 能耗（Q1调节效应） |
| 32 | cpu_total_joules | cpu_pkg_joules | 能耗 → 能耗（Q3中介效应） |
| 39 | gpu_temp_avg_celsius | gpu_total_joules | 中间变量 → 能耗（Q3中介效应） |

---

## 7. 实现建议

### 7.1 过滤函数签名

```python
def filter_causal_edges_by_whitelist(
    edges_df: pd.DataFrame
) -> pd.DataFrame:
    """
    根据白名单规则过滤因果边

    参数:
        edges_df: 包含 source_category, target_category 列的DataFrame

    返回:
        过滤后的DataFrame（仅包含合理的因果边）
    """
    pass
```

### 7.2 白名单规则编码

```python
WHITELIST_RULES = {
    # 规则组1: 超参数主效应
    ('hyperparam', 'energy'): True,
    ('hyperparam', 'mediator'): True,
    ('hyperparam', 'performance'): True,

    # 规则组2: 交互项调节效应
    ('interaction', 'energy'): True,
    ('interaction', 'mediator'): True,
    ('interaction', 'performance'): True,

    # 规则组3: 中间变量中介效应
    ('mediator', 'energy'): True,
    ('mediator', 'mediator'): True,
    ('energy', 'energy'): True,

    # 规则组4: 控制变量影响
    ('control', 'energy'): True,
    ('control', 'mediator'): True,
    ('control', 'performance'): True,
    ('mode', 'energy'): True,
    ('mode', 'mediator'): True,
    ('mode', 'performance'): True,

    # 注意: 白名单矩阵中的⚠️在实现中都视为False（禁止）
    # 理由参见第5.2节"特殊情况说明"
    # 其他所有组合默认为 False（黑名单）
}

def is_edge_allowed(source_cat: str, target_cat: str) -> bool:
    """检查因果边是否在白名单中"""
    return WHITELIST_RULES.get((source_cat, target_cat), False)
```

### 7.3 过滤流程

1. 读取 `*_causal_edges_all.csv` 或 `*_causal_edges_0.3.csv`
2. 应用白名单规则：`edges_df[edges_df.apply(lambda row: is_edge_allowed(row['source_category'], row['target_category']), axis=1)]`
3. 生成新文件：`*_causal_edges_whitelist.csv`

---

## 8. 验证计划

### 8.1 Subagent检查重点

请Subagent重点检查：

1. **白名单矩阵完整性**: 是否覆盖所有7×7=49种组合？
2. **研究问题对齐**: 白名单是否支持Q1/Q2/Q3的研究问题？
3. **逻辑一致性**:
   - 是否存在互相矛盾的规则？
   - 特殊情况的处理是否合理？
4. **遗漏情况**: 是否有合理的因果边被误判为黑名单？
5. **过度过滤**: 是否有黑名单边实际上是合理的？

### 8.2 验证数据

使用 `group1_examples_causal_edges_all.csv` 的前100行作为验证样本，检查：
- 哪些边被保留？
- 哪些边被过滤？
- 是否符合预期？

---

## 9. 下一步

1. ✅ 完成白名单设计文档
2. ⏳ **Subagent审查** ← 当前步骤
3. ⏳ 根据审查意见修正白名单
4. ⏳ 实现过滤脚本 `scripts/filter_causal_edges_by_whitelist.py`
5. ⏳ 在测试数据上验证
6. ⏳ 应用到所有6组数据
7. ⏳ 生成过滤报告

---

**维护者**: Claude
**文档版本**: v1.1
**最后更新**: 2026-01-17
**更新内容**:
- 添加 mediator → performance 到白名单（允许中间变量影响性能）
- 添加RQ2研究逻辑说明（注1），解释间接因果路径的研究价值
- 明确路径污染预防机制（禁止 energy → mediator）
