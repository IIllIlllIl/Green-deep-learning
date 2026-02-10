# DiBS步数验证测试方案

**创建日期**: 2026-02-10
**目的**: 验证DiBS是否真的执行了设定的训练步数
**背景**: 之前出现过callback设置导致未执行设定步数的问题

---

## 问题描述

在当前的DiBS训练中，`causal_discovery.py`使用了以下设置：

```python
sample_result = self.model.sample(
    key=subk,
    n_particles=self.n_particles,
    steps=self.n_steps,
    callback_every=None,  # ⚠️ 问题：设置为None
    callback=None         # ⚠️ 问题：设置为None
)
```

这可能导致DiBS没有执行设定的步数，或者无法监控实际执行的步数。

---

## 测试方案

### 方案架构

```
测试层次：
1. 快速验证（100步，10粒子，<30秒）
   └── test_dibs_quick.py

2. 标准验证（1000步，20粒子，~5分钟）
   └── test_dibs_step_verification.py

3. 监控机制
   └── DiBSStepMonitor类（callback监控）
   └── fit_with_callback方法（支持callback）
```

### 核心组件

#### 1. DiBSStepMonitor（监控器）

**位置**: `tests/test_dibs_step_verification.py`
**功能**:
- 通过callback机制监控DiBS训练步数
- 记录每步时间戳
- 验证实际步数是否符合预期

**核心方法**:
```python
monitor = DiBSStepMonitor(expected_steps=1000, verbose=True)
monitor.callback(step, *args, **kwargs)  # DiBS回调
result = monitor.verify()  # 验证结果
```

#### 2. fit_with_callback（扩展方法）

**位置**: `utils/causal_discovery.py`
**功能**:
- 扩展CausalGraphLearner，支持callback监控
- 与fit()方法功能相同，但启用callback

**用法**:
```python
learner = CausalGraphLearner(n_vars=24, n_steps=1000, ...)
graph = learner.fit_with_callback(
    data=data_df,
    callback=monitor.callback,
    callback_every=10,
    verbose=True
)
```

#### 3. 测试脚本

| 脚本 | 用途 | 配置 | 预期时间 |
|------|------|------|---------|
| `test_dibs_quick.py` | 快速验证 | 100步，10粒子 | <30秒 |
| `test_dibs_step_verification.py` | 标准验证 | 1000步，20粒子 | ~5分钟 |
| `run_dibs_step_test.sh` | Shell包装 | 可配置 | 可配置 |

---

## 使用方法

### 方法1: 快速验证（推荐首次使用）

```bash
# 切换到分析目录
cd /home/green/energy_dl/nightly/analysis

# 激活环境
conda activate causal-research

# 运行快速测试（100步，~30秒）
python3 tests/test_dibs_quick.py
```

**预期输出**:
```
✅ 快速测试通过
   DiBS正确执行了 100 步（预期 100 步）
```

### 方法2: 标准验证

```bash
# 使用shell脚本（推荐）
./tests/run_dibs_step_test.sh 1000 20

# 或直接运行Python脚本
python3 tests/test_dibs_step_verification.py --steps 1000 --particles 20
```

### 方法3: 自定义测试

```python
from tests.test_dibs_step_verification import test_dibs_with_callback

result = test_dibs_with_callback(
    n_steps=500,           # 自定义步数
    n_particles=15,        # 自定义粒子数
    callback_every=50,     # callback间隔
    verbose=True
)

if result['match']:
    print("✅ 测试通过")
else:
    print(f"❌ 测试失败: 预期{result['expected_steps']}步，实际{result['actual_steps']}步")
```

---

## 验证指标

### 测试结果字段

```python
result = {
    'expected_steps': 1000,      # 预期步数
    'actual_steps': 1000,        # 实际步数
    'match': True,               # 是否匹配
    'discrepancy': 0,            # 步数差异
    'total_time': 125.3,         # 总训练时间（秒）
    'avg_time_per_step': 0.125   # 平均每步时间（秒）
}
```

### 判定标准

- ✅ **通过**: `actual_steps == expected_steps` 且 `discrepancy == 0`
- ❌ **失败**: `actual_steps != expected_steps`

---

## 测试数据

**测试组**: `group5_mrt_oast`（最小的组）
- 样本数: 60
- 特征数: 24
- 数据文件: `analysis/data/energy_research/6groups_global_std/group5_mrt_oast_global_std.csv`

**选择理由**:
1. 样本量最小（60样本），训练速度最快
2. 具有完整的能耗和性能数据
3. 足够代表性（包含所有超参数类型）

---

## 预期行为

### 如果DiBS正常工作

```
步数监控: 0/1000 (0.0%) - 已用时间: 0.0s
步数监控: 100/1000 (10.0%) - 已用时间: 12.3s
步数监控: 200/1000 (20.0%) - 已用时间: 24.6s
...
步数监控: 1000/1000 (100.0%) - 已用时间: 123.4s

验证结果:
  预期步数: 1000
  实际步数: 1000
  匹配状态: ✅ 通过
  步数差异: 0
```

### 如果DiBS存在问题

```
验证结果:
  预期步数: 1000
  实际步数: 100    ⚠️ 实际只执行了100步
  匹配状态: ❌ 失败
  步数差异: 900
```

---

## 性能基准

基于group5_mrt_oast（60样本，24特征）的测试数据：

| 配置 | 步数 | 粒子数 | 预期时间 | 用途 |
|------|------|--------|---------|------|
| 最小 | 100 | 10 | ~30秒 | 快速验证 |
| 标准 | 1000 | 20 | ~5分钟 | 步数验证 |
| 参考 | 13000 | 50 | ~198秒 | 实际基准（来自日志） |

**注意**: 时间与样本数、特征数、粒子数、步数都相关。

---

## 后续行动

### 如果测试通过

1. ✅ 确认DiBS正常执行设定步数
2. ✅ 当前n_steps=45000的设置安全
3. ✅ 可以继续进行完整训练

### 如果测试失败

1. ❌ 需要检查DiBS库配置
2. ❌ 需要检查callback机制
3. ❌ 可能需要修改`causal_discovery.py`的sample()调用

---

## 文件清单

```
analysis/
├── tests/
│   ├── test_dibs_step_verification.py  # 主测试脚本（带callback）
│   ├── test_dibs_quick.py              # 快速验证脚本
│   └── run_dibs_step_test.sh           # Shell运行脚本
├── utils/
│   └── causal_discovery.py             # 修改：添加fit_with_callback()
└── docs/technical_reference/
    └── DIBS_STEP_VERIFICATION_TEST.md  # 本文档
```

---

## 技术细节

### DiBS callback机制

DiBS的`sample()`方法支持以下参数：

```python
dibs.sample(
    key=key,
    n_particles=50,
    steps=1000,
    callback_every=10,  # 每隔10步调用callback
    callback=my_function  # callback(step, *args, **kwargs)
)
```

**callback参数**:
- `step`: 当前训练步数
- `args/kwargs`: DiBS内部状态（可忽略）

**注意事项**:
- `callback_every=None`会导致callback不执行
- `callback=None`会导致无法监控
- callback会增加少量开销（<5%）

---

## 常见问题

### Q1: 为什么测试用最小的组？

A: 为了快速验证。group5只有60样本，训练速度最快，足够验证步数执行问题。

### Q2: callback会影响性能吗？

A: 会，但影响很小（<5%）。只在测试时使用，生产训练仍用`fit()`。

### Q3: 如果callback不工作怎么办？

A: 检查DiBS版本和安装。可能需要更新DiBS库或修改调用方式。

### Q4: 测试时间太长怎么办？

A: 使用快速测试（100步）或减少粒子数。

---

## 维护者

- **创建**: 2026-02-10
- **目的**: 验证DiBS训练步数
- **相关**: `run_dibs_6groups_global_std.py`, `causal_discovery.py`

---

## 更新日志

- 2026-02-10: 创建测试方案，添加fit_with_callback方法
