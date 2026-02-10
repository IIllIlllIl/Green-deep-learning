# DiBS步数验证测试 - 执行总结

**创建日期**: 2026-02-10
**任务**: 创建DiBS步数验证测试方案
**状态**: ✅ 完成

---

## 创建的文件

### 1. 测试脚本（analysis/tests/）

| 文件 | 行数 | 用途 |
|------|------|------|
| `test_dibs_step_verification.py` | 375 | 主测试脚本（支持callback监控） |
| `test_dibs_quick.py` | 60 | 快速验证脚本（100步，<30秒） |
| `run_dibs_step_test.sh` | 48 | Shell包装脚本（可配置参数） |
| `README_TESTS.md` | 45 | 测试索引 |

### 2. 核心修改（analysis/utils/）

| 文件 | 修改内容 | 新增行数 |
|------|---------|---------|
| `causal_discovery.py` | 添加`fit_with_callback()`方法 | ~140行 |

### 3. 文档（analysis/docs/technical_reference/）

| 文件 | 行数 | 用途 |
|------|------|------|
| `DIBS_STEP_VERIFICATION_TEST.md` | 360+ | 完整测试方案文档 |

---

## 核心组件

### DiBSStepMonitor（监控器）

**位置**: `tests/test_dibs_step_verification.py`
**功能**: 通过callback监��DiBS训练步数

```python
class DiBSStepMonitor:
    def __init__(self, expected_steps, verbose=True)
    def callback(self, step, *args, **kwargs)  # DiBS回调
    def verify(self) -> dict                    # 验证结果
```

### fit_with_callback方法

**位置**: `utils/causal_discovery.py`
**功能**: 扩展CausalGraphLearner，支持callback监控

```python
def fit_with_callback(self, data, callback, callback_every=10, verbose=True):
    # 与fit()相同，但启用callback监控
```

---

## 使用方法

### 快速验证（推荐）

```bash
cd /home/green/energy_dl/nightly/analysis
conda activate causal-research
python3 tests/test_dibs_quick.py
```

**配置**: 100步，10粒子，预期时间 < 30秒

### 标准验证

```bash
./tests/run_dibs_step_test.sh 1000 20
```

**配置**: 1000步，20粒子，预期时间 ~5分钟

### 自定义测试

```bash
# 500步，15粒子，callback间隔50
./tests/run_dibs_step_test.sh 500 15 50
```

---

## 测试设计

### 测试策略

1. **最小化测试数据**
   - 使用最小的组（group5_mrt_oast，60样本）
   - 快速迭代，快速验证

2. **多层级测试**
   - 快速验证（100步）：< 30秒
   - 标准验证（1000步）：~5分钟
   - 生产配置（13000步）：参考基准

3. **监控机制**
   - 通过callback记录每步时间戳
   - 验证实际步数是否符合预期
   - 计算性能指标

### 验证指标

```python
result = {
    'expected_steps': 1000,      # 预期步数
    'actual_steps': 1000,        # 实际步数（通过callback记录）
    'match': True,               # 是否匹配
    'discrepancy': 0,            # 步数差异
    'total_time': 125.3,         # 总训练时间
    'avg_time_per_step': 0.125   # 平均每步时间
}
```

---

## 关键修改

### causal_discovery.py修改

**添加方法**: `fit_with_callback()`

**位置**: 第210行之后

**核心代码**:

```python
# 启用callback（关键修改）
sample_result = self.model.sample(
    key=subk,
    n_particles=self.n_particles,
    steps=self.n_steps,
    callback_every=callback_every,  # ✅ 修改：从None改为参数
    callback=callback               # ✅ 修改：从None改为参数
)
```

**vs. 原始fit()方法**:

```python
# 原始代码（无监控）
sample_result = self.model.sample(
    key=subk,
    n_particles=self.n_particles,
    steps=self.n_steps,
    callback_every=None,  # ❌ 问题：无法监控
    callback=None         # ❌ 问题：无法监控
)
```

---

## 预期行为

### 如果测试通过

```
步数监控: 0/100 (0.0%) - 已用时间: 0.0s
步数监控: 100/100 (100.0%) - 已用时间: 12.3s

验证结果:
  预期步数: 100
  实际步数: 100
  匹配状态: ✅ 通过

✅ 快速测试通过
```

### 如果测试失败

```
验证结果:
  预期步数: 100
  实际步数: 10   ⚠️ 实际只执行了10步
  匹配状态: ❌ 失败
  步数差异: 90

❌ 快速测试失败
```

---

## 下一步行动

### 1. 运行快速验证（立即）

```bash
cd /home/green/energy_dl/nightly/analysis
conda activate causal-research
python3 tests/test_dibs_quick.py
```

**预期时间**: < 30秒

### 2. 根据结果决策

#### 如果测试通过

✅ DiBS正常执行设定步数
✅ 可以继续进行完整训练（n_steps=45000）
✅ 当前配置安全

#### 如果测试失败

❌ 需要检查DiBS库配置
❌ 需要检查callback机制
❌ 可能需要修改`causal_discovery.py`的sample()调用

---

## 性能基准

基于group5_mrt_oast（60样本，24特征）：

| 配置 | 步数 | 粒子数 | 预期时间 |
|------|------|--------|---------|
| 快速 | 100 | 10 | ~30秒 |
| 标准 | 1000 | 20 | ~5分钟 |
| 参考 | 13000 | 50 | ~198秒 |

**时间估算公式**:
```
时间 ≈ (步数 / 13000) × (粒子数 / 50) × 198秒
```

---

## 技术细节

### DiBS callback机制

DiBS的`sample()`方法支持callback监控：

```python
dibs.sample(
    key=key,
    n_particles=50,
    steps=1000,
    callback_every=10,    # 每隔10步调用
    callback=my_function  # callback(step, *args, **kwargs)
)
```

**关键参数**:
- `callback_every`: 调用间隔（步数）
- `callback`: 回调函数，签名为`callback(step, *args, **kwargs)`

**注意事项**:
- `callback_every=None`会导致callback不执行
- `callback=None`会导致无法监控
- callback会增加少量开销（<5%）

---

## 文件清单

```
analysis/
├── tests/
│   ├── test_dibs_step_verification.py  # ✅ 新增：主测试脚本
│   ├── test_dibs_quick.py              # ✅ 新增：快速验证
│   ├── run_dibs_step_test.sh           # ✅ 新增：Shell脚本
│   └── README_TESTS.md                 # ✅ 新增：测试索引
├── utils/
│   └── causal_discovery.py             # ✅ 修改：添加fit_with_callback()
└── docs/technical_reference/
    ├── DIBS_STEP_VERIFICATION_TEST.md  # ✅ 新增：完整文档
    └── DIBS_STEP_TEST_SUMMARY.md       # ✅ 新增：本文档
```

---

## 维护者

- **创建**: 2026-02-10
- **目的**: 验证DiBS训练步数
- **状态**: ✅ 完成，等待测试验证

---

## 相关文档

- **完整测试文档**: `docs/technical_reference/DIBS_STEP_VERIFICATION_TEST.md`
- **DiBS主脚本**: `scripts/run_dibs_6groups_global_std.py`
- **因果发现模块**: `utils/causal_discovery.py`
- **参数调优文档**: `docs/technical_reference/DIBS_PARAMETER_TUNING_ANALYSIS.md`
