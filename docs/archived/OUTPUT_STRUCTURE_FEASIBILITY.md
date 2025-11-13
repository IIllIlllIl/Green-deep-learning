# 输出文件格式优化可行性分析报告

## 分析日期
2025-11-12

## 用户需求

### 需求1: 同超参数变异的文件组织
**要求**: 每个模型的同超参数变异的日志和JSON放到同一文件夹中

**示例**:
```
当前:
results/
├── training_pytorch_resnet_cifar10_resnet20_20251112_100000.log
├── 20251112_100000_pytorch_resnet_cifar10_resnet20.json
├── energy_20251112_100000_pytorch_resnet_cifar10_resnet20_attempt0/

期望:
results/
└── pytorch_resnet_cifar10_resnet20_20251112_100000/
    ├── training.log
    ├── result.json
    └── energy/
```

### 需求2: 每次运行的总文件夹
**要求**: 每一次运行mutation生成的所有模型训练结果放到同一总文件夹中

**示例**:
```
期望:
results/
└── run_20251112_100000/  # 总文件夹（一次mutation.py运行）
    ├── pytorch_resnet_cifar10_resnet20_mutation1/
    │   ├── training.log
    │   ├── result.json
    │   └── energy/
    ├── pytorch_resnet_cifar10_resnet20_mutation2/
    │   ├── training.log
    │   ├── result.json
    │   └── energy/
    └── summary.csv  # 总结CSV
```

### 需求3: CSV总结文件
**要求**: 在总文件夹中生成CSV文件，总结所有训练的模型、超参数、性能和能耗度量

**示例**:
```csv
experiment_id,repository,model,epochs,learning_rate,dropout,...,rank1,map,cpu_energy_joules,gpu_energy_joules,...
pytorch_resnet_cifar10_resnet20_mutation1,pytorch_resnet_cifar10,resnet20,100,0.001,0.5,...,0.92,0.87,80095.55,527217.33,...
```

---

## 当前文件结构分析

### 1. 文件命名规则

**训练日志** (`build_training_command()`, Line 343):
```python
log_file = f"results/training_{repo}_{model}_{timestamp}.log"
```
格式: `training_{repo}_{model}_{timestamp}.log`

**JSON结果** (`save_results()`, Line 678):
```python
result_file = self.results_dir / f"{experiment_id}.json"
```
格式: `{experiment_id}.json` (其中 `experiment_id = f"{timestamp}_{repo}_{model}"`)

**能耗目录** (`run_experiment()`, Line 932):
```python
energy_dir = f"results/energy_{experiment_id}_attempt{retries}"
```
格式: `energy_{experiment_id}_attempt{retries}/`

### 2. 实验ID生成

**单个实验** (`run_experiment()`, Line 910):
```python
experiment_id = f"{datetime.now().strftime(self.TIMESTAMP_FORMAT)}_{repo}_{model}"
```

**并行实验** (`run_parallel_experiment()`, Line 836):
```python
experiment_id = f"{datetime.now().strftime(self.TIMESTAMP_FORMAT)}_{fg_repo}_{fg_model}_parallel"
```

### 3. 当前文件分布

**问题**:
- 所有文件平铺在 `results/` 目录下
- 同一实验的日志、JSON、能耗数据分散在不同位置
- 缺少实验批次的概念（一次运行多个实验）
- 缺少汇总CSV文件

**统计** (从实际results目录):
```
results/
├── training_*.log               # 训练日志（分散）
├── *.json                        # JSON结果（分散）
├── energy_*_attempt*/            # 能耗数据（分散）
├── background_logs_*/            # 后台日志（分散）
└── (无总结CSV)
```

---

## 可行性分析

### ✅ 可行性评估: 高度可行

所有需求都可以实现，且不会破坏现有功能。

---

## 方案1: 分层目录结构（推荐）

### 目录结构设计

```
results/
└── run_20251112_150000/                    # Session目录（一次mutation.py完整运行）
    ├── summary.csv                         # 总结CSV（所有实验汇总）
    ├── session_info.json                   # Session元数据（可选）
    │
    ├── pytorch_resnet_cifar10_resnet20_001/  # 实验1目录
    │   ├── experiment.json                 # 实验结果JSON
    │   ├── training.log                    # 训练日志
    │   ├── hyperparameters.json            # 超参数配置（可选）
    │   └── energy/                         # 能耗数据目录
    │       ├── cpu_energy.txt
    │       ├── gpu_power.csv
    │       ├── gpu_temperature.csv
    │       └── gpu_utilization.csv
    │
    ├── pytorch_resnet_cifar10_resnet20_002/  # 实验2目录
    │   ├── experiment.json
    │   ├── training.log
    │   └── energy/
    │
    └── VulBERTa_mlp_001/                   # 实验3目录
        ├── experiment.json
        ├── training.log
        └── energy/
```

### 命名规则

**Session目录**: `run_{TIMESTAMP}`
- TIMESTAMP: `%Y%m%d_%H%M%S` (会话开始时间)
- 示例: `run_20251112_150000`

**实验目录**: `{repo}_{model}_{sequence}`
- repo: 仓库名称
- model: 模型名称
- sequence: 3位序号（001, 002, 003, ...）
- 示例: `pytorch_resnet_cifar10_resnet20_001`

**文件名**:
- `experiment.json`: 实验结果（原JSON内容）
- `training.log`: 训练日志
- `energy/`: 能耗数据目录（保持原有结构）
- `summary.csv`: 会话总结CSV

---

## 方案2: 扁平目录结构（备选）

### 目录结构设计

```
results/
└── run_20251112_150000/
    ├── summary.csv
    ├── pytorch_resnet_cifar10_resnet20_001_experiment.json
    ├── pytorch_resnet_cifar10_resnet20_001_training.log
    ├── pytorch_resnet_cifar10_resnet20_001_energy/
    ├── pytorch_resnet_cifar10_resnet20_002_experiment.json
    ├── pytorch_resnet_cifar10_resnet20_002_training.log
    └── pytorch_resnet_cifar10_resnet20_002_energy/
```

**优点**: 简单，易于查找
**缺点**: 文件名较长，大量实验时显得混乱

---

## CSV总结文件设计

### 基本列

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `experiment_id` | string | 实验唯一标识 | `pytorch_resnet_cifar10_resnet20_001` |
| `timestamp` | string | ISO时间戳 | `2025-11-12T15:00:00.123456` |
| `repository` | string | 仓库名称 | `pytorch_resnet_cifar10` |
| `model` | string | 模型名称 | `resnet20` |
| `training_success` | boolean | 训练是否成功 | `true` |
| `duration_seconds` | float | 训练时长（秒） | `1234.56` |
| `retries` | int | 重试次数 | `0` |

### 超参数列（动态生成）

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `hyperparam_epochs` | int | Epochs超参数 | `100` |
| `hyperparam_learning_rate` | float | Learning rate | `0.001` |
| `hyperparam_dropout` | float | Dropout | `0.5` |
| `hyperparam_weight_decay` | float | Weight decay | `0.0001` |
| `hyperparam_seed` | int | Random seed | `42` |

### 性能指标列（动态生成）

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `perf_accuracy` | float | 准确率 | `0.92` |
| `perf_rank1` | float | Rank-1准确率 | `0.8669` |
| `perf_rank5` | float | Rank-5准确率 | `0.9510` |
| `perf_map` | float | mAP | `0.6729` |
| `perf_loss` | float | 损失值 | `0.1234` |

### 能耗指标列

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `energy_cpu_pkg_joules` | float | CPU Package能耗 | `74230.7` |
| `energy_cpu_ram_joules` | float | CPU RAM能耗 | `5864.85` |
| `energy_cpu_total_joules` | float | CPU总能耗 | `80095.55` |
| `energy_gpu_avg_watts` | float | GPU平均功率 | `246.36` |
| `energy_gpu_max_watts` | float | GPU峰值功率 | `278.74` |
| `energy_gpu_min_watts` | float | GPU最低功率 | `14.9` |
| `energy_gpu_total_joules` | float | GPU总能耗 | `527217.33` |
| `energy_gpu_temp_avg_celsius` | float | GPU平均温度 | `82.89` |
| `energy_gpu_temp_max_celsius` | float | GPU峰值温度 | `85.0` |
| `energy_gpu_util_avg_percent` | float | GPU平均利用率 | `71.29` |
| `energy_gpu_util_max_percent` | float | GPU峰值利用率 | `80.0` |

### CSV示例

```csv
experiment_id,timestamp,repository,model,training_success,duration_seconds,retries,hyperparam_epochs,hyperparam_learning_rate,hyperparam_dropout,perf_rank1,perf_rank5,perf_map,energy_cpu_total_joules,energy_gpu_total_joules,energy_gpu_avg_watts
pytorch_resnet_cifar10_resnet20_001,2025-11-12T15:00:00.123456,pytorch_resnet_cifar10,resnet20,true,1234.56,0,100,0.001,0.5,0.92,0.98,0.87,80095.55,527217.33,246.36
pytorch_resnet_cifar10_resnet20_002,2025-11-12T15:30:00.123456,pytorch_resnet_cifar10,resnet20,true,1245.67,0,150,0.0005,0.3,0.93,0.99,0.89,85000.00,540000.00,250.00
VulBERTa_mlp_001,2025-11-12T16:00:00.123456,VulBERTa,mlp,true,567.89,0,50,0.01,0.0,0.85,0.95,0.80,45000.00,320000.00,180.00
```

---

## 实现方案

### 1. 核心修改点

#### 修改1: 添加Session管理类

**位置**: `mutation.py` (新增)

**功能**:
- 创建Session目录
- 管理实验序号
- 生成CSV总结
- 维护session_info.json

```python
class ExperimentSession:
    """管理一次mutation.py运行的所有实验"""

    def __init__(self, results_dir: Path):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = results_dir / f"run_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)

        self.experiment_counter = 0
        self.experiments = []  # 存储所有实验结果

    def get_experiment_dir(self, repo: str, model: str) -> Path:
        """获取实验目录"""
        self.experiment_counter += 1
        exp_name = f"{repo}_{model}_{self.experiment_counter:03d}"
        exp_dir = self.session_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        return exp_dir

    def add_experiment(self, result: Dict):
        """添加实验结果"""
        self.experiments.append(result)

    def save_summary_csv(self):
        """生成CSV总结"""
        # 实现CSV生成逻辑
        pass
```

#### 修改2: 修改 `MutationRunner.__init__()`

**位置**: `mutation.py:72-93`

**添加**:
```python
def __init__(self, ...):
    # ... 现有代码 ...

    # Create experiment session
    self.session = ExperimentSession(self.results_dir)
    print(f"📁 Session directory: {self.session.session_dir}")
```

#### 修改3: 修改 `build_training_command()`

**位置**: `mutation.py:320-362`

**修改前**:
```python
log_file = f"results/training_{repo}_{model}_{timestamp}.log"
```

**修改后**:
```python
# Get experiment directory from session
exp_dir = self.session.get_experiment_dir(repo, model)
log_file = str(exp_dir / "training.log")
```

#### 修改4: 修改 `run_experiment()`

**位置**: `mutation.py:894-985`

**修改前**:
```python
experiment_id = f"{datetime.now().strftime(self.TIMESTAMP_FORMAT)}_{repo}_{model}"
energy_dir = f"results/energy_{experiment_id}_attempt{retries}"
```

**修改后**:
```python
# Use session experiment counter
exp_dir = self.session.get_experiment_dir(repo, model)
experiment_id = exp_dir.name  # Use directory name as ID
energy_dir = str(exp_dir / "energy")
```

#### 修改5: 修改 `save_results()`

**位置**: `mutation.py:639-683`

**修改前**:
```python
result_file = self.results_dir / f"{experiment_id}.json"
```

**修改后**:
```python
# Save to experiment directory
exp_dir = self.session.session_dir / experiment_id
result_file = exp_dir / "experiment.json"
```

#### 修改6: 在运行结束时生成CSV

**位置**: `run_mutation_experiments()`, `run_from_experiment_config()`

**添加**:
```python
# At the end of the session
self.session.save_summary_csv()
print(f"📊 Summary CSV: {self.session.session_dir / 'summary.csv'}")
```

---

### 2. 实现复杂度

| 组件 | 复杂度 | 工作量 | 风险 |
|------|--------|--------|------|
| ExperimentSession类 | 🟡 中 | ~100行 | 低 |
| 修改文件路径生成 | 🟢 低 | ~20行 | 低 |
| CSV生成逻辑 | 🟡 中 | ~80行 | 低 |
| 测试和验证 | 🟡 中 | ~50行 | 中 |
| **总计** | 🟡 中 | **~250行** | **低** |

---

## 兼容性分析

### 向后兼容性

#### 问题: 新旧文件结构不兼容

**影响**:
- 现有脚本可能依赖于旧的文件路径
- 旧的结果文件无法与新结果混合

**解决方案**:
1. **配置开关**: 添加 `--use-legacy-structure` 选项保留旧行为
2. **迁移工具**: 提供脚本将旧结果转换为新结构
3. **文档说明**: 明确说明新旧结构差异

### 增量更新

**策略**: 分阶段实施
1. **阶段1**: 实现新结构，但保留旧路径兼容（软链接）
2. **阶段2**: 仅使用新结构，提供迁移工具
3. **阶段3**: 移除旧结构支持

---

## 优势与劣势

### ✅ 优势

1. **组织清晰**
   - 每个实验的所有文件集中在一个目录
   - 一次运行的所有实验集中在一个Session目录
   - 易于查找、备份、删除

2. **便于分析**
   - CSV总结文件方便Excel/Pandas分析
   - 可快速对比不同实验的性能和能耗
   - 支持批量数据处理

3. **易于管理**
   - 删除一次运行只需删除一个Session目录
   - 归档/备份更简单（按Session打包）
   - 减少results目录的混乱

4. **可扩展性**
   - 容易添加新的文件类型（如plot图表）
   - 支持多层级分类（如按日期、项目等）
   - 便于集成可视化工具

5. **自动化友好**
   - CSV格式便于自动化分析脚本处理
   - 目录结构规范，便于批处理
   - 易于与其他工具集成

### ⚠️ 劣势

1. **向后不兼容**
   - 需要更新依赖旧路径的脚本
   - 旧结果文件需要迁移或保持隔离

2. **实现成本**
   - 需要修改多处代码（约250行）
   - 需要充分测试以避免引入bug
   - 需要更新文档

3. **路径变长**
   - 目录层级增加，路径字符串变长
   - 可能影响某些系统的路径长度限制（罕见）

4. **学习成本**
   - 用户需要适应新的目录结构
   - 需要文档说明新结构

---

## 替代方案

### 方案A: 仅添加CSV，不改变目录结构

**描述**: 保持当前的平铺结构，仅在每次运行结束后生成一个CSV总结

**优点**:
- 实现简单（~50行代码）
- 向后完全兼容
- 低风险

**缺点**:
- 不解决文件分散问题
- CSV与实验文件仍然分离
- results目录仍然混乱

### 方案B: 使用数据库替代CSV

**描述**: 使用SQLite或其他数据库存储实验结果

**优点**:
- 查询能力更强
- 支持复杂分析
- 易于扩展

**缺点**:
- 实现复杂度高
- 需要额外依赖
- 不如CSV直观

### 方案C: 仅重组目录，不添加CSV

**描述**: 实现新的目录结构，但不生成CSV总结

**优点**:
- 解决文件分散问题
- 实现较简单（~150行）

**缺点**:
- 不满足CSV总结需求
- 分析仍需手动处理

---

## 建议实施方案

### 🎯 推荐: 方案1（分层目录结构 + CSV总结）

**理由**:
1. **全面满足需求**: 同时解决文件组织和数据分析问题
2. **长期价值高**: 显著提升可维护性和可分析性
3. **风险可控**: 实现复杂度适中，可充分测试
4. **扩展性好**: 为未来功能（如可视化）打下基础

### 实施策略

#### 第1步: 实现核心功能（优先级🔴高）
- ExperimentSession类
- 修改文件路径生成逻辑
- CSV生成基础功能
- **估计时间**: 4-6小时

#### 第2步: 测试和验证（优先级🔴高）
- 单元测试（ExperimentSession类）
- 集成测试（完整实验流程）
- CSV格式验证
- **估计时间**: 2-3小时

#### 第3步: 兼容性处理（优先级🟡中）
- 添加配置开关（可选）
- 编写迁移工具（可选）
- 更新文档
- **估计时间**: 2-3小时

#### 第4步: 优化和完善（优先级🟢低）
- 添加session_info.json
- 支持中文字段名（可选）
- 性能优化
- **估计时间**: 1-2小时

**总估计时间**: 9-14小时

---

## 风险评估

### 🟢 低风险项

1. **文件路径修改**: 逻辑清晰，易于测试
2. **CSV生成**: 使用标准库，稳定可靠
3. **目录创建**: Python Path API成熟

### 🟡 中风险项

1. **并行实验支持**: 需要确保Session在并行模式下正确工作
2. **重试机制**: 失败重试时的目录和文件名处理
3. **大量实验**: 数千个实验时的性能和CSV大小

### 🔴 高风险项

**无** - 本次修改无高风险项

### 风险缓解措施

1. **充分测试**: 覆盖单实验、多实验、并行、重试等场景
2. **逐步实施**: 先实现核心功能，后添加高级特性
3. **保留选项**: 提供配置开关支持旧行为
4. **备份机制**: 在修改前备份现有results目录

---

## 测试计划

### 单元测试

1. **ExperimentSession类测试**
   - 测试Session目录创建
   - 测试实验目录生成和序号
   - 测试CSV生成（各种数据类型）
   - 测试边界情况（空实验、特殊字符等）

2. **路径生成测试**
   - 测试各种repo和model名称
   - 测试序号格式（001, 002, ...）
   - 测试路径合法性

### 集成测试

1. **单个实验测试**
   - 运行单个训练
   - 验证目录结构
   - 验证文件完整性
   - 验证CSV生成

2. **多个实验测试**
   - 运行多次变异
   - 验证序号递增
   - 验证CSV包含所有实验
   - 验证数据一致性

3. **并行实验测试**
   - 运行并行训练
   - 验证前景和背景实验的分离
   - 验证CSV正确性

4. **重试机制测试**
   - 模拟训练失败
   - 验证重试时的文件处理
   - 验证最终CSV仅包含成功的实验

5. **边界测试**
   - 大量实验（100+）
   - 特殊字符的repo/model名称
   - 磁盘空间不足
   - 权限问题

---

## 性能影响

### 预期影响

| 操作 | 当前 | 新方案 | 影响 |
|------|------|--------|------|
| 创建实验目录 | 0ms | ~1ms | 可忽略 |
| 保存JSON | ~1ms | ~1ms | 无影响 |
| 生成CSV（1个实验） | N/A | ~1ms | 新增 |
| 生成CSV（100个实验） | N/A | ~10ms | 新增 |
| 总实验时间 | 几分钟-几小时 | +几毫秒 | **0.001%** |

**结论**: 性能影响微乎其微，完全可接受。

---

## 数据迁移

### 旧结果迁移工具（可选）

**功能**: 将旧的平铺结构迁移到新的分层结构

```bash
# 迁移脚本
python scripts/migrate_results.py --source results/ --dest results_new/
```

**逻辑**:
1. 扫描所有旧的JSON文件
2. 根据timestamp分组（同一次运行）
3. 创建Session目录
4. 移动相关文件到实验目录
5. 生成CSV总结

**估计工作量**: ~100行代码，1-2小时

---

## 文档需求

### 需要更新的文档

1. **README.md**: 说明新的目录结构
2. **使用指南**: 如何查找和分析结果
3. **CSV格式说明**: 列定义和数据类型
4. **迁移指南**: 如何处理旧结果
5. **API文档**: ExperimentSession类的使用

---

## 结论

### ✅ 可行性: **高度可行**

所有用户需求都可以实现，且技术风险低。

### 📊 综合评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **技术可行性** | ⭐⭐⭐⭐⭐ | 5/5 - 技术成熟，无障碍 |
| **实现复杂度** | ⭐⭐⭐⭐☆ | 4/5 - 中等复杂度，可控 |
| **用户价值** | ⭐⭐⭐⭐⭐ | 5/5 - 显著提升可用性 |
| **风险程度** | ⭐⭐⭐⭐⭐ | 5/5 - 风险低，易测试 |
| **维护成本** | ⭐⭐⭐⭐☆ | 4/5 - 略增，但价值高 |
| **向后兼容** | ⭐⭐⭐☆☆ | 3/5 - 需要适配 |

**综合评分**: ⭐⭐⭐⭐⭐ **4.5/5 - 强烈推荐实施**

### 🎯 建议

1. **立即实施**: 方案1（分层目录 + CSV）
2. **分阶段部署**:
   - 第1阶段: 核心功能（必需）
   - 第2阶段: 测试验证（必需）
   - 第3阶段: 兼容性处理（推荐）
   - 第4阶段: 高级功能（可选）
3. **充分测试**: 覆盖各种使用场景
4. **文档先行**: 更新文档帮助用户适应

### 📝 下一步

如果您同意此方案，我将：
1. 实现 `ExperimentSession` 类
2. 修改文件路径生成逻辑
3. 实现CSV生成功能
4. 编写测试用例
5. 更新相关文档

---

**报告完成日期**: 2025-11-12
**预估总工作量**: 9-14小时
**建议优先级**: 🔴 高
**技术风险**: 🟢 低
**用户价值**: 🔴 高
