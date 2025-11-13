# 输出文件格式优化实施计划 - 剩余工作

## 已完成 ✅

1. **ExperimentSession类** - 完全实现
   - `__init__`: Session目录创建
   - `get_next_experiment_dir()`: 生成实验目录和ID
   - `add_experiment_result()`: 添加实验结果
   - `generate_summary_csv()`: 生成CSV总结

2. **MutationRunner.__init__** - 已添加 session 初始化
   - `self.session = ExperimentSession(self.results_dir)`

## 待修改的关键方法

### 1. `build_training_command()` (Line 252-294)

**当前逻辑**:
```python
timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
log_file = f"results/training_{repo}_{model}_{timestamp}.log"
```

**需要修改为**:
```python
# 使用 session 获取实验目录
exp_dir, exp_id = self.session.get_next_experiment_dir(repo, model, mode="train")
log_file = str(exp_dir / "training.log")
energy_dir_param = str(exp_dir / "energy")  # 用于传递给 run.sh
```

### 2. `save_results()` (Line 571-614)

**当前逻辑**:
```python
result_file = self.results_dir / f"{experiment_id}.json"
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)
```

**需要修改为**:
```python
# 保存到实验目录
exp_dir = self.session.session_dir / experiment_id
result_file = exp_dir / "experiment.json"
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)

# 添加到 session 用于 CSV 生成
self.session.add_experiment_result(result)
```

### 3. `run_experiment()` (Line 826-918)

**当前逻辑**:
```python
experiment_id = f"{datetime.now().strftime(self.TIMESTAMP_FORMAT)}_{repo}_{model}"
energy_dir = f"results/energy_{experiment_id}_attempt{retries}"
cmd, log_file = self.build_training_command(repo, model, mutation, energy_dir)
```

**需要修改为**:
```python
# 不再生成 experiment_id，由 build_training_command 生成
# 需要在循环外先获取实验目录
exp_dir, experiment_id = self.session.get_next_experiment_dir(repo, model, mode="train")

# 在重试循环中使用相同的 exp_dir
while not success and retries <= max_retries:
    log_file = str(exp_dir / "training.log")
    energy_dir = str(exp_dir / "energy")
    # ...
```

### 4. `run_parallel_experiment()` (Line 746-824)

**当前逻辑**:
```python
experiment_id = f"{datetime.now().strftime(self.TIMESTAMP_FORMAT)}_{fg_repo}_{fg_model}_parallel"
log_dir = self.results_dir / f"background_logs_{experiment_id}"
```

**需要修改为**:
```python
# 获取前景实验目录（带 parallel 标记）
exp_dir, experiment_id = self.session.get_next_experiment_dir(fg_repo, fg_model, mode="parallel")

# 背景训练日志放在前景实验目录的 background_logs 子目录
bg_log_dir = exp_dir / "background_logs"
bg_log_dir.mkdir(exist_ok=True, parents=True)
```

### 5. `_start_background_training()` (Line 650-707)

**当前逻辑**:
```python
log_dir = self.results_dir / f"background_logs_{experiment_id}"
```

**需要修改为** (接受 log_dir 参数):
```python
def _start_background_training(self,
                               repo: str,
                               model: str,
                               hyperparams: Dict[str, Any],
                               log_dir: Path) -> Tuple[subprocess.Popen, None]:
    # 使用传入的 log_dir，不再自己创建
    log_dir.mkdir(exist_ok=True, parents=True)
```

### 6. 在会话结束时生成 CSV

**在以下方法末尾添加**:

- `run_mutation_experiments()` (Line 920-990末尾)
- `run_from_experiment_config()` (Line 992-1192末尾)

```python
# Generate summary CSV
print("\n" + "=" * 80)
print("📊 Generating session summary...")
print("=" * 80)
csv_file = self.session.generate_summary_csv()
if csv_file:
    print(f"✅ Summary CSV: {csv_file}")
```

## 实施顺序

### 阶段1: 核心路径修改 (最关键)
1. 修改 `run_experiment()` - 使用 session
2. 修改 `build_training_command()` - 返回实验目录
3. 修改 `save_results()` - 保存到实验目录并添加到 session

### 阶段2: 并行训练支持
4. 修改 `run_parallel_experiment()` - 使用 session
5. 修改 `_start_background_training()` - 接受 log_dir 参数

### 阶段3: CSV生成
6. 在 `run_mutation_experiments()` 末尾调用 `generate_summary_csv()`
7. 在 `run_from_experiment_config()` 末尾调用 `generate_summary_csv()`

## 关键注意事项

### 1. 并行训练的背景日志位置

**要求**: 背景模型B的所有log应该在前景模型A的对应超参数文件夹中

**实现**:
```
results/
└── run_20251112_150000/
    └── pytorch_resnet_cifar10_resnet20_001_parallel/  ← 前景实验目录
        ├── training.log                ← 前景训练日志
        ├── experiment.json             ← 前景实验结果
        ├── energy/                     ← 前景能耗数据
        └── background_logs/            ← 背景训练日志目录
            ├── run_1.log
            ├── run_2.log
            └── run_3.log
```

### 2. 重试机制的处理

**问题**: 如果训练失败重试，是否创建新的实验目录？

**决策**: 不创建新目录，在同一目录中覆盖文件
- 优点: 节省空间，experiment_id 保持一致
- 缺点: 失败的日志会被覆盖

**实现**:
```python
# 在 run_experiment() 中
exp_dir, experiment_id = self.session.get_next_experiment_dir(repo, model)

while not success and retries <= max_retries:
    # 使用相同的 exp_dir
    log_file = str(exp_dir / "training.log")  # 会覆盖之前的
    energy_dir = str(exp_dir / "energy")      # 会覆盖之前的
```

### 3. 实验ID生成规则

**格式**: `{repo}_{model}_{sequence:03d}` 或 `{repo}_{model}_{sequence:03d}_parallel`

**示例**:
- `pytorch_resnet_cifar10_resnet20_001`
- `pytorch_resnet_cifar10_resnet20_002_parallel`
- `VulBERTa_mlp_003`

## 测试策略

### 单元测试 (test_output_structure.py)

```python
def test_session_creation():
    # 测试 Session 目录创建

def test_experiment_dir_generation():
    # 测试实验目录生成和序号递增

def test_csv_generation():
    # 测试 CSV 生成（空、单个、多个实验）

def test_parallel_background_logs():
    # 测试并行训练时背景日志位置
```

### 集成测试

```bash
# 测试单个实验
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs --runs 1

# 测试多个实验
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs --runs 3

# 测试并行训练
python3 mutation.py -ec settings/parallel_example.json
```

### 预期目录结构

```
results/
└── run_20251112_150000/
    ├── summary.csv
    ├── pytorch_resnet_cifar10_resnet20_001/
    │   ├── experiment.json
    │   ├── training.log
    │   └── energy/
    │       ├── cpu_energy.txt
    │       ├── gpu_power.csv
    │       ├── gpu_temperature.csv
    │       └── gpu_utilization.csv
    ├── pytorch_resnet_cifar10_resnet20_002/
    └── pytorch_resnet_cifar10_resnet20_003_parallel/
        ├── experiment.json
        ├── training.log
        ├── energy/
        └── background_logs/
            ├── run_1.log
            ├── run_2.log
            └── run_3.log
```

## 向后兼容性

**不兼容**: 新的目录结构与旧的完全不同

**迁移建议**:
1. 保留旧的 results/ 目录作为备份
2. 新运行会创建 `run_{timestamp}/` 子目录
3. 可以编写迁移脚本将旧结果转换为新结构（可选）

## 估计剩余工作量

| 任务 | 估计时间 |
|------|----------|
| 修改核心方法 (阶段1) | 1-2小时 |
| 修改并行训练 (阶段2) | 1小时 |
| 添加CSV生成调用 (阶段3) | 0.5小时 |
| 编写测试 | 1-2小时 |
| 运行集成测试和调试 | 1-2小时 |
| **总计** | **4.5-7.5小时** |

## 当前状态

✅ ExperimentSession 类完全实现
✅ MutationRunner.__init__ 已集成 session
⏳ 核心方法修改进行中
⏳ 并行训练背景日志处理待实现
⏳ CSV生成调用待添加
⏳ 测试待编写

---

**下一步**: 修改 `run_experiment()`, `build_training_command()`, `save_results()` 三个核心方法
