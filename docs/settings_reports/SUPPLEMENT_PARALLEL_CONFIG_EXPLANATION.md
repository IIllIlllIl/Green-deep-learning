# VulBERTa和Bug定位并行实验补充配置说明

**配置文件**: `settings/supplement_vulberta_buglocalization_parallel.json`
**创建日期**: 2025-12-22
**预估时长**: 16.2小时
**实验总数**: 50个

---

## 📊 当前问题分析

### 实验缺口

| 模型 | 当前有效样本 | 非并行 | 并行 | **问题** |
|------|------------|--------|------|---------|
| **Bug定位** | 40个 | 40 | **0** | ❌ **完全缺少并行实验** |
| **VulBERTa** | 52个 | 52 | **0** | ❌ **完全缺少并行实验** |

**根本原因**: 两个模型在所有超参数下都**只有非并行实验**，没有任何并行实验数据，导致无法研究并行训练对能耗和性能的影响。

---

## 🎯 补充方案设计

### 总体目标

1. **重点补充Bug定位**（40个实验）：样本量较少，优先级最高
2. **适量补充VulBERTa**（10个实验）：样本量相对充足，补充关键超参数

### 时间预算

| 模型 | 补充数量 | 单次时长 | 总时长 | 占比 |
|------|---------|---------|--------|------|
| Bug定位 | 40个 | ~18分钟 | 12.0小时 | 74.1% |
| VulBERTa | 10个 | ~25分钟 | 4.2小时 | 25.9% |
| **合计** | **50个** | - | **16.2小时** | 100% |

✅ **符合用户要求**: 15-18小时范围内

---

## 📋 详细实验配置

### 1. Bug定位并行实验（40个，12小时）

#### 1.1 Default基线实验（10个）
```json
{
  "mode": "parallel",
  "foreground": {"mode": "default"},
  "runs_per_config": 10
}
```
- **作用**: 建立并行模式的基线
- **重要性**: ⭐⭐⭐ 必须有基线才能对比变异效果

#### 1.2 kfold变异（8个）
```json
{
  "mode": "parallel",
  "foreground": {"mutate": ["kfold"]},
  "runs_per_config": 8
}
```
- **目标**: 5个唯一变异值（8次保证容错）
- **研究意义**: 交叉验证折数对并行训练能耗的影响

#### 1.3 max_iter变异（8个）
```json
{
  "mode": "parallel",
  "foreground": {"mutate": ["max_iter"]},
  "runs_per_config": 8
}
```
- **目标**: 5个唯一变异值
- **研究意义**: 最大迭代次数对并行训练时长和能耗的影响

#### 1.4 alpha变异（7个）
```json
{
  "mode": "parallel",
  "foreground": {"mutate": ["alpha"]},
  "runs_per_config": 7
}
```
- **目标**: 5个唯一变异值
- **研究意义**: L2正则化系数对并行训练的影响

#### 1.5 seed变异（7个）
```json
{
  "mode": "parallel",
  "foreground": {"mutate": ["seed"]},
  "runs_per_config": 7
}
```
- **目标**: 5个唯一变异值
- **研究意义**: 随机种子对并行训练稳定性的影响

---

### 2. VulBERTa并行实验（10个，4.2小时）

#### 2.1 epochs变异（3个）
```json
{
  "mode": "parallel",
  "foreground": {"mutate": ["epochs"]},
  "runs_per_config": 3
}
```
- **补充理由**: 非并行已有18个，补充3个并行即可
- **研究意义**: 训练轮数对并行能耗的影响

#### 2.2 learning_rate变异（3个）
```json
{
  "mode": "parallel",
  "foreground": {"mutate": ["learning_rate"]},
  "runs_per_config": 3
}
```
- **补充理由**: 非并行已有18个，补充3个并行即可
- **研究意义**: 学习率对并行训练效率的影响

#### 2.3 weight_decay变异（2个）
```json
{
  "mode": "parallel",
  "foreground": {"mutate": ["weight_decay"]},
  "runs_per_config": 2
}
```
- **补充理由**: 非并行已有18个，补充2个并行
- **研究意义**: L2正则化对并行训练的影响

#### 2.4 seed变异（2个）
```json
{
  "mode": "parallel",
  "foreground": {"mutate": ["seed"]},
  "runs_per_config": 2
}
```
- **补充理由**: 非并行已有18个，补充2个并行
- **研究意义**: 随机种子对并行训练稳定性的影响

---

## 🔧 执行方法

### 方法1: 使用主运行器（推荐）

```bash
# 进入项目目录
cd /home/green/energy_dl/nightly

# 使用sudo运行（需要perf权限）
sudo /home/green/anaconda3/envs/mutation/bin/python3 mutation/runner.py \
  --config settings/supplement_vulberta_buglocalization_parallel.json
```

### 方法2: 后台运行（推荐长时间实验）

```bash
# 后台运行，输出到日志文件
nohup sudo /home/green/anaconda3/envs/mutation/bin/python3 mutation/runner.py \
  --config settings/supplement_vulberta_buglocalization_parallel.json \
  > logs/supplement_parallel_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看进程
ps aux | grep mutation/runner.py

# 实时监控日志
tail -f logs/supplement_parallel_*.log
```

### 方法3: 分步执行（调试用）

如果担心16小时太长，可以分成两步：

**步骤1: 先运行Bug定位（12小时）**
- 手动编辑配置文件，只保留Bug定位的5个实验项
- 运行后验证数据完整性

**步骤2: 再运行VulBERTa（4.2小时）**
- 手动编辑配置文件，只保留VulBERTa的4个实验项
- 运行并验证

---

## ✅ 预期结果

### 实验完成后的数据情况

| 模型 | 补充前 | 补充后 | 增长 | 状态 |
|------|--------|--------|------|------|
| Bug定位（非并行） | 40 | 40 | - | ✅ 已有 |
| Bug定位（**并行**） | **0** | **40** | ↑ **40** | ✅ 补齐 |
| **Bug定位总计** | **40** | **80** | ↑ **100%** | ✅ 完整 |
| VulBERTa（非并行） | 52 | 52 | - | ✅ 已有 |
| VulBERTa（**并行**） | **0** | **10** | ↑ **10** | ⚠️ 部分 |
| **VulBERTa总计** | **52** | **62** | ↑ **19.2%** | ⚠️ 基本 |

### 数据质量指标（预期）

根据历史经验（91.1%有效率）：
- **Bug定位**: 40个实验 × 91.1% = **36-37个有效样本**
- **VulBERTa**: 10个实验 × 91.1% = **9个有效样本**
- **总有效样本**: 45-46个（90-92%有效率）✅

### 研究价值提升

补充并行实验后，可以研究：
1. ✅ **并行 vs 非并行**的能耗对比（之前完全无法研究）
2. ✅ **超参数在并行模式下**的因果影响（之前缺失）
3. ✅ **并行模式的权衡分析**（能耗 vs 性能）
4. ✅ 支持**分层因果分析**（按并行/非并行分组）

---

## ⚠️ 注意事项

### 1. 运行环境
- ✅ 确保有**sudo权限**（perf监控需要）
- ✅ 确保GPU驱动正常（nvidia-smi）
- ✅ 确保磁盘空间充足（至少10GB）

### 2. 实验期间
- ⚠️ **不要中断实验**（16小时较长，建议使用nohup后台运行）
- ⚠️ 避免在实验期间进行其他GPU密集型任务
- ✅ 定期检查日志，确认实验正常进行

### 3. 数据验证
实验完成后，运行以下脚本验证数据：
```bash
# 验证数据完整性
python3 tools/data_management/validate_raw_data.py

# 检查新增的并行实验
python3 << 'EOF'
import csv
bug_par = 0
vul_par = 0
with open('data/raw_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['is_parallel'] == 'True':
            if row['repository'] == 'bug-localization-by-dnn-and-rvsm':
                bug_par += 1
            elif row['repository'] == 'VulBERTa':
                vul_par += 1
print(f"Bug定位并行实验: {bug_par}")
print(f"VulBERTa并行实验: {vul_par}")
EOF
```

预期输出：
```
Bug定位并行实验: 40
VulBERTa并行实验: 10
```

---

## 📝 后续建议

### 如果时间允许，可进一步补充：

1. **VulBERTa的default并行实验**（6个，2.5小时）
   - 建立VulBERTa并行模式的完整基线

2. **VulBERTa的dropout变异**（5个，2.1小时）
   - 补齐VulBERTa的所有并行超参数

3. **增加runs_per_config**
   - 如果某些超参数的变异值不够多样化，可增加重试次数

---

## 📊 配置文件关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `experiment_name` | supplement_vulberta_buglocalization_parallel | 配置标识符 |
| `mode` | mutation | 变异模式 |
| `use_deduplication` | true | 启用去重（避免重复实验） |
| `historical_csvs` | ["data/raw_data.csv"] | 去重参考文件 |
| `max_retries` | 2 | 失败重试次数 |
| `governor` | performance | CPU性能模式 |

### 并行实验配置结构

所有并行实验使用统一的背景任务：
```json
"background": {
  "repo": "examples",
  "model": "mnist",
  "hyperparameters": {}
}
```

**背景任务选择理由**:
- `examples/mnist` 是最快的模型（~2分钟）
- 背景任务对前景任务影响较小
- 项目中所有并行实验都使用此配置（保持一致性）

---

**维护者**: Green
**Claude助手**: 配置设计与说明
**创建日期**: 2025-12-22
**配置文件**: settings/supplement_vulberta_buglocalization_parallel.json
**预估完成时间**: 执行后16.2小时
