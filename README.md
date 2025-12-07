# Mutation-Based Training Energy Profiler

自动化深度学习模型训练的超参数变异与能耗性能分析框架

**当前版本**: v4.7.1 (2025-12-07)
**状态**: ✅ Production Ready

---

## 项目概述

研究深度学习训练超参数对能耗和性能的影响。通过自动化变异超参数、监控能耗、收集性能指标,支持大规模实验研究。

### 核心功能

- ✅ **超参数变异** - 自动生成超参数变体（log-uniform/uniform分布）
- ✅ **能耗监控** - CPU (perf) + GPU (nvidia-smi),CPU误差<2%
- ✅ **并行训练** - 支持前台监控+后台负载的并行训练模式
- ✅ **完整元数据** - 并行实验记录完整前景+背景模型信息
- ✅ **去重机制** - 自动跳过重复实验，支持历史数据去重，区分并行/非并行模式
- ✅ **数据完整性** - CSV安全追加模式，防止数据覆盖丢失
- ✅ **离线训练** - 支持完全离线运行，避免网络依赖
- ✅ **快速验证** - 1-epoch配置，15-20分钟验证全部模型
- ✅ **结果组织** - 分层目录结构 + CSV汇总 + JSON详细数据
- ✅ **批量实验** - 配置文件支持复杂实验设计
- ✅ **分阶段执行** - 大规模实验分割成可管理的阶段

---

## 快速开始

### 1. 列出可用模型

```bash
python3 mutation.py --list
```

### 2. 运行单个实验

```bash
# 基本用法
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs,learning_rate -n 3
```

### 3. 运行批量实验（推荐）

```bash
# 快速验证（15-20分钟，1 epoch）
HF_HUB_OFFLINE=1 python3 mutation.py -ec settings/11_models_quick_validation_1epoch.json

# 变异实验验证（1次变异，快速测试）
sudo -E python3 mutation.py -ec settings/mutation_validation_1x.json -g performance

# 完整变异实验（3次变异，完整数据）
sudo -E python3 mutation.py -ec settings/mutation_all_models_3x_dynamic.json -g performance

# 完整基线实验（9+小时）
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance
```

### 4. 分阶段实验（大规模实验推荐）

```bash
# 阶段1-4: 已完成 ✓
# sudo -E python3 mutation.py -ec settings/stage1_nonparallel_completion.json
# sudo -E python3 mutation.py -ec settings/stage2_optimized_nonparallel_and_fast_parallel.json
# sudo -E python3 mutation.py -ec settings/stage3_4_merged_optimized_parallel.json

# 阶段7-8: 已完成 ✓ (2025-12-06~07)
# sudo -E python3 mutation.py -ec settings/stage7_nonparallel_fast_models.json
# sudo -E python3 mutation.py -ec settings/stage8_nonparallel_medium_slow_models.json

# 阶段13: 最终补充 (推荐优先执行, 12.5h, 90实验)
sudo -E python3 mutation.py -ec settings/stage13_merged_final_supplement.json

# 阶段11-12: 并行hrnet18/pcb补充 (51.7h, 40实验)
sudo -E python3 mutation.py -ec settings/stage11_parallel_hrnet18.json
sudo -E python3 mutation.py -ec settings/stage12_parallel_pcb.json

# 注: Stage9-10已归档（非并行hrnet18/pcb已达标，节省48.7小时）
```

---

## 支持的模型

| 仓库 | 模型 | 超参数 | Epochs范围 |
|------|------|--------|-----------|
| **pytorch_resnet_cifar10** | resnet20/32/44/56 | epochs, lr, weight_decay, seed | [100, 300] |
| **Person_reID_baseline_pytorch** | densenet121, hrnet18, pcb | epochs, lr, dropout, seed | [30, 90] |
| **VulBERTa** | mlp, cnn | epochs, lr, weight_decay, seed | [5, 20] |
| **MRT-OAST** | default | epochs, lr, dropout, weight_decay, seed | [5, 15] |
| **bug-localization** | default | max_iter, alpha, kfold, seed | - |
| **examples** | mnist, mnist_rnn, mnist_ff, siamese | epochs, lr, batch_size, seed | [5, 15] |

**详细范围**: [docs/MUTATION_RANGES_QUICK_REFERENCE.md](docs/MUTATION_RANGES_QUICK_REFERENCE.md)

---

## 结果输出

每次运行创建独立session,包含CSV汇总和详细JSON数据：

```
results/run_YYYYMMDD_HHMMSS/
├── summary.csv                    # 所有实验汇总
└── {repo}_{model}_{id}_parallel/  # 并行实验（或不带_parallel为顺序实验）
    ├── experiment.json            # 完整数据（超参数+性能+能耗）
    │                              # 并行实验包含foreground和background信息
    ├── training.log               # 前景训练日志
    ├── energy/                    # 能耗原始数据
    └── background_logs/           # 后台训练日志（仅并行实验）
```

**详细说明**: [docs/OUTPUT_STRUCTURE_QUICKREF.md](docs/OUTPUT_STRUCTURE_QUICKREF.md)

---

## 常用命令

```bash
# 列出所有模型
python3 mutation.py --list

# 单次训练
python3 mutation.py -r pytorch_resnet_cifar10 -m resnet20 -mt epochs -n 3

# 快速验证（15-20分钟）
HF_HUB_OFFLINE=1 python3 mutation.py -ec settings/11_models_quick_validation_1epoch.json

# 变异实验（动态生成超参数）
sudo -E python3 mutation.py -ec settings/mutation_validation_1x.json -g performance

# 完整实验（9+小时）
export HF_HUB_OFFLINE=1
sudo -E python3 mutation.py -ec settings/11_models_sequential_and_parallel_training.json -g performance

# 查看结果
cat results/run_*/summary.csv | column -t -s,
```

**完整参数说明**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

---

## 系统要求

- **Python**: 3.6+ (仅标准库)
- **能耗监控**: `perf` (CPU), `nvidia-smi` (GPU)
- **权限**: 需要sudo以获取准确的CPU能量数据

```bash
# 安装perf
sudo apt-get install linux-tools-common linux-tools-generic
sudo sysctl -w kernel.perf_event_paranoid=-1
```

---

## 📚 文档

| 文档 | 说明 |
|-----|------|
| [超参数变异范围](docs/MUTATION_RANGES_QUICK_REFERENCE.md) | 变异范围速查 ⭐⭐⭐ |
| [快速参考](docs/QUICK_REFERENCE.md) | 命令速查表 ⭐⭐ |
| [实验配置指南](docs/SETTINGS_CONFIGURATION_GUIDE.md) | Settings JSON编写 ⭐⭐ |
| [11个模型概览](docs/11_MODELS_OVERVIEW.md) | 模型详细信息 ⭐⭐ |
| [并行训练使用](docs/PARALLEL_TRAINING_USAGE.md) | 并行训练配置 ⭐⭐ |
| [输出结构](docs/OUTPUT_STRUCTURE_QUICKREF.md) | 结果目录结构 ⭐ |
| [功能总览](docs/FEATURES_OVERVIEW.md) | 所有功能说明 ⭐⭐ |
| [完整文档索引](docs/README.md) | 所有文档列表 |

---

## Settings配置文件

### 可用配置（settings/目录）

#### 基础配置
| 配置文件 | 说明 | 预计时间 |
|---------|------|---------|
| `11_models_quick_validation_1epoch.json` | 快速验证（1 epoch） | 15-20分钟 |
| `mutation_validation_1x.json` | 变异验证（1次） | 按模型而定 |
| `mutation_all_models_3x_dynamic.json` | 完整变异（3次） | 较长时间 |
| `11_models_sequential_and_parallel_training.json` | 完整基线 | 9+小时 |
| `person_reid_dropout_boundary_test.json` | Dropout边界测试 | ~6.5小时 |

#### 分阶段实验配置（推荐用于大规模实验）⭐

**分阶段配置** (v4.7.1 - 配置修复与优化):
| 配置文件 | 说明 | 预计时间 | 实验数 | 状态 |
|---------|------|---------|--------|------|
| `stage1_nonparallel_completion.json` | 阶段1: 非并行补全 | 9h | 12 | ✅ 已完成 |
| `stage2_optimized_nonparallel_and_fast_parallel.json` | 阶段2: 非并行补充 + 快速模型并行 | 7.3h | 25 | ✅ 已完成 |
| `stage3_4_merged_optimized_parallel.json` | 阶段3-4: 中速模型 + VulBERTa + densenet121 | 12.2h | 25 | ✅ 已完成 |
| `stage7_nonparallel_fast_models.json` | 阶段7: 非并行快速模型 | 0.7h | 7 | ✅ 已完成 |
| `stage8_nonparallel_medium_slow_models.json` | 阶段8: 非并行中慢速模型 | ~13h | 12 | ✅ 已完成 |
| `stage11_parallel_hrnet18.json` | 阶段11: 并行hrnet18补充 | 28.6h | 20 | ⏳ 待执行 |
| `stage12_parallel_pcb.json` | 阶段12: 并行pcb补充 | 23.1h | 20 | ⏳ 待执行 |
| `stage13_merged_final_supplement.json` | 阶段13: 最终补充 **(合并+VulBERTa/cnn)** | 12.5h | 90 | ⏳ 推荐优先 |
| **总计** | **8个阶段** | **106h** | **211** | **81/211完成** |
| ~~stage9/10~~ | ~~非并行hrnet18/pcb~~ | ~~48.7h~~ | ~~40~~ | 🗑️ 已归档（冗余） |

**v4.7.1修复与优化** (2025-12-07):
- 🔴 **严重Bug修复**: Stage7-13配置文件存在多参数混合变异问题
  - 问题: 配置使用`mutate_params=[多个参数]`导致混合变异而非单参数独立运行
  - 影响: 预期370个实验，实际只会运行108个（缺失262个，70.8%）
  - 修复: 自动拆分为单参数配置项（19项 → 62项）
  - 工具: `scripts/fix_stage_configs.py`
- ✅ **Stage7-8执行完成**:
  - Stage7: 实际0.7小时（预期38.3h），去重率96.5%，7个新实验
  - Stage8: 实际~13小时（预期35.1h），12个新实验
  - 总实验数: 400→419（新增19个实验）
- ✅ **Stage13配置合并**:
  - 合并: Stage13 + Stage14 + VulBERTa/cnn完整覆盖
  - 发现: VulBERTa/cnn模型在所有阶段中完全遗漏（0个实验）
  - 新增: 8个VulBERTa/cnn配置项（非并行4个+并行4个）
  - 配置: `stage13_merged_final_supplement.json` (18项, 90实验, 12.5h)
- ✅ **Stage9-10冗余移除**:
  - Stage9: hrnet18非并行全部达标（5-6个唯一值），已归档
  - Stage10: pcb非并行全部达标（5-6个唯一值），已归档
  - 节省时间: 48.7小时，节省实验: 40个
- 📊 **最终实验计划**:
  - 剩余3个必需阶段: Stage11, Stage12, Stage13
  - 预计时间: 64.2小时（去重后可能<20小时）
  - 预计实验: 130个
  - 完成后: 100%覆盖（90/90参数-模式组合）
- 📚 **文档完善**:
  - [JSON配置最佳实践](docs/JSON_CONFIG_BEST_PRACTICES.md) - 防止配置错误 ⭐⭐⭐
  - [Stage13-14合并报告](docs/results_reports/STAGE13_14_MERGE_AND_COMPLETION_REPORT.md) ⭐⭐
  - [Stage9-13优化报告](docs/results_reports/STAGE9_13_OPTIMIZATION_REPORT.md) - 冗余分析 ⭐⭐

**v3.0优化重点** (2025-12-05):
- ✅ **模式区分**: 修复去重机制，正确区分并行/非并行模式
- ✅ **精确规划**: 基于实际完成度（28/90组合），重新设计Stage7-13
- ✅ **效率提升**: Stage5-6被更优的Stage11-12替代（节省44.3小时）
- ✅ **全面覆盖**: 完成后达到100%（90/90组合，450个唯一值）

**详细执行计划**: [docs/settings_reports/STAGE7_13_EXECUTION_PLAN.md](docs/settings_reports/STAGE7_13_EXECUTION_PLAN.md)
**需求分析报告**: [docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md](docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md)

**归档配置** (`settings/archived/` 和备份文件):
- **Stage5-6**: 已归档（被Stage11-12替代）
- **Stage9-10**: 已归档到`redundant_stages_20251207/`（参数全部达标，节省48.7小时）
- **旧版配置**: v1.0-v2.0配置已归档
- **Stage13/14原始**: 已备份（`.bak_20251207`）：
  - `stage13_parallel_fast_models_supplement.json.bak_20251207`
  - `stage14_stage7_8_supplement.json.bak_20251207`

### 配置格式

**顺序训练**:
```json
{
  "repo": "examples",
  "model": "mnist",
  "mode": "mutation",
  "mutate": ["epochs", "learning_rate"]
}
```

**并行训练**:
```json
{
  "mode": "parallel",
  "foreground": {
    "repo": "examples",
    "model": "mnist",
    "mode": "mutation",
    "mutate": ["epochs"]
  },
  "background": {
    "repo": "Person_reID_baseline_pytorch",
    "model": "densenet121",
    "hyperparameters": {"epochs": 60, "learning_rate": 0.05}
  }
}
```

**详细指南**: [docs/SETTINGS_CONFIGURATION_GUIDE.md](docs/SETTINGS_CONFIGURATION_GUIDE.md)

---

## 版本信息

**v4.7.1** (2025-12-07)
- 🔴 **严重配置Bug修复**: Stage7-13配置文件多参数混合变异问题
  - 问题: 配置使用`mutate_params=[多个参数]`导致生成混合变异实验，而非每个参数独立运行
  - 根因: 对`runs_per_config`和`mutate_params`语义理解错误
  - 影响: 预期370个实验，实际只会运行108个（缺失262个，70.8%）
  - 修复: 使用`scripts/fix_stage_configs.py`自动拆分为单参数配置（19项 → 62项）
  - 详细分析: [Stage7-13配置Bug分析](docs/results_reports/STAGE7_13_CONFIG_BUG_ANALYSIS.md)
- ✅ **Stage7-8执行情况分析**:
  - Stage7: 虽然配置错误但去重率96.5%，大部分参数已达标（历史实验覆盖）
  - Stage8: 所有参数已超标（≥10个唯一值），无需补充
  - Stage14: 新增补充配置，仅需7个实验（2.5小时）补充MRT-OAST/default epochs
  - 执行报告: [Stage7-8修复执行报告](docs/results_reports/STAGE7_8_FIX_EXECUTION_REPORT.md)
- 📚 **配置最佳实践文档** (新增):
  - 文档: [JSON配置最佳实践](docs/JSON_CONFIG_BEST_PRACTICES.md) ⭐⭐⭐
  - 内容: 核心概念、常见错误、正确示例、验证清单、故障排查
  - 重点: "单参数原则" - 每个配置项只变异一个参数
  - 示例: 正确vs错误配置对比，参考Stage2作为最佳模板

**v4.7.0** (2025-12-06)
- ✅ **Per-experiment runs_per_config bug修复**: 修复配置文件读取逻辑
  - 问题: `runner.py`第881行全局`runs_per_config`默认为1，导致仅使用每个实验配置的runs_per_config
  - 影响: Stage7等使用per-experiment值的配置只运行1个实验而非指定的7个
  - 修复: 在mutation、parallel、default三种模式中添加per-experiment值读取逻辑
  - 修改文件: `mutation/runner.py` (lines 1001-1126, 三个实验模式)
  - 测试: 创建5个测试用例全部通过 (`tests/test_runs_per_config_fix.py`)
  - 向后兼容: 保留全局fallback机制，旧配置仍可正常运行

**v4.6.0** (2025-12-05)
- ✅ **Stage3-4执行完成**: mnist_ff剩余 + 中速模型 + VulBERTa + densenet121
  - 实际运行: 12.2小时 (预期57.1小时)
  - 实验数量: 25个实验 (预期57个)
  - 去重效果: 跳过32个重复实验 (56.1%跳过率)
  - 总实验数: 381个 (356→381)
- ✅ **去重模式区分修复**: 修复去重机制，区分并行/非并行模式
  - 问题: 原去重机制未区分模式，导致并行实验被错误跳过
  - 修复: 在去重key中包含mode信息，确保不同模式独立去重
  - 修改文件: `hyperparams.py`, `dedup.py`, `runner.py` (共约30行代码)
  - 测试验证: 创建2个测试套件（10个测试用例）全部通过
  - 向后兼容: mode参数可选，旧代码仍可正常运行
  - 详细报告: [docs/results_reports/DEDUP_MODE_FIX_REPORT.md](docs/results_reports/DEDUP_MODE_FIX_REPORT.md)
- ✅ **实验进度分析**: 当前完成度评估
  - 总实验数: 381个 (非并行261 + 并行120)
  - 参数达标率: 100% (不区分模式) / 80% (区分模式)
  - 非并行模式: 97.8% (44/45参数达标)
  - 并行模式: 62.2% (28/45参数达标)
  - 缺失: 18个参数-模式组合，需补充50个唯一值
  - 详细分析: [docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md](docs/results_reports/EXPERIMENT_REQUIREMENT_ANALYSIS.md)

**v4.5.0** (2025-12-04)
- ✅ **Stage2执行完成**: 非并行补充 + 快速模型并行实验完成
  - 实际运行: 7.3小时 (预期20-24小时)
  - 实验数量: 25个实验 (预期44个)
  - 核心目标达成: 所有29个目标参数达到5个唯一值
  - 去重效果: 跳过40个重复实验 (61.5%跳过率)
- ✅ **参数精确优化** (runs_per_config v2.0): 每个参数使用精确的runs_per_config值
  - 优化原理: runs_per_config = (5 - current_unique_count) + 1
  - 资源利用率: 从26.9%提升到>90%
  - 减少无效尝试: 从390次降低到105-145次（节省63%）
  - 节省GPU时间: 约160小时
- ✅ **配置文件合并**: 将Stage3和Stage4合并为单个配置文件
  - 合并后配置: `stage3_4_merged_optimized_parallel.json`
  - 实验项: 25个实验项，57个预期实验
  - 时间预估: 57.1小时 (基于Stage2 38.5%完成率重新预估)
- ✅ **完整文档**: 优化报告、快速参考、执行准备清单
- ✅ **配置归档**: 旧版统一runs_per_config配置移至archive/目录

**v4.4.0** (2025-12-02)
- ✅ **CSV追加bug修复**: 修复了`_append_to_summary_all()`方法导致的数据覆盖问题
  - 问题: 调用`aggregate_csvs.py`使用'w'模式���盖整个文件
  - 修复: 直接使用CSV模块的'a'模式安全追加数据
  - 测试: 创建了11个单元测试和4个手动测试验证修复
- ✅ **去重机制**: 支持基于历史CSV文件的实验去重
  - 配置: `use_deduplication: true` + `historical_csvs: ["results/summary_all.csv"]`
  - 效果: 自动跳过已存在的超参数组合，生成新的随机值
- ✅ **分阶段实验计划**: 大规模实验分割成7个可管理的阶段
  - 总计256小时实验分割为20-46小时/阶段
  - 便于监控、暂停和验证中间结果
  - 所有阶段配置文件已准备完毕
- ✅ **实验完成度分析**: 当前319个实验（214非并行 + 105并行）
  - 非并行完成度: 73.3% (33/45参数达到5个唯一值)
  - 并行完成度: 0% (0/45参数达到5个唯一值)
  - 分阶段计划将完成剩余269个实验达到100%

**v4.3.0** (2025-11-19)
- ✅ 11个模型完整支持（基线+变异）
- ✅ 动态变异系统（log-uniform/uniform分布）
- ✅ 并行训练（前景+背景GPU同时利用）
- ✅ 离线训练（HF_HUB_OFFLINE=1）
- ✅ 高精度能耗监控（CPU误差<2%）
- ✅ Dropout范围优化：统一采用d-0.2到d+0.1策略
  - MRT-OAST: [0.0, 0.3] (d=0.2)
  - Person_reID: [0.3, 0.6] (d=0.5)
  - 基于边界测试验证，dropout=0.6性能变化<1%，dropout=0.7导致严重劣化(-3%~-8%)

**完整版本历史**: [docs/FEATURES_OVERVIEW.md](docs/FEATURES_OVERVIEW.md)

---

**维护者**: Green | **状态**: ✅ Production Ready | **更新**: 2025-12-07
