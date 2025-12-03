# Claude 助手指南 - Mutation-Based Training Energy Profiler

**项目版本**: v4.5.0 (2025-12-03)
**最后更新**: 2025-12-03
**状态**: ✅ Production Ready

---

## 🎯 项目概述

### 核心目标
研究深度学习训练超参数对能耗和性能的影响。通过自动化变异超参数、监控能耗、收集性能指标，支持大规模实验研究。

### 关键特性
- ✅ **超参数变异** - 自动生成超参数变体（log-uniform/uniform分布）
- ✅ **能耗监控** - CPU (perf) + GPU (nvidia-smi), CPU误差<2%
- ✅ **并行训练** - 支持前台监控+后台负载的并行训练模式
- ✅ **去重机制** - 自动跳过重复实验，支持历史数据去重
- ✅ **数据完整性** - CSV安全追加模式，防止数据覆盖丢失
- ✅ **分阶段执行** - 大规模实验分割成可管理的阶段
- ✅ **参数精确优化** - 每个参数使用精确的runs_per_config值

### 当前状态
- **实验总数**: 331个（results/summary_all.csv）
- **完成度**: 非并行73.3%，并行0%（分阶段计划进行中）
- **优化配置**: 已创建5个优化stage配置文件，预计184-192小时完成

---

## 📁 文件结构规范

### 目录结构标准
```
energy_dl/nightly/
├── docs/                    # 所有文档文件
│   ├── archived/           # 过时文档归档
│   ├── environment/        # 环境配置文档
│   ├── results_reports/    # 实验结果报告
│   └── settings_reports/   # 配置相关报告
├── scripts/                # 辅助脚本（Python/Bash）
├── tests/                  # 测试文件
├── settings/               # 运行配置文件
│   └── archived/          # 过时配置归档
├── results/               # 实验结果数据
│   └── run_YYYYMMDD_HHMMSS/  # 每次运行的session
├── mutation/              # 核心代码
├── config/                # 模型配置
└── environment/           # 环境设置（空目录，文档已移至docs）
```

### 文件放置规则

#### 1. 文档文件 (`docs/`)
- **所有** Markdown文档必须放在`docs/`目录下
- 按类型组织子目录：
  - `docs/results_reports/` - 实验结果分析报告
  - `docs/environment/` - 环境配置说明
  - `docs/settings_reports/` - 配置相关报告
  - `docs/archived/` - 过时文档（按日期组织）

#### 2. 脚本文件 (`scripts/`)
- 数据处理脚本、分析脚本、工具脚本
- 命名规范：`{用途}_{描述}.py` 或 `.sh`
- 示例：`analyze_unique_values.py`, `verify_csv_format.py`

#### 3. 测试文件 (`tests/`)
- 单元测试、集成测试、验证测试
- 命名规范：`test_{模块}_{功能}.py`
- 示例：`test_runner_append.py`, `verify_csv_append_fix.py`

#### 4. 配置文件 (`settings/`)
- JSON实验配置文件
- 命名规范：`{stage}_{描述}.json`
- 示例：`stage2_optimized_nonparallel_and_fast_parallel.json`

#### 5. 归档文件 (`*/archived/`)
- 每个目录可包含`archived/`子目录
- 过时文件按日期或版本组织
- 保留`README_ARCHIVE.md`说明归档原因

#### 6. 代码文件 (`mutation/`)
- 核心业务逻辑
- 模块化组织：`runner.py`, `session.py`, `energy_monitor.py`等

#### 7. 数据文件 (`results/`)
- 实验结果CSV、JSON、日志文件
- 每次运行创建独立session目录：`run_YYYYMMDD_HHMMSS/`
- 全局汇总文件：`summary_all.csv`

---

## 🔧 开发工作流程

### 1. 添加新功能
1. 在`mutation/`相应模块中实现功能
2. 在`tests/`中添加测试用例
3. 在`docs/`中更新相关文档
4. 验证通过后提交

### 2. 修复Bug
1. 在`tests/`中创建复现测试
2. 在`mutation/`中修复代码
3. 运行所有测试确保无回归
4. 在`docs/results_reports/`中记录修复报告

### 3. 更新配置
1. 在`settings/`中创建新配置文件
2. 在`docs/settings_reports/`中说明配置变更
3. 测试配置有效性
4. 归档旧配置文件到`settings/archived/`

### 4. 生成报告
1. 分析脚本放在`scripts/`
2. 报告文档放在`docs/results_reports/`
3. 使用标准命名：`{主题}_{描述}.md`
4. 包含时间戳和版本信息

---

## 📊 关键文件位置

### 核心代码
- `mutation/runner.py` - 主运行器（已修复CSV追加bug）
- `mutation/session.py` - Session管理
- `mutation/energy_monitor.py` - 能耗监控
- `config/models_config.json` - 模型配置

### 重要数据文件
- `results/summary_all.csv` - 所有实验汇总（331行，37列）
- `results/summary_all.csv.backup` - 数据备份
- `settings/EXECUTION_READY.md` - 执行准备清单（已归档）

### 配置文件
- `settings/stage2_optimized_nonparallel_and_fast_parallel.json` - Stage2优化配置
- `settings/stage3_optimized_mnist_ff_and_medium_parallel.json` - Stage3优化配置
- `settings/stage4_optimized_vulberta_densenet121_parallel.json` - Stage4优化配置
- `settings/stage5_optimized_hrnet18_parallel.json` - Stage5优化配置
- `settings/stage6_optimized_pcb_parallel.json` - Stage6优化配置

### 测试文件
- `tests/verify_csv_append_fix.py` - CSV修复验证测试
- `tests/unit/test_append_to_summary_all.py` - 单元测试

### 文档索引
- `docs/FEATURES_OVERVIEW.md` - 功能特性总览
- `docs/QUICK_REFERENCE.md` - 快速参考
- `docs/SETTINGS_CONFIGURATION_GUIDE.md` - 配置指南
- `docs/results_reports/CSV_FIX_COMPREHENSIVE_SUMMARY.md` - CSV修复综合报告
- `docs/results_reports/MISSING_COLUMNS_DETAILED_ANALYSIS.md` - 缺失列详细分析

---

## 🚀 快速开始命令

### 查看项目状态
```bash
# 检查CSV格式
python3 -c "import csv; f=open('results/summary_all.csv'); r=csv.reader(f); h=next(r); rows=list(r); print(f'✓ {len(rows)} experiments, ✓ {len(h)} columns')"

# 查看最新版本
grep '当前版本' README.md
```

### 运行测试
```bash
# 运行CSV修复验证测试
python3 tests/verify_csv_append_fix.py

# 运行所有单元测试
python3 -m pytest tests/unit/
```

### 执行实验
```bash
# Stage2优化配置（20-24小时）
sudo -E python3 mutation.py -ec settings/stage2_optimized_nonparallel_and_fast_parallel.json

# 查看运行状态
ls -lht results/run_* | head -3
```

### 数据分析
```bash
# 分析唯一值数量
python3 scripts/analyze_unique_values.py

# 检查实验完成度
python3 scripts/check_completion_status.py
```

---

## ⚠️ 注意事项

### 1. CSV数据完整性
- `summary_all.csv`必须保持37列格式
- 追加数据时使用`extrasaction='ignore'`自动对齐列
- 每次修改前备份：`cp summary_all.csv summary_all.csv.backup`

### 2. 配置版本控制
- 创建新配置时保留旧版本在`settings/archived/`
- 在配置文件中添加`comment`字段说明变更原因
- 更新`docs/settings_reports/`中的相关文档

### 3. 测试要求
- 所有代码变更必须通过现有测试
- 新功能必须添加测试用例
- CSV格式变更必须运行`verify_csv_append_fix.py`

### 4. 文档同步
- 代码变更时更新相关文档
- 使用标准Markdown格式
- 在文档头部包含版本和日期信息

---

## 🔄 版本历史摘要

### v4.5.0 (2025-12-03) - 当前版本
- ✅ **CSV列不匹配修复**：修复_append_to_summary_all()使用错误的fieldnames问题
- ✅ **参数精确优化**：每个参数使用精确的runs_per_config值，资源利用率>90%

### v4.4.0 (2025-12-02)
- ✅ **CSV追加bug修复**：修复数据覆盖问题，使用安全追加模式
- ✅ **去重机制**：支持基于历史CSV文件的实验去重
- ✅ **分阶段实验计划**：大规模实验分割成可管理的阶段

### v4.3.0 (2025-11-19)
- ✅ **11个模型完整支持**：基线+变异
- ✅ **动态变异系统**：log-uniform/uniform分布
- ✅ **并行训练**：前景+背景GPU同时利用
- ✅ **高精度能耗监控**：CPU误差<2%

### 完整版本历史
查看`docs/FEATURES_OVERVIEW.md`获取完整版本历史。

---

## 📞 问题排查

### 常见问题

#### 1. CSV格式错误
```
症状: GitHub报告"row X should actually have Y columns"
原因: _append_to_summary_all()列不匹配
解决: 运行验证测试，检查runner.py第167-200行
```

#### 2. 实验重复运行
```
症状: 相同超参数重复实验
原因: 去重机制未启用或配置错误
解决: 检查配置文件中的use_deduplication和historical_csvs设置
```

#### 3. 能耗数据缺失
```
症状: energy_*列为空
原因: perf权限问题或nvidia-smi不可用
解决: 使用sudo运行，检查GPU驱动
```

#### 4. 配置执行失败
```
症状: JSON配置文件解析错误
原因: 格式错误或路径不正确
解决: 使用python -m json.tool验证JSON格式
```

### 调试命令
```bash
# 检查CSV格式
python3 tests/verify_csv_append_fix.py

# 验证JSON配置
python3 -m json.tool settings/stage2_optimized_*.json

# 检查去重效果
python3 scripts/check_deduplication.py
```

---

## 📈 项目路线图

### 近期目标（1-2周）
1. 完成Stage2-6优化配置执行（184-192小时）
2. 达到100%实验完成度（所有参数5个唯一值）
3. 进行最终数据分析

### 中期改进（1个月）
1. 实现标准37列模板，避免列不匹配问题
2. 增强性能指标提取逻辑（特别是mnist_ff的test指标）
3. 添加CSV格式自动验证和修复工具

### 长期规划（3个月）
1. 扩展到更多深度学习模型
2. 实现更精细的能耗分析功能
3. 开发Web可视化界面

---

## ✅ 质量保证清单

### 每次提交前检查
- [ ] 所有测试通过：`python3 -m pytest tests/`
- [ ] CSV格式正确：`python3 tests/verify_csv_append_fix.py`
- [ ] 文档已更新：相关Markdown文件
- [ ] 配置已归档：过时文件移至`*/archived/`
- [ ] 版本号已更新：README.md和FEATURES_OVERVIEW.md

### 每次发布前检查
- [ ] 完整功能测试：所有11个模型运行正常
- [ ] 数据完整性验证：summary_all.csv格式正确
- [ ] 性能指标提取：所有模型性能数据完整
- [ ] 能耗监控：CPU/GPU数据准确
- [ ] 文档一致性：所有文档反映最新状态

---

**维护者**: Green
**Claude助手指南版本**: 1.0
**最后更新**: 2025-12-03
**状态**: ✅ 有效 - 请根据项目发展更新此文档

> 提示：将此文件保存在项目根目录，作为Claude助手的主要参考。当项目结构或规范变更时，及时更新此文档。
