# 项目文件结构重组方案

**日期**: 2026-01-05
**版本**: v1.0
**状态**: 待执行

---

## 📋 重组目标

1. **活跃文件靠近根目录** - 减少访问层级
2. **历史数据集中归档** - 释放主目录空间
3. **功能模块清晰分离** - 提高可维护性
4. **保持向后兼容** - 更新所有引用路径

---

## 🔄 结构对比

### 当前结构问题

```
energy_dl/nightly/
├── scripts/                    # 40+个脚本（数据处理为主）
├── analysis/                   # 因果分析模块（独立）
│   └── scripts/                # 25+个分析脚本
├── docs/                       # 文档（多层嵌套）
│   ├── results_reports/        # 实验报告
│   ├── archived/               # 归档文档
│   └── environment/            # 环境文档
├── results/                    # 数据文件 + 22个历史run目录
│   ├── raw_data.csv           # ⭐ 核心数据
│   ├── data.csv               # ⭐ 核心数据
│   ├── run_20251201_*/        # 历史运行（1.8GB+）
│   └── ...
└── mutation/                   # 训练核心代码
```

**问题**：
- ❌ 活跃脚本分散在两个目录
- ❌ 核心数据文件埋在results深处
- ❌ 历史run目录占用大量空间
- ❌ 文档层级过深

### 优化后结构

```
energy_dl/nightly/
├── CLAUDE.md                   # 快速指南
├── README.md                   # 项目总览
├── mutation.py                 # 训练入口
│
├── data/                       # ⭐ 核心数据（上浮）
│   ├── raw_data.csv           # 主数据文件（836行，95.1%完整）
│   ├── data.csv               # 精简数据文件
│   ├── recoverable_energy_data.json
│   └── backups/               # 数据备份
│       └── raw_data.csv.backup_*
│
├── tools/                      # ⭐ 数据处理工具（合并+精简）
│   ├── data_management/       # 数据管理
│   │   ├── analyze_experiment_status.py      # 最近使用
│   │   ├── analyze_missing_energy_data.py    # 最近使用
│   │   ├── repair_missing_energy_data.py     # 最近使用
│   │   ├── verify_recoverable_data.py
│   │   ├── validate_raw_data.py
│   │   ├── append_session_to_raw_data.py
│   │   └── compare_data_vs_raw_data.py
│   ├── config_management/     # 配置管理
│   │   ├── generate_mutation_config.py
│   │   ├── validate_mutation_config.py
│   │   └── verify_stage_configs.py
│   └── legacy/                # 旧脚本归档
│       └── (40个历史脚本)
│
├── analysis/                   # 因果分析模块（保持独立）
│   ├── README.md
│   ├── docs/
│   │   ├── INDEX.md           # 分析模块索引
│   │   ├── QUESTION1_REGRESSION_ANALYSIS_PLAN.md
│   │   └── reports/
│   ├── scripts/               # 分析脚本（保持原位）
│   ├── utils/                 # 分析核心模块
│   ├── data/                  # 分析专用数据
│   └── results/               # 分析结果
│
├── mutation/                   # 训练核心代码（保持不变）
│   ├── runner.py
│   ├── energy.py
│   ├── hyperparams.py
│   └── ...
│
├── docs/                       # 项目文档（扁平化）
│   ├── CLAUDE_FULL_REFERENCE.md     # 完整参考
│   ├── QUICK_REFERENCE.md
│   ├── SCRIPTS_QUICKREF.md
│   ├── JSON_CONFIG_WRITING_STANDARDS.md
│   ├── reports/               # 实验报告（保持）
│   │   ├── DATA_REPAIR_REPORT_20260104.md
│   │   ├── PROJECT_PROGRESS_COMPLETE_SUMMARY.md
│   │   └── ...
│   └── archived/              # 归档文档
│
├── archives/                   # ⭐ 历史数据归档（新建）
│   ├── runs/                  # 历史运行结果
│   │   ├── run_20251201_221847/
│   │   ├── run_20251202_185830/
│   │   └── ... (22个目录)
│   ├── data_snapshots/        # 历史数据快照
│   │   ├── summary_old.csv
│   │   ├── summary_new.csv
│   │   └── collector/
│   └── README.md              # 归档说明
│
├── settings/                   # 实验配置（保持）
├── tests/                      # 测试（保持）
├── repos/                      # 训练仓库（保持）
└── environment/                # 环境配置（保持）
```

---

## 📦 详细变更清单

### 1. 数据文件上浮 ⭐⭐⭐

**原因**: 核心数据文件是最常访问的文件，应该在根目录附近

```bash
# 移动核心数据文件
data/raw_data.csv                    → data/raw_data.csv
data/data.csv                        → data/data.csv
data/recoverable_energy_data.json    → data/recoverable_energy_data.json

# 移动备份文件
data/raw_data.csv.backup_*           → data/backups/
results/raw_data.backup_*               → data/backups/
```

**影响**: 需要更新所有脚本中的数据路径

### 2. 脚本工具整合 ⭐⭐⭐

**原因**: 减少scripts/和analysis/scripts/的混淆，按功能分类

```bash
# 创建新的tools/目录，按功能分类
mkdir -p tools/{data_management,config_management,legacy}

# 数据管理工具（最近活跃的脚本）
tools/data_management/analyze_experiment_status.py           → tools/data_management/
tools/data_management/analyze_missing_energy_data.py         → tools/data_management/
tools/data_management/repair_missing_energy_data.py          → tools/data_management/
tools/data_management/verify_recoverable_data.py             → tools/data_management/
tools/data_management/validate_raw_data.py                   → tools/data_management/
tools/data_management/append_session_to_raw_data.py          → tools/data_management/
tools/data_management/compare_data_vs_raw_data.py            → tools/data_management/
scripts/check_attribute_mapping.py             → tools/data_management/
scripts/check_latest_results.py                → tools/data_management/

# 配置管理工具
tools/config_management/generate_mutation_config.py            → tools/config_management/
tools/config_management/validate_mutation_config.py            → tools/config_management/
scripts/verify_stage_configs.py                → tools/config_management/
scripts/validate_models_config.py              → tools/config_management/

# 历史脚本归档（不常用）
scripts/archived/*                             → tools/legacy/archived/
scripts/{其余30+个脚本}                         → tools/legacy/
```

**保持**: `analysis/scripts/` 保持独立（因果分析专用）

### 3. 历史运行结果归档 ⭐⭐

**原因**: 释放results/目录，历史数据集中管理

```bash
# 创建archives/目录
mkdir -p archives/runs
mkdir -p archives/data_snapshots

# 移动历史运行结果（22个目录，~1.8GB）
results/run_20251126_224751/    → archives/runs/
results/run_20251201_221847/    → archives/runs/
results/run_20251202_185830/    → archives/runs/
... (所有run_*目录)

# 移动历史数据快照
results/summary_old.csv         → archives/data_snapshots/
results/summary_new.csv         → archives/data_snapshots/
results/collector/              → archives/data_snapshots/collector/
results/archived/               → archives/data_snapshots/archived/
results/default/                → archives/data_snapshots/default/
results/mutation_1x/            → archives/data_snapshots/mutation_1x/
results/mutation_2x_*/          → archives/data_snapshots/
results/backup_archive_20251219/ → archives/data_snapshots/
```

**结果**: results/目录只保留核心文件，或者直接删除（数据已上浮到data/）

### 4. 文档结构优化 ⭐

**原因**: 减少文档嵌套，常用文档提升

```bash
# docs/目录保持相对扁平
docs/
├── CLAUDE_FULL_REFERENCE.md      # 保持
├── QUICK_REFERENCE.md             # 保持
├── SCRIPTS_QUICKREF.md            # 保持
├── JSON_CONFIG_WRITING_STANDARDS.md # 保持
├── reports/                       # 保持
│   ├── DATA_REPAIR_REPORT_20260104.md
│   ├── PROJECT_PROGRESS_COMPLETE_SUMMARY.md
│   └── ...
├── archived/                      # 保持
└── environment/                   # 保持（环境配置文档）
```

**移除**: `docs/results_reports/` 简化为 `docs/reports/`（可选）

### 5. 保持不变的目录

```bash
# 以下目录结构保持不变
analysis/          # 因果分析模块（独立）
mutation/          # 训练核心代码
settings/          # 实验配置
tests/             # 测试
repos/             # 训练仓库
environment/       # 环境配置
```

---

## 🔧 实施步骤

### 步骤1: 备份当前状态

```bash
cd /home/green/energy_dl/nightly
tar -czf ~/nightly_backup_$(date +%Y%m%d_%H%M%S).tar.gz .
```

### 步骤2: 创建新目录结构

```bash
mkdir -p data/backups
mkdir -p tools/{data_management,config_management,legacy}
mkdir -p archives/{runs,data_snapshots}
```

### 步骤3: 移动核心数据文件

```bash
# 数据文件
mv data/raw_data.csv data/
mv data/data.csv data/
mv data/recoverable_energy_data.json data/

# 备份文件
mv data/raw_data.csv.backup_* data/backups/
mv results/raw_data.backup_* data/backups/ 2>/dev/null || true
```

### 步骤4: 重组脚本目录

```bash
# 数据管理工具（活跃脚本）
mv tools/data_management/analyze_experiment_status.py tools/data_management/
mv tools/data_management/analyze_missing_energy_data.py tools/data_management/
mv tools/data_management/repair_missing_energy_data.py tools/data_management/
mv tools/data_management/verify_recoverable_data.py tools/data_management/
mv tools/data_management/validate_raw_data.py tools/data_management/
mv tools/data_management/append_session_to_raw_data.py tools/data_management/
mv tools/data_management/compare_data_vs_raw_data.py tools/data_management/
mv scripts/check_attribute_mapping.py tools/data_management/
mv scripts/check_latest_results.py tools/data_management/

# 配置管理工具
mv tools/config_management/generate_mutation_config.py tools/config_management/
mv tools/config_management/validate_mutation_config.py tools/config_management/
mv scripts/verify_stage_configs.py tools/config_management/
mv scripts/validate_models_config.py tools/config_management/

# 其余脚本归档
mv scripts/archived tools/legacy/
mv scripts/*.py tools/legacy/
mv scripts/*.sh tools/legacy/
```

### 步骤5: 归档历史运行结果

```bash
# 移动历史运行结果
mv results/run_* archives/runs/

# 移动历史数据快照
mv results/summary_old.csv archives/data_snapshots/
mv results/summary_new.csv archives/data_snapshots/
mv results/collector archives/data_snapshots/
mv results/archived archives/data_snapshots/
mv results/default archives/data_snapshots/
mv results/mutation_* archives/data_snapshots/
mv results/backup_archive_* archives/data_snapshots/
```

### 步骤6: 更新脚本路径引用

使用自动化脚本更新所有路径引用（见下一节）

### 步骤7: 验证重组结果

```bash
# 验证核心文件存在
ls -lh data/raw_data.csv
ls -lh data/data.csv

# 验证工具目录
ls tools/data_management/
ls tools/config_management/

# 验证归档目录
ls archives/runs/ | wc -l  # 应该显示22个目录

# 运行测试
python3 tools/data_management/validate_raw_data.py
```

---

## 🔍 路径更新清单

### 需要更新的文件类型

1. **Python脚本** - `data/raw_data.csv` → `data/raw_data.csv`
2. **文档** - 所有Markdown文件中的路径引用
3. **配置文件** - settings/中的JSON配置
4. **测试脚本** - tests/中的测试代码

### 自动化更新脚本

将创建 `tools/update_paths.py` 脚本来自动更新所有路径引用。

### 关键路径映射

```python
PATH_MAPPINGS = {
    # 数据文件
    'data/raw_data.csv': 'data/raw_data.csv',
    'data/data.csv': 'data/data.csv',
    'data/recoverable_energy_data.json': 'data/recoverable_energy_data.json',

    # 脚本路径
    'tools/data_management/analyze_experiment_status.py': 'tools/data_management/analyze_experiment_status.py',
    'tools/data_management/validate_raw_data.py': 'tools/data_management/validate_raw_data.py',
    # ... 更多映射

    # 相对路径导入
    '../data/raw_data.csv': '../data/raw_data.csv',
    '../../data/raw_data.csv': '../../data/raw_data.csv',
}
```

---

## 📊 预期效果

### 空间优化

- **释放空间**: ~1.8GB 历史数据移到archives/
- **核心目录**: results/ 大小从 2GB+ 降到 <10MB

### 访问优化

**之前**:
```bash
cd /home/green/energy_dl/nightly
vim data/raw_data.csv                    # 3层
vim tools/data_management/validate_raw_data.py            # 2层
```

**之后**:
```bash
cd /home/green/energy_dl/nightly
vim data/raw_data.csv                       # 2层（减少1层）
vim tools/data_management/validate_raw_data.py  # 3层（功能更清晰）
```

### 功能分类

- ✅ **data/** - 核心数据文件（清晰标识）
- ✅ **tools/** - 数据处理工具（按功能分类）
- ✅ **analysis/** - 因果分析（独立模块）
- ✅ **archives/** - 历史数据（集中管理）

---

## ⚠️ 风险与注意事项

### 高风险操作

1. **数据文件移动** - 必须确保所有备份完成
2. **路径引用更新** - 遗漏会导致脚本失败
3. **Git历史** - 如果使用Git，考虑使用 `git mv` 而不是 `mv`

### 兼容性检查

执行前必须检查：
- [ ] 所有脚本中的硬编码路径
- [ ] Markdown文档中的文件引用
- [ ] 配置文件中的路径设置
- [ ] 测试脚本中的路径

### 回滚方案

如果出现问题：
```bash
# 从备份恢复
cd /home/green/energy_dl/nightly
rm -rf *
tar -xzf ~/nightly_backup_YYYYMMDD_HHMMSS.tar.gz
```

---

## 📝 文档更新清单

### 必须更新的文档

1. **CLAUDE.md** - 项目结构快览部分
2. **docs/CLAUDE_FULL_REFERENCE.md** - 完整文件结构
3. **README.md** - 项目总览
4. **analysis/docs/INDEX.md** - 分析模块索引
5. **docs/SCRIPTS_QUICKREF.md** - 脚本快速参考
6. **所有reports/** - 文件路径引用

### 新建文档

1. **data/README.md** - 数据目录说明
2. **tools/README.md** - 工具目录说明
3. **archives/README.md** - 归档目录说明

---

## ✅ 验证清单

执行重组后，必须验证：

- [ ] 核心数据文件完整性（md5sum校验）
- [ ] 关键脚本可执行（validate_raw_data.py等）
- [ ] 文档链接正确（所有Markdown内部链接）
- [ ] 测试通过（pytest tests/）
- [ ] Git状态正常（如果使用Git）
- [ ] 磁盘空间释放（df -h检查）

---

## 🎯 总结

### 核心改进

1. ✅ **数据文件上浮** - data/ 目录清晰标识核心数据
2. ✅ **工具分类整合** - tools/ 按功能分类，易于查找
3. ✅ **历史数据归档** - archives/ 集中管理，释放主目录
4. ✅ **保持模块独立** - analysis/ 因果分析模块保持独立

### 下一步

1. 审核本方案
2. 执行重组脚本
3. 更新所有路径引用
4. 验证功能正常
5. 更新所有文档

---

**方案设计**: Claude Code
**版本**: v1.0
**日期**: 2026-01-05
