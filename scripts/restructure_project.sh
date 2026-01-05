#!/bin/bash

# 项目文件结构重组脚本
# 日期: 2026-01-05
# 版本: v1.0
# 用途: 自动化执行文件重组操作

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 工作目录
PROJECT_ROOT="/home/green/energy_dl/nightly"
BACKUP_DIR="$HOME/nightly_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 确认函数
confirm() {
    read -p "$1 (y/n): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# 备份函数
backup_project() {
    log_info "开始备份项目..."

    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="$BACKUP_DIR/nightly_backup_$TIMESTAMP.tar.gz"

    cd "$PROJECT_ROOT"
    tar -czf "$BACKUP_FILE" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='repos/*/data' \
        --exclude='repos/*/models' \
        .

    if [ -f "$BACKUP_FILE" ]; then
        BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log_success "备份完成: $BACKUP_FILE (大小: $BACKUP_SIZE)"
        echo "$BACKUP_FILE" > /tmp/nightly_last_backup.txt
        return 0
    else
        log_error "备份失败"
        return 1
    fi
}

# 创建新目录结构
create_new_structure() {
    log_info "创建新目录结构..."

    cd "$PROJECT_ROOT"

    # 创建新目录
    mkdir -p data/backups
    mkdir -p tools/data_management
    mkdir -p tools/config_management
    mkdir -p tools/legacy
    mkdir -p archives/runs
    mkdir -p archives/data_snapshots

    log_success "新目录结构创建完成"
}

# 移动核心数据文件
move_data_files() {
    log_info "移动核心数据文件..."

    cd "$PROJECT_ROOT"

    # 移动主数据文件
    if [ -f "results/raw_data.csv" ]; then
        mv results/raw_data.csv data/
        log_success "已移动: raw_data.csv → data/"
    fi

    if [ -f "results/data.csv" ]; then
        mv results/data.csv data/
        log_success "已移动: data.csv → data/"
    fi

    if [ -f "results/recoverable_energy_data.json" ]; then
        mv results/recoverable_energy_data.json data/
        log_success "已移动: recoverable_energy_data.json → data/"
    fi

    # 移动备份文件
    if ls results/raw_data.csv.backup_* 1> /dev/null 2>&1; then
        mv results/raw_data.csv.backup_* data/backups/
        log_success "已移动: 所有 raw_data.csv.backup_* → data/backups/"
    fi

    if ls results/raw_data.backup_* 1> /dev/null 2>&1; then
        mv results/raw_data.backup_* data/backups/ 2>/dev/null || true
        log_success "已移动: 所有 raw_data.backup_* → data/backups/"
    fi
}

# 移动数据管理脚本
move_data_management_scripts() {
    log_info "移动数据管理脚本..."

    cd "$PROJECT_ROOT"

    # 数据管理工具（最近活跃）
    local data_mgmt_scripts=(
        "analyze_experiment_status.py"
        "analyze_missing_energy_data.py"
        "repair_missing_energy_data.py"
        "verify_recoverable_data.py"
        "validate_raw_data.py"
        "append_session_to_raw_data.py"
        "compare_data_vs_raw_data.py"
        "check_attribute_mapping.py"
        "check_latest_results.py"
        "create_unified_data_csv.py"
        "add_new_experiments_to_raw_data.py"
        "merge_csv_to_raw_data.py"
        "update_raw_data_with_reextracted.py"
        "validate_merged_metrics.py"
        "merge_performance_metrics.py"
    )

    for script in "${data_mgmt_scripts[@]}"; do
        if [ -f "scripts/$script" ]; then
            mv "scripts/$script" tools/data_management/
            log_success "已移动: $script → tools/data_management/"
        fi
    done
}

# 移动配置管理脚本
move_config_management_scripts() {
    log_info "移动配置管理脚本..."

    cd "$PROJECT_ROOT"

    # 配置管理工具
    local config_mgmt_scripts=(
        "generate_mutation_config.py"
        "validate_mutation_config.py"
        "verify_stage_configs.py"
        "validate_models_config.py"
    )

    for script in "${config_mgmt_scripts[@]}"; do
        if [ -f "scripts/$script" ]; then
            mv "scripts/$script" tools/config_management/
            log_success "已移动: $script → tools/config_management/"
        fi
    done
}

# 移动历史脚本到legacy
move_legacy_scripts() {
    log_info "移动历史脚本到legacy..."

    cd "$PROJECT_ROOT"

    # 移动archived子目录
    if [ -d "scripts/archived" ]; then
        mv scripts/archived tools/legacy/
        log_success "已移动: scripts/archived/ → tools/legacy/"
    fi

    # 移动剩余所有脚本
    if ls scripts/*.py 1> /dev/null 2>&1; then
        mv scripts/*.py tools/legacy/
        log_success "已移动: 剩余Python脚本 → tools/legacy/"
    fi

    if ls scripts/*.sh 1> /dev/null 2>&1; then
        mv scripts/*.sh tools/legacy/
        log_success "已移动: 剩余Shell脚本 → tools/legacy/"
    fi

    # 移动__pycache__
    if [ -d "scripts/__pycache__" ]; then
        rm -rf scripts/__pycache__
        log_success "已删除: scripts/__pycache__"
    fi
}

# 归档历史运行结果
archive_historical_runs() {
    log_info "归档历史运行结果..."

    cd "$PROJECT_ROOT"

    # 移动run_*目录
    if ls -d results/run_* 1> /dev/null 2>&1; then
        mv results/run_* archives/runs/
        local run_count=$(ls -d archives/runs/run_* | wc -l)
        log_success "已归档: $run_count 个历史运行目录 → archives/runs/"
    fi
}

# 归档历史数据快照
archive_data_snapshots() {
    log_info "归档历史数据快照..."

    cd "$PROJECT_ROOT"

    # 移动历史CSV文件
    [ -f "results/summary_old.csv" ] && mv results/summary_old.csv archives/data_snapshots/
    [ -f "results/summary_new.csv" ] && mv results/summary_new.csv archives/data_snapshots/

    # 移动子目录
    [ -d "results/collector" ] && mv results/collector archives/data_snapshots/
    [ -d "results/archived" ] && mv results/archived archives/data_snapshots/
    [ -d "results/default" ] && mv results/default archives/data_snapshots/
    [ -d "results/mutation_1x" ] && mv results/mutation_1x archives/data_snapshots/

    # 移动mutation_2x和backup_archive
    if ls -d results/mutation_2x_* 1> /dev/null 2>&1; then
        mv results/mutation_2x_* archives/data_snapshots/
    fi

    if ls -d results/backup_archive_* 1> /dev/null 2>&1; then
        mv results/backup_archive_* archives/data_snapshots/
    fi

    log_success "历史数据快照已归档 → archives/data_snapshots/"
}

# 创建README文件
create_readme_files() {
    log_info "创建README文件..."

    # data/README.md
    cat > "$PROJECT_ROOT/data/README.md" << 'EOF'
# 核心数据文件

**位置**: `/home/green/energy_dl/nightly/data/`
**用途**: 存放项目核心数据文件

## 文件说明

### 主数据文件

- **raw_data.csv** - 主数据文件（836行，87列，95.1%完整性）
  - 所有实验的完整数据
  - 使用 experiment_id + timestamp 作为唯一标识
  - 最后更新: 2026-01-04

- **data.csv** - 精简数据文件（待更新）
  - 统一并行/非并行字段
  - 添加 is_parallel 列
  - 需要重新生成以反映最新数据

- **recoverable_energy_data.json** - 可恢复能耗数据
  - 253个实验的能耗数据
  - 用于数据修复

### 备份文件

- **backups/** - 数据备份目录
  - raw_data.csv.backup_* - 历史备份

## 使用示例

```python
import pandas as pd

# 读取主数据文件
df = pd.read_csv('data/raw_data.csv')

# 验证数据完整性
from tools.data_management.validate_raw_data import validate_raw_data
validate_raw_data('data/raw_data.csv')
```

## 相关工具

- `tools/data_management/validate_raw_data.py` - 数据验证
- `tools/data_management/analyze_experiment_status.py` - 实验状态分析
- `tools/data_management/repair_missing_energy_data.py` - 能耗数据修复

**最后更新**: 2026-01-05
EOF

    # tools/README.md
    cat > "$PROJECT_ROOT/tools/README.md" << 'EOF'
# 数据处理工具

**位置**: `/home/green/energy_dl/nightly/tools/`
**用途**: 数据处理和配置管理工具

## 目录结构

```
tools/
├── data_management/      # 数据管理工具
├── config_management/    # 配置管理工具
└── legacy/               # 历史脚本归档
```

## data_management/ - 数据管理工具

### 数据验证与分析

- `validate_raw_data.py` - 验证raw_data.csv完整性
- `analyze_experiment_status.py` - 分析实验状态
- `analyze_missing_energy_data.py` - 分析缺失能耗数据
- `check_attribute_mapping.py` - 检查属性映射
- `check_latest_results.py` - 检查最新结果

### 数据修复

- `repair_missing_energy_data.py` - 修复缺失能耗数据
- `verify_recoverable_data.py` - 验证可恢复数据

### 数据合并与追加

- `append_session_to_raw_data.py` - 追加新实验数据
- `merge_csv_to_raw_data.py` - 合并CSV到raw_data
- `compare_data_vs_raw_data.py` - 对比data.csv和raw_data.csv
- `create_unified_data_csv.py` - 创建统一的data.csv

## config_management/ - 配置管理工具

- `generate_mutation_config.py` - 生成变异配置
- `validate_mutation_config.py` - 验证变异配置
- `verify_stage_configs.py` - 验证阶段配置
- `validate_models_config.py` - 验证模型配置

## legacy/ - 历史脚本

包含40+个历史脚本，仅供参考。

**最后更新**: 2026-01-05
EOF

    # archives/README.md
    cat > "$PROJECT_ROOT/archives/README.md" << 'EOF'
# 历史数据归档

**位置**: `/home/green/energy_dl/nightly/archives/`
**用途**: 存放历史运行结果和数据快照

## 目录结构

```
archives/
├── runs/              # 历史运行结果（22个目录，~1.8GB）
└── data_snapshots/    # 历史数据快照
```

## runs/ - 历史运行结果

包含2025年11月-12月期间的所有实验运行结果：

- run_20251126_224751/
- run_20251201_221847/
- run_20251202_185830/
- ... (共22个目录)

每个目录包含：
- summary.csv - 该次运行的汇总数据
- 其他实验输出文件

## data_snapshots/ - 历史数据快照

- summary_old.csv - 旧版汇总数据
- summary_new.csv - 新版汇总数据
- collector/ - 数据收集器相关文件
- archived/ - 已归档的历史数据
- default/ - 默认配置运行结果
- mutation_1x/ - 1x变异运行结果
- mutation_2x_*/ - 2x变异运行结果
- backup_archive_*/ - 备份归档

## 注意事项

- ⚠️ 这些文件仅用于历史参考，不应用于当前分析
- ⚠️ 当前分析请使用 `data/raw_data.csv` (95.1%完整性)
- ⚠️ 总大小约1.8GB，如空间不足可考虑压缩或删除

**归档日期**: 2026-01-05
EOF

    log_success "README文件创建完成"
}

# 清理空目录
cleanup_empty_dirs() {
    log_info "清理空目录..."

    cd "$PROJECT_ROOT"

    # 删除空的scripts目录
    if [ -d "scripts" ] && [ -z "$(ls -A scripts)" ]; then
        rmdir scripts
        log_success "已删除空目录: scripts/"
    fi

    # 删除空的results目录（如果需要）
    if [ -d "results" ] && [ -z "$(ls -A results)" ]; then
        rmdir results
        log_success "已删除空目录: results/"
    fi
}

# 验证重组结果
verify_restructure() {
    log_info "验证重组结果..."

    cd "$PROJECT_ROOT"

    local errors=0

    # 验证核心数据文件
    if [ ! -f "data/raw_data.csv" ]; then
        log_error "缺失: data/raw_data.csv"
        ((errors++))
    else
        log_success "验证通过: data/raw_data.csv"
    fi

    if [ ! -f "data/data.csv" ]; then
        log_warning "缺失: data/data.csv (可能本来不存在)"
    else
        log_success "验证通过: data/data.csv"
    fi

    # 验证工具目录
    if [ ! -d "tools/data_management" ]; then
        log_error "缺失: tools/data_management/"
        ((errors++))
    else
        local count=$(ls tools/data_management/*.py 2>/dev/null | wc -l)
        log_success "验证通过: tools/data_management/ ($count 个脚本)"
    fi

    if [ ! -d "tools/config_management" ]; then
        log_error "缺失: tools/config_management/"
        ((errors++))
    else
        local count=$(ls tools/config_management/*.py 2>/dev/null | wc -l)
        log_success "验证通过: tools/config_management/ ($count 个脚本)"
    fi

    # 验证归档目录
    if [ ! -d "archives/runs" ]; then
        log_error "缺失: archives/runs/"
        ((errors++))
    else
        local count=$(ls -d archives/runs/run_* 2>/dev/null | wc -l)
        log_success "验证通过: archives/runs/ ($count 个历史运行)"
    fi

    # 验证README文件
    [ -f "data/README.md" ] && log_success "验证通过: data/README.md"
    [ -f "tools/README.md" ] && log_success "验证通过: tools/README.md"
    [ -f "archives/README.md" ] && log_success "验证通过: archives/README.md"

    if [ $errors -eq 0 ]; then
        log_success "所有验证通过！"
        return 0
    else
        log_error "发现 $errors 个错误"
        return 1
    fi
}

# 显示总结
show_summary() {
    echo
    echo "======================================"
    echo "  项目重组完成总结"
    echo "======================================"
    echo
    echo "✅ 新目录结构:"
    echo "   - data/              核心数据文件"
    echo "   - tools/             数据处理工具"
    echo "   - archives/          历史数据归档"
    echo
    echo "✅ 数据文件位置:"
    echo "   - data/raw_data.csv"
    echo "   - data/data.csv"
    echo
    echo "✅ 工具脚本位置:"
    echo "   - tools/data_management/"
    echo "   - tools/config_management/"
    echo "   - tools/legacy/"
    echo
    echo "✅ 历史数据位置:"
    echo "   - archives/runs/"
    echo "   - archives/data_snapshots/"
    echo
    echo "⚠️  下一步操作:"
    echo "   1. 运行路径更新脚本: python3 update_paths.py"
    echo "   2. 更新文档: 见 docs/RESTRUCTURE_PLAN_20260105.md"
    echo "   3. 验证功能: python3 tools/data_management/validate_raw_data.py"
    echo
    if [ -f "/tmp/nightly_last_backup.txt" ]; then
        echo "📦 备份位置:"
        cat /tmp/nightly_last_backup.txt
        echo
    fi
    echo "======================================"
}

# 主函数
main() {
    echo
    echo "======================================"
    echo "  项目文件结构重组脚本"
    echo "  版本: v1.0"
    echo "  日期: 2026-01-05"
    echo "======================================"
    echo

    log_warning "此脚本将重组项目文件结构"
    log_warning "请确保已阅读 docs/RESTRUCTURE_PLAN_20260105.md"
    echo

    if ! confirm "是否继续执行重组操作？"; then
        log_info "操作已取消"
        exit 0
    fi

    # 执行步骤
    backup_project || { log_error "备份失败，终止操作"; exit 1; }

    create_new_structure
    move_data_files
    move_data_management_scripts
    move_config_management_scripts
    move_legacy_scripts
    archive_historical_runs
    archive_data_snapshots
    create_readme_files
    cleanup_empty_dirs

    # 验证
    if verify_restructure; then
        show_summary
        log_success "项目重组成功完成！"
        exit 0
    else
        log_error "重组验证失败，请检查错误信息"
        log_warning "可以从备份恢复: $(cat /tmp/nightly_last_backup.txt)"
        exit 1
    fi
}

# 执行主函数
main "$@"
