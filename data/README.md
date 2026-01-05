# 核心数据文件

**位置**: `/home/green/energy_dl/nightly/data/`
**用途**: 存放项目核心数据文件

## 文件说明

### 主数据文件

- **raw_data.csv** - 主数据文件（836行，87列，95.1%完整性）
  - 所有实验的完整数据
  - 使用 experiment_id + timestamp 作为唯一标识
  - 最后更新: 2026-01-04

- **data.csv** - 精简数据文件
  - 统一并行/非并行字段
  - 添加 is_parallel 列

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
from tools.data_management.validate_raw_data import validate
```

## 相关工具

- `tools/data_management/validate_raw_data.py` - 数据验证
- `tools/data_management/analyze_experiment_status.py` - 实验状态分析
- `tools/data_management/repair_missing_energy_data.py` - 能耗数据修复

**最后更新**: 2026-01-05 (文件结构重组)
