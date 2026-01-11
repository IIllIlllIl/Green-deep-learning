# 历史数据归档

**位置**: `/home/green/energy_dl/nightly/archives/`
**用途**: 存放历史运行结果和数据快照

## 目录结构

```
archives/
├── runs/              # 历史运行结果（17个目录）
└── data_snapshots/    # 历史数据快照
```

## runs/ - 历史运行结果

包含2025年11月-12月期间的实验运行结果（17个目录）

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

**归档日期**: 2026-01-05 (文件结构重组)
