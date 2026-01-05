# 数据处理工具

**位置**: `/home/green/energy_dl/nightly/tools/`
**用途**: 数据处理和配置管理工具

## 目录结构

```
tools/
├── data_management/      # 数据管理工具 (15个脚本)
├── config_management/    # 配置管理工具 (4个脚本)
└── legacy/               # 历史脚本归档 (24个脚本)
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
- `add_new_experiments_to_raw_data.py` - 添加新实验
- `update_raw_data_with_reextracted.py` - 用重提取数据更新
- `validate_merged_metrics.py` - 验证合并指标
- `merge_performance_metrics.py` - 合并性能指标

## config_management/ - 配置管理工具

- `generate_mutation_config.py` - 生成变异配置
- `validate_mutation_config.py` - 验证变异配置
- `verify_stage_configs.py` - 验证阶段配置
- `validate_models_config.py` - 验证模型配置

## legacy/ - 历史脚本

包含24个历史脚本，仅供参考。

**最后更新**: 2026-01-05 (文件结构重组)
