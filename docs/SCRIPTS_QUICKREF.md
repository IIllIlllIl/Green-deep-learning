# Scripts 快速参考

**最后更新**: 2025-12-12
**版本**: v4.7.3

---

## 📊 脚本概览

scripts目录包含10个核心脚本，分为4类：

| 类别 | 脚本数 | 说明 |
|------|--------|------|
| 核心工具 | 3 | CSV处理和归档 |
| 配置工具 | 3 | 配置生成和验证 |
| 分析工具 | 3 | 实验数据分析 |
| 下载工具 | 1 | 预训练模型下载 |
| **总计** | **10** | - |

---

## 🔧 核心工具

### 1. merge_csv_to_raw_data.py
**功能**: 合并summary_old.csv和summary_new.csv为raw_data.csv

**用途**:
- 合并老实验（93列→80列转换）和新实验数据
- 生成476行、80列的主数据文件
- 自动验证数据完整性

**运行**:
```bash
python3 scripts/merge_csv_to_raw_data.py
```

**输出**:
- `results/raw_data.csv` - 主数据文件（476行，80列）

---

### 2. validate_raw_data.py
**功能**: 验证raw_data.csv的数据完整性和安全性

**检查项**:
- ✓ 列数（80列）
- ✓ 行数（476行）
- ✓ 训练成功率
- ✓ 能耗数据完整性（CPU + GPU）
- ✓ 性能指标完整性
- ✓ experiment_id重复分析

**运行**:
```bash
python3 scripts/validate_raw_data.py
```

**输出**: 详细验证报告（终端输出）

---

### 3. archive_summary_files.py
**功能**: 归档过时的summary文件和备份

**归档内容**:
- 5个过时summary文件（summary_all.csv, summary_all_enhanced.csv等）
- 8个过时备份文件
- 自动生成归档清单

**运行**:
```bash
python3 scripts/archive_summary_files.py
```

**输出**:
- `results/summary_archive/` - 归档目录
- `results/summary_archive/README_ARCHIVE.md` - 归档说明

---

## ⚙️ 配置工具

### 4. generate_mutation_config.py
**功能**: 生成变异实验配置文件

**用途**: 创建settings JSON配置文件

**运行**:
```bash
python3 scripts/generate_mutation_config.py
```

---

### 5. validate_mutation_config.py
**功能**: 验证变异实验配置文件

**检查项**:
- JSON格式正确性
- 必填字段完整性
- 参数值有效性

**运行**:
```bash
python3 scripts/validate_mutation_config.py <config_file.json>
```

---

### 6. verify_stage_configs.py
**功能**: 验证stage配置文件

**用途**: 检查分阶段实验配置的正确性

**运行**:
```bash
python3 scripts/verify_stage_configs.py
```

---

## 📊 分析工具

### 7. analyze_baseline.py
**功能**: 分析基线实验数据

**用途**: 统计和分析基线训练结果

**运行**:
```bash
python3 scripts/analyze_baseline.py
```

---

### 8. analyze_experiments.py
**功能**: 分析实验数据

**用途**: 综合分析实验结果（变异、基线等）

**运行**:
```bash
python3 scripts/analyze_experiments.py
```

---

### 9. analyze_archive_plan.py
**功能**: 分析归档计划

**用途**: 生成脚本和文档归档方案

**运行**:
```bash
python3 scripts/analyze_archive_plan.py
```

**输出**: 归档计划报告（终端输出）

---

## 📥 下载工具

### 10. download_pretrained_models.py
**功能**: 下载预训练模型

**用途**: 自动下载所需的预训练模型文件

**运行**:
```bash
python3 scripts/download_pretrained_models.py
```

---

## 📁 归档的脚本

22个已完成任务的脚本已归档至：
- `scripts/archived/completed_tasks_20251212/`

归档脚本分类：
- **数据重建** (7个): 93列重建、CSV转换、步骤脚本等
- **数据修复** (7个): 修复实验来源、添加字段、填充值等
- **配置修复** (1个): Stage配置修复
- **数据分离** (2个): 新老实验分离、白名单提取
- **临时分析** (3个): 列分析、字段覆盖分析、schema生成
- **验证脚本** (1个): 93列重建验证
- **已废弃** (1个): aggregate_csvs.py（已被merge_csv_to_raw_data.py替代）

查看归档清单：
```bash
cat scripts/archived/completed_tasks_20251212/README_ARCHIVE.md
```

---

## 🔗 相关文档

- [SCRIPTS_DOCUMENTATION.md](SCRIPTS_DOCUMENTATION.md) - 脚本详细文档
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 项目快速参考
- [README.md](../README.md) - 项目主文档

---

## 💡 使用建议

### 数据处理流程

1. **合并数据**:
   ```bash
   python3 scripts/merge_csv_to_raw_data.py
   ```

2. **验证数据**:
   ```bash
   python3 scripts/validate_raw_data.py
   ```

3. **归档文件**:
   ```bash
   python3 scripts/archive_summary_files.py
   ```

### 配置验证流程

1. **生成配置**:
   ```bash
   python3 scripts/generate_mutation_config.py
   ```

2. **验证配置**:
   ```bash
   python3 scripts/validate_mutation_config.py settings/your_config.json
   ```

### 数据分析流程

1. **分析基线**:
   ```bash
   python3 scripts/analyze_baseline.py
   ```

2. **分析实验**:
   ```bash
   python3 scripts/analyze_experiments.py
   ```

---

**维护者**: Green
**版本**: v4.7.3
**状态**: ✅ 已归档22个脚本，保留10个核心脚本
