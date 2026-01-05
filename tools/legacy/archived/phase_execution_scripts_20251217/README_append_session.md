# 快速参考：追加Session数据到raw_data.csv

## 🚀 最常用命令

### 1. 测试运行（推荐先执行）
```bash
python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS --dry-run
```

### 2. 实际追加
```bash
python3 tools/data_management/append_session_to_raw_data.py results/run_YYYYMMDD_HHMMSS
```

---

## 📋 典型工作流程

```bash
# 步骤1: 查找最新session
ls -td results/run_* | head -1

# 步骤2: Dry-run检查
python3 tools/data_management/append_session_to_raw_data.py $(ls -td results/run_* | head -1) --dry-run

# 步骤3: 实际执行
python3 tools/data_management/append_session_to_raw_data.py $(ls -td results/run_* | head -1)
```

---

## ✅ 预期输出

### 成功场景
```
✅ 加载现有数据: 480行
✅ examples_mnist_new_001: 训练成功: True
=== 总结 ===
新增实验: 2个
✅ 已更新: data/raw_data.csv
✅ 数据完整性验证通过
```

### 无新实验
```
⚠️  跳过 examples_mnist_001: 重复实验
⚠️  未找到新实验，无需更新
```

---

## 🔍 故障排除

### 问题: 所有实验都重复
**原因**: Session已经追加过
**解决**: 正常现象，无需操作

### 问题: 未知仓库警告
**原因**: `models_config.json` 中缺少配置
**解决**: 添加仓库配置或忽略

---

## 📚 完整文档

- **详细指南**: `docs/APPEND_SESSION_TO_RAW_DATA_GUIDE.md`
- **开发报告**: `docs/results_reports/APPEND_SESSION_SCRIPT_DEV_REPORT.md`
- **测试套件**: `tests/test_append_session_to_raw_data.py`
