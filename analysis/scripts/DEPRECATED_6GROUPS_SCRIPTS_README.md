# 已废弃的6分组脚本说明

**日期**: 2026-01-15

## 废弃原因

以下脚本使用了**错误的统一缺失率阈值(40%)**,导致大量数据丢失(从818条降至423条)。

## 已废弃脚本

1. `deprecated_generate_6groups_data.py.bak` - 原始版本
2. `deprecated_generate_6groups_dibs_data.py.bak` - DiBS版本
3. `deprecated_generate_dibs_6groups_from_data_csv.py.bak` - data.csv版本

## 正确脚本

✅ **使用**: `generate_6groups_final.py`

**特性**:
- ✅ 语义超参数合并 (alpha ≡ weight_decay)
- ✅ 按组选择相关列 (无统一阈值)
- ✅ 保留所有818条可用数据
- ✅ 添加模型变量 (One-hot n-1编码)

## 使用方法

```bash
python3 analysis/scripts/generate_6groups_final.py
```
