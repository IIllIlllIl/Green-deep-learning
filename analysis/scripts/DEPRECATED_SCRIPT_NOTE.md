# ⚠️ 废弃脚本说明

## deprecated_generate_dibs_6groups_from_data_csv.py.bak

**废弃日期**: 2026-01-15
**废弃原因**:
- 错误使用了40%缺失率阈值，导致大量数据丢失（970→423行）
- 违背了6分组的设计原则：应该保留所有可用数据，而不是设置阈值

**正确的实现方案**:
请参考 `analysis/docs/reports/6GROUPS_DATA_DESIGN_CORRECT_20260115.md`

**注意**:
- 不要使用这个脚本
- 不要恢复这个脚本
- 正确的脚本应该基于共用特征分组，保留所有非空数据