# Stage5-6配置归档说明

**归档日期**: 2025-12-05
**原因**: 被Stage11-12替代

---

## 归档原因

Stage5和Stage6的目标与新创建的Stage11-12完全相同：
- Stage5 = Stage11: hrnet18并行模式补充
- Stage6 = Stage12: pcb并行模式补充

## 为什么使用Stage11-12替代

| 对比项 | Stage5-6 (旧) | Stage11-12 (新) | 优势 |
|-------|--------------|----------------|------|
| 配置方式 | 每个参数单独变异 | 所有参数一起变异 | 更符合mutation.py设计 |
| runs设置 | 4次/参数 | 5次/配置 | 覆盖更全面 |
| 实验总数 | 32个 (16+16) | 40个 (20+20) | 实验数更多 |
| 预计时长 | 96小时 | 51.7小时 | 节省44.3小时 |
| 配置格式 | 旧格式 | 新格式 | 兼容性更好 |

## 替代配置

- Stage5 → **Stage11**: `settings/stage11_parallel_hrnet18.json`
- Stage6 → **Stage12**: `settings/stage12_parallel_pcb.json`

## 归档文件

- `stage5_optimized_hrnet18_parallel.json` (2.3KB, 创建于2025-12-03)
- `stage6_optimized_pcb_parallel.json` (2.3KB, 创建于2025-12-03)

---

**结论**: Stage5-6不再需要执行，直接使用更优化的Stage11-12配置。
