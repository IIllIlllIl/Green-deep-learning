# 配置文件归档说明 - 2025-12-08

## 归档原因
这些配置文件已被最终合并配置替代，归档以保持settings目录整洁。

## 归档配置

### stage11_supplement_parallel_hrnet18.json
- **版本**: v4.7.2
- **状态**: 已合并到最终配置
- **原因**: hrnet18并行补充已合并到`stage_final_all_remaining.json`的第二部分
- **实验数**: 8个（4参数 × 2次）

### stage12_parallel_pcb.json
- **版本**: v4.7.2-corrected
- **状态**: 已合并到最终配置
- **原因**: pcb并行补充已合并到`stage_final_all_remaining.json`的第一部分
- **实验数**: 12个（4参数 × 3次）

### stage13_final_merged.json
- **版本**: v4.7.2-final-merged
- **状态**: 已合并到最终配置
- **原因**: Stage13内容已合并到`stage_final_all_remaining.json`的第三至五部分
- **实验数**: 66个

### stage13_merged_final_supplement.json
- **版本**: v4.7.1-merged
- **状态**: 已被新版本替代
- **原因**: 被v4.7.2版本的最终配置替代

## 替代配置

所有归档配置已合并为单一配置文件：
**`settings/stage_final_all_remaining.json`**
- 版本: v4.7.2-final
- 实验数: 78个
- 预计时间: 37.8小时
- 包含5个部分: pcb并行 + hrnet18并行 + 快速模型 + MRT-OAST + VulBERTa/cnn

## 执行命令

```bash
sudo -E python3 mutation.py -ec settings/stage_final_all_remaining.json
```

---

**归档日期**: 2025-12-08
**归档者**: Green + Claude (v4.7.2最终整合)
