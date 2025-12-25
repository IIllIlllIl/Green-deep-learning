# Stage11补充执行准备清单

**日期**: 2025-12-08
**配置**: settings/stage11_supplement_parallel_hrnet18.json (已修正)
**状态**: ✅ 准备就绪

---

## ✅ 执行前检查清单

### 1. 配置验证
- [x] JSON格式验证通过
- [x] runs_per_config已修正（4 → 2）
- [x] 预计实验数正确（8个）
- [x] 预计时间合理（11.44小时）

### 2. 数据准备
- [x] summary_all.csv已备份（backup_20251208）
- [x] 当前状态已审计（9个实验，每参数3个唯一值）
- [x] 去重机制已启用
- [x] 历史数据路径正确

### 3. 系统环境
- [x] GPU可用（RTX 3080, 10GB内存）
- [x] 磁盘空间充足（1.5TB可用）
- [x] Python环境正常
- [ ] sudo权限确认（需在执行时确认）

### 4. 配置详情
```json
{
  "version": "4.7.2",
  "estimated_experiments": 8,
  "estimated_duration_hours": 11.44,
  "rationale": "方案A精确控制 - 每个参数当前3个唯一值，目标5个，需补充2个"
}
```

---

## 🚀 执行命令

```bash
# 使用sudo执行（确保能耗监控权限）
sudo -E python3 mutation.py -ec settings/stage11_supplement_parallel_hrnet18.json
```

---

## 📊 预期结果

### 实验数量
- **开始前**: 9个hrnet18并行实验
- **预计新增**: 8个实验（4参数 × 2次）
- **完成后**: 17个hrnet18并行实验

### 参数唯一值
| 参数 | 当前 | 新增 | 完成后 | 目标 |
|-----|------|-----|--------|------|
| epochs | 3 | 2 | 5 | ✅ |
| learning_rate | 3 | 2 | 5 | ✅ |
| seed | 3 | 2 | 5 | ✅ |
| dropout | 3 | 2 | 5 | ✅ |

### 时间预估
- **总时间**: ~11.44小时
- **每实验平均**: ~1.43小时
- **epochs配置**: ~2.86小时（2个实验）
- **learning_rate配置**: ~2.86小时（2个实验）
- **seed配置**: ~2.86小时（2个实验）
- **dropout配置**: ~2.86小时（2个实验）

---

## 🔍 执行中监控

### 监控命令（另开终端）
```bash
# 查看最新结果
watch -n 60 'tail -5 results/run_*/summary.csv 2>/dev/null | tail -1'

# 检查进度
watch -n 300 'ls -lt results/run_* | head -3'

# 查看GPU使用
watch -n 10 nvidia-smi
```

### 关键指标
- ✅ 每个实验成功完成（training_succeeded=True）
- ✅ CPU和GPU能耗数据记录完整
- ✅ 每个实验约1.4小时（误差±20%正常）
- ✅ 去重机制正常工作（自动跳过已有值）

---

## ✅ 完成后验证

### 验证命令
```bash
python3 -c "
import csv
with open('results/summary_all.csv') as f:
    rows = list(csv.DictReader(f))
    hrnet18_par = [r for r in rows if 'hrnet18' in r.get('model','') and 'parallel' in r.get('experiment_id','')]

    print('=' * 60)
    print('Stage11补充完成验证')
    print('=' * 60)
    print(f'总实验数: {len(hrnet18_par)} (预期17个)')
    print()

    for param in ['hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_seed', 'hyperparam_dropout']:
        values = set()
        for r in hrnet18_par:
            val = r.get(param, '')
            if val and val.strip():
                values.add(val.strip())

        param_name = param.replace('hyperparam_', '')
        status = '✅' if len(values) == 5 else '⚠️'
        print(f'{status} {param_name}: {len(values)} 个唯一值 (目标5个)')

    print('=' * 60)
"
```

### 成功标准
- [ ] 总实验数: 17个（9已有 + 8新增）
- [ ] epochs: 5个唯一值 ✅
- [ ] learning_rate: 5个唯一值 ✅
- [ ] seed: 5个唯一值 ✅
- [ ] dropout: 5个唯一值 ✅
- [ ] 所有实验training_succeeded=True
- [ ] CPU和GPU能耗数据完整

---

## 📝 完成后任务

1. [ ] 运行验证命令确认结果
2. [ ] 更新README.md（Stage11状态: 完成 ✓）
3. [ ] 更新CLAUDE.md（Stage11补充完成）
4. [ ] 归档Stage11相关文档到completed/
5. [ ] 准备Stage12执行

---

## ⚠️ 故障排查

### 问题1: 实验少于8个
- **可能原因**: 去重碰撞（极低概率）
- **解决**: 检查日志，确认随机生成是否碰撞
- **预期**: 碰撞概率<0.1%，几乎不会发生

### 问题2: 训练失败
- **可能原因**: GPU内存不足、CUDA错误
- **解决**: 检查training.log和experiment.log
- **命令**: `cat results/run_*/Person_reID_*/training.log`

### 问题3: 能耗数据缺失
- **可能原因**: sudo权限不足、perf未安装
- **解决**: 确认使用sudo -E运行
- **验证**: `sudo perf stat -e power/energy-pkg/ sleep 1`

---

## 📚 相关文档

- [Stage11实际状态修正报告](docs/results_reports/STAGE11_ACTUAL_STATE_CORRECTION.md)
- [Stage11 Bug修复报告](docs/results_reports/STAGE11_BUG_FIX_REPORT.md)
- [去重与随机变异分析](docs/results_reports/DEDUPLICATION_RANDOM_MUTATION_ANALYSIS.md)
- [Stage11快速执行指南](STAGE11_QUICK_START.md)

---

**创建者**: Green + Claude
**日期**: 2025-12-08
**状态**: ✅ 准备就绪
**下一步**: 执行 `sudo -E python3 mutation.py -ec settings/stage11_supplement_parallel_hrnet18.json`
