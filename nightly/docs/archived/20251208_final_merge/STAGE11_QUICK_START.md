# Stage11补充 - 快速执行指南

**配置文件**: `settings/stage11_supplement_parallel_hrnet18.json`
**状态**: ✅ 已修正，准备执行
**修正**: 2025-12-08 - 实际已有3个唯一值（非1个）

---

## 🚀 执行命令

```bash
# 1. 备份当前数据（推荐）
cp results/summary_all.csv results/summary_all.csv.backup_20251208

# 2. 执行补充实验
sudo -E python3 mutation.py -ec settings/stage11_supplement_parallel_hrnet18.json

# 3. 监控进度（另开终端）
watch -n 60 'tail -5 results/run_*/summary.csv 2>/dev/null | tail -1'
```

---

## 📊 预期结果

- **实验数**: 8个 (4参数 × 2次) - **已修正**
- **用时**: ~11.4小时 - **已修正**
- **最终**: 每个参数5个唯一值（3个已有 + 2个新增）

---

## ✅ 验证

执行完成后：
```bash
# 检查实验数量
python3 -c "
import csv
with open('results/summary_all.csv') as f:
    rows = list(csv.DictReader(f))
    hrnet18_par = [r for r in rows if 'hrnet18' in r.get('model','') and 'parallel' in r.get('experiment_id','')]
    print(f'hrnet18并行实验总数: {len(hrnet18_par)} (预期17个: 9已有 + 8新增)')

    # 检查每个参数唯一值数量
    for param in ['hyperparam_epochs', 'hyperparam_learning_rate', 'hyperparam_seed', 'hyperparam_dropout']:
        values = set()
        for r in hrnet18_par:
            val = r.get(param, '')
            if val and val.strip():
                values.add(val.strip())
        param_name = param.replace('hyperparam_', '')
        print(f'{param_name}: {len(values)} 个唯一值 (目标5个)')
"
```

---

**详细文档**: `docs/results_reports/STAGE11_SUPPLEMENT_EXECUTION_PLAN.md`
