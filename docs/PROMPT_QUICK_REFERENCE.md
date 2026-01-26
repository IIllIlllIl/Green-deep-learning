# 数据补完任务 - 快速参考

**版本**: 1.0
**日期**: 2026-01-26
**优先级**: 高

---

## 🎯 一句话任务

> 补完能耗数据的缺失部分，将数据完整性从95.1%提升到98%+，为因果推断分析做准备。

---

## 🚀 快速开始（3步）

### 1️⃣ 验证当前状态（5分钟）

```bash
cd /home/green/energy_dl/nightly

# 验证数据完整性
python3 tools/data_management/validate_raw_data.py

# 分析缺失数据
python3 tools/data_management/analyze_missing_energy_data.py
```

### 2️⃣ 补完数据（2-4小时）

```bash
# 方案A: 自动修复（数据存在但未提取）
python3 tools/data_management/repair_missing_energy_data.py

# 方案B: 手动追加（新实验数据）
python3 tools/data_management/append_session_to_raw_data.py \
    --experiment-dir <实验目录>

# 方案C: 重新运行（数据缺失）
python3 mutation.py --config <配置文件>
```

### 3️⃣ 验证结果（10分钟）

```bash
# 验证完整性
python3 tools/data_management/validate_raw_data.py

# 对比数据
python3 tools/data_management/compare_data_vs_raw_data.py

# 测试因果推断
cd analysis && python -c "
from utils.causal_inference import CausalInferenceEngine
import pandas as pd
data = pd.read_csv('../data/data.csv')
print(f'✅ 数据加载成功: {len(data)}行')
"
```

---

## 📁 关键文件（5个）

| 文件 | 用途 |
|------|------|
| `data/raw_data.csv` | 主数据文件（87列） |
| `tools/data_management/repair_missing_energy_data.py` | 修复脚本 |
| `tools/data_management/append_session_to_raw_data.py` | 追加脚本 |
| `docs/DATA_USAGE_GUIDE.md` | 数据使用指南 |
| `docs/PROMPT_FOR_DATA_COMPLETION.md` | 详细任务文档 |

---

## 📊 当前状态

```
总实验数: 970条
├─ 完全可用: 577条 (59.5%)
├─ 仅有能耗: 251条 (25.9%)
└─ 缺失数据: 141条 (14.6%) ⚠️

目标: 将数据完整性提升到 98%+
```

---

## ✅ 成功标准

- [ ] 数据完整性 ≥ 98%
- [ ] validate_raw_data.py 通过
- [ ] 因果推断测试通过
- [ ] 文档已更新

---

## 🆘 遇到问题？

1. **数据文件找不到** → 检查 `archives/experiments/` 目录
2. **提取失败** → 查看详细文档 `PROMPT_FOR_DATA_COMPLETION.md`
3. **格式错误** → 运行 `validate_raw_data.py` 查看详情
4. **需要更多帮助** → 查看 `docs/results_reports/DATA_REPAIR_REPORT_20260104.md`

---

## 📖 完整文档

详细信息请参考: `docs/PROMPT_FOR_DATA_COMPLETION.md`
