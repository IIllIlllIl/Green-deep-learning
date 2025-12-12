# Stage7-13 实验执行计划

**创建日期**: 2025-12-05
**版本**: v4.6.0
**目标**: 完成剩余274个唯一值，达到100%实验完成度

---

## 📊 当前状态（基于summary_all.csv）

- **总实验数**: 381个
- **参数-模式组合**: 90个
- **已完成组合**: 28个 (31.1%)
- **待补充组合**: 62个
  - 非并行模式缺失: 45个组合（225个唯一值）
  - 并行模式缺失: 17个组合（49个唯一值）
- **总需补充**: 274个唯一值

---

## 🎯 实验阶段规划

### Stage7: 非并行快速模型
**配置文件**: `settings/stage7_nonparallel_fast_models.json`
**预计时长**: 38.3小时
**预计实验数**: 199个

**目标模型**:
- examples/mnist: 4参数 × 7次 = 28实验 (1.9h)
- examples/mnist_ff: 4参数 × 7次 = 28实验 (0.1h)
- examples/mnist_rnn: 4参数 × 7次 = 28实验 (2.8h)
- examples/siamese: 4参数 × 7次 = 28实验 (4.3h)
- MRT-OAST/default: 5参数 × 7次 = 35实验 (7.8h)
- bug-localization: 4参数 × 7次 = 28实验 (10.1h)
- resnet20: 4参数 × 6次 = 24实验 (11.3h)

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/stage7_nonparallel_fast_models.json
```

---

### Stage8: 非并行中慢速模型
**配置文件**: `settings/stage8_nonparallel_medium_slow_models.json`
**预计时长**: 35.1小时
**预计实验数**: 48个

**目标模型**:
- VulBERTa/mlp: 4参数 × 7次 = 28实验 (16.6h)
- densenet121: 4参数 × 5次 = 20实验 (18.5h)

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/stage8_nonparallel_medium_slow_models.json
```

---

### Stage9: 非并行超慢速模型hrnet18
**配置文件**: `settings/stage9_nonparallel_hrnet18.json`
**预计时长**: 25.0小时
**预计实验数**: 20个

**目标模型**:
- hrnet18: 4参数 × 5次 = 20实验 (25.0h)

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/stage9_nonparallel_hrnet18.json
```

---

### Stage10: 非并行超慢速模型pcb
**配置文件**: `settings/stage10_nonparallel_pcb.json`
**预计时长**: 23.7小时
**预计实验数**: 20个

**目标模型**:
- pcb: 4参数 × 5次 = 20实验 (23.7h)

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/stage10_nonparallel_pcb.json
```

---

### Stage11: 并行模式hrnet18补充
**配置文件**: `settings/stage11_parallel_hrnet18.json`
**预计时长**: 28.6小时
**预计实验数**: 20个

**目标**: hrnet18并行模式，4参数各需补充3个唯一值

**目标模型**:
- hrnet18并行: 4参数 × 5次 = 20实验 (28.6h)

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/stage11_parallel_hrnet18.json
```

---

### Stage12: 并行模式pcb补充
**配置文件**: `settings/stage12_parallel_pcb.json`
**预计时长**: 23.1小时
**预计实验数**: 20个

**目标**: pcb并行模式，4参数各需补充3个唯一值

**目标模型**:
- pcb并行: 4参数 × 5次 = 20实验 (23.1h)

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/stage12_parallel_pcb.json
```

---

### Stage13: 并行模式快速模型补充
**配置文件**: `settings/stage13_parallel_fast_models_supplement.json`
**预计时长**: 5.0小时
**预计实验数**: 43个

**目标**: 补充快速模型并行模式的缺失参数

**目标模型**:
- mnist_ff并行: 4参数 × 5次 = 20实验 (0.7h)
- mnist并行: batch_size × 5次 = 5实验 (0.4h)
- mnist_rnn并行: batch_size × 5次 + epochs × 3次 = 8实验 (0.9h)
- siamese并行: batch_size × 5次 = 5实验 (0.9h)
- bug-loc并行: kfold × 5次 = 5实验 (2.1h)

**执行命令**:
```bash
sudo -E python3 mutation.py -ec settings/stage13_parallel_fast_models_supplement.json
```

---

## 📈 总体统计

| Stage | 配置文件 | 预计时长 | 预计实验数 | 模式 | 优先级 |
|-------|---------|---------|-----------|------|-------|
| Stage7 | stage7_nonparallel_fast_models.json | 38.3h | 199 | 非并行 | 高 |
| Stage8 | stage8_nonparallel_medium_slow_models.json | 35.1h | 48 | 非并行 | 高 |
| Stage9 | stage9_nonparallel_hrnet18.json | 25.0h | 20 | 非并行 | 中 |
| Stage10 | stage10_nonparallel_pcb.json | 23.7h | 20 | 非并行 | 中 |
| Stage11 | stage11_parallel_hrnet18.json | 28.6h | 20 | 并行 | 中 |
| Stage12 | stage12_parallel_pcb.json | 23.1h | 20 | 并行 | 中 |
| Stage13 | stage13_parallel_fast_models_supplement.json | 5.0h | 43 | 并行 | 低 |
| **总计** | **7个配置文件** | **178.8h** | **370** | - | - |

---

## ⚙️ 配置说明

### runs_per_config设置原则
- **快速模型** (0-10分钟): runs=7，确保获得5个唯一值
- **中速模型** (10-30分钟): runs=6-7，平衡时间和成功率
- **慢速模型** (30-60分钟): runs=5-7，避免过长时间
- **超慢速模型** (>60分钟): runs=5，最小化总时间
- **补充实验** (已有2/5): runs=5，补充3个即可

### 去重机制
- 所有配置启用去重：`"use_deduplication": true`
- 历史数据：`"historical_csvs": ["results/summary_all.csv"]`
- 修复后的去重逻辑正确区分并行/非并行模式

---

## ✅ 执行前检查清单

- [ ] 确认所有配置文件格式正确
- [ ] 确认去重修复已生效（mutation/dedup.py包含mode参数）
- [ ] 备份summary_all.csv：`cp results/summary_all.csv results/summary_all.csv.backup_stage7`
- [ ] 确认系统资源（磁盘空间、GPU可用）
- [ ] 准备开始执行Stage7

---

## 📝 执行建议

1. **按顺序执行**: 建议按Stage7→8→9→10→11→12→13顺序执行
2. **监控进度**: 每个Stage完成后运行`python3 scripts/analyze_from_csv.py`检查进度
3. **数据备份**: 每个Stage前备份summary_all.csv
4. **并行Stage**: Stage11-13可以在不同时间段执行（需要GPU并行）
5. **优先级**: Stage7-8为高优先级，应优先完成

---

## 🎯 预期结果

完成所有7个Stage后：
- **参数-模式组合完成度**: 90/90 (100%)
- **总实验数**: ~751个 (381现有 + 370新增)
- **所有参数**: 每个参数在两种模式下都有5个唯一值
- **研究目标**: 可以开始深入分析超参数对能耗和性能的影响

---

**维护者**: Green
**最后更新**: 2025-12-05
**状态**: ✅ 准备执行
