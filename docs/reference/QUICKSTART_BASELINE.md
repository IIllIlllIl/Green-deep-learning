# 默认值基线测试快速开始

**最后更新**: 2025-11-19

---

## 📊 查看测试结果

### 方法1: 使用分析脚本 (推荐)

```bash
cd /home/green/energy_dl/nightly
python3 scripts/analyze_baseline.py results/default_baseline_11models
```

**输出内容**:
- 基本统计（成功率、总时长）
- 能耗统计（GPU/CPU总能耗，并行vs顺序对比）
- GPU能耗排名 Top 10
- 运行时长排名 Top 10
- GPU利用率统计
- 性能指标（准确率、mAP、Rank-1/5）
- GPU温度统计

---

### 方法2: 阅读完整报告

```bash
# Markdown格式（推荐使用Markdown查看器）
cat docs/DEFAULT_BASELINE_REPORT_20251118.md | less

# 或在浏览器中查看（如果支持）
```

**报告内容**:
- 执行摘要
- 详细时间分析
- 能耗数据分析
- 性能指标分析
- 并行vs顺序对比
- hrnet18问题解决验证
- 后续工作建议

---

### 方法3: 直接查看原始数据

```bash
# 查看汇总CSV（格式化显示）
column -t -s, < results/default_baseline_11models/summary.csv | less

# 查看特定实验的详细数据
cat results/default_baseline_11models/Person_reID_baseline_pytorch_hrnet18_006/experiment.json

# 查看训练日志
less results/default_baseline_11models/Person_reID_baseline_pytorch_hrnet18_006/training.log
```

---

## 🔍 常用查询命令

### GPU能耗排序

```bash
cd results/default_baseline_11models
tail -n +2 summary.csv | awk -F, '{print $1,$32}' | sort -k2 -n -r | head -10
```

### 运行时长排序

```bash
tail -n +2 summary.csv | awk -F, '{print $1,$6}' | sort -k2 -n -r | head -10
```

### GPU利用率排序

```bash
tail -n +2 summary.csv | awk -F, '{print $1,$35}' | sort -k2 -n -r | head -10
```

### 准确率排序

```bash
tail -n +2 summary.csv | awk -F, '{print $1,$17}' | sort -k2 -n -r | grep -v "^.*,,"
```

---

## 📚 相关文档

| 文档 | 路径 | 用途 |
|------|------|------|
| **完整测试报告** | `docs/DEFAULT_BASELINE_REPORT_20251118.md` | 详细分析和建议 |
| **工作总结** | `docs/WORK_SUMMARY_20251119.md` | 本次工作概览 |
| **模型架构** | `docs/MODEL_ARCHITECTURES.md` | 11个模型详解 |
| **RVSM说明** | `docs/RVSM_EXPLAINED.md` | RVSM方法详解 |
| **hrnet18分析** | `docs/HRNET18_FAILURE_ANALYSIS_20251118.md` | 失败原因和解决方案 |
| **文档索引** | `docs/README.md` | 所有文档导航 |

---

## 📈 关键数据速览

### 测试概览

| 指标 | 值 |
|------|-----|
| 测试时间 | 2025-11-18 20:16 ~ 2025-11-19 07:49 |
| 总时长 | 11小时33分钟 |
| 实验数量 | 22 (11模型 × 2模式) |
| 成功率 | 100% (22/22) |
| 总能耗 | 2929.68 Wh |

### 能耗对比

| 模式 | GPU能耗 | CPU能耗 | 总能耗 |
|------|---------|---------|--------|
| 顺序训练 | 1153.81 Wh | 190.31 Wh | 1344.12 Wh |
| 并行训练 | 1312.26 Wh | 273.29 Wh | 1585.55 Wh |
| **增加** | **+13.7%** | **+43.6%** | **+18.0%** |

### Top 3 能耗模型

1. hrnet18_parallel: 309.15 Wh (1h 23m)
2. hrnet18_sequential: 284.96 Wh (1h 11m)  
3. pcb_sequential: 274.04 Wh (1h 12m)

### Person Re-ID 性能

| 模型 | mAP | Rank-1 | Rank-5 |
|------|-----|--------|--------|
| pcb | 77.52% | 92.49% | 97.15% |
| densenet121 | 75.32% | 90.91% | 96.35% |
| hrnet18 | 74.89% | 90.02% | 96.29% |

---

## 🛠️ 工具使用

### 分析脚本选项

```bash
# 分析默认基线
python3 scripts/analyze_baseline.py

# 分析指定目录
python3 scripts/analyze_baseline.py results/run_20251118_201629

# 查看帮助
python3 scripts/analyze_baseline.py --help
```

---

## 🎯 下一步

### 推荐操作

1. **查看完整报告**: `cat docs/DEFAULT_BASELINE_REPORT_20251118.md | less`
2. **运行分析脚本**: `python3 scripts/analyze_baseline.py`
3. **计划突变测试**: 基于基线数据设计超参数突变方案

### 突变测试准备

基于基线数据，建议的突变测试优先级：

**快速验证** (< 1分钟):
- mnist_ff (8秒)

**轻量测试** (2-10分钟):
- mnist (2分钟)
- mnist_rnn (4分钟)
- siamese (5分钟)

**中等规模** (20-60分钟):
- resnet20 (19分钟)
- MRT-OAST (21分钟)
- VulBERTa_mlp (52分钟)

**完整测试** (> 1小时):
- densenet121 (54分钟)
- hrnet18 (71分钟)
- pcb (72分钟)

---

## 📞 问题排查

### 找不到结果目录

```bash
# 检查符号链接
ls -l results/default_baseline_11models

# 应该指向
# results/default_baseline_11models -> run_20251118_201629
```

### 分析脚本报错

```bash
# 确保在项目根目录
cd /home/green/energy_dl/nightly

# 检查Python版本
python3 --version  # 应该 >= 3.6

# 手动指定完整路径
python3 scripts/analyze_baseline.py results/default_baseline_11models
```

### 查看特定实验失败原因

```bash
# 即使全部成功，也可以查看日志
tail -100 results/default_baseline_11models/*/training.log

# 查看特定实验
cat results/default_baseline_11models/Person_reID_baseline_pytorch_hrnet18_006/training.log
```

---

**生成时间**: 2025-11-19 14:30
**维护者**: Claude Code

*本快速参考卡片提供了访问和分析默认值基线测试结果的所有必要信息。*
