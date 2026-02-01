# 下一个对话Prompt - 全局标准化修复后续工作

**生成日期**: 2026-02-01
**当前状态**: 85%完成，阶段1接近尾声

---

## 🎯 项目背景

### 项目目标
实施全局标准化修复，恢复跨组可比性，基于CTF论文算法进行权衡检测分析。

### 当前进度
**总体进度**: 85%
**当前阶段**: 阶段1 - 全局标准化实施（接近完成）

```
✅ 已完成任务 (5/7):
  1.1 诊断缺失值模式              [100%]
  1.2 全局标准化数据生成           [100%]
  1.3 DiBS因果发现                [100%]
  1.5 全局标准化ATE计算            [100%]
  1.7 算法1-权衡检测（基于ATE）      [100%] ⭐刚完成

⏳ 待执行任务 (4个):
  1.4 修复group2 batch_size问题    [0%]
  1.6 敏感性分析                   [0%]
  2.1 权衡结果分析                 [0%]
  2.2 研究报告撰写                 [0%]
```

---

## 📁 关键文件路径

### 数据文件
```
analysis/
├── data/energy_research/
│   ├── 6groups_global_std/              # 全局标准化数据（49特征）
│   ├── 6groups_interaction/             # 交互项数据
│   └── interaction/whitelist_with_ate/  # 交互项ATE（95.6%完整性）⭐
│
├── results/energy_research/data/
│   ├── global_std/                      # DiBS因果图结果
│   ├── global_std_dibs_ate/             # 全局标准化ATE
│   └── tradeoff_detection/              # ⭐算法1结果（37个权衡）
│       ├── all_tradeoffs.json
│       ├── tradeoff_summary.csv
│       ├── tradeoff_detailed.csv
│       └── ALGORITHM1_ACCEPTANCE_REPORT.md
```

### 代码文件
```
analysis/
├── utils/
│   ├── causal_discovery.py              # DiBS因果发现
│   ├── causal_inference.py              # ATE计算（已修复异常值）⭐
│   └── tradeoff_detection.py            # 权衡检测（已对齐CTF）⭐
│
├── scripts/
│   ├── generate_global_std_data.py      # 全局标准化数据生成
│   ├── run_global_std_dibs.py           # DiBS执行脚本
│   ├── compute_global_std_ate.py        # ATE计算脚本
│   └── run_algorithm1_tradeoff_detection.py  # ⭐算法1执行脚本
│
└── tests/
    ├── unit/                            # ⭐单元测试（已整理）
    └── integration/                     # ⭐集成测试（已整理）
```

### 文档文件
```
analysis/docs/
├── technical_reference/
│   └── GLOBAL_STANDARDIZATION_FIX_PROGRESS.md  # ⭐进度跟踪（更新到v1.7）
│
├── reports/                             # ⭐新建（已整理）
│   ├── TRADEOFF_OPTIMIZATION_ACCEPTANCE_REPORT.md
│   ├── TRADEOFF_OPTIMIZATION_FINAL_SUMMARY.md
│   └── [其他报告...]
│
└── archived/                            # ⭐临时文档归档（已整理）
    ├── CLEANUP_PLAN.md
    ├── FILE_CLEANUP_VERIFICATION.md
    └── NEXT_CONVERSATION_PROMPT.md
```

---

## 🗂️ 文件摆放规则（重要！）

### ⚠️ 刚完成的文件结构整理

**整理时间**: 2026-02-01
**整理状态**: ✅ 完成
**整理结果**: 13个错位文件已移动到正确位置

### 文件分类规则

| 文件类型 | 正确位置 | 说明 |
|---------|---------|------|
| **项目文档** | `docs/` | 小写_下划线.md |
| **技术参考** | `docs/technical_reference/` | 技术文档 |
| **报告** | `docs/reports/` | 各类报告 ⭐新建 |
| **归档文档** | `docs/archived/` | 过时文档 |
| **数据处理脚本** | `tools/data_management/` | 动词_名词.py |
| **配置管理** | `tools/config_management/` | 配置相关 |
| **分析脚本** | `scripts/` | 动词_名词.py |
| **测试文件** | `tests/unit/` 或 `tests/integration/` | test_名词.py ⭐ |
| **历史脚本** | `tools/legacy/` | 已完成的临时脚本 ⭐ |
| **工具模块** | `utils/` | 可复用的工具类/函数 |

### 根目录规范
- **应该保留**: README.md, LICENSE
- **不应该有**: .py脚本, .md文档（除README外）
- **刚清理**: 13个错位文件已移动 ✅

---

## 🔬 刚完成的算法1（权衡检测）

### 执行时间
2026-02-01

### 关键成果
- ✅ **代码修正**: `utils/tradeoff_detection.py` 对齐CTF论文逻辑
- ✅ **检测到37个显著权衡**: 100%统计显著
- ✅ **14个能耗vs性能权衡**: 占37.8%
- ✅ **验收通过**: Subagent全面验收

### 核心发现
1. **经典权衡**: 性能提升 → 能耗增加
2. **并行化效应**: 并行模式峰值功率更低，但总能耗更高
3. **模型差异**: HRNet18、PCB、Siamese表现不同

### 输出文件
```
results/energy_research/tradeoff_detection/
├── all_tradeoffs.json          # 完整权衡JSON（10.5KB）
├── tradeoff_summary.csv        # 统计摘要
├── tradeoff_detailed.csv       # 详细权衡表（4.9KB）
└── ALGORITHM1_ACCEPTANCE_REPORT.md  # 验收报告
```

### 权衡分布
| 任务组 | 权衡数 | 比例 |
|--------|--------|------|
| group1_examples | 12 | 27.9% |
| group2_vulberta | 0 | 0% |
| group3_person_reid | 17 | 37.8% ← 最多 |
| group4_bug_localization | 0 | 0% |
| group5_mrt_oast | 4 | 10.3% |
| group6_resnet | 4 | 21.1% |

---

## 📋 待执行任务详情

### 任务1.4: 修复group2 batch_size问题（优先级：中）

**背景**: group2的batch_size可能存在零方差问题

**目标**:
1. 检查group2 batch_size数据分布
2. 诊断零方差原因
3. 实施修复方案
4. 重新运行DiBS和ATE（如需要）

**关键文件**:
- 数据: `data/energy_research/6groups_global_std/group2_vulberta_global_std.csv`
- DiBS: `results/energy_research/data/global_std/group2_vulberta_dibs_*`

**预期输出**:
- 诊断报告
- 修复脚本
- 更新的因果图

---

### 任务1.6: 敏感性分析（优先级：低）

**目标**: 对比保守填充 vs CTF删除的ATE差异

**数据源**:
- 保守填充: `6groups_global_std/`（当前使用）
- CTF删除: 需要生成（dropna策略）

**方法**:
1. 使用CTF删除策略重新生成全局标准化数据
2. 重新计算ATE
3. 对比两种策略的ATE差异
4. 评估差异对结论的影响

**关键文件**:
- `scripts/generate_global_std_data.py` - 修改参数
- `scripts/compute_global_std_ate.py` - 重新计算

---

### 任务2.1: 权衡结果分析（优先级：高）⭐

**目标**: 深入分析37个权衡关系

**分析方向**:
1. **能耗vs性能权衡**（14个）
   - 哪些干预导致权衡？
   - ATE幅度分布？
   - 模型间差异？

2. **为什么group2和group4没有权衡？**
   - 数据质量问题？
   - 因果图差异？
   - 统计功效不足？

3. **干预类型分析**
   - 超参数干预 vs 能耗指标干预
   - 可操作干预识别

4. **可视化**
   - 权衡关系网络图
   - ATE幅度分布图
   - 任务组对比图

**关键文件**:
- `results/energy_research/tradeoff_detection/tradeoff_detailed.csv`
- `results/energy_research/tradeoff_detection/all_tradeoffs.json`

**预期输出**:
- 分析脚本: `scripts/analyze_tradeoff_results.py`
- 分析报告: `docs/reports/TRADEOFF_ANALYSIS_REPORT.md`
- 可视化图表: `results/energy_research/tradeoff_detection/figures/`

---

### 任务2.2: 研究报告撰写（优先级：中）

**目标**: 基于分析结果撰写研究报告

**报告结构**:
1. 引言（背景、问题、目标）
2. 方法（全局标准化、DiBS、ATE、权衡检测）
3. 结果（37个权衡的详细分析）
4. 讨论（能耗vs性能权衡、模型差异）
5. 结论（研究问题回答、贡献）

**关键参考**:
- CTF论文: Causality-Aided Trade-off Analysis
- 进度文档: `docs/technical_reference/GLOBAL_STANDARDIZATION_FIX_PROGRESS.md`
- 验收报告: `results/energy_research/tradeoff_detection/ALGORITHM1_ACCEPTANCE_REPORT.md`

---

## 🔧 技术注意事项

### 环境配置
```bash
# 必须使用的conda环境
conda activate causal-research  # ⭐ 重要！base环境无DiBS

# 验证环境
python -c "import dibs; print('DiBS OK')"
```

### 关键代码修改

#### 1. ATE异常值修复（1.5已完成）
**文件**: `utils/causal_inference.py`
**行数**: 225-253
**功能**: 当|ATE|>10时使用简化估计
**状态**: ✅ 已修复

#### 2. Sign函数对齐CTF（1.7已完成）
**文件**: `utils/tradeoff_detection.py:create_sign_func_from_rule()`
**修改**: 添加注释说明CTF逻辑
**状态**: ✅ 已对齐

### 数据完整性

| 数据源 | 完整性 | 用途 |
|--------|--------|------|
| 交互项ATE | 95.6% | 算法1使用 ⭐ |
| 全局标准化ATE | 37.3% | 备选数据源 |
| 6组全局标准化数据 | 100% | DiBS输入 |

---

## 🚀 建议执行顺序

### 立即执行（高优先级）
1. **任务2.1**: 权衡结果深入分析
   - 为什么group2/group4没有权衡？
   - 能耗vs性能权衡详解
   - 生成可视化图表

### 短期执行（中优先级）
2. **任务1.4**: 修复group2 batch_size问题
   - 诊断零方差原因
   - 实施修复方案

3. **任务2.2**: 研究报告撰写
   - 基于分析结果撰写报告

### 可选执行（低优先级）
4. **任务1.6**: 敏感性分析
   - 对比保守填充 vs CTF删除

---

## 📝 快速启动命令

### 查看当前进度
```bash
cat docs/technical_reference/GLOBAL_STANDARDIZATION_FIX_PROGRESS.md
```

### 查看算法1结果
```bash
# 查看摘要
cat results/energy_research/tradeoff_detection/tradeoff_summary.csv

# 查看详细权衡
head -20 results/energy_research/tradeoff_detection/tradeoff_detailed.csv

# 查看验收报告
cat results/energy_research/tradeoff_detection/ALGORITHM1_ACCEPTANCE_REPORT.md
```

### 开始分析（推荐）
```bash
conda activate causal-research
cd /home/green/energy_dl/nightly/analysis

# 创建分析脚本
# scripts/analyze_tradeoff_results.py
```

---

## ⚠️ 常见问题

### Q1: 为什么group2和group4没有检测到权衡？
**A**: 可能原因：
- 数据质量问题（缺失值、方差）
- 因果图差异（边数、结构）
- ATE计算问题（统计显著性）
- 需要深入调查

### Q2: 交互项ATE vs 全局标准化ATE，用哪个？
**A**: 当前使用交互项ATE：
- 完整性更高（95.6% vs 37.3%）
- 包含超参数交互效应
- 更符合研究目标

### Q3: Sign函数对齐CTF后，有什么变化？
**A**:
- 添加详细注释说明CTF逻辑
- ATE=0时判定为恶化（保守策略）
- 正常ATE完全对齐CTF论文

---

## 📞 参考文档索引

| 文档 | 路径 | 用途 |
|------|------|------|
| 进度跟踪 | `docs/technical_reference/GLOBAL_STANDARDIZATION_FIX_PROGRESS.md` | 总体进度 |
| CTF源码 | `CTF_original/src/inf.py` | 论文参考 |
| 文件规则 | `CLAUDE.md` (项目根目录) | 文件摆放规范 |
| 验收报告 | `results/energy_research/tradeoff_detection/ALGORITHM1_ACCEPTANCE_REPORT.md` | 算法1详情 |

---

## ✅ 检查清单

开始工作前，请确认：
- [ ] 已激活causal-research环境
- [ ] 已阅读进度文档（v1.7）
- [ ] 已查看算法1结果
- [ ] 已了解文件摆放规则
- [ ] 已选择要执行的任务

---

**准备好继续了吗？建议从任务2.1（权衡结果分析）开始！** 🚀
