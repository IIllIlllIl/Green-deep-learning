# 旧版实验分析脚本归档

**归档日期**: 2025-12-06
**归档原因**: 功能重复，已被统一工具替代

---

## 归档脚本列表

| 脚本名称 | 功能 | 代码行数 | 重复度 |
|---------|------|---------|--------|
| `analyze_from_csv.py` | 从CSV分析实验完成情况 | 169行 | 90% |
| `analyze_from_json.py` | 从JSON分析实验完成情况 | 179行 | 90% |
| `analyze_missing_experiments.py` | 分析缺失实验 | 173行 | 85% |

**总代码量**: 521行
**重复度**: 约90%

---

## 功能说明

### 共同功能
所有三个脚本都实现了相同的核心功能：

1. **统计参数-模式组合的唯一值数量**
   - 数据结构: `{(repo, model, param, mode): set of unique values}`
   - 区分并行/非并行模式
   - 标准化数值（6位小数）

2. **生成完成度报告**
   - 总参数-模式组合数量
   - 已完成组合（≥5个唯一值）
   - 待补充组合列表

3. **列出缺失的参数-模式组合**
   - 按模式分组（非并行/并行）
   - 显示当前值数量和需补充数量

4. **估算需要补充的实验数**
   - 考虑去重率（50-60%）
   - 保守估计和乐观估计

### 唯一差异
- **数据源不同**:
  - `analyze_from_csv.py`: 从 `summary_all.csv` 读取
  - `analyze_from_json.py`: 遍历 `experiment.json` 文件
  - `analyze_missing_experiments.py`: 从CSV读取（与第一个功能相同）

- **输出格式略有不同**: 但本质信息相同

---

## 替代工具

**新的统一工具**: `scripts/analyze_experiments.py`

### 优势
- **统一接口**: 一个脚本支持多种数据源
- **减少代码量**: 从521行减少到约330行（减少37%）
- **灵活输出**: 支持终端、Markdown等多种格式
- **更易维护**: 单一代码库，减少bug

### 使用示例
```bash
# 从CSV分析（替代analyze_from_csv.py和analyze_missing_experiments.py）
python3 scripts/analyze_experiments.py --source csv --file results/summary_all.csv

# 从JSON分析（替代analyze_from_json.py）
python3 scripts/analyze_experiments.py --source json --dir results/

# 仅显示缺失组合
python3 scripts/analyze_experiments.py --source csv --missing-only

# 导出Markdown报告
python3 scripts/analyze_experiments.py --source csv --output report.md
```

---

## 向后兼容性

如果需要使用旧脚本：

```bash
# 恢复到scripts/目录
cp scripts/archived/legacy_analysis/*.py scripts/

# 或直接从归档目录运行
python3 scripts/archived/legacy_analysis/analyze_from_csv.py
```

---

## 整合历史

**整合日期**: 2025-12-06
**整合原因**:
- 发现90%代码重复
- 维护成本高（修改需要同步3个文件）
- 功能本质相同，仅数据源不同

**整合方法**:
1. 提取公共逻辑到ExperimentAnalyzer类
2. 实现多数据源支持（CSV/JSON）
3. 统一输出格式
4. 添加命令行参数解析

**测试验证**:
- 对比新旧工具输出一致性 ✓
- 验证CSV数据源功能 ✓
- 验证JSON数据源功能 ✓

---

**维护者**: Green
**状态**: 已归档 - 不推荐使用
**推荐**: 使用 `scripts/analyze_experiments.py`
