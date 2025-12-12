# VulBERTa/cnn 无效数据清理报告

**报告日期**: 2025-12-11
**报告版本**: v1.0
**项目版本**: v4.7.2

---

## 📊 执行摘要

在检查VulBERTa_cnn模型的运行时间异常后，发现**训练代码根本没有实现**。所有42个VulBERTa/cnn实验都是无效的，已从`summary_all.csv`中清理。

**关键行动**:
- ✅ 确认VulBERTa/cnn训练功能未实现
- ✅ 移除42条无效实验记录
- ✅ 创建数据备份
- ✅ 验证CSV格式完整性

---

## 🔍 问题发现过程

### 1. 异常症状

在分析最新实验结果时发现：
- VulBERTa/cnn 平均运行时间: **3.92秒**
- 运行时间与Epochs数量**无关** (5个epochs和18个epochs都是~3.6秒)
- 所有42个实验的 `perf_test_accuracy` 字段**全部为空**
- 对比: VulBERTa/mlp 平均运行时间 2562秒（差距**653倍**）

### 2. 代码检查

检查训练脚本 `repos/VulBERTa/train_vulberta.py`:

**第333-340行** - `train_cnn()` 函数:
```python
def train_cnn(args):
    """Train VulBERTa-CNN model"""
    print("=" * 80)
    print("Training VulBERTa-CNN Model")
    print("=" * 80)
    print("CNN training not yet implemented in this script")
    print("Please use the Jupyter notebook: Finetuning+evaluation_VulBERTa-CNN.ipynb")
    return None, None
```

**🔴 关键发现**:
- CNN训练功能**完全未实现**
- 函数只打印消息后直接返回 `None, None`
- 没有任何实际的模型训练代码

### 3. 根因分析

**为什么实验显示成功?**
1. Python脚本成功执行（打印消息然后退出）
2. 退出码为0（没有抛出异常）
3. runner.py 将其标记为 `training_success=True`

**为什么运行时间这么短?**
1. 只执行了打印语句，没有实际训练
2. 3-4秒的时间用于:
   - Python环境启动
   - 导入模块
   - 打印消息
   - 退出

**为什么性能指标为空?**
1. 没有训练过程，所以没有日志输出
2. 性能指标提取依赖日志文件中的特定模式
3. 无日志 → 无匹配 → 空值

---

## 🧹 数据清理操作

### 清理统计

| 指标 | 数值 |
|------|------|
| 原始记录数 | 518 条 |
| 移除记录数 | 42 条 (VulBERTa/cnn) |
| 保留记录数 | 476 条 |
| 清理比例 | 8.1% |
| 备份文件 | `results/summary_all.csv.backup_20251211_144013` |

### 移除的实验

**实验ID列表** (42个):
```
VulBERTa_cnn_071 ~ VulBERTa_cnn_110 (非并行)
VulBERTa_cnn_002_parallel, VulBERTa_cnn_091_parallel ~ VulBERTa_cnn_110_parallel
```

**实验特征**:
- Repository: VulBERTa
- Model: cnn
- 运行时间: 3.22 - 23.33秒
- 训练成功标志: True (误报)
- 性能指标: 全部为空

### 数据完整性验证

清理后验证:
```bash
✓ CSV格式: 476 行记录, 37 列
✓ 列数匹配: 标准37列格式
✓ 备份已创建: summary_all.csv.backup_20251211_144013
```

---

## 📈 清理后实验统计

### 更新后的模型分布

| 模型 | 实验数 | 平均运行时间 |
|------|--------|--------------|
| examples/mnist_ff | 56 | 16.72秒 |
| examples/mnist | 40 | 231.86秒 |
| examples/mnist_rnn | 43 | 337.12秒 |
| examples/siamese | 40 | 520.28秒 |
| MRT-OAST/default | 57 | 885.59秒 |
| bug-localization-by-dnn-and-rvsm/default | 40 | 1192.39秒 |
| pytorch_resnet_cifar10/resnet20 | 39 | 1676.44秒 |
| VulBERTa/mlp | 45 | 2562.34秒 |
| Person_reID_baseline_pytorch/densenet121 | 43 | 3381.41秒 |
| Person_reID_baseline_pytorch/hrnet18 | 37 | 5138.16秒 |
| Person_reID_baseline_pytorch/pcb | 36 | 5520.45秒 |

**VulBERTa/cnn**: ~~42个实验~~ → **已移除** ❌

### 实验完成度更新

清理前:
- 总实验: 518 条
- 模型数: 12 个

清理后:
- 总实验: 476 条
- 模型数: **11 个** (移除VulBERTa/cnn)
- 有效模型: MRT-OAST, bug-localization, pytorch_resnet, VulBERTa/mlp, Person_reID (3个模型), examples (4个模型)

---

## 🔧 后续行动计划

### 1. 立即任务

#### 1.1 更新模型配置
- **文件**: `mutation/models_config.json`
- **操作**: 暂时移除 VulBERTa/cnn 从 `models` 列表
- **替代方案**:
  ```json
  "VulBERTa": {
    "models": ["mlp"],  // 移除 "cnn"
    ...
  }
  ```

#### 1.2 更新实验计划
- 重新评估实验目标（原计划包含VulBERTa/cnn）
- 更新 `settings/stage_final_all_remaining.json`
- 移除所有VulBERTa/cnn相关的实验配置

### 2. 中期任务

#### 2.1 实现VulBERTa/cnn训练功能 (可选)

如果需要VulBERTa/cnn实验，有两个选项：

**选项A: 使用Jupyter Notebook** (推荐)
- 文件: `Finetuning+evaluation_VulBERTa-CNN.ipynb`
- 优点: 原作者提供的完整实现
- 缺点: 需要转换为脚本格式

**选项B: 实现train_cnn()函数**
- 参考MLP实现（第226-331行）
- 调整模型架构为CNN
- 测试验证

#### 2.2 重新运行VulBERTa/cnn实验

如果实现了CNN训练功能：
- 实验数量: 根据最新实验计划
- 预计时间: 待评估（基于CNN实现后的实际运行时间）

### 3. 长期改进

#### 3.1 增强验证机制
在 `mutation/runner.py` 中添加:
1. **训练成功验证**: 检查是否真正执行了训练
2. **性能指标验证**: 训练成功时必须有性能指标
3. **运行时间合理性检查**: 标记异常短/长的实验

示例逻辑:
```python
if training_success and not performance_metrics:
    logger.warning(f"Training marked successful but no metrics found")
    # 可以标记为可疑实验

if duration < expected_min_duration:
    logger.warning(f"Abnormally short runtime: {duration}s")
```

#### 3.2 预训练检查
在开始实验前检查:
1. 训练脚本是否存在
2. 模型实现是否完整
3. 数据集是否可用

---

## 📚 相关文档更新

### 需要更新的文档

1. **README.md**
   - 更新实验总数: 518 → 476
   - 更新模型数: 12 → 11
   - 说明VulBERTa/cnn暂时不可用

2. **CLAUDE.md**
   - 更新当前状态
   - 记录VulBERTa/cnn问题
   - 更新实验完成度统计

3. **docs/results_reports/DEFAULT_EXPERIMENTS_ANALYSIS_20251210.md**
   - 添加清理说明
   - 更新统计数据
   - 标记为已过时（被本报告替代）

### 配置文件需要检查

1. **settings/stage_final_all_remaining.json**
   - 检查是否包含VulBERTa/cnn实验
   - 如有，需要移除或注释

2. **mutation/models_config.json**
   - 考虑临时移除cnn模型
   - 或添加注释说明未实现

---

## ✅ 验证清单

- [x] 确认问题根因（训练代码未实现）
- [x] 备份原始数据
- [x] 清理无效记录（42条）
- [x] 验证CSV格式（476行，37列）
- [x] 创建清理报告
- [ ] 更新README.md
- [ ] 更新CLAUDE.md
- [ ] 检查配置文件
- [ ] 决策VulBERTa/cnn的未来计划

---

## 🎯 结论

### 关键要点

1. **问题确认**: VulBERTa/cnn训练功能未实现，所有42个实验无效
2. **数据清理**: 已安全移除无效记录，保留476条有效数据
3. **数据完整性**: CSV格式验证通过，备份已创建
4. **影响评估**:
   - 对其他11个模型无影响
   - 实验完成度需重新计算（不包含VulBERTa/cnn）
   - 实验计划需更新

### 建议

**短期**:
- ✅ 完成文档更新
- ✅ 检查并更新实验配置文件
- ✅ 重新计算实验完成度（基于11个模型）

**中长期** (可选):
- 决定是否需要VulBERTa/cnn实验
- 如需要，实现CNN训练功能
- 重新运行VulBERTa/cnn实验

---

**报告创建时间**: 2025-12-11 14:40
**数据清理时间**: 2025-12-11 14:40
**清理操作者**: Claude (v4.7.2)
**验证状态**: ✅ 完成
