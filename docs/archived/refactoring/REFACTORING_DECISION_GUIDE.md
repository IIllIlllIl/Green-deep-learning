# mutation.py 重构决策指南

## 🎯 快速决策树

```
是否有 10-12 小时可用时间？
├─ 是 → 执行【方案 A: 完整重构】⭐ 推荐
└─ 否
    ├─ 有 4-6 小时 → 执行【方案 B: 最小化重构】
    └─ 只能每次投入 2 小时 → 执行【方案 C: 增量重构】
```

---

## 📊 三种方案对比

| 维度 | 方案 A: 完整重构 | 方案 B: 最小化重构 | 方案 C: 增量重构 |
|------|-----------------|-------------------|-----------------|
| **时间投入** | 10-12 小时（1.5-2 天） | 4-6 小时（1 天） | 5×2 小时（分 5 批） |
| **完成度** | 100% | 40% | 100% |
| **测试覆盖率** | >80% | ~50% | >80% |
| **技术债** | 完全消除 | 部分消除 | 完全消除 |
| **风险** | 低（有测试保障） | 极低 | 极低 |
| **可维护性提升** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **后续扩展性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🚀 方案 A: 完整重构（推荐）

### 适用场景
- ✅ 有 1.5-2 个完整工作日
- ✅ 希望长期维护此项目
- ✅ 计划添加更多功能（如分布式训练、Web UI）
- ✅ 团队有多人协作需求

### 执行计划

**第 1 天上午（4 小时）**:
```bash
# 阶段 1: 准备 (1 小时)
mkdir -p mutation tests
git checkout -b refactor/modularize-mutation

# 阶段 2: 纯函数模块迁移 (2 小时)
# - exceptions.py
# - session.py
# - hyperparams.py (部分)

# 阶段 3: 工具模块迁移 (1 小时)
# - utils.py
```

**第 1 天下午（4 小时）**:
```bash
# 继续阶段 2 (1 小时)
# - hyperparams.py (完成)
# - energy.py

# 阶段 4: 命令执行模块迁移 (2 小时)
# - command_runner.py

# 阶段 5: 编排模块迁移 (1 小时，开始)
# - runner.py (部分)
```

**第 2 天上午（2 小时）**:
```bash
# 继续阶段 5 (30 分钟)
# - runner.py (完成)

# 阶段 6: CLI 入口迁移 (30 分钟)
# - mutation.py (重写)

# 阶段 7: 集成测试 (1 小时，开始)
```

**第 2 天下午（2 小时）**:
```bash
# 继续阶段 7: 集成测试与验证 (1 小时)
# - 端到端测试
# - 回归测试

# 阶段 8: 文档与清理 (1 小时)
# - 更新 README
# - 提交代码
```

### 成功标准
- ✅ 所有模块有单元测试（覆盖率 >80%）
- ✅ 端到端测试通过
- ✅ 输出格式与原版完全一致
- ✅ 无类型错误（mypy）
- ✅ 无风格错误（flake8）

### 预期收益
- 文件行数减少 70%（最大文件从 1,851 行降至 550 行）
- 可测试性提升 700%（从 1 个模块到 8 个独立可测模块）
- 维护成本降低 60%（单一职责，修改影响范围小）

---

## ⚡ 方案 B: 最小化重构（快速）

### 适用场景
- ✅ 时间有限（仅 4-6 小时）
- ✅ 只想解决最紧急的问题
- ✅ 计划后续再进行完整重构

### 执行计划

**第 1 天上午（3 小时）**:
```bash
# 阶段 1: 准备 (1 小时)
mkdir -p mutation tests
git checkout -b refactor/minimal-split

# 阶段 2: 核心模块迁移 (2 小时)
# - exceptions.py
# - session.py
# - hyperparams.py
```

**第 1 天下午（2-3 小时）**:
```bash
# 阶段 6: 简化 CLI 入口 (1 小时)
# - 保持 MutationRunner 在原位，但导入 session 和 hyperparams

# 测试与验证 (1-2 小时)
# - 基本功能测试
# - 回归测试
```

### 迁移示例

**mutation.py（简化版）**:
```python
#!/usr/bin/env python3
# ... imports ...
from mutation.session import ExperimentSession
from mutation.hyperparams import (
    generate_mutations,
    mutate_hyperparameter,
    format_hyperparam_value,
    build_hyperparam_args,
)

class MutationRunner:
    """保留在原位，但使用导入的模块"""

    def __init__(self, ...):
        self.session = ExperimentSession(results_dir)
        # ... rest of __init__ ...

    # generate_mutations 现在调用导入的函数
    def generate_mutations(self, repo_config, mutate_params, num_mutations=1):
        return generate_mutations(
            repo_config=repo_config,
            mutate_params=mutate_params,
            num_mutations=num_mutations,
            random_seed=self.random_seed,
        )

    # 其他方法保持不变
```

### 成功标准
- ✅ session.py 和 hyperparams.py 有单元测试
- ✅ 基本功能测试通过
- ✅ 输出格式与原版一致

### 预期收益
- 核心逻辑（超参数突变）可独立测试
- 会话管理代码隔离，便于后续扩展
- 保留完整重构的选项

---

## 🔄 方案 C: 增量重构（灵活）

### 适用场景
- ✅ 希望分批次进行，降低风险
- ✅ 每次只能投入 2-3 小时
- ✅ 希望在每批完成后立即获得收益

### 执行计划

**第 1 批（2 小时）**:
```bash
# 目标：建立基础设施
- 阶段 1: 准备
- exceptions.py
- session.py + 测试

# 验证：session 单元测试通过
```

**第 2 批（2 小时）**:
```bash
# 目标：隔离核心逻辑
- hyperparams.py + 测试
- 更新 mutation.py 导入 hyperparams

# 验证：hyperparams 单元测试通过，基本功能正常
```

**第 3 批（2.5 小时）**:
```bash
# 目标：隔离工具和解析
- utils.py + 测试
- energy.py + 测试

# 验证：工具函数和解析函数独立可用
```

**第 4 批（2.5 小时）**:
```bash
# 目标：隔离命令执行
- command_runner.py + 测试

# 验证：命令构建和执行逻辑可独立测试
```

**第 5 批（2 小时）**:
```bash
# 目标：完成重构
- runner.py（重新组织 MutationRunner）
- mutation.py（极简 CLI）
- 集成测试 + 文档

# 验证：端到端测试通过，所有功能正常
```

### 每批独立验证

每批完成后执行：
```bash
# 运行新增的单元测试
pytest tests/test_XXX.py -v

# 运行回归测试（确保基本功能不受影响）
./mutation.py --repo mnist_torch --model default --mode train --timeout 60

# 检查输出格式
cat results/latest/summary.csv
```

### 成功标准
- ✅ 每批完成后，所有测试通过
- ✅ 每批完成后，基本功能正常
- ✅ 第 5 批完成后，达到与方案 A 相同的效果

### 预期收益
- 风险分散到 5 个独立批次
- 每批完成后立即获得部分收益
- 可根据进展调整后续批次优先级

---

## 🔍 关键决策因素

### 1. 时间可用性

```
可用时间 ≥ 10 小时？
├─ 是 → 方案 A（完整重构）
└─ 否
    ├─ 可用时间 ≥ 6 小时？
    │   └─ 是 → 方案 B（最小化重构）
    └─ 否 → 方案 C（增量重构）
```

### 2. 项目维护周期

```
计划维护周期 > 6 个月？
├─ 是 → 方案 A 或 C（获得最大长期收益）
└─ 否 → 方案 B（快速见效）
```

### 3. 团队规模

```
团队人数 > 1？
├─ 是 → 方案 A 或 C（支持并行协作）
└─ 否 → 任意方案（单人项目，灵活选择）
```

### 4. 未来扩展计划

```
计划添加新功能（如分布式训练、Web UI）？
├─ 是 → 方案 A 或 C（模块化架构易扩展）
└─ 否 → 方案 B（满足当前需求即可）
```

---

## 📋 执行前检查清单

在开始任何方案前，确认以下事项：

### 必需项
- [ ] 已备份当前代码（`cp mutation.py mutation.py.backup`）
- [ ] 已创建新分支（`git checkout -b refactor/...`）
- [ ] 已安装测试依赖（`pip install pytest pytest-cov pytest-mock`）
- [ ] 已阅读完整分析报告（`docs/REFACTORING_ANALYSIS.md`）

### 推荐项
- [ ] 已准备小型测试脚本（用于快速验证）
- [ ] 已通知团队成员（如有协作）
- [ ] 已预留缓冲时间（实际时间可能比估算多 20%）
- [ ] 已设置提醒（每 2 小时运行测试验证）

### 可选项
- [ ] 已配置 CI 自动测试
- [ ] 已设置代码覆盖率工具
- [ ] 已准备回滚计划

---

## 🎯 推荐决策路径

### 标准路径（推荐 90% 的情况）

1. **评估可用时间**
   - 如果有完整的 1.5-2 天 → **选择方案 A**
   - 如果只有 1 天 → **选择方案 B**
   - 如果时间分散 → **选择方案 C**

2. **执行前验证**
   - 运行现有测试（如果有）
   - 备份代码
   - 创建分支

3. **开始重构**
   - 按照选定方案的执行计划逐步进行
   - 每完成一个模块，立即编写测试并验证
   - 遇到问题时参考完整分析报告

4. **完成验证**
   - 运行所有单元测试
   - 运行端到端测试
   - 对比新旧版本输出

5. **合并代码**
   - 代码审查（如有团队）
   - 合并到主分支
   - 更新文档

---

## 🚨 常见问题解答

### Q1: 重构会破坏现有功能吗？
**A**: 不会。通过以下措施保证兼容性：
- 保持结果 JSON 和 CSV 格式不变
- 保持 CLI 接口不变
- 每个模块迁移后立即测试
- 最终进行端到端回归测试

### Q2: 是否需要修改配置文件？
**A**: 不需要。配置文件（`config/models_config.json`、实验配置 JSON）保持不变。

### Q3: 是否需要修改 Shell 脚本？
**A**: 不需要。`run.sh`、后台训练模板脚本等保持不变。

### Q4: 重构后性能会受影响吗？
**A**: 几乎没有影响。仅增加少量导入开销（<1ms），对于训练任务可忽略不计。

### Q5: 如果中途遇到问题怎么办？
**A**:
- 方案 A/C: 回退到上一个通过测试的阶段
- 方案 B: 回退到原始 mutation.py
- 所有方案：使用 `git checkout mutation.py.backup` 恢复备份

### Q6: 是否可以混合不同方案？
**A**: 可以。例如：
- 先执行方案 B（快速见效）
- 后续逐步完成方案 A 的剩余部分

### Q7: 如何确保测试覆盖率？
**A**: 使用 pytest-cov：
```bash
pytest --cov=mutation --cov-report=html tests/
# 查看 htmlcov/index.html
```

### Q8: 是否需要重写所有测试？
**A**: 不需要。如果已有测试，可以：
1. 保持集成测试不变（测试 CLI 行为）
2. 为新模块添加单元测试
3. 逐步提升覆盖率

---

## 📞 获取更多帮助

如果在决策或执行过程中遇到问题，请参考：

1. **完整分析报告**: `docs/REFACTORING_ANALYSIS.md`
   - 详细的模块设计
   - 代码迁移示例
   - 完整的依赖关系图

2. **测试策略**: 完整分析报告中的"测试策略"部分
   - 每个模块的测试用例示例
   - Mock 使用指南

3. **风险缓解**: 完整分析报告中的"可行性评估"部分
   - 风险分析
   - 缓解措施

4. **代码示例**: 完整分析报告中的"模块详细设计"部分
   - 公共 API 设计
   - 迁移前后对比

---

## ✅ 最终建议

**对于 90% 的情况，推荐【方案 A: 完整重构】**

**理由**:
1. ✅ 投入产出比最高（10-12 小时获得 70% 代码规模减少）
2. ✅ 一次性解决所有问题，避免后续技术债
3. ✅ 风险可控（增量迁移 + 完整测试）
4. ✅ 长期收益巨大（可维护性、可测试性、可扩展性）

**立即开始**:
```bash
# 1. 备份
cp mutation.py mutation.py.backup

# 2. 创建分支
git checkout -b refactor/modularize-mutation

# 3. 阅读完整分析
cat docs/REFACTORING_ANALYSIS.md

# 4. 开始阶段 1
mkdir -p mutation tests
touch mutation/__init__.py
# ... 按照方案 A 执行 ...
```

---

**文档版本**: v1.0
**创建日期**: 2025-11-13
**相关文档**: docs/REFACTORING_ANALYSIS.md
