# 脚本复用方案 - 快速参考

## 核心变更

### 文件结构
```
nightly/
├── scripts/
│   └── background_training_template.sh  ← 新增：可复用模板脚本
├── mutation.py                           ← 修改：使用模板脚本
├── test/
│   └── test_script_reuse.py             ← 新增：测试套件
└── docs/
    ├── SCRIPT_REUSE_ANALYSIS.md         ← 新增：技术分析
    └── SCRIPT_REUSE_IMPLEMENTATION.md   ← 新增：实施总结
```

## 使用方法（无变化）

```bash
# 并行训练命令完全相同
sudo python3 mutation.py -ec settings/parallel_example.json
```

## 主要区别

### 之前
```
每次运行 → 创建临时脚本 → 使用 → 删除
```

### 之后
```
所有运行 → 复用单个模板脚本 + 不同参数
```

## 验证测试

```bash
# 运行测试套件
python3 test/test_script_reuse.py

# 预期结果
# Tests run: 26
# Successes: 26
# Failures: 0
# ✅ All tests passed!
```

## 关键修改点

### 1. mutation.py

**`_start_background_training()` (Line 662-719)**
- 使用模板脚本：`scripts/background_training_template.sh`
- 参数通过命令行传递
- 返回 `(process, None)` 而非 `(process, script_path)`

**`_stop_background_training()` (Line 721-756)**
- 仅终止进程
- 不删除脚本（模板可复用）

**`run_parallel_experiment()` (Line 820-823)**
- 移除脚本清理逻辑

### 2. 新增模板脚本

**`scripts/background_training_template.sh`**
- 接受5个参数：repo_path, train_script, train_args, log_dir, restart_delay
- 参数验证和错误处理
- 无限循环训练
- 自动日志分离

## 快速检查清单

- [✅] 模板脚本存在：`ls scripts/background_training_template.sh`
- [✅] 脚本可执行：`test -x scripts/background_training_template.sh`
- [✅] 测试全部通过：`python3 test/test_script_reuse.py`
- [✅] 无临时脚本：`ls results/background_training_*.sh 2>/dev/null | wc -l` → 0

## 优势

1. **代码简洁**: 减少脚本生成和删除逻辑
2. **维护性强**: 模板脚本可独立修改和测试
3. **资源效率**: 无临时文件，减少磁盘I/O
4. **向后兼容**: API和行为完全兼容

## 文档索引

- **技术分析**: `docs/SCRIPT_REUSE_ANALYSIS.md` - 详细的可行性分析和方案对比
- **实施总结**: `docs/SCRIPT_REUSE_IMPLEMENTATION.md` - 完整的实施细节和测试结果
- **快速参考**: `docs/SCRIPT_REUSE_QUICKREF.md` - 本文件

## 问题排查

### 模板脚本不存在
```bash
chmod +x scripts/background_training_template.sh
```

### 查看后台训练日志
```bash
cat results/background_logs_*/run_*.log
```

### 终止残留进程
```bash
ps aux | grep background_training_template.sh
kill -TERM -<PGID>
```

---

**状态**: ✅ 已完成并测试
**测试**: 26/26 通过
**可用**: 立即可用
