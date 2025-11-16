# Bug Fix: Path Duplication in run.sh

## 问题描述

**日期**: 2025-11-14
**严重性**: 🔴 Critical
**Bug ID**: #3

### 症状

运行实验时，会在项目目录下创建错误的嵌套`home`目录：
```
/home/green/energy_dl/nightly/home/green/energy_dl/nightly/results/...
```

日志文件和能耗数据被保存到错误的路径，导致：
1. 数据无法被正确读取
2. 磁盘空间浪费
3. 目录结构混乱

### 错误路径示例

Screen输出显示：
```
[Train Wrapper] CPU energy saved to: /home/green/energy_dl/nightly//home/green/energy_dl/nightly/results/run_20251113_205207/...
```

注意双斜杠 `//` 和路径重复。

---

## 根本原因

### 问题分析

**问题出现在**: `mutation/run.sh` 第39-40行

```bash
# 旧代码（错误）
LOG_FULL_PATH="$PROJECT_ROOT/$LOG_FILE"
ENERGY_FULL_PATH="$PROJECT_ROOT/$ENERGY_DIR"
```

**调用链**:

1. **runner.py** 创建绝对路径：
```python
# runner.py:432-433
log_file = str(exp_dir / "training.log")  # 绝对路径：/home/.../training.log
energy_dir = exp_dir / "energy"            # 绝对路径：/home/.../energy
```

2. **command_runner.py** 传递这些绝对路径给run.sh：
```python
# command_runner.py:93
cmd = [str(run_script), repo_path, train_script, log_file, energy_dir]
#                                                 ^^^^^^^^  ^^^^^^^^^^
#                                                 已经是绝对路径
```

3. **run.sh** 错误地再次拼接：
```bash
# run.sh (旧代码)
LOG_FULL_PATH="$PROJECT_ROOT/$LOG_FILE"
# 结果: /home/green/energy_dl/nightly + /home/green/energy_dl/nightly/results/...
#      = /home/green/energy_dl/nightly//home/green/energy_dl/nightly/results/...
```

### 为什么会这样？

Bash的路径拼接 `$A/$B`：
- 如果 `$B` 是相对路径（如 `results/file.log`），结果是 `$A/results/file.log` ✅
- 如果 `$B` 是绝对路径（如 `/home/user/file.log`），结果是 `$A//home/user/file.log` ❌

---

## 修复方案

### 代码修改

**文件1**: `mutation/run.sh`
**行号**: 35-52

```bash
# 修复后的代码
# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_FULL_PATH="$PROJECT_ROOT/$REPO_PATH"

# Handle log file path (check if already absolute)
if [[ "$LOG_FILE" = /* ]]; then
    LOG_FULL_PATH="$LOG_FILE"
else
    LOG_FULL_PATH="$PROJECT_ROOT/$LOG_FILE"
fi

# Handle energy directory path (check if already absolute)
if [[ "$ENERGY_DIR" = /* ]]; then
    ENERGY_FULL_PATH="$ENERGY_DIR"
else
    ENERGY_FULL_PATH="$PROJECT_ROOT/$ENERGY_DIR"
fi
```

**文件2**: `mutation/runner.py`
**行号**: 432-434

```python
# 修复后的代码 - 使用相对路径而非绝对路径
# Use the same experiment directory for all retry attempts
# Pass relative paths to run.sh, which will handle path resolution
log_file = f"results/{self.session.session_dir.name}/{experiment_id}/training.log"
energy_dir = f"results/{self.session.session_dir.name}/{experiment_id}/energy"
```

### 修复逻辑

**Part 1: run.sh增加绝对路径检测**
使用Bash的条件检测：
- `[[ "$PATH" = /* ]]` - 检查路径是否以 `/` 开头（绝对路径）
- 如果是绝对路径 → 直接使用
- 如果是相对路径 → 拼接PROJECT_ROOT

**Part 2: runner.py改用相对路径**
- 旧代码: `log_file = str(exp_dir / "training.log")` → 生成绝对路径
- 新代码: `log_file = f"results/{session_dir}/{exp_id}/training.log"` → 生成相对路径
- 这样确保传递给run.sh的始终是相对路径，避免路径重复

---

## 影响范围

### 受影响的运行

**运行ID**: `run_20251113_205207`
**时间**: 2025-11-13 20:52 - 2025-11-14 01:58
**实验数**: 12个

**数据状态**:
- ✅ 训练实际完成（每个实验运行了完整时间）
- ❌ 数据保存到错误路径
- ❌ 无法被程序读取
- ✅ 数据本身完整（training.log, energy数据都存在）

### 受影响的功能

- ✅ 训练执行（正常）
- ❌ 日志文件保存路径
- ❌ 能耗数据保存路径
- ❌ 结果读取和验证
- ❌ CSV生成

---

## 数据恢复

### 问题

由于路径错误，数据被保存到：
```
/home/green/energy_dl/nightly/home/green/energy_dl/nightly/results/run_20251113_205207/
```

而不是正确的：
```
/home/green/energy_dl/nightly/results/run_20251113_205207/
```

### 恢复尝试

尝试使用rsync移动数据到正确位置，但由于rsync参数问题，移动不完整。

### 最终处理

由于：
1. 数据已经因Bug #2无法使用（训练成功但无法验证）
2. 路径混乱导致数据组织不清
3. Bug已修复，可以重新运行

**决定**: 清理整个失败的运行目录

```bash
rm -rf /home/green/energy_dl/nightly/results/run_20251113_205207
```

---

## 测试验证

### 单元测试

**文件**: `tests/unit/test_runner.py`

添加了回归测试 `test_paths_are_relative_not_absolute()`，验证：
1. log_file 必须是相对路径（不以 `/` 开头）
2. energy_dir 必须是相对路径（不以 `/` 开头）
3. 路径必须以 `results/` 开头（符合预期格式）
4. 路径中不包含 `//`（无路径重复）

```python
def test_paths_are_relative_not_absolute(self):
    """
    Regression test for Bug #3: Path duplication bug

    Ensures that run_experiment generates relative paths, not absolute paths.
    This prevents path duplication in run.sh when it concatenates PROJECT_ROOT.
    """
    # ... test implementation ...

    # Critical assertions: paths must be relative, not absolute
    self.assertFalse(
        log_file.startswith('/'),
        f"log_file should be relative path, not absolute. Got: {log_file}"
    )
    self.assertFalse(
        str(energy_dir).startswith('/'),
        f"energy_dir should be relative path, not absolute. Got: {energy_dir}"
    )
```

### 集成测试

**文件**: `tests/functional/test_refactoring.py`

添加了集成测试 `test_path_handling()`，验证：
1. 不会创建嵌套的 `home/` 目录
2. 实验目录创建在正确位置
3. 路径中不包含 `//`（无路径重复）
4. 路径结构正确：`results/run_XXXXXX/repo_model_NNN`

```python
@test("8. Path Handling (Bug #3 Regression Test)")
def test_path_handling():
    """
    Integration test for Bug #3: Path duplication bug

    Verifies that:
    1. No nested 'home/' directory is created
    2. Files are created in the correct location
    3. No path duplication occurs (no '//' in paths)
    """
    # ... test implementation ...

    # Check 1: No nested 'home/' directory created
    home_dirs = list(temp_results.rglob("home"))
    assert len(home_dirs) == 0, f"Unexpected 'home/' directory created: {home_dirs}"

    # Check 3: No path duplication (no '//' in path)
    assert '//' not in exp_dir_str, f"Path duplication detected (contains '//'): {exp_dir_str}"
```

### 测试结果

```bash
# 单元测试（6个测试）
$ python3 -m unittest tests.unit.test_runner -v
...
test_paths_are_relative_not_absolute ... ok
...
Ran 6 tests in 0.010s
OK

# 功能测试（9个测试）
$ python3 tests/functional/test_refactoring.py
...
TEST: 8. Path Handling (Bug #3 Regression Test)
  ✓ No 'home/' directory created
  ✓ Experiment directory in correct location: run_20251114_160520/pytorch_resnet_cifar10_resnet20_001
  ✓ No path duplication (no '//' in paths)
  ✓ Path structure correct: run_20251114_160520/pytorch_resnet_cifar10_resnet20_001
✓ PASSED
...
🎉 ALL TESTS PASSED!
```

### 手动测试

```bash
# 测试1: 使用绝对路径（run.sh应直接使用）
./mutation/run.sh repos/test ./train.sh /absolute/path/training.log /absolute/path/energy

# 验证: 文件应该保存到 /absolute/path/... ✅

# 测试2: 使用相对路径（run.sh应拼接PROJECT_ROOT）
./mutation/run.sh repos/test ./train.sh relative/training.log relative/energy

# 验证: 文件应该保存到 $PROJECT_ROOT/relative/... ✅
```

---

## 预防措施

### 1. 代码审查清单

路径处理相关代码需要检查：
- [ ] 是否区分绝对路径和相对路径？
- [ ] 路径拼接是否会导致重复？
- [ ] 是否有路径规范化处理？

### 2. 路径处理最佳实践

**Python**:
```python
# ✅ 推荐: 使用pathlib
path = Path("/absolute/path")
full_path = base / path  # pathlib自动处理绝对路径

# ❌ 避免: 字符串拼接
full_path = base + "/" + path  # 可能导致重复
```

**Bash**:
```bash
# ✅ 推荐: 检查绝对路径
if [[ "$path" = /* ]]; then
    full_path="$path"
else
    full_path="$base/$path"
fi

# ❌ 避免: 盲目拼接
full_path="$base/$path"  # 如果path是绝对路径会出错
```

### 3. 集成测试

添加端到端测试验证：
- 实验目录创建在正确位置
- 日志文件保存在正确位置
- 能耗数据保存在正确位置
- 没有重复的路径前缀

---

## 相关Bug

此bug与以下问题相关：

1. **Bug #1**: run_training_with_monitoring参数错误
   - 导致实验无法运行

2. **Bug #2**: check_training_success签名错误
   - 导致训练完成后无法验证

3. **Bug #3**: 路径重复bug（本bug）
   - 导致数据保存到错误位置

### 综合影响

这三个bug共同导致 `run_20251113_205207` 完全失败：
- Bug #1 阻止了前几个实验运行（已修复后才运行）
- Bug #2 阻止了结果验证和保存
- Bug #3 导致数据保存到错误路径

---

## 修复检查清单

- [x] 代码修复: `mutation/run.sh:35-52` - 增加绝对路径检测
- [x] 代码修复: `mutation/runner.py:432-434` - 改用相对路径
- [x] 删除错误的home目录
- [x] 清理失败的运行数据
- [x] 添加路径处理单元测试 (`tests/unit/test_runner.py`)
- [x] 添加路径处理集成测试 (`tests/functional/test_refactoring.py`)
- [x] 更新文档
- [x] 准备重新运行实验

---

## 下一步

### 建议行动

1. ✅ **Bug已修复** - run.sh现在正确处理绝对路径
2. ✅ **清理完成** - 删除了错误的home目录和失败的运行
3. 🔜 **重新运行** - 使用修复后的代码运行边界测试

### 运行命令

```bash
# 进入screen
screen -r test

# 运行边界测试（所有bug已修复）
sudo python3 mutation.py -ec settings/boundary_test_v2.json

# 分离 (Ctrl+A+D)
```

---

## 版本信息

- **修复版本**: v4.0.5
- **修复的Bug**: Bug #3 - 路径重复
- **修复日期**: 2025-11-14
- **修复内容**:
  - Part 1: run.sh增加绝对路径检测（v4.0.4）
  - Part 2: runner.py改用相对路径（v4.0.5）
  - 添加单元测试和集成测试
- **状态**: ✅ 已修复并验证
- **测试覆盖**:
  - 单元测试: 6个测试 (test_runner.py)
  - 集成测试: 9个测试 (test_refactoring.py)
  - 全部通过 ✅

---

## 技术细节

### Bash路径拼接行为

```bash
# 示例1: 相对路径拼接（正确）
base="/home/user/project"
rel="results/file.txt"
result="$base/$rel"
# 结果: /home/user/project/results/file.txt ✅

# 示例2: 绝对路径拼接（错误）
base="/home/user/project"
abs="/home/user/project/results/file.txt"
result="$base/$abs"
# 结果: /home/user/project//home/user/project/results/file.txt ❌

# 示例3: 修复后（正确）
if [[ "$abs" = /* ]]; then
    result="$abs"  # 直接使用绝对路径
else
    result="$base/$abs"  # 拼接相对路径
fi
# 结果: /home/user/project/results/file.txt ✅
```

### Python pathlib行为

```python
from pathlib import Path

base = Path("/home/user/project")
rel = Path("results/file.txt")
abs_path = Path("/home/user/project/results/file.txt")

# pathlib自动处理
base / rel  # → /home/user/project/results/file.txt ✅
base / abs_path  # → /home/user/project/results/file.txt ✅ (自动规范化)
```

---

**总结**: 路径拼接需要区分绝对路径和相对路径，否则会导致路径重复。修复后run.sh能正确处理两种情况。
