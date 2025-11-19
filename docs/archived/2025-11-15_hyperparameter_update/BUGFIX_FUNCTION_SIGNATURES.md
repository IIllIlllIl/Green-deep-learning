# Bug Fix Summary: Function Signature Mismatch Errors

## 概述

**日期**: 2025-11-13
**版本**: v4.0.2 → v4.0.3
**修复的Bug数量**: 2个关键bug
**新增测试**: 27个（5个runner测试 + 22个energy测试）

---

## Bug #1: run_training_with_monitoring() 参数错误

### 问题
```
TypeError: CommandRunner.run_training_with_monitoring() got an unexpected keyword argument 'repo'
```

### 位置
`mutation/runner.py:440`

### 原因
`run_experiment()` 方法错误地直接传递 `repo`, `model`, `mutation` 参数给 `run_training_with_monitoring()`，但该方法需要先构建命令。

### 修复
```python
# 修复前 (错误)
exit_code, duration, energy_metrics = self.cmd_runner.run_training_with_monitoring(
    repo=repo, model=model, mutation=mutation, ...
)

# 修复后 (正确)
cmd = self.cmd_runner.build_training_command_from_dir(
    repo=repo, model=model, mutation=mutation, ...
)
exit_code, duration, energy_metrics = self.cmd_runner.run_training_with_monitoring(
    cmd=cmd, log_file=log_file, exp_dir=exp_dir, ...
)
```

---

## Bug #2: check_training_success() 和 extract_performance_metrics() 参数错误

### 问题
```
AttributeError: 'PosixPath' object has no attribute 'warning'
```

### 位置
- `mutation/runner.py:458` - `check_training_success()` 调用
- `mutation/runner.py:464` - `extract_performance_metrics()` 调用

### 原因
这两个函数的签名在重构时改变了，但runner.py中的调用代码没有更新：

**旧签名（已废弃）**:
```python
check_training_success(log_file, repo, config, project_root)
extract_performance_metrics(log_file, repo, config, project_root, logger)
```

**新签名（正确）**:
```python
check_training_success(log_file, repo, min_log_file_size_bytes, logger)
extract_performance_metrics(log_file, repo, log_patterns, logger)
```

### 修复

#### check_training_success()
```python
# 修复前 (错误)
success, error_message = check_training_success(
    log_file, repo, self.config, self.project_root
)

# 修复后 (正确)
success, error_message = check_training_success(
    log_file=log_file,
    repo=repo,
    min_log_file_size_bytes=self.MIN_LOG_FILE_SIZE_BYTES,
    logger=self.logger
)
```

#### extract_performance_metrics()
```python
# 修复前 (错误)
performance_metrics = extract_performance_metrics(
    log_file, repo, self.config, self.project_root, self.logger
)

# 修复后 (正确)
repo_config = self.config["models"][repo]
log_patterns = repo_config.get("performance_metrics", {})
performance_metrics = extract_performance_metrics(
    log_file=log_file,
    repo=repo,
    log_patterns=log_patterns,
    logger=self.logger
)
```

---

## 新增测试

### 1. Runner模块测试 (`tests/unit/test_runner.py`)

**新增**: 5个测试

| 测试名称 | 目的 |
|---------|------|
| `test_runner_initialization` | 验证Runner正确初始化 |
| `test_runner_initialization_with_seed` | 验证带种子初始化 |
| `test_run_experiment_calls_build_command` | **验证正确的命令构建流程** |
| `test_run_experiment_retries_on_failure` | 验证重试机制 |
| `test_run_experiment_signature_bug_fix` | **回归测试：防止Bug #1再次发生** |

**关键测试**:
```python
def test_run_experiment_signature_bug_fix(self):
    """确保run_training_with_monitoring使用正确的参数"""
    # 验证应该存在的参数
    self.assertIn('cmd', call_kwargs)
    self.assertIn('log_file', call_kwargs)

    # 验证不应该存在的参数（这是bug）
    self.assertNotIn('repo', call_kwargs)
    self.assertNotIn('model', call_kwargs)
```

### 2. Energy模块测试 (`tests/unit/test_energy.py`)

**新增**: 22个测试

#### TestCheckTrainingSuccess (6个测试)
- ✅ `test_function_signature` - **验证函数签名正确**
- ✅ `test_log_file_not_found` - 日志文件不存在
- ✅ `test_success_pattern_detected` - 成功模式检测
- ✅ `test_error_pattern_detected` - 错误模式检测
- ✅ `test_log_file_too_small` - 日志文件过小
- ✅ `test_with_custom_logger` - 自定义logger使用

#### TestExtractPerformanceMetrics (7个测试)
- ✅ `test_function_signature` - **验证函数签名正确**
- ✅ `test_log_file_not_found` - 日志文件不存在
- ✅ `test_metric_extraction` - 指标提取
- ✅ `test_multiple_metrics` - 多个指标提取
- ✅ `test_no_patterns` - 无模式时的行为
- ✅ `test_pattern_not_found` - 模式未找到

#### TestParseEnergyMetrics (4个测试)
- ✅ `test_function_signature` - **验证函数签名正确**
- ✅ `test_empty_directory` - 空目录处理
- ✅ `test_cpu_energy_parsing` - CPU能耗解析
- ✅ `test_gpu_power_csv_parsing` - GPU功率CSV��析

#### TestParseCsvMetricStreaming (3个测试)
- ✅ `test_file_not_found` - 文件不存在
- ✅ `test_valid_csv_parsing` - 有效CSV解析
- ✅ `test_field_not_found` - 字段不存在

#### TestEnergyFunctionIntegration (3个测试)
- ✅ `test_check_training_success_integration` - **集成测试：验证正确调用方式**
- ✅ `test_extract_performance_metrics_integration` - **集成测试：验证正确调用方式**
- ✅ `test_runner_calls_with_wrong_signature_should_fail` - **回归测试：防止Bug #2再次发生**

**关键回归测试**:
```python
def test_runner_calls_with_wrong_signature_should_fail(self):
    """确保错误的参数类型会失败"""
    with self.assertRaises((TypeError, AttributeError)):
        # 传递错误类型的参数应该失败
        check_training_success(
            str(log_file),
            "repo",
            {"models": {}},  # 错误：应该是int
            Path(".")        # 错误：应该是logger
        )
```

---

## 测试覆盖统计

### 之前
- **单元测试**: 30个
- **功能测试**: 8个
- **总计**: 38个测试

### 现在
- **单元测试**: 52个 (+22个，+73%)
- **功能测试**: 8个
- **总计**: 60个测试 (+22个，+58%)

### 按模块分布

| 模块 | 测试文件 | 测试数 | 新增 |
|------|---------|--------|------|
| **hyperparams** | test_hyperparams.py | 12 | - |
| **session** | test_session.py | 9 | - |
| **utils** | test_utils.py | 4 | - |
| **runner** | test_runner.py | 5 | ✨ +5 |
| **energy** | test_energy.py | 22 | ✨ +22 |
| **功能测试** | test_refactoring.py | 8 | - |
| **总计** | - | **60** | **+27** |

---

## 预防措施

### 1. 函数签名验证
每个关键函数都有 `test_function_signature` 测试：
```python
def test_function_signature(self):
    import inspect
    sig = inspect.signature(check_training_success)
    params = list(sig.parameters.keys())

    # 验证参数名和数量
    self.assertEqual(params[0], 'log_file')
    self.assertEqual(params[1], 'repo')
    self.assertEqual(len(params), 4)
```

### 2. 集成测试
验证runner.py中实际调用方式：
```python
def test_check_training_success_integration(self):
    """模拟runner.py的调用方式"""
    MIN_LOG_FILE_SIZE_BYTES = 1000
    mock_logger = Mock()

    success, error_msg = check_training_success(
        log_file=str(log_file),
        repo="pytorch_resnet_cifar10",
        min_log_file_size_bytes=MIN_LOG_FILE_SIZE_BYTES,
        logger=mock_logger
    )
```

### 3. 回归测试
专门测试���误的调用方式会失败：
```python
def test_runner_calls_with_wrong_signature_should_fail(self):
    """确保旧的错误调用方式会失败"""
    with self.assertRaises((TypeError, AttributeError)):
        check_training_success(log_file, repo, config, project_root)
```

---

## 测试结果

```bash
# 单元测试
python3 -m unittest discover -s tests/unit
# ✅ Ran 52 tests in 0.045s
# ✅ OK (skipped=1)

# 功能测试
python3 tests/functional/test_refactoring.py
# ✅ All 8 tests passed

# 总计: 60个测试 (59 passed, 1 skipped)
```

---

## 根本原因分析

### 为什么会发生？

1. **重构过程中的疏忽**
   - v4.0重构时改变了函数签名
   - 更新了函数实现，但忘记更新所有调用点

2. **缺少接口契约测试**
   - 重构前没有函数签名验证测试
   - 没有集成测试验证实际调用方式

3. **测试覆���不足**
   - energy模块完全没有单元测试
   - runner模块只有功能测试，没有单元测试

### 改进措施

1. ✅ **增加签名验证测试** - 每个公共函数都验证签名
2. ✅ **增加集成测试** - 验证跨模块调用的正确性
3. ✅ **增加回归测试** - 防止已修复的bug再次出现
4. ✅ **提高测试覆盖** - 从30个增至52个单元测试 (+73%)

---

## 相关文档

- [第一个Bug修复](BUGFIX_RUN_TRAINING_SIGNATURE.md)
- [重构总结](REFACTORING_SUMMARY.md)
- [配置迁移](CONFIG_MIGRATION.md)

---

## 版本更新

- **v4.0.2**: 修复Bug #1 (run_training_with_monitoring参数错误)
- **v4.0.3**: 修复Bug #2 (check_training_success/extract_performance_metrics参数错误)
- **测试版本**: v1.0 → v2.0 (新增27个测试)

---

**状态**: ✅ 所有bug已修复并验证
**测试覆盖**: 60个测试 (59 passed, 1 skipped)
**回归预防**: ✅ 已建立完整的签名验证和回归测试机制
