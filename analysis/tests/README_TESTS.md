# DiBS测试套件

本目录包含DiBS因果图学习的测试脚本。

---

## 快速开始

### 1. 快速验证（推荐首次使用，~30秒）

```bash
cd /home/green/energy_dl/nightly/analysis
conda activate causal-research
python3 tests/test_dibs_quick.py
```

**预期输出**: `✅ 快速测试通过`

### 2. 标准验证（~5分钟）

```bash
./tests/run_dibs_step_test.sh 1000 20
```

或：

```bash
python3 tests/test_dibs_step_verification.py --steps 1000 --particles 20
```

---

## 测试文件说明

| 文件 | 用途 | 快速测试 | 标准测试 |
|------|------|---------|---------|
| `test_dibs_quick.py` | 快速验证（100步） | ✅ | - |
| `test_dibs_step_verification.py` | 完整验证（可配置） | - | ✅ |
| `run_dibs_step_test.sh` | Shell包装脚本 | - | ✅ |

---

## 测试目的

验证DiBS是否真的执行了设定的训练步数。

**背景问题**: 之前出现过callback设置导致未执行设定步数的情况。

**验证方法**: 通过callback机制监控实际执行的步数。

---

## 详细文档

完整文档参见：`docs/technical_reference/DIBS_STEP_VERIFICATION_TEST.md`
