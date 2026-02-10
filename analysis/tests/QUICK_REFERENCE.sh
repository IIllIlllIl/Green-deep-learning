# DiBS步数验证测试 - 快速命令参考

## 立即执行（推荐）

```bash
# 1. 切换目录
cd /home/green/energy_dl/nightly/analysis

# 2. 激活环境
conda activate causal-research

# 3. 运行快速测试（~30秒）
python3 tests/test_dibs_quick.py
```

---

## 标准测试（~5分钟）

```bash
# 使用Shell脚本
./tests/run_dibs_step_test.sh 1000 20

# 或直接运行Python
python3 tests/test_dibs_step_verification.py --steps 1000 --particles 20
```

---

## 自定义测试

```bash
# 500步，15粒子，callback间隔50
./tests/run_dibs_step_test.sh 500 15 50

# 或
python3 tests/test_dibs_step_verification.py \
    --steps 500 \
    --particles 15 \
    --callback-every 50
```

---

## 检查测试状态

```bash
# 查看测试帮助
python3 tests/test_dibs_step_verification.py --help

# 查看测试文档
cat tests/README_TESTS.md
```

---

## 预期结果

### 成功

```
✅ 快速测试通过
   DiBS正确执行了 100 步（预期 100 步）
```

### 失败

```
❌ 测试失败: 预期100步，实际10步
```

---

## 文档位置

- **快速索引**: `tests/README_TESTS.md`
- **完整文档**: `docs/technical_reference/DIBS_STEP_VERIFICATION_TEST.md`
- **执行总结**: `docs/technical_reference/DIBS_STEP_TEST_SUMMARY.md`
