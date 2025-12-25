# Summary Append Control Feature - Update

**更新日期**: 2025-11-26
**更新内容**: 添加CLI缩写

---

## ✅ 新增功能：缩写 `-S`

为 `--skip-summary-append` 添加了缩写 `-S`，使用更便捷。

### 使用方法

```bash
# 完整参数名
python3 mutation.py -ec settings/gpu_memory_cleanup_test.json --skip-summary-append

# 使用缩写 -S (推荐)
python3 mutation.py -ec settings/gpu_memory_cleanup_test.json -S
```

### 验证结果

```bash
$ python3 mutation.py --help | grep -A 2 "skip-summary"
  -S, --skip-summary-append
                        Skip appending results to results/summary_all.csv (for
                        test/validation runs)
```

### 测试通过

```bash
$ python3 tests/unit/test_summary_append_flag.py
================================================================================
Test Summary: 6/6 Passed ✅
================================================================================
```

---

## 常用命令示例

```bash
# 正式实验（默认，添加到 summary_all.csv）
python3 mutation.py -ec settings/mutation_2x_supplement.json

# 测试/验证（使用 -S，不添加）
python3 mutation.py -ec settings/gpu_memory_cleanup_test.json -S

# 短命令示例
python3 mutation.py -r examples -m mnist_ff -mt all -n 1 -S
```

---

**状态**: ✅ 缩写添加完成、测试通过
