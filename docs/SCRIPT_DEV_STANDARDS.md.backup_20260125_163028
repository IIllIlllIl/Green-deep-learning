# 脚本开发规范

**版本**: v1.0.0
**最后更新**: 2026-01-10
**适用范围**: Energy DL 项目所有 Python 脚本

---

## 📋 概述

本文档定义了 Energy DL 项目的脚本开发标准，确保代码质量、可维护性和可复用性。

**核心原则**: 测试先行 → Dry Run → 全量执行 → 文档完整

---

## 🎯 开发流程总览

```bash
# ❌ 错误做法：直接全量执行
python3 script.py --input data.csv --output result.csv

# ✅ 正确做法：测试 → Dry Run → 全量执行

# 步骤1: 编写并运行测试
python3 test_script.py

# 步骤2: Dry Run（前10行）
python3 script.py --input data.csv --output result.csv --dry-run --limit 10

# 步骤3: 检查 Dry Run 结果
head -20 result.csv
python3 verify_result.py --file result.csv

# 步骤4: 全量执行
python3 script.py --input data.csv --output result.csv
```

---

## 📝 脚本结构标准

### 必需组件

每个脚本必须包含以下组件：

```python
#!/usr/bin/env python3
"""
脚本名称和简要描述

详细描述脚本的功能、用途和使用场景。

使用方法:
    python3 script_name.py --input data.csv --output result.csv
    python3 script_name.py --input data.csv --output result.csv --dry-run

依赖:
    - pandas
    - numpy

作者: Green
创建日期: 2026-01-10
最后更新: 2026-01-10
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_args() -> argparse.ArgumentParser:
    """
    配置命令行参数

    Returns:
        配置好的 ArgumentParser 对象
    """
    parser = argparse.ArgumentParser(
        description='脚本功能描述',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python3 script_name.py --input data.csv --output result.csv

    # Dry Run 模式
    python3 script_name.py --input data.csv --output result.csv --dry-run --limit 10

    # 详细输出
    python3 script_name.py --input data.csv --output result.csv --verbose
        """
    )

    # 必需参数
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入文件路径'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出文件路径'
    )

    # 可选参数
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry Run 模式，仅处理部分数据'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Dry Run 模式下处理的行数（默认: 10）'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )

    return parser


def validate_input(input_path: str) -> bool:
    """
    验证输入文件

    Args:
        input_path: 输入文件路径

    Returns:
        验证是否通过
    """
    path = Path(input_path)

    if not path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return False

    if not path.is_file():
        logger.error(f"输入路径不是文件: {input_path}")
        return False

    if path.stat().st_size == 0:
        logger.error(f"输入文件为空: {input_path}")
        return False

    logger.info(f"✅ 输入文件验证通过: {input_path}")
    return True


def process_data(
    input_path: str,
    output_path: str,
    dry_run: bool = False,
    limit: int = 10
) -> bool:
    """
    处理数据的主函数

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        dry_run: 是否为 Dry Run 模式
        limit: Dry Run 模式下处理的行数

    Returns:
        处理是否成功
    """
    import pandas as pd

    try:
        # 读取数据
        logger.info(f"读取输入文件: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"共读取 {len(df)} 行数据")

        # Dry Run 模式：仅处理部分数据
        if dry_run:
            logger.info(f"⚠️ Dry Run 模式：仅处理前 {limit} 行")
            df = df.head(limit)

        # 数据处理逻辑
        logger.info("开始处理数据...")
        # TODO: 在这里添加实际的处理逻辑
        result_df = df.copy()

        # 保存结果
        logger.info(f"保存结果到: {output_path}")
        result_df.to_csv(output_path, index=False)
        logger.info(f"✅ 成功保存 {len(result_df)} 行数据")

        # 输出统计信息
        logger.info(f"\n统计信息:")
        logger.info(f"  输入行数: {len(df)}")
        logger.info(f"  输出行数: {len(result_df)}")
        logger.info(f"  列数: {len(result_df.columns)}")

        return True

    except Exception as e:
        logger.error(f"❌ 处理数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """主函数"""
    # 解析参数
    parser = setup_args()
    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 显示配置
    logger.info("=" * 60)
    logger.info("脚本配置:")
    logger.info(f"  输入文件: {args.input}")
    logger.info(f"  输出文件: {args.output}")
    logger.info(f"  Dry Run: {args.dry_run}")
    if args.dry_run:
        logger.info(f"  处理行数: {args.limit}")
    logger.info("=" * 60)

    # 验证输入
    if not validate_input(args.input):
        sys.exit(1)

    # 处理数据
    success = process_data(
        input_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
        limit=args.limit
    )

    # 退出
    if success:
        logger.info("✅ 脚本执行成功")
        sys.exit(0)
    else:
        logger.error("❌ 脚本执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 🧪 测试规范

### 测试文件结构

```python
#!/usr/bin/env python3
"""
测试脚本名称

作者: Green
创建日期: 2026-01-10
"""

import unittest
import tempfile
import pandas as pd
from pathlib import Path
import sys

# 导入被测试的模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from script_name import process_data, validate_input


class TestScriptName(unittest.TestCase):
    """测试 script_name.py 的功能"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })

    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_basic_processing(self):
        """测试基本功能"""
        input_path = Path(self.temp_dir) / "input.csv"
        output_path = Path(self.temp_dir) / "output.csv"

        # 创建测试数据
        self.sample_data.to_csv(input_path, index=False)

        # 运行处理
        success = process_data(
            input_path=str(input_path),
            output_path=str(output_path),
            dry_run=False
        )

        # 验证结果
        self.assertTrue(success)
        self.assertTrue(output_path.exists())

        result_df = pd.read_csv(output_path)
        self.assertEqual(len(result_df), 5)

    def test_dry_run_mode(self):
        """测试 Dry Run 模式"""
        input_path = Path(self.temp_dir) / "input.csv"
        output_path = Path(self.temp_dir) / "output.csv"

        # 创建测试数据
        self.sample_data.to_csv(input_path, index=False)

        # 运行 Dry Run
        success = process_data(
            input_path=str(input_path),
            output_path=str(output_path),
            dry_run=True,
            limit=3
        )

        # 验证结果
        self.assertTrue(success)
        result_df = pd.read_csv(output_path)
        self.assertEqual(len(result_df), 3)

    def test_empty_input(self):
        """测试空输入"""
        input_path = Path(self.temp_dir) / "empty.csv"

        # 创建空文件
        pd.DataFrame().to_csv(input_path, index=False)

        # 验证应该失败
        is_valid = validate_input(str(input_path))
        self.assertFalse(is_valid)

    def test_nonexistent_input(self):
        """测试不存在的输入文件"""
        is_valid = validate_input("/nonexistent/file.csv")
        self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()
```

### 运行测试

```bash
# 运行单个测试文件
python3 tests/test_script_name.py

# 运行所有测试（使用 pytest）
python3 -m pytest tests/ -v

# 运行测试并显示覆盖率
python3 -m pytest tests/ --cov=. --cov-report=html
```

---

## 📋 命令行参数规范

### 必需参数

所有数据处理脚本必须支持以下参数：

| 参数 | 短选项 | 类型 | 说明 |
|------|--------|------|------|
| `--input` | `-i` | str | 输入文件路径 |
| `--output` | `-o` | str | 输出文件路径 |

### 推荐参数

强烈建议支持以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dry-run` | flag | False | Dry Run 模式 |
| `--limit` | int | 10 | Dry Run 行数限制 |
| `--verbose` / `-v` | flag | False | 详细输出 |
| `--help` / `-h` | flag | - | 显示帮助信息 |

### 可选参数

根据脚本功能添加：

```python
parser.add_argument(
    '--backup',
    action='store_true',
    help='处理前备份原文件'
)

parser.add_argument(
    '--force',
    action='store_true',
    help='覆盖已存在的输出文件'
)

parser.add_argument(
    '--filter',
    type=str,
    help='过滤条件（如 "status==completed"）'
)
```

---

## 📊 日志规范

### 日志级别使用

```python
# DEBUG: 调试信息
logger.debug(f"变量值: {variable}")

# INFO: 正常流程信息
logger.info("开始处理数据...")
logger.info(f"✅ 处理完成，共 {count} 条记录")

# WARNING: 警告信息
logger.warning(f"⚠️ 发现 {missing_count} 条缺失数据")

# ERROR: 错误信息
logger.error(f"❌ 处理失败: {error_message}")

# CRITICAL: 严重错误
logger.critical(f"🔥 系统错误: {critical_error}")
```

### 日志输出示例

```
2026-01-10 14:30:15 - INFO - ============================================================
2026-01-10 14:30:15 - INFO - 脚本配置:
2026-01-10 14:30:15 - INFO -   输入文件: data/raw_data.csv
2026-01-10 14:30:15 - INFO -   输出文件: data/processed_data.csv
2026-01-10 14:30:15 - INFO -   Dry Run: True
2026-01-10 14:30:15 - INFO -   处理行数: 10
2026-01-10 14:30:15 - INFO - ============================================================
2026-01-10 14:30:15 - INFO - ✅ 输入文件验证通过: data/raw_data.csv
2026-01-10 14:30:16 - INFO - 读取输入文件: data/raw_data.csv
2026-01-10 14:30:16 - INFO - 共读取 836 行数据
2026-01-10 14:30:16 - INFO - ⚠️ Dry Run 模式：仅处理前 10 行
2026-01-10 14:30:16 - INFO - 开始处理数据...
2026-01-10 14:30:17 - INFO - 保存结果到: data/processed_data.csv
2026-01-10 14:30:17 - INFO - ✅ 成功保存 10 行数据
2026-01-10 14:30:17 - INFO -
2026-01-10 14:30:17 - INFO - 统计信息:
2026-01-10 14:30:17 - INFO -   输入行数: 10
2026-01-10 14:30:17 - INFO -   输出行数: 10
2026-01-10 14:30:17 - INFO -   列数: 87
2026-01-10 14:30:17 - INFO - ✅ 脚本执行成功
```

---

## 🔒 数据安全规范

### 备份策略

```python
import shutil
from datetime import datetime

def backup_file(file_path: str) -> str:
    """
    备份文件

    Args:
        file_path: 原文件路径

    Returns:
        备份文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"

    shutil.copy2(file_path, backup_path)
    logger.info(f"📦 已备份文件: {backup_path}")

    return backup_path


# 使用示例
if Path(output_path).exists():
    backup_file(output_path)
```

### 错误处理

```python
def safe_process(func):
    """
    安全处理装饰器
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"❌ 文件不存在: {e}")
            return None
        except pd.errors.EmptyDataError as e:
            logger.error(f"❌ 数据文件为空: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 未知错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    return wrapper


@safe_process
def process_data(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)
```

---

## 📚 文档字符串规范

### 模块文档字符串

```python
"""
模块名称

模块的详细描述，说明模块的功能、用途和使用场景。

包含的主要类/函数:
    - function1: 功能描述
    - function2: 功能描述
    - Class1: 类描述

使用示例:
    from module_name import function1
    result = function1(input_data)

作者: Green
创建日期: 2026-01-10
最后更新: 2026-01-10
"""
```

### 函数文档字符串

```python
def function_name(
    param1: str,
    param2: int,
    param3: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    函数的简要描述（一行）

    函数的详细描述，说明功能、用途和注意事项。

    Args:
        param1: 参数1的描述
        param2: 参数2的描述
        param3: 参数3的描述（可选）

    Returns:
        返回值的描述

    Raises:
        ValueError: 在什么情况下抛出此异常
        FileNotFoundError: 在什么情况下抛出此异常

    Example:
        >>> result = function_name("test", 10)
        >>> print(result)
        {'status': 'success', 'count': 10}
    """
    pass
```

---

## 🎯 代码质量检查清单

提交代码前，确认以下项目：

### 必需项 ⭐⭐⭐
- [ ] 包含文档字符串（模块、类、函数）
- [ ] 支持 `--dry-run` 参数
- [ ] 支持 `--help` 参数
- [ ] 包含日志输出
- [ ] 包含错误处理
- [ ] 编写了单元测试
- [ ] 测试通过

### 推荐项 ⭐⭐
- [ ] 支持 `--verbose` 参数
- [ ] 包含输入验证
- [ ] 包含数据备份功能
- [ ] 使用类型注解
- [ ] 代码符合 PEP 8 规范
- [ ] 包含使用示例

### 可选项 ⭐
- [ ] 支持进度条显示
- [ ] 支持多种输出格式
- [ ] 包含性能优化
- [ ] 支持并行处理

---

## 📝 脚本命名规范

### 文件命名

```
动词_名词_[修饰词].py

示例:
✅ analyze_experiment_status.py
✅ validate_raw_data.py
✅ append_session_to_raw_data.py
✅ create_unified_data_csv.py

❌ script.py
❌ temp123.py
❌ new_script_final_v2.py
```

### 测试文件命名

```
test_[脚本名称].py

示例:
✅ test_analyze_experiment_status.py
✅ test_validate_raw_data.py
```

---

## 🔄 脚本复用检查

**在创建新脚本前，必须先检查是否有可复用的现有脚本！**

详细指南: [脚本复用检查指南](../CLAUDE.md#-脚本复用检查指南)

---

## 📚 相关文档

- [开发工作流程](DEVELOPMENT_WORKFLOW.md) - 整体开发流程
- [独立验证规范](INDEPENDENT_VALIDATION_GUIDE.md) - 验证指南
- [脚本快速参考](SCRIPTS_QUICKREF.md) - 现有脚本列表

---

## 🔄 版本历史

### v1.0.0 (2026-01-10)
- 初始版本
- 从 CLAUDE.md 拆分独立
- 添加完整代码模板和示例

---

**维护者**: Green
**文档类型**: 开发规范
**强制执行**: 是 ⭐⭐⭐
