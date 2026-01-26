# 项目文档索引

**最后更新**: 2026-01-26
**项目状态**: 生产就绪

---

## 🚀 快速开始

### 新手入门
1. **5分钟指南** → `CLAUDE.md`
2. **快速健康检查** → `bash tools/quick_health_check.sh`
3. **快速参考（数据补完）** → `PROMPT_QUICK_REFERENCE.md`

---

## 📚 核心文档

### 项目概述
| 文档 | 说明 | 优先级 |
|------|------|--------|
| [README.md](README.md) | 项目总览 | ⭐⭐⭐ |
| [CLAUDE.md](CLAUDE.md) | 5分钟快速指南 | ⭐⭐⭐ |
| [CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) | 完整参考文档 | ⭐⭐ |

### 数据相关
| 文档 | 说明 | 优先级 |
|------|------|--------|
| [DATA_USAGE_GUIDE.md](docs/DATA_USAGE_GUIDE.md) | 数据使用指南（必读） | ⭐⭐⭐ |
| [DATA_REPAIR_REPORT_20260104.md](docs/results_reports/DATA_REPAIR_REPORT_20260104.md) | 数据修复报告 | ⭐⭐ |
| [PROMPT_FOR_DATA_COMPLETION.md](docs/PROMPT_FOR_DATA_COMPLETION.md) | 数据补完详细任务 | ⭐⭐⭐ |
| [PROMPT_QUICK_REFERENCE.md](docs/PROMPT_QUICK_REFERENCE.md) | 数据补完快速参考 | ⭐⭐⭐ |

### ATE集成（2026-01-26完成）
| 文档 | 说明 | 优先级 |
|------|------|--------|
| [ATE_INTEGRATION_COMPLETION_REPORT_20260126.md](docs/current_plans/ATE_INTEGRATION_COMPLETION_REPORT_20260126.md) | 实施完成报告 | ⭐⭐⭐ |
| [CTF_STYLE_ATE_QUICK_START_20260126.md](docs/current_plans/CTF_STYLE_ATE_QUICK_START_20260126.md) | 快速使用指南 | ⭐⭐⭐ |
| [ATE_PROJECT_STATUS_20260126.md](docs/current_plans/ATE_PROJECT_STATUS_20260126.md) | 项目状态 | ⭐⭐ |
| [ATE_INTEGRATION_IMPLEMENTATION_PLAN_20260125.md](docs/current_plans/ATE_INTEGRATION_IMPLEMENTATION_PLAN_20260125.md) | 实施方案 | ⭐ |
| [ATE_INTEGRATION_FEASIBILITY_REPORT_20260125.md](docs/current_plans/ATE_INTEGRATION_FEASIBILITY_REPORT_20260125.md) | 可行性报告 | ⭐ |

### 开发工作流
| 文档 | 说明 | 优先级 |
|------|------|--------|
| [DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) | 开发工作流规范 | ⭐⭐ |

---

## 🛠️ 工具脚本

### 数据管理（`tools/data_management/`）
| 脚本 | 用途 | 使用频率 |
|------|------|----------|
| `validate_raw_data.py` | 验证数据完整性 | 高 |
| `analyze_missing_energy_data.py` | 分析缺失数据 | 中 |
| `repair_missing_energy_data.py` | 修复缺失数据 | 中 |
| `append_session_to_raw_data.py` | 追加新实验数据 | 中 |
| `create_unified_data_csv.py` | 创建统一数据文件 | 低 |

### 因果推断（`analysis/`）
| 文件 | 说明 | 状态 |
|------|------|------|
| `utils/causal_inference.py` | 因果推断引擎 | ✅ 完成 |
| `tests/test_ctf_style_ate.py` | ATE功能测试 | ✅ 通过 |

---

## 📊 项目数据

### 数据文件
| 文件 | 大小 | 记录数 | 完整性 |
|------|------|--------|--------|
| `data/raw_data.csv` | 538KB | 970行 | 95.1% |
| `data/data.csv` | 389KB | 971行 | 95.1% |

### 数据可用性
- **完全可用**: 577条 (59.5%)
- **仅有能耗**: 251条 (25.9%)
- **缺失数据**: 141条 (14.6%)

---

## 🎯 当前任务

### 优先级1: 数据补完
**目标**: 将数据完整性提升到98%+

**文档**: [PROMPT_FOR_DATA_COMPLETION.md](docs/PROMPT_FOR_DATA_COMPLETION.md)

**快速参考**: [PROMPT_QUICK_REFERENCE.md](docs/PROMPT_QUICK_REFERENCE.md)

**预计时间**: 4-6小时

### 优先级2: 因果分析
**状态**: 工具已就绪，等待数据补完

**文档**: [CTF_STYLE_ATE_QUICK_START_20260126.md](docs/current_plans/CTF_STYLE_ATE_QUICK_START_20260126.md)

**预计时间**: 2-4小时

---

## 📖 按主题查找

### 数据问题？
1. 查看 [DATA_USAGE_GUIDE.md](docs/DATA_USAGE_GUIDE.md)
2. 运行 `python3 tools/data_management/validate_raw_data.py`
3. 参考 [PROMPT_FOR_DATA_COMPLETION.md](docs/PROMPT_FOR_DATA_COMPLETION.md)

### 如何使用ATE？
1. 查看 [CTF_STYLE_ATE_QUICK_START_20260126.md](docs/current_plans/CTF_STYLE_ATE_QUICK_START_20260126.md)
2. 参考示例代码
3. 运行测试验证

### 如何开发新功能？
1. 阅读 [DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md)
2. 遵循测试→Dry Run→执行流程
3. 更新相关文档

### 遇到问题？
1. 查看 [CLAUDE_FULL_REFERENCE.md](docs/CLAUDE_FULL_REFERENCE.md) 的常见问题章节
2. 搜索相关文档
3. 查看历史报告

---

## 📝 文档规范

### 文件命名规则
- **指南**: `GUIDE_<主题>.md`
- **报告**: `<主题>_REPORT_<日期>.md`
- **方案**: `<主题>_PLAN_<日期>.md`
- **Prompt**: `PROMPT_FOR_<任务>.md`

### 文件位置规则
| 类型 | 位置 | 示例 |
|------|------|------|
| 项目文档 | `docs/` | `README.md` |
| 数据报告 | `docs/results_reports/` | `DATA_REPAIR_REPORT_*.md` |
| 当前方案 | `docs/current_plans/` | `ATE_*.md` |
| 指南 | `docs/guides/` | `APPEND_*.md` |
| Prompt | `docs/` | `PROMPT_*.md` |

---

## 🔄 文档更新历史

### 2026-01-26
- ✅ 完成ATE集成实施
- ✅ 创建数据补完Prompt
- ✅ 更新项目状态文档

### 2026-01-04
- ✅ 数据完整性修复（69.7% → 95.1%）
- ✅ 创建数据修复报告

---

## 📞 获取帮助

### 快速问题
- 查看本索引
- 查看 `CLAUDE.md`
- 运行 `tools/quick_health_check.sh`

### 复杂问题
- 查看详细参考文档
- 查看历史报告
- 查看代码注释

### 联系维护者
- 项目维护者: Green
- 文档版本: 按文件头部标注

---

**索引版本**: 1.0
**最后更新**: 2026-01-26
**维护者**: Claude Code (Sonnet 4.5)
