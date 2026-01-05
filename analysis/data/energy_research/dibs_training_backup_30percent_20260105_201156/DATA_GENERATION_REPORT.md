# 6任务组DiBS训练数据生成报告

**生成时间**: 2026-01-05 19:55:53
**数据源**: /home/green/energy_dl/nightly/analysis/data/energy_research/raw/energy_data_original.csv
**总样本数**: 836行
**成功率**: 6/6 (100%)

## 任务组详情

| 任务组 | 样本数 | 特征数 | 保留率 | 缺失率 | 状态 |
|--------|--------|--------|--------|--------|------|
| examples（图像分类-小型） | 259 | 17 | 41.5% | 5.61% | ✓ |
| VulBERTa（代码漏洞检测） | 152 | 13 | 31.7% | 13.16% | ✓ |
| Person_reID（行人重识别） | 146 | 20 | 48.8% | 4.18% | ✓ |
| bug-localization（缺陷定位） | 142 | 12 | 29.3% | 3.23% | ✓ |
| MRT-OAST（缺陷定位） | 88 | 15 | 36.6% | 7.12% | ✓ |
| pytorch_resnet（图像分类-ResNet） | 49 | 15 | 36.6% | 1.36% | ✓ |

## 数据处理流程

1. 按repository过滤任务组数据
2. 选择数值型列
3. 移除全NaN列
4. 移除缺失率>30.0%的列
5. 移除零方差列（常数列）
6. 填充缺失值（均值填充）
7. 标准化（Z-score）

## 输出文件

- `group1_examples.csv` - 259行 × 17列
- `group2_vulberta.csv` - 152行 × 13列
- `group3_person_reid.csv` - 146行 × 20列
- `group4_bug_localization.csv` - 142行 × 12列
- `group5_mrt_oast.csv` - 88行 × 15列
- `group6_resnet.csv` - 49行 × 15列

---

**报告生成**: 2026-01-05 19:55:53
