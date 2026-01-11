# DiBS数据需求分析报告

**生成时间**: 2026-01-11

## 数据概况

- 总实验数: 970
- 能耗数据完整: 839 (86.5%)
- 性能数据完整: 737 (76.0%)
- 能耗+性能都完整: 717 (73.9%)

## 12分组方案样本量

| 任务组 | 场景 | 能耗+性能都完整 | 建议补充 |
|--------|------|----------------|----------|
| group1a_examples | 非并行 | 126 | 0 |
| group1a_examples | 并行 | 119 | 0 |
| group1b_resnet | 非并行 | 31 | 69 |
| group1b_resnet | 并行 | 30 | 70 |
| group2_person_reid | 非并行 | 93 | 7 |
| group2_person_reid | 并行 | 90 | 10 |
| group3_vulberta | 非并行 | 25 | 75 |
| group3_vulberta | 并行 | 47 | 53 |
| group4_bug_localization | 非并行 | 25 | 75 |
| group4_bug_localization | 并行 | 65 | 35 |
| group5_mrt_oast | 非并行 | 36 | 64 |
| group5_mrt_oast | 并行 | 30 | 70 |

## 总体建议

- 当前可用: 717 个实验
- 推荐目标: 1200 个实验（100/组）
- 需要补充: 约 483 个实验
