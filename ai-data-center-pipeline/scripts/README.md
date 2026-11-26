# 数据中心方向：脚本与产物

负责方向：**AI 数据中心产能管线与实际兑现**。分析文字见 [`../README.md`](../README.md)。

## 运行顺序

```powershell
python 02_metrics.py      # 清洗 + 指标 -> ../results/*.csv
python 03_charts.py       # 图 01/02/03 -> ../figures/（PNG + SVG）
python 04_stages.py       # 状态分类 + 图 04/05
python 06_audit.py        # 审计 + 敏感性 + 人工判读合并 -> ../results/

python 01_inspect.py <csv>          # 可选：表结构体检
python 05_trace.py "Colossus 2"     # 可选：逐站点追踪
```

依赖：pandas / numpy / matplotlib（Python 3.7 + pandas 1.1 实测通过）。

## 产物位置（只写本目录，不动仓库其它位置）

| 脚本 | 写到哪里 |
|---|---|
| `01_inspect.py` | `../results/reports/_schema_report.txt` |
| `02_metrics.py` | `../results/site_level.csv`、`annual.csv`、`summary.json` |
| `03_charts.py` | `../figures/01-gap-cumulative.*`、`02-site-realization.*`、`03-lag-distribution.*` |
| `04_stages.py` | `../results/site_stage.csv`、`stage_counts.csv`、`group_compare.csv`；`../figures/04-site-timeline.*`、`05-owner-split.*` |
| `06_audit.py` | `../results/audit.json`、`row_audit.csv`、`state_audit.csv`、`stage_counts_refined.csv`、`definition_sensitivity.csv`、`cutoff_sensitivity.csv`、`caliber_comparison.csv` |

路径全部由 `_paths.py` 决定：输入默认读 `../data/`（自包含副本），可用 `DCP_BASE` 覆盖；
出图统一走 `_paths.save(fig, "04")`，自动同时导出 PNG 与 SVG。

## 三条硬规则（改脚本前先读）

1. **`Date > 2026-10-21` 的 98 行是 projection**，不是观测。任何容量汇总都必须区分 `is_projection`。
2. **时间线每行是累计值**，不能逐年相加；年度新增要靠"年末快照差分 + 站点间结转"。
3. **功率口径与算力口径分开报**：IT 功率管线倍数 2.689×，算力口径 4.552×，差异来自承诺隐含的芯片密度假设（809 → 1344 H100e/MW）。
