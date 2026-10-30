# 中美 AGI 竞争与 AI 金融泡沫

> 数据分析与可视化课程项目 · 小组仓库
>
> 用可复现的公开数据，量化**中美算力竞争**与**AI 资本泡沫**两条线索，并检验二者之间的关联。

---

## 一、研究问题

本项目不做"泡沫何时破裂"的预测，只回答三个**可证伪**的问题：

1. **竞争形态**：中美在前沿算力上的差距在扩大还是收窄？竞争主要发生在"峰值"还是"规模扩散"上？
2. **成本效率**：单位算力价格在怎样变化？同样的资本投入换到的算力是变多还是变少？
3. **估值背离**：AI 相关的资本开支与基础设施承诺，相对于可观测的收入/盈利是否存在缺口？缺口有多大？

## 二、分析框架

```
算力竞赛  →  资本开支  →  估值  →  融资结构  →  风险
   ↑            ↑          ↑         ↑           ↑
芯片销量    云厂商 capex  EV/Sales   债务/信贷   与 2000 年
数据中心    数据中心成本   P/E     私募信贷     dot-com 对照
```

每一环都对应到具体数据集，避免"两张不相关的图硬拼在一起"。

## 三、目录结构

```
datasets/
├── chip_sales/          # AI 芯片出货与算力供给（Epoch AI）
├── data_centers/        # 大型 AI 数据中心站点与建设进度（Epoch AI）
└── models/              # 中美 notable 模型训练算力（Epoch AI）
```

## 四、数据清单

| 文件 | 规模 | 覆盖范围 | 关键字段 | 用途 |
|---|---|---|---|---|
| `chip_sales/timelines_by_chip.csv` | 171 × 24 | 2022Q1 – 2026Q3 | 数量、H100e、成本、TDP，均含 5/95 分位 | 算力供给的季度批次主表 |
| `chip_sales/cumulative_timelines.csv` | 265 × 19 | 同上 | 按芯片型号的累计值 | 单型号累计算力曲线 |
| `chip_sales/cumulative_timelines_by_designer.csv` | 75 × 15 | 同上 | 按设计商累计的 H100e 与 Power (MW) | 厂商/国别累计算力曲线 |
| `chip_sales/chip_types.csv` | 32 × 14 | 2018 – 2026 | 8-bit OP/s、H100e、单价、`Primarily for Chinese market` | 芯片代际效率与对华特供识别 |
| `chip_sales/organizations.csv` | 12 | — | 设计商国别与角色 | 与行情/财报表关联 |
| `data_centers/data_centers.csv` | 95 × 16 | 当前快照 | H100e、功率 MW、资本成本、业主、用户、国别 | 站点级横截面 |
| `data_centers/data_center_timelines.csv` | 567 × 14 | 2018 – 2030 | 建设状态、IT 功率、成本、水耗 | **产能管线倍数（核心图）** |
| `data_centers/data_center_chip_quantities.csv` | 239 × 11 | — | 站点内芯片型号与数量 | 芯片去向追踪 |
| `models/epoch_us_cn_subset.csv` | 642 × 8 | 1950 – 2026 | 训练算力 FLOP、机构、国别、开源与否 | 中美模型算力长波图 |

> 说明：`data_centers` 原始包中的 `data_center_chillers.csv`、`data_center_cooling_towers.csv` 为 Epoch 的估算方法论附件（冷却设备规格表），与本项目主题无关，未纳入仓库。

## 五、数据来源与许可

全部数据来自 **Epoch AI**，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可——可自由使用、分发、再发布，需署名。

| 数据集 | 页面 | 下载 |
|---|---|---|
| AI chip sales | https://epoch.ai/data/ai-chip-sales | `/data/ai_chip_sales.zip` |
| AI data centers | https://epoch.ai/data/ai-data-centers | `/data/data_centers/data_centers.zip` |
| AI models | https://epoch.ai/data/ai-models | `/data/notable_ai_models.csv`、`/data/frontier_ai_models.csv` |

**数据冻结日：2026-10-21。** 上述数据集仍在频繁更新（芯片销量表每日更新、模型表每周更新），历史值也会被回溯修正。重新下载时请更新本节日期，并在图表中注明"数据截至 X 日"，否则不同成员产出的图会对不上。

## 六、已知口径与陷阱

使用前务必阅读，这些是答辩时最容易被追问的点。

### 模型表（`models/`）

- `Frontier_model` 字段只有 `True` 和空值，**没有 `False`**，且 74 条标记中仅 2 条为中国模型 → 不可直接用于国别对比，须按年度自算最大值。
- `Open_weights` 存在 `Yes` / `No` / 空 三种取值，**空值表示未知而非闭源**。
- `Domain` 为逗号分隔多值（57 种组合），需拆分后使用。
- `Country` 指**研发地**而非母公司总部（如 Microsoft Research Asia 记为 CN）。
- 2025 年美国条目数较 2024 明显下降，疑为收录滞后，**作图前须截断至数据完备月份**。

### 芯片表（`chip_sales/`）

- `Cost Estimate (USD)` 为芯片采购价，**不含服务器、系统、机房成本**，不可与云厂商 capex 直接画等号，只能比较趋势同步性。
- `Incomplete` 标记的 8 条记录覆盖时段不完整，聚合时须剔除或注明。
- `Cost per chip (approx.)` 中 Google TPU、Amazon Trainium 为内部成本估算，**非市场可购价格**，横比时须与商用芯片区分。

### 数据中心表（`data_centers/`）

- `Current total capital cost` 由功率派生：经核验 `cost ÷ power` 恒为 **0.0379 $B/MW**（95 行中唯一值）。**不可将其与功率做回归或相关性分析**，属循环论证。
- `Construction status` 为自由文本（含长段卫星影像描述与链接），不可直接分组，须人工归纳状态标签。
- `data_center_timelines` 中 98 行日期晚于冻结日，**属预测而非观测**，作图须标注为 projection。
- `Owner` / `Users` 字段带置信度后缀（如 `Nvidia #confident`、`ByteDance #likely`），分组前须剥离，否则同一实体被拆成多类。
- 官方自述覆盖率为 **44%**，美国站点 77/95，中国仅 5 站 → **任何国别结论都必须限定为"在本样本中"**。

### 其他

- 仓库当前仅有数据，暂无清洗脚本。原始 zip 包不入库，仅保留解压后的 CSV。

## 七、协作约定

- **分支**：结构性改动（移动文件、改目录、改字段口径）走 `chore/…`、`feat/…` 分支并发 PR；仅新增单个数据文件可直接提交 main。
- **提交信息**：`chore(data): …`、`feat(fig): …`、`docs: …`，说明"为什么"而非"做了什么"。
- **数据冻结**：任何重新下载都必须同步更新第五节日期，并在该次提交信息中注明。
- **受限数据隔离**：Wind / Choice / Crunchbase / Statista 等订阅数据**不得提交**，统一放 `data/external/`（该目录已加入 `.gitignore`，仅保留 `.gitkeep`）。
- **派生文件**：由脚本生成的表放 `data/processed/`，不手工维护第二份真相。

## 八、待补充数据

**竞争侧**：Stanford AI Index（中美投资/论文/人才）、中国信通院算力白皮书、BIS 出口管制事件表（用于曲线上标断点）。

**金融侧**：云厂商 capex（SEC EDGAR XBRL）、AI 相关上市公司行情与估值（yfinance / akshare）、一级市场融资（AI Index 汇总）、AI 相关债务与信用利差（FRED）、数据中心用电与并网（IEA / LBNL）、叙事热度（SEC 10-K "AI" 词频、GDELT）。

## 九、首轮探查结论（待复核）

以下数字由 2026-10-21 版本数据算出，供论证方向参考，正式分析请以脚本产出为准：

- 单位算力价格（$/H100e）从 2022 年约 3.57 万美元降至 2026 年约 1.31 万美元，四年下降约 63%。
- 数据中心现有 14.2 GW，已宣布的最终建成规模 38.2 GW，**管线倍数为 2.69×**。
- 数据中心总资本成本约 $36,473/H100e，芯片本身约 $25,000/H100e，非芯片（土建、供电、冷却）约占三成。
- 对华特供芯片存在显著单位算力溢价：`H20` 约 $80,230/H100e，为 `H100`（$25,000）的 3.2 倍；国产 `Ascend 910C` 已降至约 $26,316，接近持平。

## 十、引用

```
Epoch AI, "AI chip sales". Published online at epoch.ai.
Retrieved from https://epoch.ai/data/ai-chip-sales [online resource].

Epoch AI, "AI data centers". Published online at epoch.ai.
Retrieved from https://epoch.ai/data/ai-data-centers [online resource].

Epoch AI, "AI models". Published online at epoch.ai.
Retrieved from https://epoch.ai/data/ai-models [online resource].
```
