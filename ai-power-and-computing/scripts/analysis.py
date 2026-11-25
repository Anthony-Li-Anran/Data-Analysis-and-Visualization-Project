import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 csv（row_audit 和其他文件放同一个文件夹）
df_main = pd.read_csv("data_centers.csv")
df_chip = pd.read_csv("data_center_chip_quantities.csv")
df_time = pd.read_csv("data_center_timelines.csv")
df_audit = pd.read_csv("row_audit.csv")

print("==== data_centers（主候选表） ====")
print("行数,列数：", df_main.shape)
print("列名：")
for col in df_main.columns:
    print(col)

print("\n==== data_center_chip_quantities（芯片数量表） ====")
print("行数,列数：", df_chip.shape)
print("列名：")
for col in df_chip.columns:
    print(col)

print("\n==== data_center_timelines（时间线表） ====")
print("行数,列数：", df_time.shape)
print("列名：")
for col in df_time.columns:
    print(col)

# ========== 冻结日切分 ==========
df_merge = pd.concat([df_time, df_audit[["is_projection"]]], axis=1)
df_actual = df_merge[df_merge["is_projection"] == False].copy()

print("\n==== 冻结日切分处理结果 ====")
print(f"原始时间线总行数：{len(df_merge)}")
print(f"剔除 projection 行数：{df_merge['is_projection'].sum()}")
print(f"保留真实观测行数：{len(df_actual)}")


# 探查：查看缺失、分布
core_cols = ["H100 equivalents", "Power (MW)", "IT power (MW)"]
print("====【原始全部样本（含projection）】缺失统计====")
print(df_time[core_cols].isna().sum())
print("\n====【原始全部样本（含projection）】描述统计====")
print(df_time[core_cols].describe())

print("\n" + "="*60)
print("====【冻结日后过滤：仅真实观测 df_actual】缺失统计====")
print(df_actual[core_cols].isna().sum())
print("\n====【冻结日后过滤：仅真实观测 df_actual】描述统计====")
print(df_actual[core_cols].describe())

# （含projection，保留用于对比，后续分析不再用这组）
df_full = df_time.dropna(subset=["H100 equivalents", "Power (MW)", "IT power (MW)"])

# 过滤：算力、总功率、IT功率必须>0；避免除法报错
df_full = df_full[(df_full["H100 equivalents"] > 0)
                  & (df_full["Power (MW)"] > 0)
                  & (df_full["IT power (MW)"] > 0)]

# 派生指标
df_full["eff_total"] = df_full["H100 equivalents"] / df_full["Power (MW)"]    #总功率口径能源效率
df_full["eff_it"] = df_full["H100 equivalents"] / df_full["IT power (MW)"]   #IT硬件口径能源效率
df_full["pue"] = df_full["Power (MW)"] / df_full["IT power (MW)"]            #PUE电源使用效率

# 仅真实观测 df_actual（正式后续分析使用）
df_actual_full = df_actual.dropna(subset=["H100 equivalents", "Power (MW)", "IT power (MW)"])

df_actual_full = df_actual_full[(df_actual_full["H100 equivalents"] > 0)
                                & (df_actual_full["Power (MW)"] > 0)
                                & (df_actual_full["IT power (MW)"] > 0)]

df_actual_full["eff_total"] = df_actual_full["H100 equivalents"] / df_actual_full["Power (MW)"]
df_actual_full["eff_it"] = df_actual_full["H100 equivalents"] / df_actual_full["IT power (MW)"]
df_actual_full["pue"] = df_actual_full["Power (MW)"] / df_actual_full["IT power (MW)"]


# 输出样本规模对比
print(f"原始总记录数：{len(df_time)}")
print(f"【含projection】拓展分析可用记录数 df_full：{len(df_full)}")
print(f"【仅真实观测】拓展分析可用记录数 df_actual_full：{len(df_actual_full)}")

#预览数据
preview_cols = ["H100 equivalents", "Power (MW)", "IT power (MW)",
                "eff_total", "eff_it", "pue"]
print("\n=====仅真实观测样本预览 df_actual_full =====")
print(df_actual_full[preview_cols].head())

# 导出带派生指标的 timeline 清洗结果
df_full.to_csv("timeline_cleaned.csv", index=False, encoding="utf-8-sig")

df_actual_full.to_csv("timeline_cleaned_actual.csv", index=False, encoding="utf-8-sig")

print("已导出：timeline_cleaned_actual.csv")
print(f"行数：{len(df_actual_full)}，仅含冻结日前真实观测样本")

# 精简清洗数据
df_out = df_full[["Data center", "H100 equivalents", "Power (MW)", "IT power (MW)", "eff_total", "eff_it", "pue"]]
df_out.to_csv("timeline_slim.csv", index=False, encoding="utf-8-sig")

# 精简清洗数据（仅真实观测）
df_out = df_actual_full[["Data center", "H100 equivalents", "Power (MW)", "IT power (MW)", "eff_total", "eff_it", "pue"]]
df_out.to_csv("timeline_slim_actual.csv", index=False, encoding="utf-8-sig")

print(f"精简表导出成功，文件名：timeline_slim_actual.csv，行数：{len(df_out)}")

# 原始全部样本（含 projection，留作对比）】主线数据集：宽松筛选，不用 IT power
df_clean = df_time.dropna(subset=["H100 equivalents", "Power (MW)"])
df_clean = df_clean[(df_clean["H100 equivalents"] > 0) & (df_clean["Power (MW)"] > 0)]
df_clean["eff_total"] = df_clean["H100 equivalents"] / df_clean["Power (MW)"]

print(f"【含 projection】主线样本数量：{len(df_clean)}")
print("\n====【含 projection】主线数据描述统计====")
print(df_clean[["H100 equivalents", "Power (MW)", "eff_total"]].describe())

# 真实观测 主线数据集：宽松筛选，不用 IT power
df_clean_actual = df_actual.dropna(subset=["H100 equivalents", "Power (MW)"])
df_clean_actual = df_clean_actual[(df_clean_actual["H100 equivalents"] > 0) & (df_clean_actual["Power (MW)"] > 0)]
df_clean_actual["eff_total"] = df_clean_actual["H100 equivalents"] / df_clean_actual["Power (MW)"]

print("\n" + "="*60)
print(f"【仅真实观测】主线样本数量 df_clean_actual：{len(df_clean_actual)}")
print("\n====【仅真实观测】主线数据描述统计====")
print(df_clean_actual[["H100 equivalents", "Power (MW)", "eff_total"]].describe())

import matplotlib.pyplot as plt

# 含 projection
plt.figure(figsize=(8,5))
plt.scatter(df_clean["H100 equivalents"], df_clean["Power (MW)"], alpha=0.6)
plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("Power (MW)（总功率 MW）")
plt.title("【对照｜含预测行】等效 H100 总算力 — 数据中心总功率")
plt.grid(alpha=0.3)
plt.savefig("scatter_with_proj.png", dpi=300, bbox_inches="tight")
plt.show()

r_pearson_old = df_clean["H100 equivalents"].corr(df_clean["Power (MW)"], method="pearson")
print(f"【含 projection】皮尔逊相关系数：{r_pearson_old:.3f}")


# 真实观测
plt.figure(figsize=(8,5))
plt.scatter(df_clean_actual["H100 equivalents"], df_clean_actual["Power (MW)"], alpha=0.6)
plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("Power (MW)（总功率 MW）")
plt.title("【冻结日过滤｜仅真实观测】等效 H100 总算力 — 数据中心总功率")
plt.grid(alpha=0.3)
plt.savefig("scatter_actual_only.png", dpi=300, bbox_inches="tight")
plt.show()

r_pearson_new = df_clean_actual["H100 equivalents"].corr(df_clean_actual["Power (MW)"], method="pearson")
print(f"【仅真实观测】皮尔逊相关系数：{r_pearson_new:.3f}")

import matplotlib.pyplot as plt

# 含 projection
plt.figure(figsize=(8,5))
plt.scatter(df_clean["H100 equivalents"], df_clean["Power (MW)"], alpha=0.6)
plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("Power (MW)（总功率 MW）")
plt.title("【对照｜含预测行】等效 H100 总算力 — 数据中心总功率")
plt.grid(alpha=0.3)
plt.savefig("scatter_power_with_proj.png", dpi=300, bbox_inches="tight")
plt.show()

r_pearson_old = df_clean["H100 equivalents"].corr(df_clean["Power (MW)"], method="pearson")
print(f"【含 projection】皮尔逊相关系数：{r_pearson_old:.3f}")

# 冻结日过滤，仅真实观测
plt.figure(figsize=(8,5))
plt.scatter(df_clean_actual["H100 equivalents"], df_clean_actual["Power (MW)"], alpha=0.6)
plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("Power (MW)（总功率 MW）")
plt.title("【冻结日过滤｜仅真实观测】等效 H100 总算力 — 数据中心总功率")
plt.grid(alpha=0.3)
plt.savefig("scatter_power_actual.png", dpi=300, bbox_inches="tight")
plt.show()

r_pearson_new = df_clean_actual["H100 equivalents"].corr(df_clean_actual["Power (MW)"], method="pearson")
print(f"【仅真实观测】皮尔逊相关系数：{r_pearson_new:.3f}")

import matplotlib.pyplot as plt

# ==========【对照｜含 projection】==========
plt.figure(figsize=(8, 5))
plt.scatter(df_full["H100 equivalents"], df_full["IT power (MW)"], alpha=0.6)
plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("IT power (MW)（IT 硬件功率 MW）")
plt.title("【对照｜含预测行】等效 H100 总算力 — IT 硬件功率")
plt.grid(alpha=0.3)
plt.savefig("scatter_itpower_with_proj.png", dpi=300, bbox_inches="tight")
plt.show()

r_pearson_it_old = df_full["H100 equivalents"].corr(df_full["IT power (MW)"], method="pearson")
print(f"【含 projection】等效算力与 IT 硬件功率皮尔逊相关系数：{r_pearson_it_old:.3f}")

# 真实观测
plt.figure(figsize=(8, 5))
plt.scatter(df_actual_full["H100 equivalents"], df_actual_full["IT power (MW)"], alpha=0.6)
plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("IT power (MW)（IT 硬件功率 MW）")
plt.title("【冻结日过滤｜仅真实观测】等效 H100 总算力 — IT 硬件功率")
plt.grid(alpha=0.3)
plt.savefig("scatter_itpower_actual.png", dpi=300, bbox_inches="tight")
plt.show()

r_pearson_it_new = df_actual_full["H100 equivalents"].corr(df_actual_full["IT power (MW)"], method="pearson")
print(f"【仅真实观测】等效算力与 IT 硬件功率皮尔逊相关系数：{r_pearson_it_new:.3f}")

import matplotlib.pyplot as plt

# ==========【对照｜含 projection】保留不动 ==========
plt.figure(figsize=(8, 5))
plt.scatter(df_full["pue"], df_full["eff_total"], alpha=0.6)
plt.xlabel("pue（电源使用效率）")
plt.ylabel("eff_total（总口径能源效率）")
plt.title("【对照｜含预测行】pue — 总口径能源效率")
plt.grid(alpha=0.3)
plt.savefig("scatter_pue_eff_with_proj.png", dpi=300, bbox_inches="tight")
plt.show()

r_pearson_pue_old = df_full["pue"].corr(df_full["eff_total"], method="pearson")
print(f"【含 projection】pue 与总口径能源效率皮尔逊相关系数：{r_pearson_pue_old:.3f}")

# 真实观测
plt.figure(figsize=(8, 5))
plt.scatter(df_actual_full["pue"], df_actual_full["eff_total"], alpha=0.6)
plt.xlabel("pue（电源使用效率）")
plt.ylabel("eff_total（总口径能源效率）")
plt.title("【冻结日过滤｜仅真实观测】pue — 总口径能源效率")
plt.grid(alpha=0.3)
plt.savefig("scatter_pue_eff_actual.png", dpi=300, bbox_inches="tight")
plt.show()

r_pearson_pue_new = df_actual_full["pue"].corr(df_actual_full["eff_total"], method="pearson")
print(f"【仅真实观测】pue 与总口径能源效率皮尔逊相关系数：{r_pearson_pue_new:.3f}")

# 基于 df_full（含 projection）】
# 新建稳健子集 df_robust_power，原始 df_full 不动
df_robust_power = df_full[
    (df_full["H100 equivalents"] > 0)
    & (df_full["H100 equivalents"] < 4.5e6)
    & (df_full["Power (MW)"] < 1800)
]

# 输出样本量对比
print(f"【对照｜含 projection】剔除前样本量：{len(df_full)}")
print(f"【对照｜含 projection】剔除后（df_robust_power）样本量：{len(df_robust_power)}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.scatter(df_robust_power["H100 equivalents"], df_robust_power["Power (MW)"], alpha=0.6)

plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("Power (MW)（总功率 MW）")
plt.title("【对照｜含预测行｜剔除离群点】等效 H100 总算力 — 总功率")
plt.grid(alpha=0.3)

plt.savefig("scatter_power_robust_with_proj.png", dpi=300, bbox_inches="tight")
plt.show()

r_robust_power_old = df_robust_power["H100 equivalents"].corr(df_robust_power["Power (MW)"], method="pearson")
print(f"【对照｜含 projection】剔除离群点后算力与总功率皮尔逊相关系数：{r_robust_power_old:.3f}")


# 基于 df_actual_full｜冻结日真实观测
df_robust_power_actual = df_actual_full[
    (df_actual_full["H100 equivalents"] > 0)
    & (df_actual_full["H100 equivalents"] < 4.5e6)
    & (df_actual_full["Power (MW)"] < 1800)
]

print("\n" + "="*60)
print(f"【正式｜仅真实观测】剔除前样本量：{len(df_actual_full)}")
print(f"【正式｜仅真实观测】剔除后（df_robust_power_actual）样本量：{len(df_robust_power_actual)}")

plt.figure(figsize=(8, 5))
plt.scatter(df_robust_power_actual["H100 equivalents"], df_robust_power_actual["Power (MW)"], alpha=0.6)

plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("Power (MW)（总功率 MW）")
plt.title("【冻结日过滤｜仅真实观测｜剔除离群点】等效 H100 总算力 — 总功率")
plt.grid(alpha=0.3)

plt.savefig("scatter_power_robust_actual.png", dpi=300, bbox_inches="tight")
plt.show()

r_robust_power_new = df_robust_power_actual["H100 equivalents"].corr(df_robust_power_actual["Power (MW)"], method="pearson")
print(f"【正式｜仅真实观测】剔除离群点后算力与总功率皮尔逊相关系数：{r_robust_power_new:.3f}")

# 基于 df_full（含 projection）
# 重新构造稳健子集，收紧门槛
df_robust_power_new = df_full[
    (df_full["H100 equivalents"] > 0)
    & (df_full["H100 equivalents"] < 3.2e6)
    & (df_full["Power (MW)"] < 1400)
]

print(f"【对照｜含 projection】剔除前：{len(df_full)}")
print(f"【对照｜含 projection】新稳健子集 df_robust_power_new：{len(df_robust_power_new)}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.scatter(df_robust_power_new["H100 equivalents"], df_robust_power_new["Power (MW)"], alpha=0.6)

plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("Power (MW)（总功率 MW）")
plt.title("【对照｜含预测行｜收紧阈值】等效 H100 总算力 — 总功率")
plt.grid(alpha=0.3)

plt.savefig("scatter_power_robust_tight_with_proj.png", dpi=300, bbox_inches="tight")
plt.show()

r_robust_tight_old = df_robust_power_new["H100 equivalents"].corr(df_robust_power_new["Power (MW)"], method="pearson")
print(f"【对照｜含 projection】收紧阈值后算力与总功率皮尔逊相关系数：{r_robust_tight_old:.3f}")

# 基于 df_actual_full｜冻结日真实观测
df_robust_power_tight_actual = df_actual_full[
    (df_actual_full["H100 equivalents"] > 0)
    & (df_actual_full["H100 equivalents"] < 3.2e6)
    & (df_actual_full["Power (MW)"] < 1400)
]

print("\n" + "="*60)
print(f"【正式｜仅真实观测】剔除前：{len(df_actual_full)}")
print(f"【正式｜仅真实观测】收紧阈值稳健子集：{len(df_robust_power_tight_actual)}")

plt.figure(figsize=(8, 5))
plt.scatter(df_robust_power_tight_actual["H100 equivalents"], df_robust_power_tight_actual["Power (MW)"], alpha=0.6)

plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("Power (MW)（总功率 MW）")
plt.title("【冻结日过滤｜仅真实观测｜收紧阈值】等效 H100 总算力 — 总功率")
plt.grid(alpha=0.3)

plt.savefig("scatter_power_robust_tight_actual.png", dpi=300, bbox_inches="tight")
plt.show()

r_robust_tight_new = df_robust_power_tight_actual["H100 equivalents"].corr(df_robust_power_tight_actual["Power (MW)"], method="pearson")
print(f"【正式｜仅真实观测】收紧阈值后算力与总功率皮尔逊相关系数：{r_robust_tight_new:.3f}")

# 基于 df_full（含 projection）
df_robust_it_new = df_full[
    (df_full["H100 equivalents"] > 0)
    & (df_full["H100 equivalents"] < 3.2e6)
    & (df_full["IT power (MW)"] < 1400)
]

# 打印样本数量，方便前后对比
print(f"【对照｜含 projection｜算力—IT 功率】剔除前样本量：{len(df_full)}")
print(f"【对照｜含 projection｜算力—IT 功率】新稳健子集 df_robust_it_new 样本量：{len(df_robust_it_new)}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.scatter(df_robust_it_new["H100 equivalents"], df_robust_it_new["IT power (MW)"], alpha=0.6)
plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("IT power (MW)（IT 硬件功率 MW）")
plt.title("【对照｜含预测行｜收紧阈值】等效 H100 总算力 — IT 硬件功率")
plt.grid(alpha=0.3)
plt.savefig("scatter_it_robust_tight_with_proj.png", dpi=300, bbox_inches="tight")
plt.show()
r_robust_it_old = df_robust_it_new["H100 equivalents"].corr(df_robust_it_new["IT power (MW)"], method="pearson")
print(f"【对照｜含 projection】收紧阈值后算力与 IT 硬件功率皮尔逊相关系数：{r_robust_it_old:.3f}")

# 基于 df_actual_full｜冻结日真实观测
df_robust_it_tight_actual = df_actual_full[
    (df_actual_full["H100 equivalents"] > 0)
    & (df_actual_full["H100 equivalents"] < 3.2e6)
    & (df_actual_full["IT power (MW)"] < 1400)
]

print("\n" + "="*60)
print(f"【正式｜仅真实观测｜算力—IT 功率】剔除前样本量：{len(df_actual_full)}")
print(f"【正式｜仅真实观测｜算力—IT 功率】收紧阈值稳健子集 df_robust_it_tight_actual 样本量：{len(df_robust_it_tight_actual)}")

plt.figure(figsize=(8, 5))
plt.scatter(df_robust_it_tight_actual["H100 equivalents"], df_robust_it_tight_actual["IT power (MW)"], alpha=0.6)

plt.xlabel("H100 equivalents（等效 H100 总算力）")
plt.ylabel("IT power (MW)（IT 硬件功率 MW）")
plt.title("【冻结日过滤｜仅真实观测｜收紧阈值】等效 H100 总算力 — IT 硬件功率")
plt.grid(alpha=0.3)

plt.savefig("scatter_it_robust_tight_actual.png", dpi=300, bbox_inches="tight")
plt.show()

r_robust_it_new = df_robust_it_tight_actual["H100 equivalents"].corr(df_robust_it_tight_actual["IT power (MW)"], method="pearson")
print(f"【正式｜仅真实观测】收紧阈值后算力与 IT 硬件功率皮尔逊相关系数：{r_robust_it_new:.3f}")

print(df_full.columns)

df_full["每兆瓦 IT 对应算力"] = df_full["H100 equivalents"] / df_full["IT power (MW)"]

# 冻结过滤
df_actual_full["每兆瓦 IT 对应算力"] = df_actual_full["H100 equivalents"] / df_actual_full["IT power (MW)"]

print(sum(df_actual_full["IT power (MW)"] <= 0))

# 含 projection
print("====【对照｜含预测行】各数据中心样本数量 ====")
print(df_full.groupby("Data center").size())

# 冻结日过滤｜仅真实观测
print("\n" + "=" * 60)
print("====【冻结日过滤｜仅真实观测】各数据中心样本数量 ====")
print(df_actual_full.groupby("Data center").size())

# 含 prediction
机房汇总 = df_full.groupby("Data center")["每兆瓦 IT 对应算力"].agg(["count", "mean", "median"])

# 打印结果，方便对比各个机房水平
print("\n====【对照｜含预测行】各机房｜每兆瓦 IT 对应算力 ====")
print(机房汇总)

# 冻结日过滤｜仅真实观测
df_actual_full["每兆瓦 IT 对应算力"] = df_actual_full["H100 equivalents"] / df_actual_full["IT power (MW)"]

机房汇总_actual = df_actual_full.groupby("Data center")["每兆瓦 IT 对应算力"].agg(["count", "mean", "median"])
print("\n" + "=" * 60)
print("====【冻结日过滤｜仅真实观测】各机房｜每兆瓦 IT 对应算力 ====")
print(机房汇总_actual)


机房汇总_actual = df_actual_full.groupby("Data center")["每兆瓦 IT 对应算力"].agg(["count", "median"])
机房汇总_median_sort = 机房汇总_actual.sort_values("median", ascending=True)
plt.figure(figsize=(12, 16), constrained_layout=True)
plt.barh(机房汇总_median_sort.index,
         机房汇总_median_sort["median"],
         alpha=0.65)
整体中位数 = 机房汇总_median_sort["median"].median()
plt.axvline(x = 整体中位数, color = "crimson", linestyle = "--", label = f"全部机房中位数：{整体中位数:.2f}")

plt.xlabel("每兆瓦 IT 对应的算力")
plt.ylabel("数据中心")
plt.title("【冻结日过滤｜仅真实观测】各数据中心 — 每兆瓦 IT 对应算力（中位数，由低到高）")
plt.grid(axis = "x", alpha = 0.3)
plt.legend()

plt.savefig("dc_barh_median_actual.png", dpi = 300)
plt.show()

# 含预测行
数据中心能耗汇总 = df_full.groupby("Data center")[["IT power (MW)", "Power (MW)", "pue"]].agg(["mean"])

print("\n====【对照｜含预测行】各数据中心能耗与 PUE 均值 ====")
print(数据中心能耗汇总)

# 冻结日过滤｜仅真实观测
数据中心能耗汇总_actual = df_actual_full.groupby("Data center")[["IT power (MW)", "Power (MW)", "pue"]].agg(["count", "mean"])
print("\n" + "=" * 60)
print("====【冻结日过滤｜仅真实观测】各数据中心能耗与 PUE ====")
print(数据中心能耗汇总_actual)

# 冻结日过滤｜仅真实观测
# 提取 pue 均值序列
pue_序列_actual = 数据中心能耗汇总_actual["pue"]["mean"]
pue_min_actual = pue_序列_actual.idxmin(), pue_序列_actual.min()
pue_max_actual = pue_序列_actual.idxmax(), pue_序列_actual.max()

print("\n====【冻结日过滤｜仅真实观测】PUE 极值 + 样本量 ====")
print("PUE 最小机房：", pue_min_actual[0], "｜值：", f"{pue_min_actual[1]:.4f}")
print("PUE 最大机房：", pue_max_actual[0], "｜值：", f"{pue_max_actual[1]:.4f}")

极值行 = 数据中心能耗汇总_actual.loc[[pue_min_actual[0], pue_max_actual[0]]]
print("\n==== 极值机房详细信息 ====")
print(极值行)

# 冻结日过滤｜仅真实观测
pue_sort = 数据中心能耗汇总_actual.sort_values(("pue", "mean"), ascending=True)

plt.figure(figsize=(12,16), constrained_layout=True)
plt.barh(pue_sort.index, pue_sort[("pue", "mean")], alpha=0.65)

# 标记最大最小
plt.axvline(x = 1.09375, color = "green", linestyle = "--", label = f"最低 PUE：1.09")
plt.axvline(x = 1.515152, color = "crimson", linestyle = "--", label = f"最高 PUE：1.52")

plt.xlabel("平均 PUE")
plt.ylabel("数据中心")
plt.title("【冻结日过滤｜仅真实观测】各数据中心 — 平均 PUE 对比")
plt.grid(axis = "x", alpha = 0.3)
plt.legend()
plt.savefig("barh_pue_actual.png", dpi = 300)
plt.show()

# 含预测行
plt.figure(figsize=(10, 6), constrained_layout=True)
plt.scatter(df_full["IT power (MW)"], df_full["Power (MW)"], alpha=0.6)
plt.xlabel("IT power (MW)（IT 硬件功率）")
plt.ylabel("Power (MW)（数据中心总功率）")
plt.title("【对照｜含预测行】IT 功率 — 数据中心总功率")
plt.grid(alpha=0.3)
plt.savefig("scatter_it_total_with_proj.png", dpi=300)
plt.show()

# 冻结日过滤｜仅真实观测
plt.figure(figsize=(10, 6), constrained_layout=True)
plt.scatter(df_actual_full["IT power (MW)"], df_actual_full["Power (MW)"], alpha=0.6)

plt.xlabel("IT power (MW)（IT 硬件功率）")
plt.ylabel("Power (MW)（数据中心总功率）")
plt.title("【冻结日过滤｜仅真实观测】IT 功率 — 数据中心总功率")
plt.grid(alpha=0.3)

plt.savefig("scatter_it_total_actual.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6), constrained_layout=True)
plt.scatter(df_actual_full["IT power (MW)"], df_actual_full["Power (MW)"], alpha=0.6)
plt.plot([0, 2100], [0, 2100], color="orange", linestyle="--", label="y=x 参考线")
plt.xlabel("IT power (MW)（IT 硬件功率）")
plt.ylabel("Power (MW)（数据中心总功率）")
plt.title("【冻结日过滤｜仅真实观测】IT 功率 — 数据中心总功率")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("scatter_it_total_ref_actual.png", dpi=300)
plt.show()

import pandas as pd

df_actual_full = pd.read_csv("timeline_cleaned_actual.csv", encoding="utf-8-sig")

med_eff_it = df_actual_full["eff_it"].median()   # 每1 MW IT硬件功率可以扛多少H100e
med_eff_total = df_actual_full["eff_total"].median()  # 每1 MW机房总供电（IT+制冷配电）可以扛多少H100e


H_add = 48.0 * 10**6

add_it_power_mw = H_add / med_eff_it # IT硬件本身耗电
add_total_power_mw = H_add / med_eff_total # 整套机房向电网取电，含制冷配电损耗（总用电需求）

print("===== 预测结果 =====")
print(f"eff_it 中位数：{med_eff_it:.2f}  H100e / MW‑IT")
print(f"eff_total 中位数：{med_eff_total:.2f} H100e / MW‑总功率")
print(f"\n承诺新增等效 H100：{H_add:,}")
print(f"👉 仅 IT 硬件功率：{add_it_power_mw:.2f} MW")
print(f"👉 机房总供电功率：{add_total_power_mw:.2f} MW")

pred_result = pd.DataFrame({
    "项目": [
        "承诺新增 H100e",
        "eff_it 中位数",
        "eff_total 中位数",
        "新增 IT 功率(MW)",
        "新增总功率(MW)"
    ],
    "数值": [
        H_add,
        med_eff_it,
        med_eff_total,
        add_it_power_mw,
        add_total_power_mw
    ]
})
pred_result.to_csv("power_prediction_local.csv", index=False, encoding="utf-8-sig")
print("\n✅ 结果保存为：power_prediction_local.csv")

import pandas as pd
df_actual_full = pd.read_csv("timeline_cleaned_actual.csv", encoding="utf-8-sig")
print("全部列名：")
print(df_actual_full.columns.tolist())

import pandas as pd
import matplotlib.pyplot as plt

df_actual_full = pd.read_csv("timeline_cleaned_actual.csv", encoding="utf-8-sig")

size_col = "H100 equivalents"

corr_size_effit = df_actual_full[[size_col, "eff_it"]].corr().iloc[0, 1]
corr_size_efftotal = df_actual_full[[size_col, "eff_total"]].corr().iloc[0, 1]

print("==== 规模效应检验：算力规模 — 能效相关系数 ====")
print(f"算力规模 — eff_it 相关系数：{corr_size_effit:.3f}")
print(f"算力规模 — eff_total 相关系数：{corr_size_efftotal:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(df_actual_full[size_col], df_actual_full["eff_it"], alpha=0.6)
ax1.set_xlabel("机房算力规模 H100 equivalents")
ax1.set_ylabel("eff_it (H100e / MW‑IT)")
ax1.set_title(f"机房规模 vs IT硬件能效，r = {corr_size_effit:.3f}")

ax2.scatter(df_actual_full[size_col], df_actual_full["eff_total"], alpha=0.6)
ax2.set_xlabel("机房算力规模 H100 equivalents")
ax2.set_ylabel("eff_total (H100e / MW‑总功率)")
ax2.set_title(f"机房规模 vs 机房总能效，r = {corr_size_efftotal:.3f}")

plt.tight_layout()
plt.savefig("scale_efficiency_scatter.png", dpi=300)
plt.show()
print("\n✅ 图表已保存：scale_efficiency_scatter.png")

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_actual_full = pd.read_csv("timeline_cleaned_actual.csv", encoding="utf-8-sig")
size_col = "H100 equivalents"
H_add = 48.0 * 10**6   # 情景假设新增算力

# 1. 按算力中位数划分大小规模组
size_median = df_actual_full[size_col].median()
df_small = df_actual_full[df_actual_full[size_col] <= size_median].copy()
df_large = df_actual_full[df_actual_full[size_col] > size_median].copy()

print(f"全样本算力中位数(分组阈值): {size_median:.1f} H100 equivalents")
print(f"小规模组样本量: {len(df_small)}；大规模组样本量: {len(df_large)}")

# 2. 各组计算能效中位数
med_it_small = df_small["eff_it"].median()
med_total_small = df_small["eff_total"].median()
med_it_all = df_actual_full["eff_it"].median()
med_total_all = df_actual_full["eff_total"].median()
med_it_large = df_large["eff_it"].median()
med_total_large = df_large["eff_total"].median()

# 3. 三情景电力预测计算
scenario = pd.DataFrame({
    "情景": ["保守(小规模)", "基准(全样本)", "乐观(大规模)"],
    "eff_it_med": [med_it_small, med_it_all, med_it_large],
    "eff_total_med": [med_total_small, med_total_all, med_total_large]
})
scenario["IT_power_MW"] = H_add / scenario["eff_it_med"]
scenario["Total_power_MW"] = H_add / scenario["eff_total_med"]

print("\n=====三情景预测结果======")
print(scenario.round(2))
scenario.to_csv("three_scenario_prediction.csv", index=False, encoding="utf-8-sig")
print("\n✅三情景结果保存：three_scenario_prediction.csv")

# 4. 组内相关系数
corr_small_it = df_small[[size_col, "eff_it"]].corr().iloc[0, 1]
corr_small_total = df_small[[size_col, "eff_total"]].corr().iloc[0, 1]
corr_large_it = df_large[[size_col, "eff_it"]].corr().iloc[0, 1]
corr_large_total = df_large[[size_col, "eff_total"]].corr().iloc[0, 1]

print("\n====组内规模‑能效相关系数====")
print(f"小规模组 r(eff_it):{corr_small_it:.3f}, r(eff_total):{corr_small_total:.3f}")
print(f"大规模组 r(eff_it):{corr_large_it:.3f}, r(eff_total):{corr_large_total:.3f}")

# 图1：边际密度散点图
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.08, hspace=0.08)
ax_scatter = fig.add_subplot(gs[1, 0])
ax_xdens = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
ax_ydens = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

ax_scatter.scatter(df_small[size_col], df_small["eff_it"], alpha=0.6, c="#1f77b4", label="小规模机房")
ax_scatter.scatter(df_large[size_col], df_large["eff_it"], alpha=0.6, c="#ff7f0e", label="大规模机房")
ax_scatter.axvline(x=size_median, ls="--", c="gray", label=f"分组阈值 {size_median:.1f}")
ax_scatter.set_xlabel("机房算力规模 H100 equivalents")
ax_scatter.set_ylabel("eff_it (H100e / MW‑IT)")
ax_scatter.legend()

ax_xdens.hist(df_small[size_col], bins=25, alpha=0.5, color="#1f77b4", density=True)
ax_xdens.hist(df_large[size_col], bins=25, alpha=0.5, color="#ff7f0e", density=True)
ax_xdens.set_xticks([])
ax_xdens.set_yticks([])

ax_ydens.hist(df_small["eff_it"], bins=25, alpha=0.5, color="#1f77b4", density=True, orientation="horizontal")
ax_ydens.hist(df_large["eff_it"], bins=25, alpha=0.5, color="#ff7f0e", density=True, orientation="horizontal")
ax_ydens.set_xticks([])
ax_ydens.set_yticks([])

fig.suptitle("机房算力规模与 IT 能效（带边际密度，大小机房分组）", y=0.94)
fig.savefig("scatter_marginal_density.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
print("✅图1已保存 scatter_marginal_density.png")

# 图2：三情景预测柱状对比图
fig2, (ax_bar1, ax_bar2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(scenario))
width = 0.4

ax_bar1.bar(x, scenario["IT_power_MW"], width)
ax_bar1.set_xticks(x)
ax_bar1.set_xticklabels(scenario["情景"])
ax_bar1.set_ylabel("预测 IT 硬件功率 MW")
ax_bar1.set_title("三情景：所需 IT 硬件功率")

ax_bar2.bar(x, scenario["Total_power_MW"], width, color="#2ca02c")
ax_bar2.set_xticks(x)
ax_bar2.set_xticklabels(scenario["情景"])
ax_bar2.set_ylabel("预测机房总供电功率 MW")
ax_bar2.set_title("三情景：机房总供电功率（含制冷配电）")

plt.tight_layout()
fig2.savefig("three_scenario_bar.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig2)
print("✅图2已保存 three_scenario_bar.png")