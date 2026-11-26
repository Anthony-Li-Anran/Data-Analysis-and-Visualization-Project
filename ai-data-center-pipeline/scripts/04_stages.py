# -*- coding: utf-8 -*-
"""
04_stages.py —— 补 PDF「主要研究内容」里还没落地的三块:
    (1) 建设状态分类            -> data/processed/stage_counts.csv
    (2) 规划->建设->投运 时间线  -> figures/data_centers/fig4_timeline.png
    (3) 项目差异的分组维度      -> data/processed/group_compare.csv + fig5_groups.png

口径继承 02_metrics.py:
    actual    : Date <= 冻结日 的观测行
    planned   : 该站点全部行(含 projection)的最高累计规模
    delivery  : actual / planned, 只对"存在 projection 行"的站点有意义
                （无 projection 的站点 planned 恒等于 actual, 兑现率=100% 属构造性, 需单列）
"""
from __future__ import print_function
import io, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import _paths
import _stage

BASE = _paths.BASE
DAT = _paths.RESULTS
_paths.ensure(DAT)

CUTOFF = pd.Timestamp("2026-10-21")
H100 = "H100 equivalents"
ITMW = "IT power (MW)"
MW = "Power (MW)"
OPS = "Buildings operational"
M_DAYS = 30.44

for cand in ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DengXian"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
                     "axes.titlesize": 15, "axes.labelsize": 12,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                     "legend.fontsize": 11, "figure.dpi": 150})


def load():
    tl = pd.read_csv(os.path.join(BASE, "data_center_timelines.csv"))
    tl["Date"] = pd.to_datetime(tl["Date"], errors="coerce")
    tl = tl.dropna(subset=["Date"]).sort_values(["Data center", "Date"])
    tl["proj"] = tl["Date"] > CUTOFF
    dc = pd.read_csv(os.path.join(BASE, "data_centers.csv"))
    return tl, dc


# ---------------------------------------------------------------- (1) 状态分类
def stage_table(tl, site):
    rows = []
    for name, sub in tl.groupby("Data center"):
        act = sub[~sub["proj"]]
        plan = sub[H100].max()
        st = _stage.classify(act, plan)
        p = site[site["site"] == name].iloc[0]
        rows.append({"site": name, "stage": st, "planned_h100e": plan,
                     "actual_h100e": p["actual_h100e"],
                     "has_projection": p["n_obs_projected"] > 0,
                     "delivery": p["realization"],
                     "lag_months": p["lag_months"],
                     "span_months": p["span_months"],
                     "country": p["country"], "owner": p["owner"]})
    d = pd.DataFrame(rows)
    order = ["已投运（达产）", "部分投运", "在建（已有机电/建筑）", "在建（土建阶段）", "前期/土地平整", "无观测"]
    summ = (d.groupby("stage")
             .agg(站点数=("site", "size"),
                  承诺容量_M=("planned_h100e", lambda s: s.sum() / 1e6),
                  已投运_M=("actual_h100e", lambda s: s.sum() / 1e6),
                  观察跨度中位月=("span_months", "median"))
             .reindex([o for o in order if o in set(d["stage"])]))
    summ["容量占比"] = summ["承诺容量_M"] / (d["planned_h100e"].sum() / 1e6)
    return d, summ


# ---------------------------------------------------------------- (2) 时间线
def fig_timeline(tl, site, topn=22):
    import matplotlib.dates as mdates

    def num(x):
        return mdates.date2num(pd.Timestamp(x).to_pydatetime())

    d = site.nlargest(topn, "planned_h100e").iloc[::-1]
    fig, ax = plt.subplots(figsize=(12.4, 8.2))
    for i, (_, r) in enumerate(d.iterrows()):
        sub = tl[tl["Data center"] == r["site"]]
        act = sub[~sub["proj"]]
        proj = sub[sub["proj"]]
        d0, d1 = num(act["Date"].min()), num(act["Date"].max())
        if pd.notna(r["first_op"]):
            op = num(r["first_op"])
            ax.plot([d0, op], [i, i], color="#b8c4d0", lw=6,
                    solid_capstyle="butt", zorder=2)
            ax.plot([op, d1], [i, i], color="#2b6cb0", lw=6,
                    solid_capstyle="butt", zorder=3)
            ax.plot([op], [i], "o", color="#2b6cb0", ms=7, zorder=5)
        else:
            ax.plot([d0, d1], [i, i], color="#d9a19a", lw=6,
                    solid_capstyle="butt", zorder=2)
        if len(proj):
            pe = num(proj["Date"].max())
            ax.plot([d1, pe], [i, i], color="#c0392b", lw=1.8,
                    ls=(0, (3, 2)), zorder=4)
            ax.plot([pe], [i], ">", color="#c0392b", ms=6, zorder=5)

    ax.axvline(num(CUTOFF), color="#555555", ls="--", lw=1.2, zorder=6)
    ax.text(num(CUTOFF) + 30, len(d) - 0.4, " 数据冻结日 2026-10-21", fontsize=9.5,
            color="#555555", va="top")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([s if len(s) <= 26 else s[:25] + "…" for s in d["site"]],
                       fontsize=9.5)
    ax.set_xlim(num("2021-06-01"), num("2029-12-31"))
    ax.set_ylim(-0.8, len(d) - 0.15)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("数据中心：从开工到投运，以及尚未兑现的承诺", loc="left", pad=14)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#b8c4d0", lw=6, label="建设期（首次观测→首次投运）"),
        Line2D([0], [0], color="#2b6cb0", lw=6, label="已投运（首次投运→当前）"),
        Line2D([0], [0], color="#d9a19a", lw=6, label="零投运（仅在建/前期）"),
        Line2D([0], [0], color="#c0392b", lw=1.8, ls="--",
               label="未来承诺进度（projection）"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.055),
              frameon=False, ncol=2, fontsize=10)
    fig.text(0.008, 0.012,
             "数据：Epoch AI Data Center Timelines（%.0f-10-21）。仅画承诺容量最大的 %d 个站点。"
             % (CUTOFF.year, topn), fontsize=9, color="#666666")
    fig.subplots_adjust(left=0.245, right=0.975, top=0.925, bottom=0.165)
    out = _paths.save(fig, "04")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- (3) 分组差异
OWNER_MAP = [
    (r"openai|stargate", "OpenAI"),
    (r"microsoft", "Microsoft"),
    (r"google", "Google"),
    (r"amazon|aws", "Amazon"),
    (r"meta", "Meta"),
    (r"oracle", "Oracle（含代建 Stargate）"),
    (r"coreweave", "CoreWeave"),
    (r"spacex|xa?i\b", "xAI/SpaceX"),
]


def owner_clean(s):
    if pd.isna(s):
        return "未披露"
    s = re.sub(r"#\w+", "", str(s)).strip()
    low = s.lower()
    for pat, lab in OWNER_MAP:
        if re.search(pat, low):
            return lab
    first = s.split(",")[0].strip()
    return first or "未披露"


def fig_owner(grp, topn=10):
    g = grp.nlargest(topn, "承诺容量_M").iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    y = np.arange(len(g))
    ax.barh(y, g["已投运_M"], color="#2b6cb0", height=0.66, zorder=3, label="已投运")
    ax.barh(y, g["承诺容量_M"] - g["已投运_M"], left=g["已投运_M"],
            color="#f3bdb7", height=0.66, zorder=2, label="承诺但未投运")
    ax.set_yticks(y)
    ax.set_yticklabels(g.index, fontsize=10.5)
    for i, (a, t) in enumerate(zip(g["已投运_M"], g["承诺容量_M"])):
        ax.text(t + 0.12, i, "%.0f%%" % (100 * a / t if t else 0), va="center",
                fontsize=10.5, color="#333333")
    ax.set_xlabel("容量（百万 H100 等效）")
    ax.set_xlim(0, g["承诺容量_M"].max() * 1.14)
    ax.set_title("按业主类别：承诺 vs 已投运（前 %d 类）" % topn, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.008, 0.012,
             "数据：Epoch AI。业主取 `Owner` 字段并剥离置信度后缀；右侧为容量加权兑现率。",
             fontsize=9, color="#666666")
    fig.subplots_adjust(left=0.26, right=0.975, top=0.92, bottom=0.115)
    out = _paths.save(fig, "05")
    plt.close(fig)
    return out


def cohort_lag(site):
    d = site[site["is_operational"] & site["lag_months"].notna()].copy()
    d["cohort"] = d["first_obs"].astype(str).str[:4]
    out = (d.groupby("cohort")
             .agg(n=("site", "size"),
                  滞后中位=("lag_months", "median"),
                  滞后均值=("lag_months", "mean"),
                  承诺容量_M=("planned_h100e", lambda s: s.sum() / 1e6)))
    return out


def main():
    tl, dc = load()
    site = pd.read_csv(os.path.join(DAT, "site_level.csv"))
    st_df, st_summ = stage_table(tl, site)
    st_df["owner_clean"] = st_df["owner"].map(owner_clean)

    print("=== (1) 建设状态分类 ===")
    print(st_summ.to_string(float_format=lambda v: "%.2f" % v))
    st_summ.to_csv(os.path.join(DAT, "stage_counts.csv"), encoding="utf-8-sig")

    print()
    print("=== 兑现率的构造性问题 ===")
    hp = st_df[st_df["has_projection"]]
    np_ = st_df[~st_df["has_projection"]]
    print("有 projection 的站点: %d 个, 兑现率中位 %.1f%%" % (len(hp), 100 * hp["delivery"].median()))
    print("无 projection 的站点: %d 个, 兑现率中位 %.1f%%（planned==actual, 属构造性 100%%）"
          % (len(np_), 100 * np_["delivery"].median()))
    print("无 projection 站点里实际达产(=承诺)的: %d 个" % int((np_["delivery"] >= 0.999).sum()))
    print()
    op = st_df[(st_df["actual_h100e"] > 0) & st_df["has_projection"]]
    print("已投运且有扩张排期的站点 %d 个, 其兑现率中位 %.1f%%, 均值 %.1f%%"
          % (len(op), 100 * op["delivery"].median(), 100 * op["delivery"].mean()))

    # 分组
    grp = (st_df.groupby("owner_clean")
             .agg(站点数=("site", "size"),
                  承诺容量_M=("planned_h100e", lambda s: s.sum() / 1e6),
                  已投运_M=("actual_h100e", lambda s: s.sum() / 1e6)))
    grp["兑现率"] = grp["已投运_M"] / grp["承诺容量_M"]
    grp = grp.sort_values("承诺容量_M", ascending=False)
    print()
    print("=== 按业主类别 ===")
    print(grp.to_string(float_format=lambda v: "%.2f" % v))
    grp.to_csv(os.path.join(DAT, "group_compare.csv"), encoding="utf-8-sig")

    print()
    print("=== 按首次观测年份 cohort（已投运站点） ===")
    co = cohort_lag(site)
    print(co.to_string(float_format=lambda v: "%.1f" % v))

    print()
    print("=== 按国家（样本内） ===")
    ct = (st_df.groupby("country")
            .agg(站点数=("site", "size"),
                 承诺容量_M=("planned_h100e", lambda s: s.sum() / 1e6),
                 已投运_M=("actual_h100e", lambda s: s.sum() / 1e6)))
    ct["兑现率"] = ct["已投运_M"] / ct["承诺容量_M"]
    print(ct.sort_values("站点数", ascending=False).to_string(float_format=lambda v: "%.2f" % v))

    out1 = fig_timeline(tl, site)
    out2 = fig_owner(grp)
    print()
    print("[ok]", out1)
    print("[ok]", out2)
    st_df.to_csv(os.path.join(DAT, "site_stage.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
