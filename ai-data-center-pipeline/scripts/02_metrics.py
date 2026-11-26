# -*- coding: utf-8 -*-
"""
02_metrics.py —— 清洗 + 指标计算
关键判断: 时间线表混了两类行
    actual   : Date <= 采集日(2026-10-21)  -> 卫星/许可证据观测到的真实累计状态
    projected: Date >  采集日            -> 未来里程碑(Epoch 的估计/公司公开排期)
观测不可能发生在未来, 所以这个切分是硬边界, 不是人工挑选。

产出:
    data/site_level.csv     站点级指标
    data/annual.csv         年度新增算力: 已兑现 vs 承诺
    data/summary.json       报告要引用的头部数字
"""
from __future__ import print_function
import io, os, json
import numpy as np
import pandas as pd

import _paths

BASE = _paths.BASE
DAT = _paths.RESULTS
REP = _paths.REPORTS
_paths.ensure(DAT, REP)

CUTOFF = pd.Timestamp("2026-10-21")   # 数据集采集日(文件时间)
H100 = "H100 equivalents"
CUR = "Current H100 equivalents (from Data center)"
ITMW = "IT power (MW)"
MW = "Power (MW)"
OPS = "Buildings operational"
M_DAYS = 30.44


def load():
    tl = pd.read_csv(os.path.join(BASE, "data_center_timelines.csv"))
    dc = pd.read_csv(os.path.join(BASE, "data_centers.csv"))
    tl["Date"] = pd.to_datetime(tl["Date"], errors="coerce")
    tl = tl.dropna(subset=["Date"]).sort_values(["Data center", "Date"])
    tl["is_projected"] = tl["Date"] > CUTOFF
    return tl, dc


def site_metrics(tl, dc):
    rows = []
    for name, sub in tl.groupby("Data center"):
        sub = sub.sort_values("Date")
        act = sub[~sub["is_projected"]]
        proj = sub[sub["is_projected"]]

        first_obs = act["Date"].iloc[0] if len(act) else pd.NaT
        last_obs = act["Date"].iloc[-1] if len(act) else pd.NaT
        op = act[act[H100] > 0]
        first_op = op["Date"].iloc[0] if len(op) else pd.NaT

        actual = act[H100].max() if len(act) else np.nan      # 截至采集日的实际规模
        planned = sub[H100].max()                            # 含投影的规划规模

        rec = {
            "site": name,
            "first_obs": first_obs,
            "last_obs": last_obs,
            "n_obs_actual": len(act),
            "n_obs_projected": len(proj),
            "span_months": ((last_obs - first_obs).days / M_DAYS
                            if pd.notna(first_obs) and pd.notna(last_obs) else np.nan),
            "first_op": first_op,
            "actual_h100e": actual,
            "planned_h100e": planned,
            "power_mw_actual": act[MW].max() if len(act) else np.nan,
            "it_power_mw_actual": act[ITMW].max() if len(act) else np.nan,
            "buildings_actual": act[OPS].max() if len(act) else np.nan,
            "last_projected_date": proj["Date"].max() if len(proj) else pd.NaT,
            "is_operational": bool(pd.notna(first_op)),
        }
        # 从首次观测到首次投运的滞后(月); 未投运者用观察跨度作为下界
        rec["lag_months"] = ((first_op - first_obs).days / M_DAYS
                             if pd.notna(first_op) else np.nan)
        rec["pending_h100e"] = planned - (actual if pd.notna(actual) else 0)
        site = dc[dc["Name"] == name]
        rec["owner"] = site["Owner"].iloc[0] if len(site) else None
        rec["country"] = site["Country"].iloc[0] if len(site) else None
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["realization"] = df["actual_h100e"] / df["planned_h100e"]
    df["realization"] = df["realization"].replace([np.inf, -np.inf], np.nan)
    return df.sort_values("planned_h100e", ascending=False)


def annual_series(tl):
    """年度新增算力: 已兑现 vs 承诺, 单位 H100e。

    做法: 先构造 (站点 x 年份) 的"年末累计规模"矩阵, 逐年结转(forward fill),
    再横向求和。只在有观测/排期的年份取该站点当年级别的最大值。
    """
    years = list(range(2018, 2031))
    sites = sorted(tl["Data center"].unique())
    act_mat = pd.DataFrame(index=sites, columns=years, dtype=float)
    all_mat = pd.DataFrame(index=sites, columns=years, dtype=float)

    for name, sub in tl.groupby("Data center"):
        sub = sub.sort_values("Date")
        a_y = sub[~sub["is_projected"]].groupby(
            sub[~sub["is_projected"]]["Date"].dt.year)[H100].max()
        p_y = sub[sub["is_projected"]].groupby(
            sub[sub["is_projected"]]["Date"].dt.year)[H100].max()
        for y in years:
            if y in a_y.index:
                act_mat.loc[name, y] = a_y[y]
            if y in a_y.index or y in p_y.index:
                all_mat.loc[name, y] = max(a_y.get(y, np.nan) if y in a_y.index else -1,
                                           p_y.get(y, np.nan) if y in p_y.index else -1)

    # 逐年结转: 用截至该年已知的最高累计值(避免数据修订造成负增长)
    act_mat = act_mat.ffill(axis=1).fillna(0.0)
    all_mat = all_mat.ffill(axis=1).fillna(0.0)
    for y in years:                                   # 单调化
        if y > years[0]:
            act_mat[y] = np.maximum(act_mat[y], act_mat[y - 1])
            all_mat[y] = np.maximum(all_mat[y], all_mat[y - 1])

    cum_a = act_mat.sum(axis=0)
    cum_p = all_mat.sum(axis=0)
    rows = []
    for i, y in enumerate(years):
        a, p = float(cum_a[y]), float(cum_p[y])
        prev_a = float(cum_a[years[i - 1]]) if i else 0.0
        prev_p = float(cum_p[years[i - 1]]) if i else 0.0
        rows.append({
            "year": y,
            "cum_actual_h100e": a,
            "cum_committed_h100e": p,
            "add_actual_h100e": max(a - prev_a, 0.0),
            "add_committed_h100e": max(p - prev_p, 0.0),
            "gap_h100e": p - a,
        })
    return pd.DataFrame(rows)


def main():
    tl, dc = load()
    site = site_metrics(tl, dc)
    ann = annual_series(tl)
    site.to_csv(os.path.join(DAT, "site_level.csv"), index=False, encoding="utf-8-sig")
    ann.to_csv(os.path.join(DAT, "annual.csv"), index=False, encoding="utf-8-sig")

    op = site[site["is_operational"]]
    nonop = site[~site["is_operational"]]
    summ = {
        "cutoff": str(CUTOFF.date()),
        "n_sites": int(len(site)),
        "n_operational": int(len(op)),
        "n_not_operational": int(len(nonop)),
        "actual_h100e_total": float(site["actual_h100e"].sum()),
        "planned_h100e_total": float(site["planned_h100e"].sum()),
        "pending_h100e_total": float(site["planned_h100e"].sum() - site["actual_h100e"].sum()),
        "realization_rate": float(site["actual_h100e"].sum() / site["planned_h100e"].sum()),
        "actual_power_mw_total": float(site["power_mw_actual"].sum()),
        "planned_power_mw_total": float(tl[MW].max() and site["power_mw_actual"].sum()),
        "lag_median_months": float(op["lag_months"].median()),
        "lag_p10_months": float(op["lag_months"].quantile(.1)),
        "lag_p90_months": float(op["lag_months"].quantile(.9)),
        "lag_max_months": float(op["lag_months"].max()),
        "nonoperational_span_median": float(nonop["span_months"].median()),
        "median_realization": float(site["realization"].median()),
        "actual_add_2024_2026": float(ann[ann["year"].between(2024, 2026)]["add_actual_h100e"].sum()),
        "committed_add_2027_2030": float(ann[ann["year"].between(2027, 2030)]["add_committed_h100e"].sum()),
    }
    with io.open(os.path.join(DAT, "summary.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(summ, ensure_ascii=False, indent=2))

    print("站点 %d | 已投运 %d | 零投运 %d"
          % (summ["n_sites"], summ["n_operational"], summ["n_not_operational"]))
    print("实际合计 %.0f H100e | 规划合计 %.0f H100e | 兑现率 %.1f%%"
          % (summ["actual_h100e_total"], summ["planned_h100e_total"],
             summ["realization_rate"] * 100))
    print("滞后中位 %.1f 月 | 零投运站点观察跨度中位 %.1f 月"
          % (summ["lag_median_months"], summ["nonoperational_span_median"]))
    print("2024-2026 实际新增 %.0f H100e | 2027-2030 承诺新增 %.0f H100e"
          % (summ["actual_add_2024_2026"], summ["committed_add_2027_2030"]))
    print()
    print(ann.to_string(index=False))


if __name__ == "__main__":
    main()
